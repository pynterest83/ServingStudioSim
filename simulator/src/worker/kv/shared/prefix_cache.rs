//! Session-scoped retained KV for autoregressive prefix reuse.
//!
//! Entries are count-only because traces carry token counts, not token content.
//! A lookup is destructive: retained KV ownership moves to the admitted request,
//! so this cache never models inter-request sharing or ref-counted prefix pages.
//!
//! ## Reuse quantum
//!
//! Full-attention KV is per-token, so any prefix length is reusable — that is a
//! quantum of one. A recurrent state is a single rolling snapshot instead, only
//! written at multiples of the model's checkpoint interval, so its reuse
//! quantizes to that interval and each retained entry additionally carries one
//! `charge_per_checkpoint` for every snapshot it keeps. Both knobs are
//! constructor parameters; `(1, 0)` is exactly the pre-hybrid behavior.
//!
//! ## Which boundaries survive
//!
//! [`CheckpointRetention::EveryQuantum`] (the default) keeps a snapshot at
//! every quantum boundary of the retained length, and any of them can serve a
//! later request. [`CheckpointRetention::EagleReplayBoundaries`] instead keeps
//! only the few positions vLLM retains for a speculating hybrid with the EAGLE
//! block drop (two per lookup granularity): see that variant for the rule. Only one entry is kept
//! per session (the latest completed request's), so checkpoints an earlier
//! turn left behind are never resumed from, even where vLLM's LRU pool might
//! still hold them.

use std::collections::{BTreeMap, HashMap};

use serde::Deserialize;

/// Whether completed-session KV may be retained after a request finishes.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum PrefixCacheMode {
    /// Drop completed-session KV. This is an explicit no-reuse baseline.
    Disabled,
    /// Retain into whatever physical attention slack is currently available.
    #[default]
    Opportunistic,
}

/// Victim selection when retained session KV must be evicted.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum PrefixCachePolicy {
    /// Hits refresh recency; the least recently used session is evicted first.
    #[default]
    Lru,
    /// Hits do not refresh order; the oldest inserted session is evicted first.
    Fifo,
    /// The least frequently consumed session is evicted, with age as the tie-break.
    Lfu,
    /// The largest retained session is evicted, with age as the tie-break.
    LargestFirst,
}

/// Validated runtime configuration for the retained-session tier.
///
/// The default has no carved-out capacity: completed prefixes may occupy all
/// physical attention slack, while active/promised/held KV remains strictly
/// higher priority. `max_retained_bytes` is only an optional experiment ceiling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefixCacheConfig {
    Disabled,
    Opportunistic {
        policy: PrefixCachePolicy,
        max_retained_bytes: Option<u64>,
    },
}

impl Default for PrefixCacheConfig {
    fn default() -> Self {
        Self::Opportunistic {
            policy: PrefixCachePolicy::Lru,
            max_retained_bytes: None,
        }
    }
}

/// Construction-time token-space form consumed by `FullAttnKv`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PrefixCacheTokenConfig {
    pub(crate) max_retained_tokens: u64,
    pub(crate) policy: PrefixCachePolicy,
}

impl PrefixCacheConfig {
    /// Convert the optional byte ceiling with the same model KV layout used for
    /// total attention capacity. An uncapped opportunistic cache resolves to the
    /// full capacity; runtime physical slack still shrinks it below that value.
    pub(crate) fn resolve_tokens(
        self,
        total_capacity_tokens: u64,
        total_kv_bytes_per_token: u64,
        num_attn_shards: u16,
    ) -> PrefixCacheTokenConfig {
        match self {
            Self::Disabled => PrefixCacheTokenConfig {
                max_retained_tokens: 0,
                policy: PrefixCachePolicy::Lru,
            },
            Self::Opportunistic {
                policy,
                max_retained_bytes,
            } => {
                let max_retained_tokens = max_retained_bytes
                    .map(|bytes| {
                        bytes.saturating_mul(u64::from(num_attn_shards.max(1)))
                            / total_kv_bytes_per_token.max(1)
                    })
                    .unwrap_or(total_capacity_tokens)
                    .min(total_capacity_tokens);
                PrefixCacheTokenConfig {
                    max_retained_tokens,
                    policy,
                }
            }
        }
    }
}

/// Which positions of a completed request a later request can resume from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum CheckpointRetention {
    /// Every quantum boundary of the retained length, one charged snapshot
    /// each. At quantum one this is plain per-token full-attention reuse.
    #[default]
    EveryQuantum,
    /// vLLM v0.31's hybrid prefix caching with `mamba_cache_mode = "align"`,
    /// an EAGLE/MTP drafter (`use_eagle_block_drop`) and no retention interval.
    ///
    /// States are written only inside the prompt, so a completed request of
    /// prompt length `P` (fresh plus declared prefix, output excluded) retains
    /// the replay boundaries of `KVCacheCoordinator.get_replay_boundaries`,
    /// with `Q` the quantum:
    ///
    /// ```text
    /// resend    = floor((P - 1) / Q) * Q - Q     (the same prompt sent again)
    /// extension = floor(P / Q) * Q - Q           (a longer sibling)
    /// ```
    ///
    /// each clamped at 0, one charged state per distinct nonzero boundary.
    /// The attention KV kept is the block above the higher boundary, which the
    /// EAGLE lookup matches and drops back from. A later request whose lookup
    /// may reach `M = min(declared prefix, its prompt - 1)` tokens resumes at
    /// the largest retained boundary not above `floor(M / Q) * Q - Q`
    /// (`find_longest_cache_hit`), or recomputes everything. Output tokens
    /// are therefore always recomputed by the next turn.
    ///
    /// A match unit `u` finer than `Q` ([`PrefixCache::with_match_unit`],
    /// vLLM's `--prefix-match-unit`) adds the same two boundaries at `u`
    /// granularity, `floor((P - 1) / u) * u - u` and `floor(P / u) * u - u`
    /// (the scheduler's `tail_boundary` chunk stop writes that state), and the
    /// lookup candidate becomes `floor(M / u) * u - u`; the `Q` boundaries
    /// stay. Each distinct nonzero boundary is one charged state, and the
    /// attention KV kept extends to the `Q`-page holding the unit above the
    /// highest boundary (vLLM caches the prompt's partial tail page,
    /// `_cache_partial_tail_block`). At `u = Q` both sets coincide.
    EagleReplayBoundaries,
}

/// When a completed request's resumable state enters the evictable tier, and
/// how that tier picks victims.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum CheckpointRelease {
    /// The whole entry, context KV and snapshots, is retained when the request
    /// completes and is evicted as one unit by the replacement policy.
    #[default]
    OnCompletion,
    /// vLLM v0.31's block pool under `mamba_cache_mode = "align"`.
    ///
    /// `MambaManager.remove_skipped_blocks` frees a prompt's boundary state
    /// blocks into the free queue once the request computes past them, at the
    /// end of its prefill, while the attention blocks return only when the
    /// request finishes (`single_type_kv_cache_manager.py` `free`, reversed).
    /// The free queue is one LRU in free order (`block_pool.py` `free_blocks`
    /// appends cached blocks, `get_new_blocks` pops the head), so an entry is
    /// two segments: its snapshots, stamped at the prefill end
    /// ([`PrefixCache::insert_checkpoints`]), and its context KV, stamped at
    /// completion ([`PrefixCache::complete_context`]). Eviction takes the
    /// oldest segment. Losing the snapshots loses the hit; the context KV of
    /// such a dead entry stays in the queue at its own position as an orphan
    /// segment that only occupies capacity until it is reached. Only the LRU
    /// policy is defined in this mode.
    AtPrefillEnd,
    /// What-if: freed at the prefill end exactly as [`Self::AtPrefillEnd`],
    /// but snapshots still cached when the request completes move to the
    /// completion position together with its context KV, as if the engine
    /// re-freed (touched) them then. Snapshots evicted before completion stay
    /// lost. Only the LRU policy is defined in this mode.
    AtPrefillEndRefreshedAtCompletion,
}

impl CheckpointRelease {
    /// Snapshots enter the tier at the prefill end, as a segment of their own.
    pub(crate) fn at_prefill_end(self) -> bool {
        matches!(
            self,
            Self::AtPrefillEnd | Self::AtPrefillEndRefreshedAtCompletion
        )
    }
}

/// The resumable positions of one entry under
/// [`CheckpointRetention::EagleReplayBoundaries`]: the resend and extension
/// boundaries at the quantum, then at the match unit. Zero means no state.
type ReplayBoundaries = [u64; 4];

#[derive(Clone, Copy, Debug)]
struct PrefixEntry {
    /// Retained context KV, the occupancy half of the entry.
    tokens: u64,
    /// Charged snapshots this entry keeps.
    checkpoints: u64,
    /// Replay boundaries under
    /// [`CheckpointRetention::EagleReplayBoundaries`]; `None` when every
    /// quantum boundary up to `tokens` is resumable.
    replay_boundaries: Option<ReplayBoundaries>,
    insertion_sequence: u64,
    last_access_sequence: u64,
    frequency: u64,
    /// [`CheckpointRelease::AtPrefillEnd`] only: the snapshots are retained but
    /// the context KV is still owned by the running request, so `tokens` is
    /// not yet charged and the entry cannot serve a hit.
    context_pending: bool,
    /// When the snapshot segment entered the tier. Equals
    /// `last_access_sequence` under [`CheckpointRelease::OnCompletion`].
    checkpoint_sequence: u64,
}

/// The next segment the replacement policy gives up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Victim {
    /// A live entry; under [`CheckpointRelease::AtPrefillEnd`] its snapshot
    /// segment, which kills the entry and orphans its context KV.
    Entry(u32),
    /// The context KV of a dead entry, keyed by its sequence.
    Orphan(u64),
}

/// Replacement metadata that follows a destructively consumed session until
/// its active request returns KV to the cache.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PrefixCacheReturnMetadata {
    insertion_sequence: u64,
    frequency: u64,
}

/// One authoritative cache-entry transition returned to the KV owner.
///
/// Logging consumes these receipts instead of reconstructing mutations from
/// sampled occupancy. That keeps `PrefixCache` as the only cache ledger while
/// still exposing enough information for an exact operation replay.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PrefixCacheMutation {
    pub(crate) kind: PrefixCacheMutationKind,
    pub(crate) session_id: u32,
    pub(crate) entry_tokens_before: u64,
    pub(crate) entry_tokens_after: u64,
    pub(crate) cache_used_before: u64,
    pub(crate) cache_used_after: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PrefixCacheMutationKind {
    SameSessionReplacement,
    CapacityEviction,
    ReplacementPolicyEviction,
    Retain,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PrefixCacheTakeResult {
    pub(crate) hit_tokens: u32,
    pub(crate) entry_tokens: u64,
    pub(crate) cache_used_before: u64,
    pub(crate) cache_used_after: u64,
    pub(crate) return_metadata: Option<PrefixCacheReturnMetadata>,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct PrefixCacheInsertResult {
    pub(crate) mutations: Vec<PrefixCacheMutation>,
    pub(crate) retained_tokens: u64,
}

/// What one insert keeps: context KV, charged snapshots and, under
/// [`CheckpointRetention::EagleReplayBoundaries`], the resumable positions.
struct RetainedSpan {
    tokens: u64,
    checkpoints: u64,
    replay_boundaries: Option<ReplayBoundaries>,
}

/// One attention partition's evictable retained-session tier.
pub(crate) struct PrefixCache {
    max_retained_tokens: u64,
    used_tokens: u64,
    /// Snapshots behind `used_tokens`, tracked incrementally so `used_charge`
    /// stays O(1) inside the eviction loops.
    used_checkpoints: u64,
    policy: PrefixCachePolicy,
    /// Reuse granularity in context tokens. `1` = any prefix length is reusable.
    quantum_tokens: u64,
    /// Extra occupancy every retained snapshot costs on top of its context KV.
    charge_per_checkpoint: u64,
    retention: CheckpointRetention,
    /// Lookup granularity under [`CheckpointRetention::EagleReplayBoundaries`];
    /// divides `quantum_tokens`, and equals it unless a finer unit is set.
    match_unit_tokens: u64,
    entries: HashMap<u32, PrefixEntry>,
    next_sequence: u64,
    release: CheckpointRelease,
    /// [`CheckpointRelease::AtPrefillEnd`] only: context KV of dead entries,
    /// by the sequence it entered the tier at, and its total.
    orphans: BTreeMap<u64, u64>,
    orphan_tokens: u64,
}

impl PrefixCache {
    pub(crate) fn new(
        max_retained_tokens: u64,
        policy: PrefixCachePolicy,
        quantum_tokens: u32,
        charge_per_checkpoint: u64,
    ) -> Self {
        Self {
            max_retained_tokens,
            used_tokens: 0,
            used_checkpoints: 0,
            policy,
            quantum_tokens: u64::from(quantum_tokens.max(1)),
            charge_per_checkpoint,
            retention: CheckpointRetention::EveryQuantum,
            match_unit_tokens: u64::from(quantum_tokens.max(1)),
            entries: HashMap::new(),
            next_sequence: 0,
            release: CheckpointRelease::OnCompletion,
            orphans: BTreeMap::new(),
            orphan_tokens: 0,
        }
    }

    /// Select when snapshots enter the tier and how victims are picked (see
    /// [`CheckpointRelease`]). Set it before the first insert.
    pub(crate) fn with_checkpoint_release(mut self, release: CheckpointRelease) -> Self {
        debug_assert!(self.entries.is_empty());
        assert!(
            release == CheckpointRelease::OnCompletion || self.policy == PrefixCachePolicy::Lru,
            "checkpoint release at the prefill end is defined only for the LRU policy"
        );
        self.release = release;
        self
    }

    /// Select which boundaries of a completed request survive. Changes the
    /// meaning of [`Self::insert`]'s `tokens`; set it before the first insert.
    pub(crate) fn with_checkpoint_retention(mut self, retention: CheckpointRetention) -> Self {
        debug_assert!(self.entries.is_empty());
        self.retention = retention;
        self
    }

    /// Match prefixes at `match_unit_tokens`, a divisor of the quantum, under
    /// [`CheckpointRetention::EagleReplayBoundaries`] (see that variant). The
    /// quantum itself is the default. Set it before the first insert.
    pub(crate) fn with_match_unit(mut self, match_unit_tokens: u32) -> Self {
        let unit = u64::from(match_unit_tokens);
        assert!(
            unit > 0 && self.quantum_tokens.is_multiple_of(unit),
            "prefix match unit {unit} must divide the reuse quantum {}",
            self.quantum_tokens
        );
        debug_assert!(self.entries.is_empty());
        self.match_unit_tokens = unit;
        self
    }

    /// Context tokens held. Deliberately test-only: every production decision
    /// is about occupancy, which is [`Self::used_charge`]. The two differ once
    /// retained snapshots carry a charge of their own.
    #[cfg(test)]
    pub(crate) fn used_tokens(&self) -> u64 {
        self.used_tokens
    }

    /// What this tier actually occupies: retained context KV plus one charge per
    /// retained snapshot. Equals [`Self::used_tokens`] when there is no per-
    /// checkpoint charge.
    pub(crate) fn used_charge(&self) -> u64 {
        self.used_tokens + self.used_checkpoints * self.charge_per_checkpoint + self.orphan_tokens
    }

    /// Round a length down to the coarsest boundary this cache can resume from.
    fn floor_to_quantum(&self, tokens: u64) -> u64 {
        tokens / self.quantum_tokens * self.quantum_tokens
    }

    /// The largest quantized length not exceeding `tokens` whose charge fits
    /// `capacity`. Each retained snapshot costs `quantum + charge_per_checkpoint`,
    /// so the count is a plain division; at quantum one with no charge this is
    /// `min(tokens, capacity)`, the pre-hybrid rule.
    fn largest_fitting_length(&self, tokens: u64, capacity: u64) -> u64 {
        let per_checkpoint = self.quantum_tokens + self.charge_per_checkpoint;
        let checkpoints = (tokens / self.quantum_tokens).min(capacity / per_checkpoint);
        checkpoints * self.quantum_tokens
    }

    /// Pure preview used by admission's token-budget and capacity gates.
    ///
    /// A request declaring `requested_tokens` can only resume at a boundary it
    /// shares with the retained entry, so the hit is the smaller of the two
    /// quantized lengths. At quantum one this is the plain `min`.
    ///
    /// Under [`CheckpointRetention::EagleReplayBoundaries`] `requested_tokens`
    /// is the longest prefix the engine's lookup may reach (the caller caps the
    /// declared prefix at the prompt minus its last token), and the hit is the
    /// replay boundary that variant describes.
    pub(crate) fn peek(&self, session_id: u32, requested_tokens: u32) -> u32 {
        self.entries
            .get(&session_id)
            .filter(|entry| !entry.context_pending)
            .map(|entry| match entry.replay_boundaries {
                None => entry
                    .tokens
                    .min(self.floor_to_quantum(u64::from(requested_tokens)))
                    as u32,
                Some(boundaries) => {
                    // The attention lookup matches whole match units and drops
                    // the last one; the recurrent finder then needs a retained
                    // state at or below that candidate.
                    let unit = self.match_unit_tokens;
                    let candidate =
                        (u64::from(requested_tokens) / unit * unit).saturating_sub(unit);
                    boundaries
                        .into_iter()
                        .filter(|&boundary| boundary <= candidate)
                        .max()
                        .unwrap_or(0) as u32
                }
            })
            .unwrap_or(0)
    }

    /// Move one retained session into an active request.
    ///
    /// The whole entry is removed even when the request declares a shorter
    /// prefix: count-only traces cannot prove that the unused remainder is a
    /// separately reusable page range.
    pub(crate) fn take(&mut self, session_id: u32, requested_tokens: u32) -> PrefixCacheTakeResult {
        let cache_used_before = self.used_tokens;
        let hit_tokens = self.peek(session_id, requested_tokens);
        let removed_entry = self.remove(session_id);
        let entry_tokens = removed_entry.map(|entry| entry.tokens).unwrap_or(0);
        let return_metadata = removed_entry.map(|entry| PrefixCacheReturnMetadata {
            insertion_sequence: entry.insertion_sequence,
            frequency: entry.frequency.saturating_add(1),
        });
        PrefixCacheTakeResult {
            hit_tokens,
            entry_tokens,
            cache_used_before,
            cache_used_after: self.used_tokens,
            return_metadata,
        }
    }

    /// Retain a completed request's physically resident KV.
    ///
    /// `tokens` is the request's resident context under
    /// [`CheckpointRetention::EveryQuantum`] and its prompt length (fresh plus
    /// declared prefix) under [`CheckpointRetention::EagleReplayBoundaries`].
    /// `physical_limit` is the current slack left by active/promised/held KV in
    /// the partition. An optional experiment ceiling was already resolved into
    /// `max_retained_tokens`; the inserted entry is clamped to the smaller limit.
    pub(crate) fn insert(
        &mut self,
        session_id: u32,
        tokens: u64,
        physical_limit: u64,
        return_metadata: Option<PrefixCacheReturnMetadata>,
    ) -> PrefixCacheInsertResult {
        let mut mutations = Vec::new();
        if let Some(mutation) =
            self.remove_with_receipt(session_id, PrefixCacheMutationKind::SameSessionReplacement)
        {
            mutations.push(mutation);
        }
        let effective_capacity = self.max_retained_tokens.min(physical_limit);
        mutations.extend(self.shrink_to(effective_capacity));
        let retained = self.retained_span(tokens, effective_capacity);
        let retained_tokens = retained.tokens;
        if retained_tokens == 0 {
            return PrefixCacheInsertResult {
                mutations,
                retained_tokens,
            };
        }
        let retained_charge = retained_tokens + retained.checkpoints * self.charge_per_checkpoint;
        while self.used_charge().saturating_add(retained_charge) > effective_capacity {
            let Some(victim) = self.victim() else {
                break;
            };
            if let Some(mutation) =
                self.evict(victim, PrefixCacheMutationKind::ReplacementPolicyEviction)
            {
                mutations.push(mutation);
            }
        }
        let sequence = self.take_sequence();
        let insertion_sequence = return_metadata
            .map(|value| value.insertion_sequence)
            .unwrap_or(sequence);
        let frequency = return_metadata.map(|value| value.frequency).unwrap_or(0);
        let cache_used_before = self.used_tokens;
        self.entries.insert(
            session_id,
            PrefixEntry {
                tokens: retained_tokens,
                checkpoints: retained.checkpoints,
                replay_boundaries: retained.replay_boundaries,
                insertion_sequence,
                last_access_sequence: sequence,
                frequency,
                context_pending: false,
                checkpoint_sequence: sequence,
            },
        );
        self.used_tokens += retained_tokens;
        self.used_checkpoints += retained.checkpoints;
        mutations.push(PrefixCacheMutation {
            kind: PrefixCacheMutationKind::Retain,
            session_id,
            entry_tokens_before: 0,
            entry_tokens_after: retained_tokens,
            cache_used_before,
            cache_used_after: self.used_tokens,
        });
        PrefixCacheInsertResult {
            mutations,
            retained_tokens,
        }
    }

    /// What an insert of `tokens` keeps within `capacity` under the selected
    /// retention.
    fn retained_span(&self, tokens: u64, capacity: u64) -> RetainedSpan {
        match self.retention {
            CheckpointRetention::EveryQuantum => {
                // Only whole snapshots are resumable, so a tail shorter than
                // one quantum is dropped rather than retained as unusable KV.
                let tokens = self.largest_fitting_length(tokens, capacity);
                RetainedSpan {
                    tokens,
                    checkpoints: tokens / self.quantum_tokens,
                    replay_boundaries: None,
                }
            }
            CheckpointRetention::EagleReplayBoundaries => self.eagle_replay_span(tokens, capacity),
        }
    }

    /// The replay boundaries of a prompt of `prompt_tokens`, resend and
    /// extension if they fit `capacity`, else the resend boundaries alone, else
    /// nothing. A boundary at 0 needs no state and keeps nothing worth
    /// retaining.
    fn eagle_replay_span(&self, prompt_tokens: u64, capacity: u64) -> RetainedSpan {
        let quantum = self.quantum_tokens;
        let unit = self.match_unit_tokens;
        // One step below the last whole step a lookup reaching `reach` sees.
        let boundary = |reach: u64, step: u64| (reach / step * step).saturating_sub(step);
        let resend_reach = prompt_tokens.saturating_sub(1);
        let span = |boundaries: ReplayBoundaries| {
            let mut states = boundaries;
            states.sort_unstable();
            let checkpoints = states
                .iter()
                .enumerate()
                .filter(|&(index, &state)| state > 0 && (index == 0 || states[index - 1] != state))
                .count() as u64;
            // The attention pages up to the match unit the highest boundary
            // drops back from; at `unit == quantum` the block above it.
            let top = states[3];
            RetainedSpan {
                tokens: (top + unit).div_ceil(quantum) * quantum,
                checkpoints,
                replay_boundaries: Some(boundaries),
            }
        };
        let resend = [
            boundary(resend_reach, quantum),
            boundary(resend_reach, unit),
        ];
        let both = span([
            resend[0],
            boundary(prompt_tokens, quantum),
            resend[1],
            boundary(prompt_tokens, unit),
        ]);
        let resend_only = span([resend[0], resend[0], resend[1], resend[1]]);
        [both, resend_only]
            .into_iter()
            .filter(|span| span.checkpoints > 0)
            .find(|span| span.tokens + span.checkpoints * self.charge_per_checkpoint <= capacity)
            .unwrap_or(RetainedSpan {
                tokens: 0,
                checkpoints: 0,
                replay_boundaries: None,
            })
    }

    pub(crate) fn shrink_to(&mut self, physical_limit: u64) -> Vec<PrefixCacheMutation> {
        let effective_capacity = self.max_retained_tokens.min(physical_limit);
        let mut mutations = Vec::new();
        while self.used_charge() > effective_capacity {
            let Some(victim) = self.victim() else {
                break;
            };
            if let Some(mutation) = self.evict(victim, PrefixCacheMutationKind::CapacityEviction) {
                mutations.push(mutation);
            }
        }
        mutations
    }

    /// Retain a prompt's snapshots at its prefill end
    /// ([`CheckpointRelease::AtPrefillEnd`]). The context KV that makes them
    /// resumable is recorded but stays uncharged, and the entry serves no hit,
    /// until [`Self::complete_context`]. `prompt_tokens` is fresh plus declared
    /// prefix, as for [`Self::insert`]; `physical_limit` is the slack the hard
    /// tier leaves.
    pub(crate) fn insert_checkpoints(
        &mut self,
        session_id: u32,
        prompt_tokens: u64,
        physical_limit: u64,
    ) -> PrefixCacheInsertResult {
        debug_assert!(self.release.at_prefill_end());
        let mut mutations = Vec::new();
        if let Some(mutation) =
            self.remove_with_receipt(session_id, PrefixCacheMutationKind::SameSessionReplacement)
        {
            mutations.push(mutation);
        }
        // Which boundaries the prompt leaves does not depend on the slack now:
        // the engine wrote them; whether they survive is the LRU's business.
        let retained = self.retained_span(prompt_tokens, self.max_retained_tokens);
        let checkpoint_charge = retained.checkpoints * self.charge_per_checkpoint;
        let effective_capacity = self.max_retained_tokens.min(physical_limit);
        if retained.checkpoints == 0 || checkpoint_charge > effective_capacity {
            return PrefixCacheInsertResult {
                mutations,
                retained_tokens: 0,
            };
        }
        while self.used_charge().saturating_add(checkpoint_charge) > effective_capacity {
            let Some(victim) = self.victim() else {
                break;
            };
            if let Some(mutation) =
                self.evict(victim, PrefixCacheMutationKind::ReplacementPolicyEviction)
            {
                mutations.push(mutation);
            }
        }
        let sequence = self.take_sequence();
        self.entries.insert(
            session_id,
            PrefixEntry {
                tokens: retained.tokens,
                checkpoints: retained.checkpoints,
                replay_boundaries: retained.replay_boundaries,
                insertion_sequence: sequence,
                last_access_sequence: sequence,
                frequency: 0,
                context_pending: true,
                checkpoint_sequence: sequence,
            },
        );
        self.used_checkpoints += retained.checkpoints;
        let cache_used = self.used_tokens;
        mutations.push(PrefixCacheMutation {
            kind: PrefixCacheMutationKind::Retain,
            session_id,
            entry_tokens_before: 0,
            entry_tokens_after: 0,
            cache_used_before: cache_used,
            cache_used_after: cache_used,
        });
        PrefixCacheInsertResult {
            mutations,
            retained_tokens: 0,
        }
    }

    /// The request whose snapshots [`Self::insert_checkpoints`] retained has
    /// finished: its context KV joins the tier as the entry's newer segment and
    /// the entry can serve a hit. If the snapshots were evicted meanwhile, the
    /// context KV still enters the queue, as an orphan. Freed blocks never
    /// evict anything in the engine, so this only evicts when the hard tier
    /// keeps the freed context for itself (`physical_limit`). Under
    /// [`CheckpointRelease::AtPrefillEndRefreshedAtCompletion`] surviving
    /// snapshots move to this completion position too.
    pub(crate) fn complete_context(
        &mut self,
        session_id: u32,
        prompt_tokens: u64,
        physical_limit: u64,
    ) -> PrefixCacheInsertResult {
        debug_assert!(self.release.at_prefill_end());
        let mut mutations = Vec::new();
        let effective_capacity = self.max_retained_tokens.min(physical_limit);
        let live = self
            .entries
            .get(&session_id)
            .is_some_and(|entry| entry.context_pending);
        let tokens = match self.entries.get(&session_id) {
            Some(entry) if live => entry.tokens,
            _ => {
                self.retained_span(prompt_tokens, self.max_retained_tokens)
                    .tokens
            }
        };
        while self.used_charge().saturating_add(tokens) > effective_capacity {
            let Some(victim) = self.victim() else {
                break;
            };
            if let Some(mutation) =
                self.evict(victim, PrefixCacheMutationKind::ReplacementPolicyEviction)
            {
                mutations.push(mutation);
            }
        }
        let fits = self.used_charge().saturating_add(tokens) <= effective_capacity;
        let sequence = self.take_sequence();
        let still_live = self
            .entries
            .get(&session_id)
            .is_some_and(|entry| entry.context_pending);
        if !fits {
            if still_live {
                // The snapshots alone resume nothing.
                if let Some(mutation) = self.evict(
                    Victim::Entry(session_id),
                    PrefixCacheMutationKind::CapacityEviction,
                ) {
                    mutations.push(mutation);
                }
            }
            return PrefixCacheInsertResult {
                mutations,
                retained_tokens: 0,
            };
        }
        if !still_live {
            if tokens > 0 {
                self.orphans.insert(sequence, tokens);
                self.orphan_tokens += tokens;
            }
            return PrefixCacheInsertResult {
                mutations,
                retained_tokens: 0,
            };
        }
        let cache_used_before = self.used_tokens;
        let entry = self
            .entries
            .get_mut(&session_id)
            .expect("a live pending entry");
        entry.context_pending = false;
        entry.last_access_sequence = sequence;
        if self.release == CheckpointRelease::AtPrefillEndRefreshedAtCompletion {
            entry.checkpoint_sequence = sequence;
        }
        self.used_tokens += tokens;
        mutations.push(PrefixCacheMutation {
            kind: PrefixCacheMutationKind::Retain,
            session_id,
            entry_tokens_before: 0,
            entry_tokens_after: tokens,
            cache_used_before,
            cache_used_after: self.used_tokens,
        });
        PrefixCacheInsertResult {
            mutations,
            retained_tokens: tokens,
        }
    }

    /// The full attention blocks a finished request frees beyond the context
    /// segment [`Self::complete_context`] retains: its output, and the prompt
    /// above the replay boundary. vLLM caches them (`cache_blocks` hashes every
    /// computed full block) and `free_blocks` appends them in reverse order,
    /// tail first, so they enter the LRU just ahead of the context segment.
    /// They resume nothing, so they join as an orphan. Call it right before
    /// [`Self::complete_context`]; `context_tokens` is prompt plus output.
    pub(crate) fn queue_freed_tail(
        &mut self,
        session_id: u32,
        prompt_tokens: u64,
        context_tokens: u64,
        physical_limit: u64,
    ) -> Vec<PrefixCacheMutation> {
        debug_assert!(self.release.at_prefill_end());
        let retained = match self.entries.get(&session_id) {
            Some(entry) if entry.context_pending => entry.tokens,
            _ => {
                self.retained_span(prompt_tokens, self.max_retained_tokens)
                    .tokens
            }
        };
        let tail = self
            .floor_to_quantum(context_tokens)
            .saturating_sub(retained);
        let mut mutations = Vec::new();
        if tail == 0 {
            return mutations;
        }
        let effective_capacity = self.max_retained_tokens.min(physical_limit);
        while self.used_charge().saturating_add(tail) > effective_capacity {
            let Some(victim) = self.victim() else {
                break;
            };
            if let Some(mutation) =
                self.evict(victim, PrefixCacheMutationKind::ReplacementPolicyEviction)
            {
                mutations.push(mutation);
            }
        }
        if self.used_charge().saturating_add(tail) <= effective_capacity {
            let sequence = self.take_sequence();
            self.orphans.insert(sequence, tail);
            self.orphan_tokens += tail;
        }
        mutations
    }

    /// Give up one victim. Under [`CheckpointRelease::AtPrefillEnd`] evicting
    /// an entry means its snapshot segment, the older one: the entry dies and
    /// its charged context KV stays queued at its own position as an orphan.
    fn evict(
        &mut self,
        victim: Victim,
        kind: PrefixCacheMutationKind,
    ) -> Option<PrefixCacheMutation> {
        match victim {
            Victim::Entry(session_id) => {
                let cache_used_before = self.used_tokens;
                let entry = self.remove(session_id)?;
                if self.release.at_prefill_end() && !entry.context_pending {
                    self.orphans
                        .insert(entry.last_access_sequence, entry.tokens);
                    self.orphan_tokens += entry.tokens;
                }
                Some(PrefixCacheMutation {
                    kind,
                    session_id,
                    entry_tokens_before: entry.tokens,
                    entry_tokens_after: 0,
                    cache_used_before,
                    cache_used_after: self.used_tokens,
                })
            }
            Victim::Orphan(sequence) => {
                let tokens = self
                    .orphans
                    .remove(&sequence)
                    .expect("an orphan victim is queued");
                self.orphan_tokens -= tokens;
                None
            }
        }
    }

    fn victim(&self) -> Option<Victim> {
        if self.release.at_prefill_end() {
            // One LRU over segments: an entry's oldest is its snapshot segment.
            let entry = self
                .entries
                .iter()
                .map(|(session_id, entry)| (entry.checkpoint_sequence, *session_id))
                .min();
            let orphan = self.orphans.keys().next().copied();
            return match (entry, orphan) {
                (Some((entry_sequence, session_id)), Some(orphan_sequence)) => {
                    Some(if orphan_sequence < entry_sequence {
                        Victim::Orphan(orphan_sequence)
                    } else {
                        Victim::Entry(session_id)
                    })
                }
                (Some((_, session_id)), None) => Some(Victim::Entry(session_id)),
                (None, Some(orphan_sequence)) => Some(Victim::Orphan(orphan_sequence)),
                (None, None) => None,
            };
        }
        self.policy_victim().map(Victim::Entry)
    }

    fn policy_victim(&self) -> Option<u32> {
        match self.policy {
            PrefixCachePolicy::Lru => self
                .entries
                .iter()
                .min_by_key(|(session_id, entry)| (entry.last_access_sequence, **session_id)),
            PrefixCachePolicy::Fifo => self
                .entries
                .iter()
                .min_by_key(|(session_id, entry)| (entry.insertion_sequence, **session_id)),
            PrefixCachePolicy::Lfu => self.entries.iter().min_by_key(|(session_id, entry)| {
                (entry.frequency, entry.insertion_sequence, **session_id)
            }),
            PrefixCachePolicy::LargestFirst => {
                self.entries.iter().min_by_key(|(session_id, entry)| {
                    (
                        std::cmp::Reverse(entry.tokens),
                        entry.insertion_sequence,
                        **session_id,
                    )
                })
            }
        }
        .map(|(session_id, _)| *session_id)
    }

    fn remove(&mut self, session_id: u32) -> Option<PrefixEntry> {
        let entry = self.entries.remove(&session_id)?;
        if !entry.context_pending {
            self.used_tokens = self.used_tokens.saturating_sub(entry.tokens);
        }
        self.used_checkpoints = self.used_checkpoints.saturating_sub(entry.checkpoints);
        Some(entry)
    }

    fn remove_with_receipt(
        &mut self,
        session_id: u32,
        kind: PrefixCacheMutationKind,
    ) -> Option<PrefixCacheMutation> {
        let cache_used_before = self.used_tokens;
        let entry = self.remove(session_id)?;
        Some(PrefixCacheMutation {
            kind,
            session_id,
            entry_tokens_before: entry.tokens,
            entry_tokens_after: 0,
            cache_used_before,
            cache_used_after: self.used_tokens,
        })
    }

    fn take_sequence(&mut self) -> u64 {
        let sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.saturating_add(1);
        sequence
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Full-attention settings: every prefix length is resumable and a retained
    /// entry costs exactly its tokens. The quantized/charged behaviour is
    /// covered by the hybrid store's own tests.
    const TOKEN_QUANTUM: u32 = 1;
    const NO_EXTRA_CHARGE: u64 = 0;

    #[test]
    fn runtime_config_resolves_disabled_uncapped_and_bounded_token_limits() {
        assert_eq!(
            PrefixCacheConfig::Disabled.resolve_tokens(1_000, 2_000, 2),
            PrefixCacheTokenConfig {
                max_retained_tokens: 0,
                policy: PrefixCachePolicy::Lru,
            }
        );
        assert_eq!(
            PrefixCacheConfig::default().resolve_tokens(1_000, 2_000, 2),
            PrefixCacheTokenConfig {
                max_retained_tokens: 1_000,
                policy: PrefixCachePolicy::Lru,
            }
        );
        assert_eq!(
            PrefixCacheConfig::Opportunistic {
                policy: PrefixCachePolicy::Fifo,
                max_retained_bytes: Some(400_000),
            }
            .resolve_tokens(1_000, 2_000, 2),
            PrefixCacheTokenConfig {
                max_retained_tokens: 400,
                policy: PrefixCachePolicy::Fifo,
            }
        );
    }

    #[test]
    fn take_transfers_ownership_instead_of_sharing() {
        let mut cache =
            PrefixCache::new(100, PrefixCachePolicy::Lru, TOKEN_QUANTUM, NO_EXTRA_CHARGE);
        cache.insert(7, 80, 100, None);
        assert_eq!(cache.take(7, 50).hit_tokens, 50);
        assert_eq!(cache.peek(7, 50), 0);
        assert_eq!(cache.used_tokens(), 0);
    }

    #[test]
    fn lru_evicts_the_oldest_retained_session() {
        let mut cache =
            PrefixCache::new(100, PrefixCachePolicy::Lru, TOKEN_QUANTUM, NO_EXTRA_CHARGE);
        cache.insert(1, 60, 100, None);
        cache.insert(2, 60, 100, None);
        assert_eq!(cache.peek(1, 60), 0);
        assert_eq!(cache.peek(2, 60), 60);
    }

    #[test]
    fn physical_slack_can_shrink_below_the_configured_ceiling() {
        let mut cache =
            PrefixCache::new(100, PrefixCachePolicy::Fifo, TOKEN_QUANTUM, NO_EXTRA_CHARGE);
        cache.insert(1, 40, 100, None);
        cache.insert(2, 40, 100, None);
        cache.shrink_to(40);
        assert_eq!(cache.used_tokens(), 40);
        assert_eq!(cache.peek(1, 40), 0);
        assert_eq!(cache.peek(2, 40), 40);
    }

    #[test]
    fn lru_refreshes_returned_metadata_but_fifo_preserves_first_residency() {
        for (policy, expected_victim) in [(PrefixCachePolicy::Lru, 2), (PrefixCachePolicy::Fifo, 1)]
        {
            let mut cache = PrefixCache::new(100, policy, TOKEN_QUANTUM, NO_EXTRA_CHARGE);
            cache.insert(1, 40, 100, None);
            cache.insert(2, 40, 100, None);
            let take_result = cache.take(1, 40);
            cache.insert(1, 40, 100, take_result.return_metadata);
            cache.insert(3, 40, 100, None);
            assert_eq!(cache.peek(expected_victim, 40), 0);
        }
    }

    #[test]
    fn largest_first_evicts_the_biggest_retained_session() {
        let mut cache = PrefixCache::new(
            100,
            PrefixCachePolicy::LargestFirst,
            TOKEN_QUANTUM,
            NO_EXTRA_CHARGE,
        );
        cache.insert(1, 70, 100, None);
        cache.insert(2, 20, 100, None);
        cache.insert(3, 30, 100, None);
        assert_eq!(cache.peek(1, 70), 0);
        assert_eq!(cache.peek(2, 20), 20);
        assert_eq!(cache.peek(3, 30), 30);
    }

    // ── EAGLE replay boundaries ──────────────────────────────────────────────

    /// GLM-5.3-Flash on H200: a 1152-token hybrid block, and one retained
    /// boundary costs three state pages.
    const BLOCK: u32 = 1_152;
    const BLOCK_TOKENS: u64 = BLOCK as u64;
    const CHECKPOINT_CHARGE: u64 = 3 * BLOCK_TOKENS;

    fn eagle_cache(max_retained_tokens: u64) -> PrefixCache {
        PrefixCache::new(
            max_retained_tokens,
            PrefixCachePolicy::Lru,
            BLOCK,
            CHECKPOINT_CHARGE,
        )
        .with_checkpoint_retention(CheckpointRetention::EagleReplayBoundaries)
    }

    #[test]
    fn an_unaligned_prompt_retains_one_boundary_one_block_below_its_last() {
        let mut cache = eagle_cache(u64::MAX);
        // P = 20000: resend and extension both floor to 17 blocks (19584), so
        // the single retained boundary is 16 blocks.
        let insert = cache.insert(7, 20_000, u64::MAX, None);
        assert_eq!(insert.retained_tokens, 17 * BLOCK_TOKENS);
        assert_eq!(
            cache.used_charge(),
            17 * BLOCK_TOKENS + CHECKPOINT_CHARGE,
            "one state, not one per block"
        );
        // Next turn declares 20376: candidate = 17 blocks - 1 = 16 blocks.
        assert_eq!(cache.peek(7, 20_376), 16 * BLOCK);
        // A longer sibling sharing exactly the prompt resumes there too.
        assert_eq!(cache.peek(7, 20_000), 16 * BLOCK);
        // Truncated below one block above the boundary: candidate 15 blocks.
        assert_eq!(cache.peek(7, 17 * BLOCK - 1), 0);
        assert_eq!(cache.peek(7, 17 * BLOCK), 16 * BLOCK);
    }

    #[test]
    fn an_aligned_prompt_keeps_both_the_resend_and_the_extension_boundary() {
        let mut cache = eagle_cache(u64::MAX);
        let prompt = 18 * BLOCK_TOKENS;
        let insert = cache.insert(7, prompt, u64::MAX, None);
        assert_eq!(insert.retained_tokens, prompt);
        assert_eq!(cache.used_charge(), prompt + 2 * CHECKPOINT_CHARGE);
        // An extension reaches all 18 blocks: drop one, resume at 17.
        assert_eq!(cache.peek(7, 18 * BLOCK), 17 * BLOCK);
        // A resend of the same prompt reaches only P - 1 (17 whole blocks):
        // drop one, resume at the resend boundary, 16 blocks.
        assert_eq!(cache.peek(7, 18 * BLOCK - 1), 16 * BLOCK);
        assert_eq!(cache.peek(7, 17 * BLOCK - 1), 0);
    }

    #[test]
    fn a_hit_never_passes_the_retained_prompt_boundaries() {
        let mut cache = eagle_cache(u64::MAX);
        cache.insert(7, 20_000, u64::MAX, None);
        // A declaration far past the prompt (it included the output the next
        // turn recomputes) still resumes at the prompt's boundary.
        assert_eq!(cache.peek(7, 40_000), 16 * BLOCK);
    }

    #[test]
    fn a_prompt_shorter_than_two_blocks_retains_nothing_resumable() {
        let mut cache = eagle_cache(u64::MAX);
        assert_eq!(
            cache
                .insert(7, 2 * BLOCK_TOKENS - 1, u64::MAX, None)
                .retained_tokens,
            0
        );
        assert_eq!(cache.used_charge(), 0);
        // Two blocks exactly: the resend boundary is 0, the extension one block.
        let insert = cache.insert(7, 2 * BLOCK_TOKENS, u64::MAX, None);
        assert_eq!(insert.retained_tokens, 2 * BLOCK_TOKENS);
        assert_eq!(cache.used_charge(), 2 * BLOCK_TOKENS + CHECKPOINT_CHARGE);
        assert_eq!(cache.peek(7, 2 * BLOCK), BLOCK);
        assert_eq!(cache.peek(7, 2 * BLOCK - 1), 0);
    }

    #[test]
    fn a_tight_cache_falls_back_to_the_resend_boundary_alone() {
        let resend_only = 17 * BLOCK_TOKENS + CHECKPOINT_CHARGE;
        let mut cache = eagle_cache(resend_only);
        let insert = cache.insert(7, 18 * BLOCK_TOKENS, u64::MAX, None);
        assert_eq!(insert.retained_tokens, 17 * BLOCK_TOKENS);
        assert_eq!(cache.used_charge(), resend_only);
        assert_eq!(cache.peek(7, 18 * BLOCK), 16 * BLOCK);

        let mut cache = eagle_cache(resend_only - 1);
        assert_eq!(
            cache
                .insert(7, 18 * BLOCK_TOKENS, u64::MAX, None)
                .retained_tokens,
            0
        );
        assert_eq!(cache.used_charge(), 0);
    }

    // ── A finer prefix match unit (vLLM `--prefix-match-unit 64`) ────────────

    const UNIT: u32 = 64;

    fn unit_cache(max_retained_tokens: u64) -> PrefixCache {
        eagle_cache(max_retained_tokens).with_match_unit(UNIT)
    }

    #[test]
    fn a_match_unit_equal_to_the_block_is_the_default_rule() {
        for prompt in [20_000, 18 * BLOCK_TOKENS, 2 * BLOCK_TOKENS, 26_600] {
            let mut default = eagle_cache(u64::MAX);
            let mut explicit = eagle_cache(u64::MAX).with_match_unit(BLOCK);
            assert_eq!(
                default.insert(7, prompt, u64::MAX, None),
                explicit.insert(7, prompt, u64::MAX, None)
            );
            assert_eq!(default.used_charge(), explicit.used_charge());
            for reach in [prompt as u32 - 1, prompt as u32, prompt as u32 + 1_200] {
                assert_eq!(default.peek(7, reach), explicit.peek(7, reach));
            }
        }
    }

    #[test]
    fn a_unit_of_64_resumes_one_unit_below_the_previous_prompt() {
        // Measured on H200 (MTP 5, align mode, --prefix-match-unit 64): the
        // next turn after a prompt of P hits floor(P / 64) * 64 - 64.
        for (prompt, hit) in [(20_000, 19_904), (26_600, 26_496), (21_200, 21_120)] {
            let mut cache = unit_cache(u64::MAX);
            cache.insert(7, prompt, u64::MAX, None);
            assert_eq!(cache.peek(7, prompt as u32 + 1_199), hit, "P = {prompt}");
        }
    }

    #[test]
    fn a_unit_of_64_keeps_the_block_boundaries_and_charges_each_distinct_state() {
        let mut cache = unit_cache(u64::MAX);
        // P = 20000: block boundaries 16 blocks (resend = extension), unit
        // boundaries 19904 (resend = extension). The partial tail page holding
        // 19904..19968 is kept, so the attention KV covers 18 blocks.
        let insert = cache.insert(7, 20_000, u64::MAX, None);
        assert_eq!(insert.retained_tokens, 18 * BLOCK_TOKENS);
        assert_eq!(
            cache.used_charge(),
            18 * BLOCK_TOKENS + 2 * CHECKPOINT_CHARGE
        );
        // A reach whose candidate falls below the unit boundary resumes at
        // the block boundary, and one below that recomputes.
        assert_eq!(cache.peek(7, 19_967), 16 * BLOCK);
        assert_eq!(cache.peek(7, 16 * BLOCK + UNIT), 16 * BLOCK);
        assert_eq!(cache.peek(7, 16 * BLOCK + UNIT - 1), 0);
        // A resend of the same prompt reaches 19999, still 312 whole units.
        assert_eq!(cache.peek(7, 19_999), 19_904);
    }

    #[test]
    fn a_unit_aligned_prompt_keeps_both_unit_boundaries() {
        let mut cache = unit_cache(u64::MAX);
        // P = 19968 = 312 units: resend 19840, extension 19904; block
        // boundaries both 16 blocks: three distinct charged states.
        let insert = cache.insert(7, 19_968, u64::MAX, None);
        assert_eq!(insert.retained_tokens, 18 * BLOCK_TOKENS);
        assert_eq!(
            cache.used_charge(),
            18 * BLOCK_TOKENS + 3 * CHECKPOINT_CHARGE
        );
        assert_eq!(cache.peek(7, 19_968), 19_904);
        assert_eq!(cache.peek(7, 19_967), 19_840);
    }

    #[test]
    fn a_unit_of_64_makes_a_prompt_shorter_than_two_blocks_resumable() {
        let mut cache = unit_cache(u64::MAX);
        let insert = cache.insert(7, 1_500, u64::MAX, None);
        assert_eq!(insert.retained_tokens, 2 * BLOCK_TOKENS);
        assert_eq!(cache.used_charge(), 2 * BLOCK_TOKENS + CHECKPOINT_CHARGE);
        assert_eq!(cache.peek(7, 2_000), 1_408);
    }

    #[test]
    fn eviction_returns_every_charged_state() {
        let mut cache = eagle_cache(u64::MAX);
        cache.insert(1, 18 * BLOCK_TOKENS, u64::MAX, None);
        cache.insert(2, 20_000, u64::MAX, None);
        let take = cache.take(1, 18 * BLOCK);
        assert_eq!(take.hit_tokens, 17 * BLOCK);
        assert_eq!(cache.used_charge(), 17 * BLOCK_TOKENS + CHECKPOINT_CHARGE);
        cache.shrink_to(0);
        assert_eq!(cache.used_charge(), 0);
    }

    #[test]
    fn mutation_receipts_form_an_exact_occupancy_replay() {
        let mut cache =
            PrefixCache::new(100, PrefixCachePolicy::Lru, TOKEN_QUANTUM, NO_EXTRA_CHARGE);
        let first_insert = cache.insert(1, 60, 100, None);
        let second_insert = cache.insert(2, 60, 100, None);
        let take = cache.take(2, 40);

        let mutations: Vec<PrefixCacheMutation> = first_insert
            .mutations
            .into_iter()
            .chain(second_insert.mutations)
            .collect();
        assert_eq!(mutations.len(), 3);
        for mutation in &mutations {
            assert_eq!(
                mutation.cache_used_after,
                mutation.cache_used_before - mutation.entry_tokens_before
                    + mutation.entry_tokens_after
            );
        }
        assert_eq!(mutations[0].cache_used_before, 0);
        assert_eq!(mutations[0].cache_used_after, 60);
        assert_eq!(
            mutations[1].kind,
            PrefixCacheMutationKind::ReplacementPolicyEviction
        );
        assert_eq!(mutations[1].session_id, 1);
        assert_eq!(mutations[2].cache_used_before, 0);
        assert_eq!(mutations[2].cache_used_after, 60);
        assert_eq!(take.hit_tokens, 40);
        assert_eq!(take.entry_tokens, 60);
        assert_eq!(take.cache_used_before, 60);
        assert_eq!(take.cache_used_after, 0);
    }

    /// P = 40 at a 16-token quantum leaves one replay boundary, 16, kept with
    /// 32 attention tokens; each boundary state is charged 10.
    fn prefill_end_cache(capacity: u64) -> PrefixCache {
        PrefixCache::new(capacity, PrefixCachePolicy::Lru, 16, 10)
            .with_checkpoint_retention(CheckpointRetention::EagleReplayBoundaries)
            .with_checkpoint_release(CheckpointRelease::AtPrefillEnd)
    }

    #[test]
    fn prefill_end_snapshots_serve_no_hit_until_the_context_returns() {
        let mut cache = prefill_end_cache(1_000);
        cache.insert_checkpoints(7, 40, 1_000);
        assert_eq!(
            cache.used_charge(),
            10,
            "only the boundary state is charged"
        );
        assert_eq!(
            cache.peek(7, 39),
            0,
            "the running request still owns the context"
        );
        let completed = cache.complete_context(7, 40, 1_000);
        assert_eq!(completed.retained_tokens, 32);
        assert_eq!(cache.used_charge(), 42);
        assert_eq!(cache.peek(7, 39), 16);
    }

    #[test]
    fn a_long_decode_loses_its_snapshots_before_a_younger_entry() {
        // Session 1 ends its prefill first but decodes longest: its snapshots
        // are the oldest segment although its context returned last.
        let mut cache = prefill_end_cache(1_000);
        cache.insert_checkpoints(1, 40, 1_000);
        cache.insert_checkpoints(2, 40, 1_000);
        cache.complete_context(2, 40, 1_000);
        cache.complete_context(1, 40, 1_000);
        assert_eq!(cache.used_charge(), 84);
        cache.shrink_to(80);
        assert_eq!(cache.peek(1, 39), 0, "session 1's snapshots went first");
        assert_eq!(cache.peek(2, 39), 16);
        assert_eq!(
            cache.used_charge(),
            74,
            "the dead entry's context stays queued as an orphan"
        );
        // Completion-time retention evicts the other way round.
        let mut whole_entries = PrefixCache::new(1_000, PrefixCachePolicy::Lru, 16, 10)
            .with_checkpoint_retention(CheckpointRetention::EagleReplayBoundaries);
        whole_entries.insert(2, 40, 1_000, None);
        whole_entries.insert(1, 40, 1_000, None);
        whole_entries.shrink_to(80);
        assert_eq!(whole_entries.peek(1, 39), 16);
        assert_eq!(whole_entries.peek(2, 39), 0);
    }

    #[test]
    fn an_orphaned_context_is_evicted_in_its_own_turn() {
        let mut cache = prefill_end_cache(1_000);
        cache.insert_checkpoints(1, 40, 1_000); // segment 0
        cache.insert_checkpoints(2, 40, 1_000); // segment 1
        cache.complete_context(2, 40, 1_000); // segment 2
        cache.complete_context(1, 40, 1_000); // segment 3
        cache.shrink_to(80); // session 1's snapshots; its context orphaned at 3
                             // Session 2's snapshots (1) are older than the orphan (3): session 2
                             // dies next, and its own context (2) is then reclaimed before the
                             // orphan.
        cache.shrink_to(60);
        assert_eq!(cache.peek(2, 39), 0);
        assert_eq!(
            cache.used_charge(),
            32,
            "only session 1's orphaned context is left"
        );
        cache.shrink_to(0);
        assert_eq!(cache.used_charge(), 0);
    }

    #[test]
    fn a_context_returning_after_its_snapshots_died_is_queued_as_an_orphan() {
        let mut cache = prefill_end_cache(1_000);
        cache.insert_checkpoints(1, 40, 1_000);
        cache.shrink_to(0);
        assert_eq!(cache.used_charge(), 0);
        let completed = cache.complete_context(1, 40, 1_000);
        assert_eq!(completed.retained_tokens, 0);
        assert_eq!(cache.peek(1, 39), 0);
        assert_eq!(cache.used_charge(), 32);
    }

    fn refreshed_cache(capacity: u64) -> PrefixCache {
        PrefixCache::new(capacity, PrefixCachePolicy::Lru, 16, 10)
            .with_checkpoint_retention(CheckpointRetention::EagleReplayBoundaries)
            .with_checkpoint_release(CheckpointRelease::AtPrefillEndRefreshedAtCompletion)
    }

    #[test]
    fn a_completion_refresh_moves_the_snapshots_behind_newer_segments() {
        // The long-decode order of `a_long_decode_loses_its_snapshots_before_a_younger_entry`:
        // refreshed at completion, session 1's snapshots are now the newest
        // segment, so session 2 (completed earlier) goes first.
        let mut cache = refreshed_cache(1_000);
        cache.insert_checkpoints(1, 40, 1_000);
        cache.insert_checkpoints(2, 40, 1_000);
        cache.complete_context(2, 40, 1_000);
        cache.complete_context(1, 40, 1_000);
        assert_eq!(cache.used_charge(), 84);
        cache.shrink_to(80);
        assert_eq!(cache.peek(1, 39), 16, "the refreshed snapshots survive");
        assert_eq!(cache.peek(2, 39), 0);
        assert_eq!(
            cache.used_charge(),
            74,
            "session 2's context stays queued as an orphan"
        );
        // Session 2's orphan (completion 2) precedes session 1 (completion 3).
        cache.shrink_to(50);
        assert_eq!(cache.peek(1, 39), 16);
        assert_eq!(cache.used_charge(), 42);
    }

    #[test]
    fn a_completion_refresh_does_not_resurrect_evicted_snapshots() {
        let mut cache = refreshed_cache(1_000);
        cache.insert_checkpoints(1, 40, 1_000);
        cache.shrink_to(0);
        let completed = cache.complete_context(1, 40, 1_000);
        assert_eq!(completed.retained_tokens, 0);
        assert_eq!(cache.peek(1, 39), 0);
        assert_eq!(
            cache.used_charge(),
            32,
            "only the context returns, as an orphan"
        );
    }

    #[test]
    fn a_finished_requests_freed_tail_queues_ahead_of_its_context() {
        let mut cache = prefill_end_cache(1_000);
        cache.insert_checkpoints(7, 40, 1_000);
        // 40 prompt + 30 output: four full 16-token blocks, two of them kept
        // as the context segment, two (the output) queued as an orphan.
        cache.queue_freed_tail(7, 40, 70, 1_000);
        assert_eq!(cache.used_charge(), 10 + 32);
        cache.complete_context(7, 40, 1_000);
        assert_eq!(cache.used_charge(), 10 + 32 + 32);
        assert_eq!(
            cache.peek(7, 39),
            16,
            "the tail resumes nothing but costs nothing to the hit"
        );
        // The snapshots go first, then the tail, then the (orphaned) context.
        cache.shrink_to(64);
        assert_eq!(cache.used_charge(), 64);
        cache.shrink_to(63);
        assert_eq!(cache.used_charge(), 32);
    }
}
