//! KV accounting for a hybrid model: per-token full attention **plus** a fixed
//! per-request recurrent (SSM + causal-conv) state, both living in the same
//! physical attention memory.
//!
//! ## Two ledgers, one capacity
//!
//! | tier | contents | evictable | enters [`KvStore::fits`] |
//! |---|---|---|---|
//! | hard | full-attention KV (context + future decode) + **one** live recurrent state per resident request | no | yes |
//! | soft | retained prefix KV and its snapshots; snapshots a live decode drops while running (none under the EAGLE replay rule) | yes | no |
//!
//! The hard tier is what admission reserves atomically: one request's footprint
//! is `post_prefill_context + remaining_output + state_tokens`, all three
//! checked and reserved in a single [`KvStore::reserve`]. The recurrent half is
//! paid once at admission and never grows with decode — a recurrent state is a
//! rolling snapshot, not a per-token page.
//!
//! The soft tier is clamped to whatever capacity the hard tier leaves, exactly
//! like the pre-existing `prefix_cache_physical_limit` rule, so the total
//! constraint covers active + reserved + retained + held and their recurrent
//! shares without a second admission ledger.
//!
//! Chunked prefill ([`ChunkedPrefillKv`]) keeps that single reservation: the
//! state is promised with the request's first chunk and committed resident with
//! its last. Where the engine can only checkpoint state on block boundaries, the
//! lifecycle, not this store, clips chunk ends to them.
//!
//! ## Why prefix reuse quantizes
//!
//! A full-attention cache can resume from any token: page `i` is independent of
//! page `j`. A recurrent state cannot — the state at token `T` is only
//! recoverable from a snapshot taken at `T`, and a *later* snapshot is useless
//! to an *earlier* resume point. vLLM's hybrid allocator therefore snapshots
//! only at multiples of an aligned block size and quantizes hits to it (see
//! `arch::qwen36_local` for the alignment formula and its verification against
//! measured vLLM numbers). This store passes that interval to
//! [`PrefixCache`] as its reuse quantum, and the arch's retained-checkpoint
//! size as its per-snapshot charge, so a retained prefix pays for the snapshots
//! that make it resumable.
//!
//! By default every interval boundary of the completed request's resident
//! context is resumable. A speculating vLLM baseline with the EAGLE block drop
//! ([`HybridGdnKv::with_eagle_replay_boundaries`]) keeps only the prompt's two
//! replay boundaries and backs every hit off one block
//! ([`CheckpointRetention::EagleReplayBoundaries`]).
//!
//! ## Not modeled
//!
//! Live snapshots yield to admission but are never *proactively* shed: there is
//! no retention-interval or boundary-protection policy picking which snapshot to
//! drop first. vLLM's own work in that direction is
//! <https://github.com/vllm-project/vllm/pull/43447> and Marconi
//! (<https://arxiv.org/abs/2411.19379>).

use std::collections::HashMap;

use crate::common::{RequestId, SessionInput, Time};
use crate::log::{
    KvSampler, KvSubmit, PrefixCacheActivation, PrefixCacheEvent, PrefixCacheEventKind,
    PrefixCacheEvictionReason, PrefixCacheLogger, PrefixCacheRetentionReason,
};
use crate::worker::kv::shared::prefix_cache::{
    CheckpointRelease, CheckpointRetention, PrefixCache, PrefixCacheMutation,
    PrefixCacheReturnMetadata,
};
use crate::worker::kv::shared::prefix_cache_journal::PrefixCacheJournal;
use crate::worker::kv::shared::request_ledger::RequestLedger;
use crate::worker::kv::shared::resident_partition::ResidentPartitionState;
use crate::worker::kv::{
    ChunkedPrefillKv, IterWorkerKv, KvStore, PrefixCacheTokenConfig, PrefixKv,
    ResolvedPrefillContext,
};
use crate::worker::shared::advance_scope::{AdvanceScope, PartitionId};

/// Hard-tier charge of one request, in full-attention token equivalents.
pub struct HybridKvFootprint {
    post_prefill_context_tokens: u32,
    /// Decode tokens the reservation covers: the whole remaining output, or a
    /// bounded-future estimate plus one page.
    future_decode_tokens: u32,
    /// The whole model's recurrent state for this one request. Constant.
    state_tokens: u64,
}

impl HybridKvFootprint {
    #[inline]
    fn reserved_charge(&self) -> u64 {
        u64::from(self.post_prefill_context_tokens)
            + u64::from(self.future_decode_tokens)
            + self.state_tokens
    }
}

pub struct HybridGdnKv {
    partitions: Vec<ResidentPartitionState>,
    ledger: RequestLedger,
    prefix_caches: Vec<PrefixCache>,
    journal: PrefixCacheJournal,
    sampler: Option<KvSampler>,
    /// One request's recurrent state, expressed in full-attention token
    /// equivalents so a single capacity number governs both kinds of memory.
    state_tokens_per_request: u64,
    /// Context-token spacing of resumable snapshots. `0` = no recurrent state.
    checkpoint_interval_tokens: u32,
    /// vLLM's EAGLE replay boundaries: retain from the prompt only, and cap a
    /// lookup at the prompt minus its last token.
    eagle_replay_boundaries: bool,
    /// When a prompt's snapshots enter the retained tier (see
    /// [`CheckpointRelease`]).
    checkpoint_release: CheckpointRelease,
    /// vLLM v1 block allocation instead of an up-front reservation (see
    /// [`IncrementalBlocks`]). `None` keeps the reserved footprint.
    incremental: Option<IncrementalBlocks>,
}

/// vLLM v1's `KVCacheManager.allocate_slots` ledger: a running request holds
/// only the whole blocks its scheduled tokens reach, plus the speculative
/// lookahead slots and its recurrent state, and nothing is reserved for the
/// rest of its prompt or its output. Every allocation that the free blocks
/// cannot cover evicts retained KV from the head of the LRU queue
/// (`BlockPool.get_new_blocks`); one that the free and evictable blocks
/// together cannot cover fails, and the lifecycle preempts.
struct IncrementalBlocks {
    /// The hash block size, in tokens: the attention allocation granularity.
    block_tokens: u64,
    /// Slots allocated past every scheduled position (`num_lookahead_tokens`).
    lookahead_tokens: u64,
    /// Attention tokens each running request holds, in whole blocks; its
    /// recurrent state comes on top.
    request_tokens: HashMap<RequestId, (PartitionId, u64)>,
    /// Per partition: everything allocated, attention plus states.
    partition_tokens: Vec<u64>,
}

impl IncrementalBlocks {
    fn whole_blocks(&self, tokens: u64) -> u64 {
        tokens.div_ceil(self.block_tokens) * self.block_tokens
    }

    fn free(&mut self, request: RequestId, state_tokens: u64) {
        if let Some((partition, tokens)) = self.request_tokens.remove(&request) {
            self.partition_tokens[partition as usize] -= tokens + state_tokens;
        }
    }
}

impl KvStore for HybridGdnKv {
    type Footprint = HybridKvFootprint;

    fn num_partitions(&self) -> usize {
        self.partitions.len()
    }

    fn footprint(
        &self,
        _request: RequestId,
        post_prefill_context_tokens: u32,
        remaining_output_tokens: u32,
    ) -> Self::Footprint {
        HybridKvFootprint {
            post_prefill_context_tokens,
            future_decode_tokens: remaining_output_tokens,
            state_tokens: self.state_tokens_per_request,
        }
    }

    /// Both halves of the footprint are tested against one capacity, against
    /// both the immediate resident total and the projected decode peak. The
    /// soft tier is deliberately absent: it yields to this decision.
    fn fits(&self, partition: PartitionId, footprint: &Self::Footprint) -> bool {
        let partition_state = &self.partitions[partition as usize];
        let reserved_charge = self.ledger.partition_promised(partition)
            + self.ledger.partition_held(partition)
            + footprint.reserved_charge();
        partition_state.resident_tokens() + reserved_charge <= partition_state.capacity_tokens()
            && partition_state.projected_peak_kv() + reserved_charge
                <= partition_state.capacity_tokens()
    }

    fn reserve(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        footprint: Self::Footprint,
        now: Time,
    ) {
        // Incremental allocation reserves nothing up front: each scheduled
        // step allocates what it reaches (`allocate_incremental`).
        let promised = if self.incremental.is_some() {
            0
        } else {
            footprint.reserved_charge()
        };
        self.ledger.promise(request, partition, promised);
        let evictions = self.trim_prefix_cache_to_physical_slack(partition);
        for eviction in evictions {
            self.journal.record_mutation(
                request,
                partition,
                now,
                eviction,
                PrefixCacheEventKind::Evict(PrefixCacheEvictionReason::ActiveKvPressure),
                0,
            );
        }
    }

    fn commit_resident(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        post_prefill_context_tokens: u64,
        remaining_output_tokens: u32,
    ) {
        self.ledger.forget_promise(request);
        // `ResidentPartitionState` charges `state_tokens_per_request` here and
        // releases it in `release_decode`; it is never advanced.
        self.partitions[partition as usize].begin_decode(
            request,
            post_prefill_context_tokens,
            remaining_output_tokens,
        );
    }

    fn advance(&mut self, scope: AdvanceScope<'_>, steps: u32) {
        for _ in 0..steps {
            match scope {
                AdvanceScope::WholePartition(partition) => {
                    self.partitions[partition as usize].advance_decodes();
                }
                AdvanceScope::RequestSubset {
                    partition,
                    request_ids,
                } => {
                    debug_assert!(
                        self.ledger.all_placed_on(partition, request_ids),
                        "advance scope contains a request outside KV partition {partition}"
                    );
                    self.partitions[partition as usize].advance_subset(request_ids);
                }
            }
        }
    }

    fn release(&mut self, request: RequestId, partition: PartitionId) {
        if let Some(incremental) = self.incremental.as_mut() {
            incremental.free(request, self.state_tokens_per_request);
        }
        self.ledger.take_held(request);
        self.ledger.forget_promise(request);
        self.ledger.forget_chunked_prefill(request);
        let current_kv = self.partitions[partition as usize]
            .decode_current_kv(request)
            .unwrap_or(0);
        self.partitions[partition as usize].release_decode(request, current_kv);
        self.ledger.forget_placement(request);
        self.ledger.take_prefill_context(request);
    }

    fn sample_submit(&mut self, partition: PartitionId, now: Time) {
        if self.sampler.is_none() {
            return;
        }
        let retained_prefix_kv = self.prefix_caches[partition as usize].used_charge();
        let live_checkpoints = self.clamped_live_checkpoint_charge(partition);
        let held = self.ledger.partition_held(partition);
        let partition_state = &self.partitions[partition as usize];
        if let Some(incremental) = &self.incremental {
            let allocated = incremental.partition_tokens[partition as usize] + held;
            let submit = KvSubmit {
                active_kv: allocated + retained_prefix_kv + live_checkpoints,
                retained_prefix_kv,
                projected_peak: allocated + retained_prefix_kv + live_checkpoints,
                promised_kv: 0,
            };
            self.sampler
                .as_mut()
                .unwrap()
                .submit(partition, submit, now);
            return;
        }
        let submit = KvSubmit {
            active_kv: partition_state.resident_tokens()
                + held
                + retained_prefix_kv
                + live_checkpoints,
            retained_prefix_kv,
            projected_peak: partition_state.projected_peak_kv()
                + held
                + retained_prefix_kv
                + live_checkpoints,
            promised_kv: self.ledger.partition_promised(partition),
        };
        self.sampler
            .as_mut()
            .unwrap()
            .submit(partition, submit, now);
    }
}

impl IterWorkerKv for HybridGdnKv {
    fn drain_ready(&mut self) {
        for (partition, request) in self.ledger.drain_promised() {
            self.partitions[partition as usize].add_prefill_admit(request);
        }
    }

    fn clear_prefill_admits(&mut self, partition: PartitionId) {
        self.partitions[partition as usize].clear_prefill_admits();
    }

    fn has_live_decode(&self, partition: PartitionId) -> bool {
        self.partitions[partition as usize].has_live_decode()
    }

    fn live_decode_count(&self, partition: PartitionId) -> u32 {
        self.partitions[partition as usize].live_decode_count()
    }

    fn has_prefill_admit(&self, partition: PartitionId) -> bool {
        self.partitions[partition as usize].has_prefill_admit()
    }

    fn status_active(&self, partition: PartitionId) -> u32 {
        self.partitions[partition as usize].live_decode_count()
            + self.ledger.partition_promised_count(partition)
            + self.partitions[partition as usize].prefill_admit_count()
    }

    fn release_external(&mut self, request: RequestId, current_kv: u64) -> Option<PartitionId> {
        if let Some(incremental) = self.incremental.as_mut() {
            incremental.free(request, self.state_tokens_per_request);
        }
        let Some(partition) = self.ledger.forget_placement(request) else {
            self.ledger.take_held(request);
            self.ledger.forget_chunked_prefill(request);
            self.ledger.take_prefill_context(request);
            return None;
        };
        let partition_state = &mut self.partitions[partition as usize];
        partition_state.release_decode(request, current_kv);
        partition_state.remove_prefill_admit(request);
        self.ledger.forget_promise(request);
        self.ledger.forget_chunked_prefill(request);
        self.ledger.take_prefill_context(request);
        Some(partition)
    }

    fn visit_prefill_admits(&self, partition: PartitionId, mut visitor: impl FnMut(RequestId)) {
        for request in self.partitions[partition as usize].iter_prefill_admits() {
            visitor(request);
        }
    }

    fn visit_decode_members(
        &self,
        partition: PartitionId,
        mut visitor: impl FnMut(RequestId, u64),
    ) {
        // Context tokens, never charge: this feeds `decode_kv_lens` in the arch
        // input and therefore the attention cost model.
        for (request, current_kv) in self.partitions[partition as usize].decode_members() {
            visitor(request, current_kv);
        }
    }
}

impl PrefixKv for HybridGdnKv {
    fn retained_prefix_partition(&self, session_input: SessionInput) -> Option<PartitionId> {
        let SessionInput::Session {
            session_id,
            declared_prefix_tokens,
            ..
        } = session_input
        else {
            return None;
        };

        let mut best_partition = None;
        let mut best_resident_tokens = 0;
        for (partition_index, prefix_cache) in self.prefix_caches.iter().enumerate() {
            // No fresh prompt length here: rank by the uncapped declaration.
            let resident_tokens = prefix_cache.peek(session_id, declared_prefix_tokens);
            if resident_tokens > best_resident_tokens {
                best_resident_tokens = resident_tokens;
                best_partition = Some(
                    PartitionId::try_from(partition_index)
                        .expect("HybridGdnKv partition count must fit PartitionId"),
                );
            }
        }
        best_partition
    }

    fn preview_prefill_context(
        &self,
        partition: PartitionId,
        fresh_prompt_tokens: u32,
        session_input: SessionInput,
    ) -> ResolvedPrefillContext {
        let resident_prefix_tokens = match session_input {
            SessionInput::Standalone => 0,
            SessionInput::Session {
                session_id,
                declared_prefix_tokens,
                ..
            } => self.prefix_caches[partition as usize].peek(
                session_id,
                self.lookup_reach(
                    declared_prefix_tokens,
                    declared_prefix_tokens.saturating_add(fresh_prompt_tokens),
                ),
            ),
        };
        ResolvedPrefillContext::new(session_input, resident_prefix_tokens, fresh_prompt_tokens)
    }

    fn reserve_prefill_context(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        mut resolved_prefill: ResolvedPrefillContext,
        footprint: Self::Footprint,
        now: Time,
    ) {
        if let Some(session_id) = resolved_prefill.session_id() {
            let lookup_reach = self.lookup_reach(
                resolved_prefill.declared_prefix_tokens(),
                resolved_prefill.post_prefill_context_tokens(),
            );
            let take_result = self.prefix_caches[partition as usize].take(session_id, lookup_reach);
            // `ResolvedPrefillContext::new` borrows one token back from a fully
            // resident prefix so that a zero-length prefill never reaches the
            // model. The snapshot still physically sits at the boundary, so the
            // resume point is off by that one borrowed token. Cost is unchanged
            // (a GDN prefill only sees the append length and whether a state
            // exists), so the accounting keeps the boundary and tolerates the
            // one-token gap rather than inventing an unaligned snapshot.
            debug_assert!(
                take_result
                    .hit_tokens
                    .abs_diff(resolved_prefill.resident_prefix_tokens())
                    <= 1,
                "prefill context changed between preview and reservation"
            );
            self.journal.record(PrefixCacheEvent {
                partition_id: partition,
                time: now,
                request_id: request,
                session_id,
                kind: PrefixCacheEventKind::Activate(if take_result.hit_tokens > 0 {
                    PrefixCacheActivation::Hit
                } else {
                    PrefixCacheActivation::Miss
                }),
                entry_tokens_before: take_result.entry_tokens,
                entry_tokens_after: 0,
                cache_used_before: take_result.cache_used_before,
                cache_used_after: take_result.cache_used_after,
                requested_tokens: u64::from(resolved_prefill.declared_prefix_tokens()),
                hit_tokens: u64::from(take_result.hit_tokens),
            });
            resolved_prefill =
                resolved_prefill.with_cache_return_metadata(take_result.return_metadata);
        }
        self.ledger.set_prefill_context(request, resolved_prefill);
        self.reserve(request, partition, footprint, now);
    }

    fn resolved_prefill_context(&self, request: RequestId) -> ResolvedPrefillContext {
        self.ledger
            .prefill_context(request)
            .expect("resolved prefill context must exist for an admitted fresh request")
    }

    fn release_retaining_prefix(&mut self, request: RequestId, partition: PartitionId, now: Time) {
        let resolved_prefill = self.ledger.prefill_context(request);
        let prompt_tokens =
            resolved_prefill.map(|value| u64::from(value.post_prefill_context_tokens()));
        // Under the EAGLE replay rule states exist only inside the prompt, so
        // the prompt, not the decoded context, bounds what survives.
        let retained_tokens = if self.eagle_replay_boundaries {
            prompt_tokens
        } else {
            self.partitions[partition as usize]
                .decode_current_kv(request)
                .or(prompt_tokens)
        }
        .unwrap_or(0);
        let session_id = resolved_prefill.and_then(ResolvedPrefillContext::session_id);
        let cache_return_metadata =
            resolved_prefill.and_then(ResolvedPrefillContext::cache_return_metadata);
        self.release(request, partition);
        if let Some(session_id) = session_id {
            self.retain_prefix(
                request,
                partition,
                session_id,
                retained_tokens,
                cache_return_metadata,
                now,
                PrefixCacheRetentionReason::RequestComplete,
            );
        }
    }
}

/// Chunked prefill over the same two tiers. The recurrent state is part of the
/// one footprint reserved at admission, stays promised while chunks run, and
/// becomes resident exactly once, when the last chunk commits the request.
impl ChunkedPrefillKv for HybridGdnKv {
    fn bounded_future_footprint(
        &self,
        _request: RequestId,
        post_prefill_context_tokens: u32,
        remaining_output_tokens: u32,
        max_future_tokens: u32,
        page_size: u32,
    ) -> Self::Footprint {
        let future_decode_tokens = remaining_output_tokens
            .min(max_future_tokens)
            .checked_add(page_size)
            .expect("bounded-future KV footprint overflow");
        HybridKvFootprint {
            post_prefill_context_tokens,
            future_decode_tokens,
            state_tokens: self.state_tokens_per_request,
        }
    }

    /// Like [`KvStore::fits`], only the hard tier is protected: retained
    /// prefixes and live snapshots yield.
    fn fits_bounded_future(
        &self,
        partition: PartitionId,
        footprint: &Self::Footprint,
        max_future_tokens: u32,
        new_token_ratio: f64,
        decode_step_tokens: u32,
    ) -> bool {
        let partition_state = &self.partitions[partition as usize];
        let protected_tokens = partition_state.resident_tokens()
            + self.ledger.partition_promised(partition)
            + self.ledger.partition_held(partition)
            + footprint.reserved_charge();
        let mut running_future =
            partition_state.bounded_future_decode_tokens(max_future_tokens, new_token_ratio);
        if decode_step_tokens > 0 {
            // Bounded-future admission runs on one-token pages only.
            let next_step = partition_state.next_decode_allocation_tokens(1, decode_step_tokens);
            running_future = running_future.max(next_step as f64);
        }
        protected_tokens as f64 + running_future < partition_state.capacity_tokens() as f64
    }

    /// The live-snapshot staircase keeps its precedence over retained prefixes
    /// but, being soft, never counts toward the shortfall.
    fn prepare_next_decode(
        &mut self,
        partition: PartitionId,
        page_size: u32,
        step_tokens: u32,
        now: Time,
    ) -> u64 {
        assert!(page_size > 0, "decode page size must be positive");
        let partition_state = &self.partitions[partition as usize];
        let required = partition_state.next_decode_allocation_tokens(page_size, step_tokens);
        if required == 0 {
            return 0;
        }
        let trigger = partition_state
            .first_decode_request()
            .expect("positive decode allocation requires a live request");
        let protected = partition_state.resident_tokens()
            + self.ledger.partition_promised(partition)
            + self.ledger.partition_held(partition);
        let capacity = partition_state.capacity_tokens();
        let slack_after_step = capacity.saturating_sub(protected.saturating_add(required));
        let live_checkpoints = self.live_checkpoint_charge(partition).min(slack_after_step);
        let cache_limit = slack_after_step - live_checkpoints;
        let evictions = self.prefix_caches[partition as usize].shrink_to(cache_limit);
        for eviction in evictions {
            self.journal.record_mutation(
                trigger,
                partition,
                now,
                eviction,
                PrefixCacheEventKind::Evict(PrefixCacheEvictionReason::ActiveKvPressure),
                0,
            );
        }
        let cache_tokens = self.prefix_caches[partition as usize].used_charge();
        let available = capacity.saturating_sub(protected.saturating_add(cache_tokens));
        required.saturating_sub(available)
    }

    fn visit_decode_states(
        &self,
        partition: PartitionId,
        mut visitor: impl FnMut(RequestId, u64, u32),
    ) {
        for (request, current_kv, remaining_decode) in
            self.partitions[partition as usize].decode_states()
        {
            visitor(request, current_kv, remaining_decode);
        }
    }

    fn reserve_chunked_prefill_context(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        resolved_prefill: ResolvedPrefillContext,
        footprint: Self::Footprint,
        now: Time,
    ) {
        self.reserve_prefill_context(request, partition, resolved_prefill, footprint, now);
        self.ledger.promote_promise_to_chunked_prefill(request);
    }

    fn schedule_prefill_chunk(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        chunk_tokens: u32,
    ) {
        debug_assert!(self.ledger.has_chunked_prefill(request));
        self.ledger.schedule_prefill_chunk(request, chunk_tokens);
        self.partitions[partition as usize].add_prefill_admit(request);
    }

    fn complete_prefill_chunk(&mut self, request: RequestId) {
        self.ledger.complete_prefill_chunk(request);
    }

    fn finish_chunked_prefill(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        remaining_output_tokens: u32,
        now: Time,
    ) {
        let resolved_prefill = self
            .ledger
            .prefill_context(request)
            .expect("chunked prefill context must exist");
        let context_tokens = resolved_prefill.post_prefill_context_tokens();
        self.ledger.forget_chunked_prefill(request);
        self.commit_resident(
            request,
            partition,
            u64::from(context_tokens),
            remaining_output_tokens,
        );
        if self.checkpoint_release == CheckpointRelease::AtPrefillEnd {
            if let Some(session_id) = resolved_prefill.session_id() {
                self.retain_checkpoints(
                    request,
                    partition,
                    session_id,
                    u64::from(context_tokens),
                    now,
                );
            }
        }
    }

    fn incremental_prompt_fits(
        &self,
        partition: PartitionId,
        prompt_tokens: u32,
        first_chunk_end_tokens: u32,
    ) -> bool {
        let incremental = self
            .incremental
            .as_ref()
            .expect("incremental admission needs `with_incremental_allocation`");
        // `allocate_slots(full_sequence_must_fit=True)`: the whole prompt must
        // fit in the free blocks, evictable cached ones included, without
        // counting what earlier admissions will still allocate.
        let attention =
            incremental
                .whole_blocks(u64::from(prompt_tokens))
                .max(incremental.whole_blocks(
                    u64::from(first_chunk_end_tokens) + incremental.lookahead_tokens,
                ));
        let partition_state = &self.partitions[partition as usize];
        incremental.partition_tokens[partition as usize]
            + self.ledger.partition_held(partition)
            + attention
            + self.state_tokens_per_request
            <= partition_state.capacity_tokens()
    }

    fn allocate_incremental(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        context_tokens: u64,
        now: Time,
    ) -> bool {
        let state_tokens = self.state_tokens_per_request;
        let capacity = self.partitions[partition as usize].capacity_tokens();
        let held = self.ledger.partition_held(partition);
        let incremental = self
            .incremental
            .as_mut()
            .expect("incremental admission needs `with_incremental_allocation`");
        let target = incremental.whole_blocks(context_tokens + incremental.lookahead_tokens);
        let (current, first) = match incremental.request_tokens.get(&request) {
            Some(&(owner, tokens)) => {
                debug_assert_eq!(owner, partition);
                (tokens, false)
            }
            None => (0, true),
        };
        let added = target.saturating_sub(current) + if first { state_tokens } else { 0 };
        if added == 0 {
            return true;
        }
        if incremental.partition_tokens[partition as usize] + held + added > capacity {
            return false;
        }
        incremental
            .request_tokens
            .insert(request, (partition, current.max(target)));
        incremental.partition_tokens[partition as usize] += added;
        let evictions = self.trim_prefix_cache_to_physical_slack(partition);
        for eviction in evictions {
            self.journal.record_mutation(
                request,
                partition,
                now,
                eviction,
                PrefixCacheEventKind::Evict(PrefixCacheEvictionReason::ActiveKvPressure),
                0,
            );
        }
        true
    }

    fn hold_trailing_decode_row(&mut self, request: RequestId, partition: PartitionId) {
        self.partitions[partition as usize].hold_trailing_row(request);
    }
}

impl HybridGdnKv {
    /// `state_tokens_per_request`, `checkpoint_interval_tokens` and
    /// `checkpoint_charge_tokens` come from the arch
    /// (`recurrent_state_bytes_per_request` divided by
    /// `total_kv_bytes_per_token`, `recurrent_checkpoint_interval_tokens`, and
    /// the bytes of one retained checkpoint in the same token unit).
    /// `checkpoint_charge_tokens` is what one retained prefix snapshot costs;
    /// the resident state and the live-decode staircase keep
    /// `state_tokens_per_request`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        num_partitions: usize,
        kv_capacity: u64,
        state_tokens_per_request: u64,
        checkpoint_interval_tokens: u32,
        checkpoint_charge_tokens: u64,
        prefix_cache: PrefixCacheTokenConfig,
        sampler: Option<KvSampler>,
        prefix_cache_logger: Option<PrefixCacheLogger>,
    ) -> Self {
        // A model with no recurrent state degrades to full-attention accounting
        // rather than treating every single token as a snapshot boundary. The
        // recipe should not select this store for such a model.
        let (quantum_tokens, charge_per_checkpoint) = if checkpoint_interval_tokens == 0 {
            (1, 0)
        } else {
            (checkpoint_interval_tokens, checkpoint_charge_tokens)
        };
        Self {
            partitions: (0..num_partitions)
                .map(|_| ResidentPartitionState::new(kv_capacity, state_tokens_per_request))
                .collect(),
            ledger: RequestLedger::new(num_partitions),
            prefix_caches: (0..num_partitions)
                .map(|_| {
                    PrefixCache::new(
                        prefix_cache.max_retained_tokens.min(kv_capacity),
                        prefix_cache.policy,
                        quantum_tokens,
                        charge_per_checkpoint,
                    )
                })
                .collect(),
            journal: PrefixCacheJournal::new(prefix_cache_logger),
            sampler,
            state_tokens_per_request,
            checkpoint_interval_tokens,
            eagle_replay_boundaries: false,
            checkpoint_release: CheckpointRelease::OnCompletion,
            incremental: None,
        }
    }

    /// Allocate as vLLM v1 does ([`IncrementalBlocks`]): attention in whole
    /// checkpoint-interval blocks plus `lookahead_tokens` slots, the recurrent
    /// state with the first allocation, nothing reserved ahead. Needs a
    /// recurrent checkpoint interval, which is the block size.
    pub(crate) fn with_incremental_allocation(mut self, lookahead_tokens: u32) -> Self {
        assert!(
            self.checkpoint_interval_tokens > 0,
            "incremental allocation needs the checkpoint interval as its block size"
        );
        self.incremental = Some(IncrementalBlocks {
            block_tokens: u64::from(self.checkpoint_interval_tokens),
            lookahead_tokens: u64::from(lookahead_tokens),
            request_tokens: HashMap::new(),
            partition_tokens: vec![0; self.partitions.len()],
        });
        self
    }

    /// Retain a prompt's snapshots when its prefill ends and its context KV
    /// when it completes, as two LRU segments ([`CheckpointRelease::AtPrefillEnd`]).
    /// Needs [`Self::with_eagle_replay_boundaries`] first: only the replay
    /// boundaries are freed into the pool at the prefill end.
    pub(crate) fn with_checkpoint_release_at_prefill_end(mut self) -> Self {
        assert!(
            self.eagle_replay_boundaries,
            "checkpoint release at the prefill end is modeled only under EAGLE replay boundaries"
        );
        self.prefix_caches = self
            .prefix_caches
            .into_iter()
            .map(|cache| cache.with_checkpoint_release(CheckpointRelease::AtPrefillEnd))
            .collect();
        self.checkpoint_release = CheckpointRelease::AtPrefillEnd;
        self
    }

    /// Retain and resume as vLLM does for a speculating hybrid with the EAGLE
    /// block drop ([`CheckpointRetention::EagleReplayBoundaries`]): only the
    /// prompt's replay boundaries survive a request, and a lookup reaches at
    /// most the prompt minus its last token. Needs a recurrent checkpoint
    /// interval.
    pub(crate) fn with_eagle_replay_boundaries(mut self) -> Self {
        assert!(
            self.checkpoint_interval_tokens > 0,
            "EAGLE replay boundaries need a recurrent checkpoint interval"
        );
        self.prefix_caches = self
            .prefix_caches
            .into_iter()
            .map(|cache| {
                cache.with_checkpoint_retention(CheckpointRetention::EagleReplayBoundaries)
            })
            .collect();
        self.eagle_replay_boundaries = true;
        self
    }

    /// vLLM's `--prefix-match-unit` under the EAGLE replay rule: lookups match
    /// at `match_unit_tokens`, a divisor of the checkpoint interval, and each
    /// prompt also retains its replay boundaries at that granularity
    /// ([`PrefixCache::with_match_unit`]). Needs
    /// [`Self::with_eagle_replay_boundaries`] first.
    pub(crate) fn with_prefix_match_unit(mut self, match_unit_tokens: u32) -> Self {
        assert!(
            self.eagle_replay_boundaries,
            "a prefix match unit is modeled only under EAGLE replay boundaries"
        );
        self.prefix_caches = self
            .prefix_caches
            .into_iter()
            .map(|cache| cache.with_match_unit(match_unit_tokens))
            .collect();
        self
    }

    /// How far a lookup for a prompt of `prompt_tokens` may reach into its
    /// declared prefix. vLLM caps every hit at `num_tokens - 1` so the last
    /// prompt token is computed for logits; the default rule instead borrows
    /// that token in `ResolvedPrefillContext::new`, which keeps it unchanged.
    fn lookup_reach(&self, declared_prefix_tokens: u32, prompt_tokens: u32) -> u32 {
        if self.eagle_replay_boundaries {
            declared_prefix_tokens.min(prompt_tokens.saturating_sub(1))
        } else {
            declared_prefix_tokens
        }
    }

    /// Hard-tier occupancy: whichever of the immediate and the projected total
    /// is larger, including everything reserved but not yet resident.
    fn non_cache_peak(&self, partition: PartitionId) -> u64 {
        if let Some(incremental) = &self.incremental {
            return incremental.partition_tokens[partition as usize]
                + self.ledger.partition_held(partition);
        }
        let reserved =
            self.ledger.partition_promised(partition) + self.ledger.partition_held(partition);
        let partition_state = &self.partitions[partition as usize];
        (partition_state.resident_tokens() + reserved)
            .max(partition_state.projected_peak_kv() + reserved)
    }

    /// Capacity the hard tier is not using — the whole soft budget.
    fn soft_slack(&self, partition: PartitionId) -> u64 {
        self.partitions[partition as usize]
            .capacity_tokens()
            .saturating_sub(self.non_cache_peak(partition))
    }

    /// Snapshots live decodes have dropped behind them, priced as full state
    /// copies. Every `checkpoint_interval_tokens` of context a decode produces
    /// leaves one more resumable snapshot on the way.
    ///
    /// Under the EAGLE replay rule there is no staircase: vLLM's align mode
    /// writes reusable states only inside the prompt (the replay boundaries
    /// [`PrefixCache`] charges on release), so a running decode leaves nothing
    /// behind beyond the one live state the hard tier already holds.
    fn live_checkpoint_charge(&self, partition: PartitionId) -> u64 {
        if self.eagle_replay_boundaries {
            return 0;
        }
        self.partitions[partition as usize]
            .live_checkpoint_count(self.checkpoint_interval_tokens)
            .saturating_mul(self.state_tokens_per_request)
    }

    /// [`Self::live_checkpoint_charge`] clamped to the slack the hard tier
    /// leaves. This is a soft, evictable staircase that never participates in
    /// [`KvStore::fits`]. It therefore cannot starve a resident request, only
    /// the retained-prefix tier.
    fn clamped_live_checkpoint_charge(&self, partition: PartitionId) -> u64 {
        self.live_checkpoint_charge(partition)
            .min(self.soft_slack(partition))
    }

    /// What the retained-prefix tier may occupy: the soft budget minus the
    /// live-decode staircase, which is nearer-term useful and so wins.
    fn prefix_cache_physical_limit(&self, partition: PartitionId) -> u64 {
        self.soft_slack(partition)
            .saturating_sub(self.clamped_live_checkpoint_charge(partition))
    }

    fn trim_prefix_cache_to_physical_slack(
        &mut self,
        partition: PartitionId,
    ) -> Vec<PrefixCacheMutation> {
        let physical_limit = self.prefix_cache_physical_limit(partition);
        self.prefix_caches[partition as usize].shrink_to(physical_limit)
    }

    fn retain_prefix(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        session_id: u32,
        tokens: u64,
        cache_return_metadata: Option<PrefixCacheReturnMetadata>,
        now: Time,
        retain_reason: PrefixCacheRetentionReason,
    ) {
        let physical_limit = self.prefix_cache_physical_limit(partition);
        let insert_result = match self.checkpoint_release {
            CheckpointRelease::OnCompletion => self.prefix_caches[partition as usize].insert(
                session_id,
                tokens,
                physical_limit,
                cache_return_metadata,
            ),
            CheckpointRelease::AtPrefillEnd => self.prefix_caches[partition as usize]
                .complete_context(session_id, tokens, physical_limit),
        };
        for mutation in insert_result.mutations {
            let event_kind = PrefixCacheJournal::retain_event_kind(mutation.kind, retain_reason);
            self.journal
                .record_mutation(request, partition, now, mutation, event_kind, tokens);
        }
        if insert_result.retained_tokens == 0 {
            let cache_used = self.prefix_caches[partition as usize].used_charge();
            self.journal.record(PrefixCacheEvent {
                partition_id: partition,
                time: now,
                request_id: request,
                session_id,
                kind: PrefixCacheEventKind::Retain(PrefixCacheRetentionReason::NoCacheCapacity),
                entry_tokens_before: 0,
                entry_tokens_after: 0,
                cache_used_before: cache_used,
                cache_used_after: cache_used,
                requested_tokens: tokens,
                hit_tokens: 0,
            });
        }
    }
}

impl HybridGdnKv {
    /// The prefill-end half of [`CheckpointRelease::AtPrefillEnd`]: the
    /// prompt's replay-boundary snapshots enter the LRU now.
    fn retain_checkpoints(
        &mut self,
        request: RequestId,
        partition: PartitionId,
        session_id: u32,
        prompt_tokens: u64,
        now: Time,
    ) {
        let physical_limit = self.prefix_cache_physical_limit(partition);
        let insert_result = self.prefix_caches[partition as usize].insert_checkpoints(
            session_id,
            prompt_tokens,
            physical_limit,
        );
        for mutation in insert_result.mutations {
            let event_kind = PrefixCacheJournal::retain_event_kind(
                mutation.kind,
                PrefixCacheRetentionReason::RequestComplete,
            );
            self.journal.record_mutation(
                request,
                partition,
                now,
                mutation,
                event_kind,
                prompt_tokens,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::worker::kv::PrefixCachePolicy;

    /// Qwen3.6-35B-A3B TP1: 30 GDN layers of state over a 10-GQA-layer token
    /// page, and the vLLM block alignment that follows from it.
    const STATE_TOKENS: u64 = 3_168;
    const CHECKPOINT_INTERVAL: u32 = 2_048;

    fn uncapped(max_retained_tokens: u64) -> PrefixCacheTokenConfig {
        PrefixCacheTokenConfig {
            max_retained_tokens,
            policy: PrefixCachePolicy::Lru,
        }
    }

    fn store(kv_capacity: u64, state_tokens: u64, interval: u32) -> HybridGdnKv {
        HybridGdnKv::new(
            1,
            kv_capacity,
            state_tokens,
            interval,
            state_tokens,
            uncapped(kv_capacity),
            None,
            None,
        )
    }

    // ── hard tier ────────────────────────────────────────────────────────────

    #[test]
    fn admission_reserves_context_future_decode_and_state_in_one_check() {
        let kv_store = store(1_000, 100, 512);
        // 800 context + 100 decode + 100 state == capacity, to the token.
        assert!(kv_store.fits(0, &kv_store.footprint(RequestId(0), 800, 100)));
        assert!(
            !kv_store.fits(0, &kv_store.footprint(RequestId(0), 801, 100)),
            "one token over capacity must be refused",
        );
        assert!(
            !kv_store.fits(0, &kv_store.footprint(RequestId(0), 800, 101)),
            "future decode tokens are part of the same atomic reservation",
        );
        let stateless = store(1_000, 0, 0);
        assert!(
            stateless.fits(0, &stateless.footprint(RequestId(0), 900, 100)),
            "the same request fits only once the recurrent state is priced out",
        );
    }

    #[test]
    fn a_promised_state_blocks_the_next_request_before_it_is_resident() {
        let mut kv_store = store(1_000, 100, 512);
        let first = kv_store.footprint(RequestId(0), 400, 0);
        kv_store.reserve(RequestId(0), 0, first, Time::ZERO);
        // 500 promised (400 + 100 state) leaves 500, so 400 context + 100 state fits...
        assert!(kv_store.fits(0, &kv_store.footprint(RequestId(1), 400, 0)));
        // ...but 401 does not, proving the state rode along in the promise.
        assert!(!kv_store.fits(0, &kv_store.footprint(RequestId(1), 401, 0)));
    }

    #[test]
    fn state_is_charged_once_at_commit_never_advanced_and_fully_released() {
        let mut kv_store = store(100_000, STATE_TOKENS, CHECKPOINT_INTERVAL);
        let footprint = kv_store.footprint(RequestId(0), 1_000, 20);
        kv_store.reserve(RequestId(0), 0, footprint, Time::ZERO);
        kv_store.drain_ready();
        kv_store.commit_resident(RequestId(0), 0, 1_000, 20);
        assert_eq!(
            kv_store.partitions[0].resident_tokens(),
            1_000 + STATE_TOKENS
        );

        kv_store.advance(AdvanceScope::WholePartition(0), 10);
        assert_eq!(
            kv_store.partitions[0].resident_tokens(),
            1_010 + STATE_TOKENS,
            "decode grows context only",
        );
        assert_eq!(
            kv_store.partitions[0].decode_members().collect::<Vec<_>>(),
            [(RequestId(0), 1_010)],
            "decode_kv_lens must stay pure context",
        );

        kv_store.release(RequestId(0), 0);
        assert_eq!(kv_store.partitions[0].resident_tokens(), 0);
        assert_eq!(kv_store.ledger.partition_promised(0), 0);
    }

    // ── chunked prefill ──────────────────────────────────────────────────────

    #[test]
    fn a_chunked_prefill_keeps_its_state_promised_until_the_last_chunk_commits_it() {
        let mut kv_store = store(100_000, STATE_TOKENS, CHECKPOINT_INTERVAL);
        let resolved = kv_store.preview_prefill_context(0, 5_000, SessionInput::Standalone);
        let footprint = kv_store.footprint(RequestId(0), 5_000, 20);
        kv_store.reserve_chunked_prefill_context(RequestId(0), 0, resolved, footprint, Time::ZERO);
        kv_store.drain_ready();
        let promised = 5_000 + 20 + STATE_TOKENS;

        for chunk in [2_048, 2_048] {
            kv_store.schedule_prefill_chunk(RequestId(0), 0, chunk);
            assert_eq!(kv_store.ledger.partition_promised(0), promised);
            assert_eq!(kv_store.partitions[0].resident_tokens(), 0);
            kv_store.complete_prefill_chunk(RequestId(0));
            kv_store.clear_prefill_admits(0);
        }
        kv_store.schedule_prefill_chunk(RequestId(0), 0, 904);
        kv_store.complete_prefill_chunk(RequestId(0));
        kv_store.finish_chunked_prefill(RequestId(0), 0, 20, Time::ZERO);
        assert_eq!(kv_store.ledger.partition_promised(0), 0);
        assert_eq!(
            kv_store.partitions[0].resident_tokens(),
            5_000 + STATE_TOKENS,
            "state becomes resident exactly once"
        );

        kv_store.release(RequestId(0), 0);
        assert_eq!(kv_store.partitions[0].resident_tokens(), 0);
    }

    #[test]
    fn a_request_cancelled_mid_prefill_returns_its_whole_reservation() {
        let mut kv_store = store(100_000, STATE_TOKENS, CHECKPOINT_INTERVAL);
        let resolved = kv_store.preview_prefill_context(0, 5_000, SessionInput::Standalone);
        let footprint = kv_store.footprint(RequestId(0), 5_000, 20);
        kv_store.reserve_chunked_prefill_context(RequestId(0), 0, resolved, footprint, Time::ZERO);
        kv_store.schedule_prefill_chunk(RequestId(0), 0, 2_048);
        kv_store.complete_prefill_chunk(RequestId(0));

        kv_store.release(RequestId(0), 0);
        assert_eq!(kv_store.ledger.partition_promised(0), 0);
        assert_eq!(
            kv_store.status_active(0),
            1,
            "the prefill admit clears with the iteration"
        );
        kv_store.clear_prefill_admits(0);
        assert_eq!(kv_store.status_active(0), 0);
    }

    #[test]
    fn a_bounded_future_footprint_still_carries_the_whole_state() {
        let kv_store = store(100_000, STATE_TOKENS, CHECKPOINT_INTERVAL);
        let footprint = kv_store.bounded_future_footprint(RequestId(0), 1_000, 500, 64, 1);
        assert_eq!(footprint.reserved_charge(), 1_000 + 64 + 1 + STATE_TOKENS);
        assert!(kv_store.fits_bounded_future(0, &footprint, 64, 0.7, 1));
        let tight = store(1_000 + 65 + STATE_TOKENS, STATE_TOKENS, CHECKPOINT_INTERVAL);
        assert!(!tight.fits_bounded_future(0, &footprint, 64, 0.7, 1));
    }

    // ── soft tier: quantized reuse ───────────────────────────────────────────

    /// Reproduces the hit rates reported for a hybrid Qwen3 model in
    /// <https://github.com/vllm-project/vllm/issues/40696>, whose aligned block
    /// size is 528 tokens. A prompt shorter than one block gets nothing; longer
    /// prompts reuse only whole blocks, so the hit rate saws downward as the
    /// prompt grows past a boundary.
    #[test]
    fn quantized_reuse_reproduces_the_measured_vllm_hit_rates() {
        const BLOCK: u32 = 528;
        let expected = [
            (479u32, 0u32, 0.0f64),
            (552, 528, 95.4),
            (597, 528, 88.2),
            (979, 528, 53.7),
        ];
        for (prompt_tokens, expected_hit, measured_percent) in expected {
            let mut kv_store = store(10_000_000, STATE_TOKENS, BLOCK);
            // First turn retains what it can; the second turn replays it.
            kv_store.prefix_caches[0].insert(7, u64::from(prompt_tokens), 10_000_000, None);
            let hit = kv_store.prefix_caches[0].peek(7, prompt_tokens);
            assert_eq!(
                hit, expected_hit,
                "prompt {prompt_tokens}: block-aligned reuse",
            );
            let modeled_percent = 100.0 * f64::from(hit) / f64::from(prompt_tokens);
            assert!(
                (modeled_percent - measured_percent).abs() < 0.5,
                "prompt {prompt_tokens}: modeled {modeled_percent:.1}% vs measured {measured_percent}%",
            );
        }
    }

    #[test]
    fn a_retained_prefix_pays_for_the_snapshots_that_make_it_resumable() {
        let mut kv_store = store(10_000_000, STATE_TOKENS, CHECKPOINT_INTERVAL);
        let session = SessionInput::Session {
            session_id: 7,
            session_start_time: Time::ZERO,
            declared_prefix_tokens: 0,
        };
        let resolved_prefill = kv_store.preview_prefill_context(0, 5_000, session);
        let footprint = kv_store.footprint(RequestId(0), 5_000, 0);
        kv_store.reserve_prefill_context(RequestId(0), 0, resolved_prefill, footprint, Time::ZERO);
        kv_store.drain_ready();
        kv_store.commit_resident(RequestId(0), 0, 5_000, 0);
        kv_store.release_retaining_prefix(RequestId(0), 0, Time::from_ms(1.0));

        // 5_000 tokens hold two whole 2_048-token snapshots; the 904-token tail
        // is unusable because no snapshot sits at its end.
        assert_eq!(kv_store.prefix_caches[0].used_tokens(), 2 * 2_048);
        assert_eq!(
            kv_store.prefix_caches[0].used_charge(),
            2 * 2_048 + 2 * STATE_TOKENS,
        );
        assert_eq!(kv_store.prefix_caches[0].peek(7, 5_000), 2 * 2_048);
        assert_eq!(
            kv_store.prefix_caches[0].peek(7, 2_047),
            0,
            "a later snapshot cannot serve an earlier resume point",
        );
    }

    /// The reusable length is bounded by the attention KV **and** by the last
    /// recurrent snapshot. Rather than taking a `min` of two independently
    /// tracked lengths, one entry carries both halves at one aligned length:
    /// they are sized together, charged together, and evicted together, so they
    /// cannot drift apart.
    #[test]
    fn attention_kv_and_recurrent_snapshots_are_retained_at_one_aligned_length() {
        let interval = 2_048u64;
        // Room for exactly two snapshots' worth of both halves, plus change.
        let capacity = 2 * (interval + STATE_TOKENS) + 500;
        let mut kv_store = store(capacity, STATE_TOKENS, interval as u32);

        // Ask to retain five intervals; only two fit, and the entry is cut at a
        // snapshot boundary rather than at whatever the byte budget allowed.
        let insert_result = kv_store.prefix_caches[0].insert(7, 5 * interval, capacity, None);
        assert_eq!(insert_result.retained_tokens, 2 * interval);
        assert_eq!(
            kv_store.prefix_caches[0].used_charge(),
            2 * (interval + STATE_TOKENS),
            "both halves of every retained snapshot are paid for",
        );
        assert_eq!(kv_store.prefix_caches[0].used_tokens() % interval, 0);

        // Eviction takes both halves at once: no orphaned KV, no orphaned state.
        kv_store.prefix_caches[0].shrink_to(0);
        assert_eq!(kv_store.prefix_caches[0].used_tokens(), 0);
        assert_eq!(kv_store.prefix_caches[0].used_charge(), 0);
        assert_eq!(kv_store.prefix_caches[0].peek(7, 5 * interval as u32), 0);
    }

    #[test]
    fn a_prefix_shorter_than_one_snapshot_interval_is_not_retained() {
        let mut kv_store = store(10_000_000, STATE_TOKENS, CHECKPOINT_INTERVAL);
        let insert_result = kv_store.prefix_caches[0].insert(7, 2_047, 10_000_000, None);
        assert_eq!(insert_result.retained_tokens, 0);
        assert_eq!(kv_store.prefix_caches[0].used_charge(), 0);
    }

    #[test]
    fn the_soft_tier_yields_entirely_when_the_hard_tier_fills_the_partition() {
        let capacity = 40_000;
        let mut kv_store = store(capacity, STATE_TOKENS, CHECKPOINT_INTERVAL);
        kv_store.prefix_caches[0].insert(7, 4_096, capacity, None);
        assert!(kv_store.prefix_caches[0].used_charge() > 0);

        // Fill the hard tier: 8 residents × (2_000 context + 3_168 state) ≈ 41k,
        // so admission has to reclaim the whole retained tier to make room.
        for index in 0..8u32 {
            let footprint = kv_store.footprint(RequestId(index), 2_000, 0);
            kv_store.reserve(RequestId(index), 0, footprint, Time::ZERO);
        }
        assert_eq!(kv_store.prefix_cache_physical_limit(0), 0);
        assert_eq!(kv_store.prefix_caches[0].used_charge(), 0);
    }

    // ── soft tier: EAGLE replay boundaries ───────────────────────────────────

    /// GLM-5.3-Flash on H200 with MTP 5: 1152-token blocks, 19 resident state
    /// pages per decoding request, 3 per retained boundary.
    const GLM_BLOCK: u32 = 1_152;
    const GLM_STATE_TOKENS: u64 = 19 * GLM_BLOCK as u64;
    const GLM_CHECKPOINT_TOKENS: u64 = 3 * GLM_BLOCK as u64;

    fn glm_store(eagle: bool) -> HybridGdnKv {
        let kv_store = HybridGdnKv::new(
            1,
            10_000_000,
            GLM_STATE_TOKENS,
            GLM_BLOCK,
            GLM_CHECKPOINT_TOKENS,
            uncapped(10_000_000),
            None,
            None,
        );
        if eagle {
            kv_store.with_eagle_replay_boundaries()
        } else {
            kv_store
        }
    }

    fn session(declared_prefix_tokens: u32) -> SessionInput {
        SessionInput::Session {
            session_id: 7,
            session_start_time: Time::ZERO,
            declared_prefix_tokens,
        }
    }

    /// Run one request of the session to completion, decoding `output` tokens,
    /// and return its resolved prefill.
    fn run_round(
        kv_store: &mut HybridGdnKv,
        request: u32,
        fresh: u32,
        declared: u32,
        output: u32,
    ) -> ResolvedPrefillContext {
        let resolved = kv_store.preview_prefill_context(0, fresh, session(declared));
        let context = resolved.post_prefill_context_tokens();
        let footprint = kv_store.footprint(RequestId(request), context, output);
        kv_store.reserve_prefill_context(RequestId(request), 0, resolved, footprint, Time::ZERO);
        kv_store.drain_ready();
        kv_store.commit_resident(RequestId(request), 0, u64::from(context), output);
        kv_store.advance(AdvanceScope::WholePartition(0), output);
        kv_store.release_retaining_prefix(RequestId(request), 0, Time::from_ms(1.0));
        resolved
    }

    #[test]
    fn the_glm_store_charges_three_pages_per_retained_boundary() {
        let mut kv_store = glm_store(true);
        run_round(&mut kv_store, 0, 20_000, 0, 500);
        assert_eq!(
            kv_store.prefix_caches[0].used_charge(),
            17 * u64::from(GLM_BLOCK) + GLM_CHECKPOINT_TOKENS,
        );
    }

    #[test]
    fn eagle_replay_retains_from_the_prompt_and_recomputes_the_output() {
        let mut kv_store = glm_store(true);
        run_round(&mut kv_store, 0, 20_000, 0, 500);
        // The output decoded past the prompt left no state behind.
        assert_eq!(kv_store.prefix_caches[0].used_tokens(), 17 * 1_152);
        // The next turn declares prompt + output as its prefix.
        let next = kv_store.preview_prefill_context(0, 376, session(20_500));
        assert_eq!(next.resident_prefix_tokens(), 16 * 1_152);
        assert_eq!(next.prefill_tokens_to_compute(), 20_876 - 16 * 1_152);

        // The default rule resumes from the decoded context instead.
        let mut kv_store = glm_store(false);
        run_round(&mut kv_store, 0, 20_000, 0, 500);
        let next = kv_store.preview_prefill_context(0, 376, session(20_500));
        assert_eq!(next.resident_prefix_tokens(), 17 * 1_152);
    }

    #[test]
    fn eagle_replay_chains_across_turns_and_caps_a_resend_at_its_last_token() {
        let mut kv_store = glm_store(true);
        run_round(&mut kv_store, 0, 18 * GLM_BLOCK, 0, 100);
        // An exact resend (nothing fresh) reaches only P - 1: the resend
        // boundary, two blocks below the prompt.
        let resend = run_round(&mut kv_store, 1, 0, 18 * GLM_BLOCK, 100);
        assert_eq!(resend.resident_prefix_tokens(), 16 * GLM_BLOCK);
        assert_eq!(resend.prefill_tokens_to_compute(), 2 * GLM_BLOCK);
        // The resend retained the same prompt again; an extension now resumes
        // at the extension boundary, one block below the prompt.
        let extension = run_round(&mut kv_store, 2, 1_000, 18 * GLM_BLOCK, 100);
        assert_eq!(extension.resident_prefix_tokens(), 17 * GLM_BLOCK);
        // A truncated history below both boundaries recomputes everything.
        let truncated = kv_store.preview_prefill_context(0, 50, session(10_000));
        assert_eq!(truncated.resident_prefix_tokens(), 0);
    }

    #[test]
    fn a_prefix_match_unit_resumes_one_unit_below_each_previous_prompt() {
        // The measured H200 chain (--prefix-match-unit 64): prompts of 20000,
        // 21200 and 22400 tokens, each turn declaring the previous prompt plus
        // its output, resume at floor(P / 64) * 64 - 64 of the previous prompt.
        let mut kv_store = glm_store(true).with_prefix_match_unit(64);
        run_round(&mut kv_store, 0, 20_000, 0, 500);
        let second = run_round(&mut kv_store, 1, 700, 20_500, 500);
        assert_eq!(second.resident_prefix_tokens(), 19_904);
        let third = run_round(&mut kv_store, 2, 700, 21_700, 500);
        assert_eq!(third.resident_prefix_tokens(), 21_120);
        // Without the unit the same chain resumes on the block boundaries.
        let mut kv_store = glm_store(true);
        run_round(&mut kv_store, 0, 20_000, 0, 500);
        let second = run_round(&mut kv_store, 1, 700, 20_500, 500);
        assert_eq!(second.resident_prefix_tokens(), 16 * GLM_BLOCK);
    }

    #[test]
    fn eagle_replay_decodes_leave_no_live_staircase() {
        // A decode crossing many block boundaries past its prompt drops
        // nothing reusable under the EAGLE replay rule; the default rule
        // charges one state per crossed boundary.
        for (eagle, expected_checkpoints) in [(true, 0), (false, 21)] {
            let mut kv_store = glm_store(eagle);
            let resolved = kv_store.preview_prefill_context(0, 20 * GLM_BLOCK, session(0));
            let footprint = kv_store.footprint(RequestId(0), 20 * GLM_BLOCK, 2 * GLM_BLOCK);
            kv_store.reserve_prefill_context(RequestId(0), 0, resolved, footprint, Time::ZERO);
            kv_store.drain_ready();
            kv_store.commit_resident(RequestId(0), 0, u64::from(20 * GLM_BLOCK), 2 * GLM_BLOCK);
            let limit_before = kv_store.prefix_cache_physical_limit(0);
            kv_store.advance(AdvanceScope::WholePartition(0), GLM_BLOCK + 1);
            assert_eq!(
                kv_store.clamped_live_checkpoint_charge(0),
                expected_checkpoints * GLM_STATE_TOKENS,
            );
            if eagle {
                assert_eq!(
                    kv_store.prefix_cache_physical_limit(0),
                    limit_before,
                    "decoding past a block boundary leaves the retained tier untouched",
                );
            }
        }
    }

    // ── soft tier: the live-decode staircase ─────────────────────────────────

    #[test]
    fn live_decode_snapshots_shrink_the_retained_tier_but_never_admission() {
        let capacity = 100_000;
        let interval = 2_048u32;
        let mut kv_store = store(capacity, STATE_TOKENS, interval);
        let footprint = kv_store.footprint(RequestId(0), 2_047, 4);
        kv_store.reserve(RequestId(0), 0, footprint, Time::ZERO);
        kv_store.drain_ready();
        kv_store.commit_resident(RequestId(0), 0, 2_047, 4);

        assert_eq!(kv_store.clamped_live_checkpoint_charge(0), 0);
        let limit_before = kv_store.prefix_cache_physical_limit(0);
        let fits_before = kv_store.fits(0, &kv_store.footprint(RequestId(1), 90_000, 0));

        // One more token crosses the 2_048 boundary and leaves a snapshot behind.
        kv_store.advance(AdvanceScope::WholePartition(0), 1);
        assert_eq!(kv_store.clamped_live_checkpoint_charge(0), STATE_TOKENS);
        assert_eq!(
            kv_store.prefix_cache_physical_limit(0),
            limit_before - STATE_TOKENS,
            "the new snapshot comes straight out of the retained-prefix budget",
        );
        assert_eq!(
            kv_store.fits(0, &kv_store.footprint(RequestId(1), 90_000, 0)),
            fits_before,
            "an evictable snapshot must not change an admission decision",
        );
    }

    #[test]
    fn the_staircase_is_clamped_to_the_slack_it_is_allowed_to_use() {
        // Capacity leaves almost nothing after the hard tier, but the resident
        // request has crossed many snapshot boundaries. The soft charge is
        // clamped rather than overcommitting physical memory.
        let mut kv_store = store(30_000, STATE_TOKENS, 512);
        let footprint = kv_store.footprint(RequestId(0), 25_000, 5);
        kv_store.reserve(RequestId(0), 0, footprint, Time::ZERO);
        kv_store.drain_ready();
        kv_store.commit_resident(RequestId(0), 0, 25_000, 5);

        let slack = kv_store.soft_slack(0);
        assert_eq!(slack, 30_000 - 25_000 - STATE_TOKENS - 5);
        assert!(
            kv_store.partitions[0].live_checkpoint_count(512) * STATE_TOKENS > slack,
            "the unclamped staircase must exceed the slack for this to prove anything",
        );
        assert_eq!(kv_store.clamped_live_checkpoint_charge(0), slack);
        assert_eq!(kv_store.prefix_cache_physical_limit(0), 0);
    }

    #[test]
    fn a_zero_interval_model_degrades_to_plain_full_attention_accounting() {
        let mut kv_store = store(10_000, 0, 0);
        kv_store.prefix_caches[0].insert(7, 137, 10_000, None);
        assert_eq!(kv_store.prefix_caches[0].peek(7, 137), 137);
        assert_eq!(kv_store.prefix_caches[0].used_charge(), 137);
        assert_eq!(kv_store.clamped_live_checkpoint_charge(0), 0);
    }

    // ── incremental allocation (vLLM v1) ─────────────────────────────────────

    /// 400-token pool, 20 tokens of state per request, 16-token blocks, a
    /// 10-token retained boundary state, EAGLE replay boundaries, 5
    /// lookahead slots.
    fn incremental_store(capacity: u64) -> HybridGdnKv {
        HybridGdnKv::new(1, capacity, 20, 16, 10, uncapped(capacity), None, None)
            .with_eagle_replay_boundaries()
            .with_incremental_allocation(5)
    }

    /// Admit, prefill in one chunk and decode one request of session 7 with a
    /// 40-token prompt through the incremental API, stopping before release.
    fn incremental_round_to_decode(kv_store: &mut HybridGdnKv, request: u32) {
        let resolved = kv_store.preview_prefill_context(0, 40, session(0));
        let footprint = kv_store.footprint(RequestId(request), 40, 4);
        kv_store.reserve_chunked_prefill_context(
            RequestId(request),
            0,
            resolved,
            footprint,
            Time::ZERO,
        );
        assert!(kv_store.allocate_incremental(RequestId(request), 0, 40, Time::ZERO));
        kv_store.schedule_prefill_chunk(RequestId(request), 0, 40);
        kv_store.drain_ready();
        kv_store.complete_prefill_chunk(RequestId(request));
        kv_store.finish_chunked_prefill(RequestId(request), 0, 4, Time::from_ms(1.0));
        kv_store.clear_prefill_admits(0);
        kv_store.advance(AdvanceScope::WholePartition(0), 4);
    }

    #[test]
    fn incremental_allocation_takes_whole_blocks_plus_lookahead_and_the_state_once() {
        let mut kv_store = incremental_store(1_000);
        let resolved = kv_store.preview_prefill_context(0, 100, SessionInput::Standalone);
        let footprint = kv_store.footprint(RequestId(0), 100, 500);
        kv_store.reserve_chunked_prefill_context(RequestId(0), 0, resolved, footprint, Time::ZERO);
        assert_eq!(
            kv_store.ledger.partition_promised(0),
            0,
            "nothing is reserved ahead"
        );
        assert!(kv_store.allocate_incremental(RequestId(0), 0, 20, Time::ZERO));
        // 20 + 5 lookahead -> two 16-token blocks, plus the 20-token state.
        assert_eq!(kv_store.non_cache_peak(0), 32 + 20);
        assert!(kv_store.allocate_incremental(RequestId(0), 0, 30, Time::ZERO));
        assert_eq!(kv_store.non_cache_peak(0), 48 + 20);
        assert!(kv_store.allocate_incremental(RequestId(0), 0, 10, Time::ZERO));
        assert_eq!(
            kv_store.non_cache_peak(0),
            48 + 20,
            "allocations never shrink"
        );
        // A request that cannot fit allocates nothing.
        assert!(!kv_store.allocate_incremental(RequestId(1), 0, 912, Time::ZERO));
        assert_eq!(kv_store.non_cache_peak(0), 48 + 20);
        kv_store.release(RequestId(0), 0);
        assert_eq!(kv_store.non_cache_peak(0), 0);
    }

    #[test]
    fn incremental_admission_checks_the_whole_prompt_against_unheld_blocks() {
        let mut kv_store = incremental_store(1_000);
        assert!(kv_store.allocate_incremental(RequestId(0), 0, 75, Time::ZERO)); // 80 + 20
                                                                                 // 1_000 - 100 held: an 880-token prompt (880 + 20) fits, 881 does not,
                                                                                 // whatever its first chunk.
        assert!(kv_store.incremental_prompt_fits(0, 880, 16));
        assert!(!kv_store.incremental_prompt_fits(0, 881, 16));
        // The future decode of the running request is not reserved.
        let mut decoding = incremental_store(1_000);
        incremental_round_to_decode(&mut decoding, 0);
        assert!(decoding.incremental_prompt_fits(0, 880, 16));
    }

    #[test]
    fn an_incremental_step_evicts_retained_kv_and_fails_only_on_held_blocks() {
        let mut kv_store = incremental_store(400);
        incremental_round_to_decode(&mut kv_store, 0);
        kv_store.release_retaining_prefix(RequestId(0), 0, Time::from_ms(2.0));
        assert_eq!(kv_store.non_cache_peak(0), 0);
        assert_eq!(kv_store.prefix_caches[0].used_charge(), 32 + 10);
        assert_eq!(
            kv_store
                .preview_prefill_context(0, 10, session(40))
                .resident_prefix_tokens(),
            16
        );
        // Retained KV is free capacity: 368 + 20 of 400 fits by evicting it.
        assert!(kv_store.allocate_incremental(RequestId(1), 0, 350, Time::from_ms(3.0)));
        assert_eq!(kv_store.prefix_caches[0].used_charge(), 0);
        assert_eq!(
            kv_store
                .preview_prefill_context(0, 10, session(40))
                .resident_prefix_tokens(),
            0
        );
        assert!(!kv_store.allocate_incremental(RequestId(2), 0, 1, Time::from_ms(3.0)));
    }

    #[test]
    fn prefill_end_release_retains_the_snapshots_before_the_request_finishes() {
        let mut kv_store = incremental_store(10_000).with_checkpoint_release_at_prefill_end();
        incremental_round_to_decode(&mut kv_store, 0);
        assert_eq!(
            kv_store.prefix_caches[0].used_charge(),
            10,
            "the boundary state entered the pool at the prefill end"
        );
        assert_eq!(
            kv_store
                .preview_prefill_context(0, 10, session(40))
                .resident_prefix_tokens(),
            0
        );
        kv_store.release_retaining_prefix(RequestId(0), 0, Time::from_ms(2.0));
        assert_eq!(kv_store.prefix_caches[0].used_charge(), 32 + 10);
        assert_eq!(
            kv_store
                .preview_prefill_context(0, 10, session(40))
                .resident_prefix_tokens(),
            16
        );
    }
}
