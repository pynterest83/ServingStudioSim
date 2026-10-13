//! Host (CPU) KV tier: vLLM v0.31's native CPU offload,
//! `SimpleCPUOffloadConnector` (`--kv-offloading-backend native` with
//! `VLLM_USE_SIMPLE_KV_OFFLOAD=1`, eager mode), for a hybrid arch under the
//! EAGLE replay rule.
//!
//! What the connector does (`v1/simple_kv_offload/manager.py`, `worker.py`):
//!
//! - **Capacity.** `cpu_bytes_to_use` (the total `--kv-offloading-size`) is
//!   split evenly over the replica's ranks; each rank pins
//!   `floor(bytes / page_bytes)` blocks of the GPU page size and runs its own
//!   `BlockPool`, an LRU over cached blocks (`_derive_cpu_config`,
//!   `register_kv_caches`). One block holds one attention page or one
//!   recurrent-state page.
//! - **Stores.** Every scheduler step queues, for each scheduled request, the
//!   full-attention blocks its *previous* steps completed and the GPU has
//!   hashed, skipping blocks the host already holds or is copying
//!   (`_prepare_eager_store_specs`, `_classify_store_candidate`). Mamba
//!   `align` state blocks are not positional; the GPU hands every retained
//!   boundary state it caches to the connector, which stores it in the same
//!   step (`boundary_state_offloads`). A finishing request flushes the blocks
//!   it completed since (`_queue_finished_eager_store`). Host blocks are
//!   allocated, evicting the LRU, when the store is queued; they serve hits
//!   only once the D2H copy has completed (`_process_store_completion`). The
//!   copy runs on a low-priority stream after the step's compute, in FIFO
//!   order, overlapping the next step; nothing waits for it.
//! - **Loads.** On a GPU prefix miss the scheduler asks the host for a deeper
//!   hit with the same hybrid coordinator and EAGLE block drop
//!   (`get_num_new_matched_tokens`); the request then waits in
//!   `WAITING_FOR_REMOTE_KVS` while its blocks are copied H2D, and is
//!   scheduled by the first decision that has seen the copy reported done.
//!   Hit blocks are touched, which moves them to the LRU tail.
//! - **Inclusive.** Host blocks stay after a load; a GPU hit never consults
//!   the host.
//!
//! This tier keeps that ledger in token space, per session (traces carry
//! counts, not content, and sessions share no content): present attention
//! runs and boundary states, each an LRU item. Eviction takes whole blocks
//! from the head of the LRU; losing a block in the middle of a session's
//! prefix bounds its contiguous extent, and losing a state removes that resume
//! point, exactly as a missing hash breaks a vLLM lookup.

use std::collections::{BTreeMap, HashMap, VecDeque};

use crate::common::{RequestId, Time};

/// Byte and rate facts of the host tier, as configured.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HostOffloadConfig {
    /// vLLM's `--kv-offloading-size`, in bytes, summed over the replica's ranks.
    pub cpu_bytes_total: u64,
    /// Per-rank host-to-device bandwidth with every rank copying at once.
    pub h2d_bytes_per_s: f64,
    /// Per-rank device-to-host bandwidth with every rank copying at once.
    pub d2h_bytes_per_s: f64,
    /// Fixed cost of one copy, on top of bytes over bandwidth.
    pub transfer_latency: Time,
}

/// One serialized copy direction (vLLM's `load_stream` / `store_stream`).
#[derive(Clone, Copy, Debug)]
struct TransferQueue {
    bytes_per_s: f64,
    latency: Time,
    free_at: Time,
}

impl TransferQueue {
    /// Queue a copy of `bytes` submitted at `now`; returns its start and end.
    fn submit(&mut self, now: Time, bytes: u64) -> (Time, Time) {
        let start = now.max(self.free_at);
        let done = start + self.latency + Time::from_s(bytes as f64 / self.bytes_per_s);
        self.free_at = done;
        (start, done)
    }
}

/// What one LRU key names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HostBlock {
    /// The attention run of `session` starting at this context position.
    Run { session: u32, start: u64 },
    /// The recurrent state of `session` at this boundary.
    State { session: u32, boundary: u64 },
}

/// One session's host-resident blocks.
#[derive(Debug, Default)]
struct HostSession {
    /// Present attention runs: start → (end, LRU sequence). Disjoint.
    runs: BTreeMap<u64, (u64, u64)>,
    /// Present boundary states: boundary → LRU sequence.
    states: BTreeMap<u64, u64>,
    /// Attention ranges being copied to the host.
    copying_runs: Vec<(u64, u64)>,
    /// Boundary states being copied to the host.
    copying_states: Vec<u64>,
}

impl HostSession {
    fn is_empty(&self) -> bool {
        self.runs.is_empty()
            && self.states.is_empty()
            && self.copying_runs.is_empty()
            && self.copying_states.is_empty()
    }

    /// End of the present attention extent starting at `from`: `from` itself
    /// when the block at `from` is absent.
    fn contiguous_end(&self, from: u64) -> u64 {
        let mut end = match self.runs.range(..=from).next_back() {
            Some((_, &(run_end, _))) if run_end > from => run_end,
            _ => return from,
        };
        while let Some(&(next_end, _)) = self.runs.get(&end) {
            end = next_end;
        }
        end
    }

    /// Sub-ranges of `[start, end)` neither present nor being copied.
    fn missing_ranges(&self, start: u64, end: u64) -> Vec<(u64, u64)> {
        let mut covered: Vec<(u64, u64)> = self
            .runs
            .range(..end)
            .map(|(&run_start, &(run_end, _))| (run_start, run_end))
            .chain(self.copying_runs.iter().copied())
            .filter(|&(run_start, run_end)| run_end > start && run_start < end)
            .collect();
        covered.sort_unstable();
        let mut missing = Vec::new();
        let mut cursor = start;
        for (run_start, run_end) in covered {
            if run_start > cursor {
                missing.push((cursor, run_start.min(end)));
            }
            cursor = cursor.max(run_end);
            if cursor >= end {
                break;
            }
        }
        if cursor < end {
            missing.push((cursor, end));
        }
        missing
    }
}

/// A store whose D2H copy is in flight; its blocks are allocated but serve no
/// hit until `done`.
#[derive(Debug)]
struct PendingStore {
    session: u32,
    runs: Vec<(u64, u64)>,
    states: Vec<u64>,
    done: Time,
}

/// One queued store, for logging.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct HostStoreReceipt {
    /// Token equivalents copied: attention tokens plus each state's charge.
    pub(crate) tokens: u64,
    pub(crate) bytes: u64,
    pub(crate) start: Time,
    pub(crate) done: Time,
}

/// One load issued for an admitted request, for logging.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct HostLoadReceipt {
    pub(crate) tokens: u64,
    pub(crate) bytes: u64,
    pub(crate) start: Time,
    pub(crate) done: Time,
}

/// Blocks the LRU gave up, for logging.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HostEviction {
    pub(crate) session: u32,
    /// Token equivalents freed.
    pub(crate) tokens: u64,
}

/// One partition's host tier (one rank's pinned block pool; the ranks of a
/// tensor-parallel replica hold identical layouts and move in lockstep).
pub(crate) struct HostKvTier {
    capacity_tokens: u64,
    /// Attention block size in tokens: the hybrid checkpoint interval.
    block_tokens: u64,
    /// Token equivalents of one boundary's recurrent state (all its groups).
    state_tokens: u64,
    /// Bytes one token equivalent occupies on one rank.
    bytes_per_token: f64,
    /// Present and in-flight token equivalents.
    used_tokens: u64,
    sessions: HashMap<u32, HostSession>,
    /// Oldest first: (sequence, position) → block.
    lru: BTreeMap<(u64, u64), HostBlock>,
    next_sequence: u64,
    in_flight: VecDeque<PendingStore>,
    d2h: TransferQueue,
    h2d: TransferQueue,
    /// Admitted requests whose load has not been consumed: when it lands.
    loads: HashMap<RequestId, Time>,
}

impl HostKvTier {
    /// `capacity_tokens` should be whole blocks (`block_tokens` each).
    pub(crate) fn new(
        capacity_tokens: u64,
        block_tokens: u64,
        state_tokens: u64,
        bytes_per_token: f64,
        config: HostOffloadConfig,
    ) -> Self {
        assert!(block_tokens > 0, "host tier needs a block size");
        assert!(
            config.h2d_bytes_per_s > 0.0 && config.d2h_bytes_per_s > 0.0,
            "host tier bandwidths must be positive"
        );
        Self {
            capacity_tokens,
            block_tokens,
            state_tokens,
            bytes_per_token,
            used_tokens: 0,
            sessions: HashMap::new(),
            lru: BTreeMap::new(),
            next_sequence: 0,
            in_flight: VecDeque::new(),
            d2h: TransferQueue {
                bytes_per_s: config.d2h_bytes_per_s,
                latency: config.transfer_latency,
                free_at: Time::ZERO,
            },
            h2d: TransferQueue {
                bytes_per_s: config.h2d_bytes_per_s,
                latency: config.transfer_latency,
                free_at: Time::ZERO,
            },
            loads: HashMap::new(),
        }
    }

    pub(crate) fn used_tokens(&self) -> u64 {
        self.used_tokens
    }

    fn take_sequence(&mut self) -> u64 {
        let sequence = self.next_sequence;
        self.next_sequence += 1;
        sequence
    }

    fn bytes(&self, tokens: u64) -> u64 {
        (tokens as f64 * self.bytes_per_token).round() as u64
    }

    /// Register every store whose copy completed by `now`, oldest first. Its
    /// blocks join the LRU tail (`_process_store_completion` frees them into
    /// the pool), states before attention as the connector lists them.
    pub(crate) fn settle(&mut self, now: Time) {
        while self
            .in_flight
            .front()
            .is_some_and(|store| store.done <= now)
        {
            let store = self.in_flight.pop_front().expect("checked front");
            for &boundary in &store.states {
                let sequence = self.take_sequence();
                let session = self.sessions.entry(store.session).or_default();
                session
                    .copying_states
                    .retain(|&copying| copying != boundary);
                session.states.insert(boundary, sequence);
                self.lru.insert(
                    (sequence, boundary),
                    HostBlock::State {
                        session: store.session,
                        boundary,
                    },
                );
            }
            let sequence = self.take_sequence();
            for &(start, end) in &store.runs {
                let session = self.sessions.entry(store.session).or_default();
                session
                    .copying_runs
                    .retain(|&copying| copying != (start, end));
                session.runs.insert(start, (end, sequence));
                self.lru.insert(
                    (sequence, start),
                    HostBlock::Run {
                        session: store.session,
                        start,
                    },
                );
            }
        }
    }

    /// The deepest resume point the host serves `session` beyond a GPU hit of
    /// `gpu_hit_tokens`, for a lookup that may reach `reach_tokens` (the
    /// declared prefix capped at the prompt minus its last token): the hybrid
    /// coordinator matches whole attention blocks from the GPU hit on, the
    /// EAGLE drop gives up the last one, and the recurrent finder needs a
    /// state at or below what is left. `0` when nothing beyond the GPU hit is
    /// resumable.
    pub(crate) fn peek(&self, session: u32, gpu_hit_tokens: u32, reach_tokens: u32) -> u32 {
        let Some(entry) = self.sessions.get(&session) else {
            return 0;
        };
        let from = u64::from(gpu_hit_tokens);
        let reach = u64::from(reach_tokens) / self.block_tokens * self.block_tokens;
        let matched = entry.contiguous_end(from).min(reach);
        let candidate = matched.saturating_sub(self.block_tokens);
        if candidate <= from {
            return 0;
        }
        entry
            .states
            .range(from + 1..=candidate)
            .next_back()
            .map_or(0, |(&boundary, _)| boundary as u32)
    }

    /// Queue the D2H copy of `session`'s attention `[start, end)` (whole
    /// blocks) and recurrent states at `boundaries`, skipping what the host
    /// holds or is copying. Host blocks are allocated now, evicting the LRU;
    /// what still does not fit is dropped (`num_free` in the connector),
    /// attention from the end first. Returns `None` when nothing is copied.
    pub(crate) fn store(
        &mut self,
        now: Time,
        session: u32,
        start: u64,
        end: u64,
        boundaries: &[u64],
        evictions: &mut Vec<HostEviction>,
    ) -> Option<HostStoreReceipt> {
        let block = self.block_tokens;
        let (start, end) = (start / block * block, end / block * block);
        let entry = self.sessions.entry(session).or_default();
        let mut runs = if start < end {
            entry.missing_ranges(start, end)
        } else {
            Vec::new()
        };
        let mut states: Vec<u64> = boundaries
            .iter()
            .copied()
            .filter(|boundary| {
                *boundary > 0
                    && !entry.states.contains_key(boundary)
                    && !entry.copying_states.contains(boundary)
            })
            .collect();
        states.sort_unstable();
        states.dedup();
        let mut tokens =
            runs.iter().map(|(s, e)| e - s).sum::<u64>() + states.len() as u64 * self.state_tokens;
        if tokens == 0 {
            return None;
        }
        self.evict_to_fit(tokens, evictions);
        let free = self.capacity_tokens.saturating_sub(self.used_tokens);
        while tokens > free {
            if let Some(last) = runs.last_mut() {
                last.1 -= block;
                tokens -= block;
                if last.0 == last.1 {
                    runs.pop();
                }
            } else if states.pop().is_some() {
                tokens -= self.state_tokens;
            } else {
                break;
            }
        }
        if tokens == 0 {
            return None;
        }
        self.used_tokens += tokens;
        let entry = self.sessions.entry(session).or_default();
        entry.copying_runs.extend(runs.iter().copied());
        entry.copying_states.extend(states.iter().copied());
        let bytes = self.bytes(tokens);
        let (copy_start, done) = self.d2h.submit(now, bytes);
        self.in_flight.push_back(PendingStore {
            session,
            runs,
            states,
            done,
        });
        Some(HostStoreReceipt {
            tokens,
            bytes,
            start: copy_start,
            done,
        })
    }

    /// Issue the H2D copy for an admitted `request` that resumes `session` at
    /// `hit_tokens` from the host after a GPU hit of `gpu_hit_tokens`: the
    /// attention blocks between them and the state at the hit. The copied
    /// blocks move to the LRU tail (touched while pinned, freed when the load
    /// ends).
    pub(crate) fn load(
        &mut self,
        now: Time,
        request: RequestId,
        session: u32,
        gpu_hit_tokens: u32,
        hit_tokens: u32,
    ) -> HostLoadReceipt {
        let (from, hit) = (u64::from(gpu_hit_tokens), u64::from(hit_tokens));
        debug_assert!(hit > from);
        let sequence = self.take_sequence();
        let run_sequence = self.take_sequence();
        let entry = self
            .sessions
            .get_mut(&session)
            .expect("a host load needs a host entry");
        let state_sequence = entry
            .states
            .get_mut(&hit)
            .expect("a host hit has its boundary state");
        self.lru.remove(&(*state_sequence, hit));
        *state_sequence = sequence;
        self.lru.insert(
            (sequence, hit),
            HostBlock::State {
                session,
                boundary: hit,
            },
        );
        // Re-stamp the attention blocks [from, hit) as one run.
        let overlapping: Vec<(u64, u64, u64)> = entry
            .runs
            .range(..hit)
            .filter(|(_, &(end, _))| end > from)
            .map(|(&start, &(end, run_sequence))| (start, end, run_sequence))
            .collect();
        for (start, end, old_sequence) in overlapping {
            entry.runs.remove(&start);
            self.lru.remove(&(old_sequence, start));
            if start < from {
                entry.runs.insert(start, (from, old_sequence));
                self.lru
                    .insert((old_sequence, start), HostBlock::Run { session, start });
            }
            if end > hit {
                entry.runs.insert(hit, (end, old_sequence));
                self.lru.insert(
                    (old_sequence, hit),
                    HostBlock::Run {
                        session,
                        start: hit,
                    },
                );
            }
        }
        entry.runs.insert(from, (hit, run_sequence));
        self.lru.insert(
            (run_sequence, from),
            HostBlock::Run {
                session,
                start: from,
            },
        );
        let tokens = hit - from + self.state_tokens;
        let bytes = self.bytes(tokens);
        let (start, done) = self.h2d.submit(now, bytes);
        self.loads.insert(request, done);
        HostLoadReceipt {
            tokens,
            bytes,
            start,
            done,
        }
    }

    /// When `request`'s load lands, while it has not started running.
    pub(crate) fn load_ready_at(&self, request: RequestId) -> Option<Time> {
        self.loads.get(&request).copied()
    }

    /// `request` started running, or left: its load is no longer awaited.
    pub(crate) fn forget_load(&mut self, request: RequestId) {
        self.loads.remove(&request);
    }

    /// Evict LRU blocks until `tokens` more fit or nothing evictable is left.
    fn evict_to_fit(&mut self, tokens: u64, evictions: &mut Vec<HostEviction>) {
        while self.used_tokens + tokens > self.capacity_tokens {
            let Some((&(sequence, position), &block)) = self.lru.iter().next() else {
                return;
            };
            match block {
                HostBlock::State { session, boundary } => {
                    self.lru.remove(&(sequence, position));
                    let entry = self.sessions.get_mut(&session).expect("lru session");
                    entry.states.remove(&boundary);
                    self.used_tokens -= self.state_tokens;
                    if entry.is_empty() {
                        self.sessions.remove(&session);
                    }
                    evictions.push(HostEviction {
                        session,
                        tokens: self.state_tokens,
                    });
                }
                HostBlock::Run { session, start } => {
                    self.lru.remove(&(sequence, position));
                    let entry = self.sessions.get_mut(&session).expect("lru session");
                    let (end, _) = entry.runs.remove(&start).expect("lru run");
                    let deficit = self.used_tokens + tokens - self.capacity_tokens;
                    let blocks = deficit.div_ceil(self.block_tokens);
                    let freed = (blocks * self.block_tokens).min(end - start);
                    let new_start = start + freed;
                    if new_start < end {
                        entry.runs.insert(new_start, (end, sequence));
                        self.lru.insert(
                            (sequence, new_start),
                            HostBlock::Run {
                                session,
                                start: new_start,
                            },
                        );
                    }
                    self.used_tokens -= freed;
                    if entry.is_empty() {
                        self.sessions.remove(&session);
                    }
                    evictions.push(HostEviction {
                        session,
                        tokens: freed,
                    });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLOCK: u64 = 16;
    const STATE: u64 = 48;

    /// 1 byte per token at 1e6 B/s both ways plus 10 µs: a 16-token block
    /// copies in 26 µs.
    fn tier(capacity_blocks: u64) -> HostKvTier {
        HostKvTier::new(
            capacity_blocks * BLOCK,
            BLOCK,
            STATE,
            1.0,
            HostOffloadConfig {
                cpu_bytes_total: 0,
                h2d_bytes_per_s: 1e6,
                d2h_bytes_per_s: 1e6,
                transfer_latency: Time::from_us(10),
            },
        )
    }

    fn store_prompt(tier: &mut HostKvTier, now: Time, session: u32, prompt: u64) {
        // The EAGLE replay boundaries of a prompt of `prompt` tokens.
        let resend = ((prompt - 1) / BLOCK * BLOCK).saturating_sub(BLOCK);
        let extension = (prompt / BLOCK * BLOCK).saturating_sub(BLOCK);
        tier.store(
            now,
            session,
            0,
            prompt,
            &[resend, extension],
            &mut Vec::new(),
        );
    }

    #[test]
    fn a_store_serves_hits_only_once_its_copy_completed() {
        let mut tier = tier(100);
        // 40 tokens: two blocks plus the state at 16.
        let receipt = tier
            .store(Time::ZERO, 7, 0, 40, &[16], &mut Vec::new())
            .unwrap();
        assert_eq!(receipt.tokens, 2 * BLOCK + STATE);
        assert_eq!(receipt.done, Time::from_us(10 + 2 * BLOCK + STATE));
        assert_eq!(tier.used_tokens(), 2 * BLOCK + STATE, "allocated at once");
        assert_eq!(tier.peek(7, 0, 40), 0, "still copying");
        tier.settle(receipt.done);
        // Reach 40 matches 32 tokens, the EAGLE drop leaves 16: the state.
        assert_eq!(tier.peek(7, 0, 40), 16);
        // A shorter reach drops below the only state.
        assert_eq!(tier.peek(7, 0, 31), 0);
    }

    #[test]
    fn stores_are_fifo_and_skip_what_the_host_holds() {
        let mut tier = tier(100);
        let first = tier
            .store(Time::ZERO, 7, 0, 32, &[16], &mut Vec::new())
            .unwrap();
        // Issued while the first copies: it waits for the stream.
        let second = tier
            .store(Time::from_us(1), 8, 0, 16, &[], &mut Vec::new())
            .unwrap();
        assert_eq!(second.start, first.done);
        // Re-storing session 7's prompt while it copies copies nothing.
        assert!(tier
            .store(Time::from_us(2), 7, 0, 32, &[16], &mut Vec::new())
            .is_none());
        tier.settle(second.done);
        // A longer turn copies only the new block and state.
        let third = tier
            .store(second.done, 7, 0, 48, &[16, 32], &mut Vec::new())
            .unwrap();
        assert_eq!(third.tokens, BLOCK + STATE);
    }

    #[test]
    fn the_lru_evicts_whole_blocks_oldest_first_and_breaks_the_prefix() {
        // 10 blocks; session 7's 48-token prompt takes 3 blocks and the
        // states at 16 and 32 (6 blocks).
        let mut tier = tier(10);
        store_prompt(&mut tier, Time::ZERO, 7, 48);
        tier.settle(Time::from_s(1.0));
        assert_eq!(tier.used_tokens(), 3 * BLOCK + 2 * STATE);
        assert_eq!(tier.peek(7, 0, 48), 32);
        // The oldest item, the state at 16, goes first: the deeper hit stays.
        let mut evictions = Vec::new();
        tier.store(Time::from_s(1.0), 8, 0, 48, &[], &mut evictions);
        assert_eq!(
            evictions,
            [HostEviction {
                session: 7,
                tokens: STATE
            }]
        );
        assert_eq!(tier.peek(7, 0, 48), 32);
        // Then the state at 32, which takes the hit with it.
        tier.settle(Time::from_s(2.0));
        let mut evictions = Vec::new();
        tier.store(Time::from_s(2.0), 9, 0, 32, &[], &mut evictions);
        assert_eq!(
            evictions,
            [HostEviction {
                session: 7,
                tokens: STATE
            }]
        );
        assert_eq!(tier.peek(7, 0, 48), 0);
        // Then session 7's attention, from its front, one block at a time.
        tier.settle(Time::from_s(3.0));
        let mut evictions = Vec::new();
        tier.store(Time::from_s(3.0), 10, 0, 64, &[], &mut evictions);
        assert_eq!(
            evictions,
            [HostEviction {
                session: 7,
                tokens: 2 * BLOCK
            }]
        );
        assert_eq!(tier.used_tokens(), 10 * BLOCK);
        assert_eq!(tier.sessions[&7].runs.get(&32), Some(&(48, 2)));
    }

    #[test]
    fn a_hit_moves_its_blocks_to_the_lru_tail_and_times_the_load() {
        let mut tier = tier(20);
        store_prompt(&mut tier, Time::ZERO, 7, 48);
        tier.settle(Time::from_s(1.0));
        store_prompt(&mut tier, Time::from_s(1.0), 8, 32);
        tier.settle(Time::from_s(2.0));
        // Session 7 is older; a hit on it at 32 refreshes [0, 32) and the
        // state at 32.
        assert_eq!(tier.peek(7, 0, 48), 32);
        let receipt = tier.load(Time::from_s(2.0), RequestId(3), 7, 0, 32);
        assert_eq!(receipt.tokens, 32 + STATE);
        assert_eq!(
            receipt.done,
            Time::from_s(2.0) + Time::from_us(10 + 32 + STATE)
        );
        assert_eq!(tier.load_ready_at(RequestId(3)), Some(receipt.done));
        // Pressure evicts session 7's untouched state at 16 and block
        // [32, 48) first, then session 8, and spares the refreshed blocks.
        let mut evictions = Vec::new();
        tier.store(Time::from_s(3.0), 9, 0, 224, &[], &mut evictions);
        assert_eq!(
            evictions,
            [
                HostEviction {
                    session: 7,
                    tokens: STATE
                },
                HostEviction {
                    session: 7,
                    tokens: BLOCK
                },
                HostEviction {
                    session: 8,
                    tokens: STATE
                },
                HostEviction {
                    session: 8,
                    tokens: BLOCK
                },
            ]
        );
        assert!(tier.sessions[&7].states.contains_key(&32));
        assert_eq!(tier.sessions[&7].contiguous_end(0), 32);
        // The EAGLE drop needed the evicted block above the state.
        assert_eq!(tier.peek(7, 0, 48), 0);
        tier.forget_load(RequestId(3));
        assert_eq!(tier.load_ready_at(RequestId(3)), None);
    }

    #[test]
    fn a_lookup_starts_at_the_gpu_hit_and_must_go_deeper() {
        let mut tier = tier(100);
        // A 64-token prompt: resend boundary 32, extension boundary 48.
        store_prompt(&mut tier, Time::ZERO, 7, 64);
        tier.settle(Time::from_s(1.0));
        assert_eq!(tier.peek(7, 0, 64), 48);
        assert_eq!(
            tier.peek(7, 0, 63),
            32,
            "a resend reaches the resend boundary"
        );
        assert_eq!(tier.peek(7, 16, 64), 48);
        assert_eq!(tier.peek(7, 48, 64), 0, "nothing beyond the GPU hit");
        // A load from a GPU hit copies only the blocks beyond it.
        let receipt = tier.load(Time::from_s(1.0), RequestId(0), 7, 16, 48);
        assert_eq!(receipt.tokens, 32 + STATE);
    }

    #[test]
    fn a_store_that_cannot_fit_is_trimmed_from_its_attention_end() {
        // 4 blocks; the in-flight store pins everything it allocated.
        let mut tier = tier(4);
        let receipt = tier
            .store(Time::ZERO, 7, 0, 96, &[16], &mut Vec::new())
            .unwrap();
        // The state (3 blocks) and one attention block fit.
        assert_eq!(receipt.tokens, STATE + BLOCK);
        assert!(tier
            .store(Time::ZERO, 8, 0, 16, &[], &mut Vec::new())
            .is_none());
    }
}
