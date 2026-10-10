//! `OutputLag` — the whole-iteration shell's one-step output delay under async
//! scheduling.
//!
//! Owns only when a finished step's output becomes visible outside the engine:
//! the client-observed token times and the completion events. It does not own
//! progress, KV, or batch membership: the lifecycle still retires every step's
//! tokens and releases its KV when the step ends, so the scheduler's next
//! decision sees them.
//!
//! vLLM's async engine loop (`EngineCore.step_with_batch_queue`,
//! `vllm/v1/engine/core.py` ~L760-876) launches step k+1 before it processes
//! step k's sampled output, and processes it only when the queue must drain:
//! with k+1 running, step k's tokens and finished requests reach the client at
//! the end of k+1. When the engine launches nothing after k, it processes k's
//! output at once.
//!
//! Cadence, driven by `IterBatchWorker::tick` at each step end:
//!
//! 1. `publish_held`: the previous step's output, held because this step was
//!    launched after it, becomes visible now.
//! 2. `snapshot_members` → lifecycle `complete_iteration` into `step_events` →
//!    `hold_step_output`: this step's output is held.
//! 3. If the next decision launches nothing, `publish_held_now` releases it at
//!    this step's end, unchanged.
//!
//! Invariant: output is held for at most one step; it is published either at
//! the end of the next step or at the end of its own.

use crate::common::{RequestId, SharedRequests, Time};
use crate::worker::kv::IterWorkerKv;
use crate::worker::types::IterBatchPlan;

/// Tokens one request produced in the held step.
struct HeldOutput {
    request: RequestId,
    tokens: u32,
    /// The step produced the request's first token, so TTFT moves too.
    includes_first: bool,
}

pub(super) struct OutputLag<Event> {
    held_outputs: Vec<HeldOutput>,
    held_events: Vec<Event>,
    /// Scratch: each member of the step being completed and the tokens it had
    /// emitted before the step's output was retired.
    members: Vec<(RequestId, u32)>,
    /// Scratch: the completing step's events, before they are held.
    pub(super) step_events: Vec<Event>,
}

impl<Event> OutputLag<Event> {
    pub(super) fn new() -> Self {
        Self {
            held_outputs: Vec::new(),
            held_events: Vec::new(),
            members: Vec::new(),
            step_events: Vec::new(),
        }
    }

    /// The previous step's output becomes visible at `now`, the end of the
    /// step launched after it.
    pub(super) fn publish_held(
        &mut self,
        requests: &SharedRequests,
        now: Time,
        events: &mut Vec<Event>,
    ) {
        if !self.held_outputs.is_empty() {
            let mut store = requests.borrow_mut();
            for held in self.held_outputs.drain(..) {
                store[held.request].delay_latest_outputs(held.tokens, held.includes_first, now);
            }
        }
        events.append(&mut self.held_events);
    }

    /// Record how many tokens each member of the completing step had emitted,
    /// before the lifecycle retires the step.
    pub(super) fn snapshot_members<K: IterWorkerKv>(
        &mut self,
        kv_store: &K,
        batch_plan: &IterBatchPlan,
        requests: &SharedRequests,
    ) {
        self.members.clear();
        let store = requests.borrow();
        for partition in 0..kv_store.num_partitions() as u16 {
            kv_store.visit_prefill_admits(partition, |request| {
                self.members
                    .push((request, store[request].progress.output_tokens_emitted));
            });
            if batch_plan.partition_runs_decode(partition) {
                kv_store.visit_decode_members(partition, |request, _| {
                    self.members
                        .push((request, store[request].progress.output_tokens_emitted));
                });
            }
        }
    }

    /// Hold the output the lifecycle just retired: every member that emitted
    /// tokens, and the step's events.
    pub(super) fn hold_step_output(&mut self, requests: &SharedRequests) {
        debug_assert!(self.held_outputs.is_empty() && self.held_events.is_empty());
        let store = requests.borrow();
        for &(request, emitted_before) in &self.members {
            let emitted = store[request].progress.output_tokens_emitted;
            if emitted > emitted_before {
                self.held_outputs.push(HeldOutput {
                    request,
                    tokens: emitted - emitted_before,
                    includes_first: emitted_before == 0,
                });
            }
        }
        std::mem::swap(&mut self.held_events, &mut self.step_events);
    }

    /// The engine launched no next step: the held output is processed at the
    /// end of its own step, where its tokens are already stamped.
    pub(super) fn publish_held_now(&mut self, events: &mut Vec<Event>) {
        self.held_outputs.clear();
        events.append(&mut self.held_events);
    }
}
