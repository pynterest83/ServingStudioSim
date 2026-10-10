//! `AsyncEngineLoop` — the whole-iteration shell's model of vLLM's async engine
//! loop (`EngineCore.step_with_batch_queue`, `vllm/v1/engine/core.py`
//! ~L760-876).
//!
//! That loop launches step k, schedules step k+1 at once, and blocks on k's
//! result. When k ends it processes k's output, polls its input queue, and
//! schedules k+2 while k+1 runs. This file owns the two shell-side facts of
//! that loop: when an arriving message enters the engine, and when a finished
//! step's output becomes visible. It does not own batch formation. The shell
//! asks for k+1's batch at the end of k, and the lifecycle answers from what
//! was known when k launched (`ChunkedPrefillAdmission::with_async_scheduling`).
//!
//! Input, driven by `IterBatchWorker::enqueue` and `tick`:
//!
//! 1. `pass_or_hold`: an idle engine runs its first input at once. Every later
//!    message, including one at the same instant, waits in `input_queue`.
//! 2. `messages_after_launch`: at the end of step k the shell first forms
//!    k+1's batch, then hands the messages that arrived during k to admission.
//!    They are queued at that boundary and first eligible for k+2. After an
//!    idle engine's first decision no step has ended: the messages that came
//!    with its first input wait for the end of that first step.
//! 3. `messages_for_redecision`: if the decision for k+1 launches nothing, the
//!    engine waits for k's output instead, picks every message up, and decides
//!    again at once.
//!
//! Output (`snapshot_members` → lifecycle `complete_iteration` into
//! `step_events` → `defer_step_output` → `publish_processed`): step k's tokens
//! and completion events become visible at the end of k plus
//! `output_processing`. Progress and KV still retire at the end of k. With no
//! processing delay the shell completes straight into the event sink.
//!
//! Invariants: a message waits for at most one step, and every step's output
//! is published exactly once, in step order.

use std::collections::VecDeque;

use crate::common::{RequestId, SharedRequests, Time};
use crate::worker::kv::IterWorkerKv;
use crate::worker::types::IterBatchPlan;

pub(super) struct AsyncEngineLoop<Msg, Event> {
    /// Messages the engine has not picked up yet (EngineCore's input queue).
    input_queue: Vec<Msg>,
    /// The engine woke on an input while idle and launches it at this
    /// instant; later inputs wait for that step to end.
    woken: bool,
    /// A step ended at this tick and no decision has been made since.
    step_ended: bool,
    /// From the end of a step to the moment its output is visible.
    output_processing: Time,
    /// Scratch: each member of the step being completed and the tokens it had
    /// emitted before the step's output was retired.
    members: Vec<(RequestId, u32)>,
    /// Scratch: the completing step's events, before they are deferred.
    step_events: Vec<Event>,
    /// Completed steps' events and the time they become visible, in order.
    processing: VecDeque<(Time, Vec<Event>)>,
}

impl<Msg, Event> AsyncEngineLoop<Msg, Event> {
    pub(super) fn new(output_processing: Time) -> Self {
        Self {
            input_queue: Vec::new(),
            woken: false,
            step_ended: false,
            output_processing,
            members: Vec::new(),
            step_events: Vec::new(),
            processing: VecDeque::new(),
        }
    }

    /// Return `msg` when an idle engine picks it up at once; otherwise hold
    /// it until the running (or just launched) step ends.
    pub(super) fn pass_or_hold(&mut self, msg: Msg, step_running: bool) -> Option<Msg> {
        if step_running || self.woken {
            self.input_queue.push(msg);
            return None;
        }
        debug_assert!(self.input_queue.is_empty());
        self.woken = true;
        Some(msg)
    }

    pub(super) fn held_messages(&self) -> usize {
        self.input_queue.len()
    }

    /// Every held message, for a caller that must reach one now (cancellation).
    pub(super) fn take_all_messages(&mut self) -> std::vec::Drain<'_, Msg> {
        self.input_queue.drain(..)
    }

    pub(super) fn note_step_end(&mut self) {
        self.step_ended = true;
    }

    /// A decision launched a step. If it was made at a step's end, the
    /// messages that arrived during that step are picked up now, for the next
    /// decision.
    pub(super) fn messages_after_launch(&mut self) -> std::vec::Drain<'_, Msg> {
        self.woken = false;
        if std::mem::take(&mut self.step_ended) {
            self.input_queue.drain(..)
        } else {
            self.input_queue.drain(0..0)
        }
    }

    /// A decision launched nothing. If a step just ended (or messages wait),
    /// the engine picks every message up and decides again at once; `None`
    /// means it goes idle.
    pub(super) fn messages_for_redecision(&mut self) -> Option<std::vec::Drain<'_, Msg>> {
        self.woken = false;
        let redecide = std::mem::take(&mut self.step_ended) || !self.input_queue.is_empty();
        redecide.then(|| self.input_queue.drain(..))
    }

    /// Whether step output waits for processing (otherwise the shell completes
    /// straight into the event sink).
    pub(super) fn delays_output(&self) -> bool {
        self.output_processing > Time::ZERO
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

    /// The sink the lifecycle completes the step into.
    pub(super) fn step_events(&mut self) -> &mut Vec<Event> {
        &mut self.step_events
    }

    /// The step that ended at `now` is retired: its tokens become visible, and
    /// its events are published, after the output processing delay.
    pub(super) fn defer_step_output(&mut self, requests: &SharedRequests, now: Time) {
        let visible = now + self.output_processing;
        let mut store = requests.borrow_mut();
        for &(request, emitted_before) in &self.members {
            let record = &mut store[request];
            let emitted = record.progress.output_tokens_emitted;
            if emitted > emitted_before {
                record.delay_latest_outputs(emitted - emitted_before, emitted_before == 0, visible);
            }
        }
        if !self.step_events.is_empty() {
            self.processing
                .push_back((visible, std::mem::take(&mut self.step_events)));
        }
    }

    /// Publish every step output whose processing finished by `now`.
    pub(super) fn publish_processed(&mut self, now: Time, events: &mut Vec<Event>) {
        while self
            .processing
            .front()
            .is_some_and(|(visible, _)| *visible <= now)
        {
            let (_, mut step_events) = self.processing.pop_front().expect("checked front");
            events.append(&mut step_events);
        }
    }

    /// When the next deferred step output becomes visible.
    pub(super) fn next_publication(&self) -> Option<Time> {
        self.processing.front().map(|(visible, _)| *visible)
    }
}
