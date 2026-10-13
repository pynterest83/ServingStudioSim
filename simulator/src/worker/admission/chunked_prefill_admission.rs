//! Hard-capped chunked-prefill lifecycle for the whole-iteration shell.
//!
//! Pending policies own fresh and retracted requests. Once a prompt starts, its
//! continuation stays in a partition-local started list, in start order, and
//! never rotates through the policy again. Without a chunk-end quantum a partial
//! chunk always spends the rest of the budget, so that list holds at most one
//! request. A chunk-end quantum (vLLM's Mamba `align` cache mode) clips chunks
//! to recurrent-state checkpoint boundaries; the leftover budget then admits
//! further prompts and several may be partial at once. KV ownership is policy-defined:
//! historical deployments reserve the complete request footprint, whereas a
//! bounded-future deployment pairs a waiting-request estimate with decode-time
//! physical allocation checks and retraction. Under `Mix` the decode check runs
//! before admission and a retracting step admits nothing (vLLM); under
//! `SeparatePrefillPriority` it runs on decode-only iterations (SGLang).
//!
//! Batch composition is independent of KV membership. `Mix` lets resident
//! decode share the remaining chunk budget. `SeparatePrefillPriority` emits a
//! prefill-only iteration whenever a prompt chunk can run, leaving decode
//! resident and unadvanced until a later decode-only iteration. This is the
//! mechanism selected by SGLang when `enable_mixed_chunk` is false; admission
//! capacity estimation and decode retraction are separate policies.
//!
//! Under async scheduling (`with_async_scheduling`) vLLM forms the batch for
//! step k+1 when step k launches, without k's output. The whole-iteration shell
//! still asks for that batch at the end of k, so this lifecycle answers from
//! what was known when k launched:
//!
//! - a request that finishes at k only because drafts were accepted keeps one
//!   trailing verify row in k+1: it stays KV-resident and billed for that
//!   iteration, emits nothing, and is released at its end;
//! - KV that k's other completions free is released only after the decision
//!   for k+1, so it first serves k+2;
//! - a request admitted by that decision is stamped first scheduled when k
//!   launched, the moment vLLM made it;
//! - each request's EngineCore `QUEUED` time is the decision boundary at which
//!   the shell delivered it, or its arrival at an idle engine.
//!
//! The shell owns the other half: it delivers a request that arrived during k
//! only after the decision for k+1, so it first runs in k+2.
//!
//! Under incremental KV admission (`KvAdmissionConfig::Incremental`, vLLM v1)
//! nothing is reserved ahead: each decision first allocates the next step of
//! every running decode, then of every started prompt, in admission order,
//! preempting the most recently admitted running request whenever one of them
//! cannot be allocated (`preempt_newest_running`); a decision that preempted
//! admits nobody. A waiting prompt is admitted when the KV store says its
//! whole prompt fits the blocks no running request holds; its chunk is then
//! allocated like any other. A preempted decode recomputes its prompt and
//! output from the front of the queue; a preempted unfinished prompt restarts.
//! `with_max_running_requests` (vLLM's `max_num_seqs`) caps admission in
//! every KV mode.
//!
//! With a host KV tier (vLLM's CPU offload) a waiting prompt whose deepest
//! prefix sits in host memory is admitted without running: the KV store
//! copies the prefix in, and the prompt waits outside the running set (vLLM's
//! `WAITING_FOR_REMOTE_KVS`, never preempted, no budget spent) until a
//! decision sees the copy done (`ChunkedPrefillKv::host_load_ready_at`). It is
//! then scheduled ahead of fresh requests, in admission order, and stamped
//! first scheduled. At least one step passes in between.

use std::cmp::Reverse;
use std::collections::HashMap;

use crate::common::{RequestId, SessionInput, Time, UnifiedStage};
use crate::worker::config::{
    BatchPolicy, BoundedFutureKvAdmissionConfig, DecodeRetractionPolicy, KvAdmissionConfig,
};
use crate::worker::kv::{ChunkedPrefillKv, ResolvedPrefillContext};
use crate::worker::shared::context::WorkerContext;
use crate::worker::types::{IterBatchPlan, WorkerEventCommon, WorkerMsgCommon};

use super::{
    AdmissionCandidate, DecodeCompletion, EnqueueSequence, IterAdmission, LoadBalance,
    PendingOrderPolicy, SingleTokenDecodeCompletion,
};

pub struct ChunkedPrefillAdmission<P: PendingOrderPolicy, D = SingleTokenDecodeCompletion> {
    partition_policies: Vec<(P, P::Context)>,
    enqueue_sequence: EnqueueSequence,
    max_batch_tokens: u32,
    batch_policy: BatchPolicy,
    kv_admission: KvAdmissionConfig,
    current_new_token_ratio: f64,
    balance: LoadBalance,
    /// Per partition: prompts that have started and still have prefill tokens
    /// left to schedule, in start order.
    started_prefills: Vec<Vec<AdmissionCandidate>>,
    /// Per partition: admitted prompts that wait for a host KV load before
    /// they first run, in admission order. Not running: they spend no budget,
    /// count against no running cap, and are never preempted.
    loading_prefills: Vec<Vec<AdmissionCandidate>>,
    /// Context-token spacing that every non-final chunk must end on, when the
    /// engine can only checkpoint recurrent state at those boundaries.
    chunk_end_quantum: Option<u32>,
    /// vLLM's `use_eagle_block_drop` (speculative drafter with prefix caching):
    /// the prompt's last cacheable boundary backs off one `chunk_end_quantum`,
    /// because full attention prunes the last matching block. A framework
    /// constraint, enabled only to reproduce a vLLM baseline.
    eagle_block_drop: bool,
    /// vLLM's `--prefix-match-unit` when finer than `chunk_end_quantum`: the
    /// prompt also stops a chunk at its partial-tail hash boundary, where the
    /// engine writes the finer resumable state.
    partial_tail_unit: Option<u32>,
    prefill_episodes: HashMap<RequestId, bool>,
    /// Order in which each running request was last admitted from the waiting
    /// queue; FCFS retraction evicts the most recent one.
    admission_order: HashMap<RequestId, u64>,
    next_admission: u64,
    /// How far one resident decode moves per iteration. Ordinary decode retires
    /// one token; speculative decode retires up to the verify width.
    decode_completion: D,
    /// Budget tokens charged to every scheduled request on top of its own rows
    /// (see `SpeculativeUnifiedModel::drafting_slots_per_request`). Zero for
    /// ordinary decode and sequential drafters.
    drafting_slots: u32,
    /// vLLM's async scheduling: a request finished by accepted drafts runs one
    /// trailing verify row in the next iteration. A framework constraint,
    /// enabled only to reproduce a vLLM baseline.
    async_scheduling: bool,
    /// Per partition: requests that finished last iteration, emitted their
    /// completion, and stay KV-resident for one trailing verify row this
    /// iteration. Released, without a token, when that iteration completes.
    trailing_decode_rows: Vec<Vec<RequestId>>,
    /// Scratch for `DecodeCompletion::complete_decodes`.
    finished_by_accepted_drafts: Vec<RequestId>,
    /// Async scheduling only: the latest batch decision boundary. The shell
    /// delivers a request either right after a decision, which is then its
    /// `QUEUED` time, or to an idle engine at its arrival.
    last_decision_boundary: Time,
    /// Async scheduling only: when the previous decision launched a step, the
    /// time it did. The next decision is the one vLLM made at that launch, so
    /// it stamps its admissions first scheduled then. `None` after a decision
    /// that launched nothing: the next one is made at once.
    pipelined_decision_time: Option<Time>,
    /// Async scheduling only: per request and partition, completions whose KV
    /// the next decision must still see held. Released when it is made.
    deferred_releases: Vec<(RequestId, u16)>,
    /// vLLM's `max_num_seqs`: most running requests (live decodes, trailing
    /// rows and started prompts) after which no waiting request is admitted.
    max_running_requests: Option<u32>,
}

impl<P: PendingOrderPolicy> ChunkedPrefillAdmission<P, SingleTokenDecodeCompletion> {
    pub(crate) fn new(
        partition_policies: Vec<(P, P::Context)>,
        max_batch_tokens: u32,
        batch_policy: BatchPolicy,
        kv_admission: KvAdmissionConfig,
        balance: LoadBalance,
    ) -> Self {
        Self::with_decode_completion(
            partition_policies,
            max_batch_tokens,
            batch_policy,
            kv_admission,
            balance,
            SingleTokenDecodeCompletion,
            0,
        )
    }
}

impl<P: PendingOrderPolicy, D> ChunkedPrefillAdmission<P, D> {
    pub(crate) fn with_decode_completion(
        partition_policies: Vec<(P, P::Context)>,
        max_batch_tokens: u32,
        batch_policy: BatchPolicy,
        kv_admission: KvAdmissionConfig,
        balance: LoadBalance,
        decode_completion: D,
        drafting_slots: u32,
    ) -> Self {
        assert!(max_batch_tokens > 0, "chunked prefill cap must be positive");
        assert!(
            !partition_policies.is_empty(),
            "chunked prefill requires at least one partition"
        );
        assert!(
            !matches!(kv_admission, KvAdmissionConfig::BoundedFuture(_))
                || partition_policies.len() == 1,
            "bounded-future KV admission is validated only for one attention partition"
        );
        assert!(
            !matches!(
                kv_admission,
                KvAdmissionConfig::BoundedFuture(config) if config.page_size != 1
            ),
            "bounded-future currently requires page_size=1; larger pages need allocated-length state"
        );
        assert!(
            !matches!(kv_admission, KvAdmissionConfig::Incremental)
                || partition_policies.len() == 1,
            "incremental KV admission is validated only for one attention partition"
        );
        let current_new_token_ratio = match kv_admission {
            KvAdmissionConfig::FullFootprint | KvAdmissionConfig::Incremental => 1.0,
            KvAdmissionConfig::BoundedFuture(config) => config.initial_new_token_ratio,
        };
        Self {
            partition_policies,
            enqueue_sequence: EnqueueSequence::default(),
            max_batch_tokens,
            batch_policy,
            kv_admission,
            current_new_token_ratio,
            balance,
            started_prefills: Vec::new(),
            loading_prefills: Vec::new(),
            chunk_end_quantum: None,
            eagle_block_drop: false,
            partial_tail_unit: None,
            prefill_episodes: HashMap::new(),
            admission_order: HashMap::new(),
            next_admission: 0,
            decode_completion,
            drafting_slots,
            async_scheduling: false,
            trailing_decode_rows: Vec::new(),
            finished_by_accepted_drafts: Vec::new(),
            last_decision_boundary: Time::ZERO,
            pipelined_decision_time: None,
            deferred_releases: Vec::new(),
            max_running_requests: None,
        }
    }

    /// Admit no waiting request while `max_running` requests run (vLLM's
    /// `max_num_seqs`, checked against `len(self.running)`).
    pub(crate) fn with_max_running_requests(mut self, max_running: u32) -> Self {
        assert!(max_running > 0, "max running requests must be positive");
        self.max_running_requests = Some(max_running);
        self
    }

    /// End every non-final chunk on a multiple of `quantum` context tokens.
    pub(crate) fn with_chunk_end_quantum(mut self, quantum: u32) -> Self {
        assert!(quantum > 0, "chunk-end quantum must be positive");
        self.chunk_end_quantum = Some(quantum);
        self
    }

    /// Back the prompt's last cacheable chunk end off one quantum, as vLLM's
    /// `_mamba_block_aligned_split` does under `use_eagle_block_drop`. Only
    /// meaningful together with [`Self::with_chunk_end_quantum`].
    pub(crate) fn with_eagle_block_drop(mut self) -> Self {
        assert!(
            self.chunk_end_quantum.is_some(),
            "eagle block drop needs a chunk-end quantum"
        );
        self.eagle_block_drop = true;
        self
    }

    /// Also stop a prompt's chunk at its partial-tail boundary for a prefix
    /// match unit `unit` finer than the chunk-end quantum, as vLLM's
    /// `_mamba_block_aligned_split` does (`tail_boundary`). A unit equal to the
    /// quantum adds nothing. Only meaningful together with
    /// [`Self::with_chunk_end_quantum`].
    pub(crate) fn with_prefix_match_unit(mut self, unit: u32) -> Self {
        let quantum = self
            .chunk_end_quantum
            .expect("a prefix match unit needs a chunk-end quantum");
        assert!(
            unit > 0 && quantum.is_multiple_of(unit),
            "prefix match unit {unit} must divide the chunk-end quantum {quantum}"
        );
        self.partial_tail_unit = (unit < quantum).then_some(unit);
        self
    }

    /// Reproduce vLLM's async scheduling: `EngineCore.step_with_batch_queue`
    /// (`core.py` ~L760-876) schedules step k+1 right after it launched step
    /// k and processes k's output only afterwards. Each `form_batch` called at
    /// the end of k therefore answers from what was known when k launched:
    ///
    /// - A decoding request whose round k finished it is scheduled again at
    ///   k+1 unless it was sure to finish at k even with every draft rejected
    ///   (`scheduler.py:635-647`: at most one output token left). That row is
    ///   billed and holds its KV through k+1 but retires nothing and is not a
    ///   verify round.
    /// - KV freed by k's other completions is released after the decision, so
    ///   a request waiting for it starts in k+2.
    /// - Admissions are stamped first scheduled (`UnifiedStage::Prefill`) at
    ///   k's launch, when vLLM decided them; after a decision that launched
    ///   nothing, the next one is made, and stamped, at once.
    ///
    /// It also stamps `queued_time` as EngineCore does: its loop picks new
    /// requests up only between steps, so the async shell hands a request that
    /// arrived while a step ran over at that step's end boundary, and one that
    /// arrived at an idle engine at its arrival. Waiting for KV or budget after
    /// that stays in server TTFT. (A synchronous vLLM picks requests up the
    /// same way; that is not applied with the option off, where `queued_time`
    /// is the arrival.)
    ///
    /// Full-footprint or incremental KV only: retracting a finished trailing
    /// row is not a modeled transition, so preemption never picks one.
    pub(crate) fn with_async_scheduling(mut self) -> Self {
        assert!(
            matches!(
                self.kv_admission,
                KvAdmissionConfig::FullFootprint | KvAdmissionConfig::Incremental
            ),
            "async scheduling is modeled only with full-footprint or incremental KV admission"
        );
        self.async_scheduling = true;
        self
    }

    fn choose_partition<K: ChunkedPrefillKv>(
        &mut self,
        kv_store: &K,
        session_input: SessionInput,
    ) -> usize {
        kv_store
            .retained_prefix_partition(session_input)
            .map(usize::from)
            .unwrap_or_else(|| self.balance.choose(self.partition_policies.len()))
    }

    fn bounded_config(&self) -> Option<BoundedFutureKvAdmissionConfig> {
        match self.kv_admission {
            KvAdmissionConfig::FullFootprint | KvAdmissionConfig::Incremental => None,
            KvAdmissionConfig::BoundedFuture(config) => Some(config),
        }
    }

    fn incremental(&self) -> bool {
        matches!(self.kv_admission, KvAdmissionConfig::Incremental)
    }

    /// Incremental KV: allocate every live decode's next step, `query_width`
    /// rows past its context, in admission order. A decode that cannot be
    /// allocated preempts the newest running request until it can or it was
    /// the newest itself. Returns whether anything was preempted.
    fn allocate_decode_steps<K: ChunkedPrefillKv>(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        now: Time,
        query_width: u32,
    ) -> bool {
        let mut preempted = false;
        for partition_index in 0..kv_store.num_partitions() {
            let partition = partition_index as u16;
            let mut decodes = Vec::new();
            kv_store.visit_decode_states(partition, |request, current_kv, remaining| {
                // A trailing row already holds its last allocation.
                if remaining > 0 {
                    decodes.push((self.admission_order[&request], request, current_kv));
                }
            });
            decodes.sort_unstable();
            let mut preempted_requests = Vec::new();
            for (_, request, current_kv) in decodes {
                if preempted_requests.contains(&request) {
                    continue;
                }
                while !kv_store.allocate_incremental(
                    request,
                    partition,
                    current_kv + u64::from(query_width),
                    now,
                ) {
                    let victim = self.preempt_newest_running(kv_store, context, partition, now);
                    preempted = true;
                    preempted_requests.push(victim);
                    if victim == request {
                        break;
                    }
                }
            }
        }
        preempted
    }

    /// vLLM's FCFS preemption under incremental KV (`scheduler.py`
    /// `self.running.pop()`): the running request admitted last, a live decode
    /// or a started prompt, frees its KV and returns to the queue front.
    /// Trailing rows are never chosen. Returns the victim.
    fn preempt_newest_running<K: ChunkedPrefillKv>(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        partition: u16,
        now: Time,
    ) -> RequestId {
        let mut newest: Option<(u64, RequestId, bool)> = None;
        kv_store.visit_decode_states(partition, |request, _, remaining| {
            let order = self.admission_order[&request];
            if remaining > 0 && newest.is_none_or(|(best, _, _)| order > best) {
                newest = Some((order, request, false));
            }
        });
        for candidate in &self.started_prefills[partition as usize] {
            let order = self.admission_order[&candidate.request_id];
            if newest.is_none_or(|(best, _, _)| order > best) {
                newest = Some((order, candidate.request_id, true));
            }
        }
        let (_, victim, started) =
            newest.expect("an allocation shortfall needs a running request to preempt");
        if started {
            self.restart_started_prefill(kv_store, context, partition, victim, now);
        } else {
            // The prompt's snapshots and context return to the retained tier
            // as at completion, so recomputation can resume from its prefix.
            kv_store.release_retaining_prefix(victim, partition, now);
            self.requeue_retracted(
                kv_store,
                context,
                partition,
                victim,
                DecodeRetractionPolicy::Fcfs,
                now,
            );
        }
        victim
    }

    /// A preempted prompt that had not finished its prefill: free its KV and
    /// queue it again at the front to start over. The prefix it took from
    /// the retained tier is not returned (vLLM would return the attention
    /// blocks; the state it resumed from was already freed into the pool).
    fn restart_started_prefill<K: ChunkedPrefillKv>(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        partition: u16,
        request: RequestId,
        now: Time,
    ) {
        kv_store.release(request, partition);
        self.started_prefills[partition as usize]
            .retain(|candidate| candidate.request_id != request);
        self.admission_order.remove(&request);
        let reprocessed = self
            .prefill_episodes
            .remove(&request)
            .expect("a started prompt has a prefill episode");
        let (fresh_prompt_tokens, remaining_output_tokens, session_input, conversation_start_time) = {
            let mut store = context.requests.borrow_mut();
            let record = &mut store[request];
            record.record_retraction();
            record.restart_unfinished_prefill();
            context.stamp_stage(record, now, UnifiedStage::Pending as u16);
            let definition = &record.request.definition;
            (
                definition
                    .prompt_tokens
                    .checked_add(record.progress.output_tokens_emitted)
                    .expect("reprocessed input length overflow"),
                definition
                    .target_output_tokens
                    .saturating_sub(record.progress.output_tokens_emitted),
                definition.session,
                definition
                    .session
                    .session_start_or(record.request.core.arrival_time),
            )
        };
        let resident_prefix_tokens =
            kv_store.resident_prefix_tokens(fresh_prompt_tokens, session_input);
        let candidate = if reprocessed {
            self.enqueue_sequence.freeze_retracted(
                request,
                fresh_prompt_tokens,
                remaining_output_tokens,
                session_input,
                conversation_start_time,
                resident_prefix_tokens,
            )
        } else {
            self.enqueue_sequence.freeze(
                request,
                fresh_prompt_tokens,
                remaining_output_tokens,
                session_input,
                conversation_start_time,
                resident_prefix_tokens,
            )
        };
        let (policy, policy_context) = &mut self.partition_policies[partition as usize];
        policy.push_front(candidate, policy_context);
    }

    /// Requests counted against `max_running_requests` before this decision
    /// admits anyone: live decodes and trailing rows, started prompts, and
    /// completions whose release waits for the decision (vLLM removes a
    /// finished request from `running` only when it processes that output).
    fn running_count<K: ChunkedPrefillKv>(&self, kv_store: &K, partition: usize) -> u32 {
        let deferred = self
            .deferred_releases
            .iter()
            .filter(|(_, owner)| usize::from(*owner) == partition)
            .count() as u32;
        kv_store.live_decode_count(partition as u16)
            + self.started_prefills[partition].len() as u32
            + deferred
    }

    /// Incremental KV: give each started prompt, in start order, its next
    /// chunk of `budget` and allocate it. A chunk that cannot be allocated
    /// preempts the newest running request until it can; if the prompt was
    /// the newest itself, vLLM stops scheduling running requests for the step.
    /// Returns whether anything was preempted.
    fn schedule_started_prefills_incremental<K: ChunkedPrefillKv>(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        partition_index: usize,
        budget: &mut u32,
        now: Time,
    ) -> bool {
        let partition = partition_index as u16;
        let mut preempted = false;
        let mut index = 0;
        while index < self.started_prefills[partition_index].len() {
            let candidate = self.started_prefills[partition_index][index];
            let resolved_prefill = kv_store.resolved_prefill_context(candidate.request_id);
            let chunk_tokens = next_chunk_tokens(
                resolved_prefill,
                budget.saturating_sub(self.drafting_slots),
                self.chunk_end_quantum,
                self.eagle_block_drop,
                self.partial_tail_unit,
                self.max_batch_tokens,
            );
            if chunk_tokens == 0 {
                index += 1;
                continue;
            }
            let chunk_end = u64::from(resolved_prefill.active_chunk().0 + chunk_tokens);
            let mut runs = true;
            while !kv_store.allocate_incremental(candidate.request_id, partition, chunk_end, now) {
                let victim = self.preempt_newest_running(kv_store, context, partition, now);
                preempted = true;
                if victim == candidate.request_id {
                    runs = false;
                    break;
                }
            }
            if !runs {
                break;
            }
            kv_store.schedule_prefill_chunk(candidate.request_id, partition, chunk_tokens);
            *budget -= chunk_tokens + self.drafting_slots;
            if chunk_tokens < resolved_prefill.remaining_prefill_tokens() {
                index += 1;
            } else {
                self.started_prefills[partition_index].remove(index);
            }
        }
        preempted
    }

    fn decay_new_token_ratio(&mut self, config: BoundedFutureKvAdmissionConfig) {
        let decay = (config.initial_new_token_ratio - config.minimum_new_token_ratio)
            / f64::from(config.new_token_ratio_decay_steps);
        self.current_new_token_ratio =
            (self.current_new_token_ratio - decay).max(config.minimum_new_token_ratio);
    }

    fn update_ratio_after_retraction<K: ChunkedPrefillKv>(
        &mut self,
        kv_store: &K,
        context: &WorkerContext,
        config: BoundedFutureKvAdmissionConfig,
    ) {
        let store = context.requests.borrow();
        let mut decoded_tokens = 0u64;
        let mut max_new_tokens = 0u64;
        let mut requests = 0u64;
        for partition in 0..kv_store.num_partitions() as u16 {
            kv_store.visit_decode_states(partition, |request, _, _| {
                let record = &store[request];
                decoded_tokens += u64::from(record.progress.output_tokens_emitted);
                max_new_tokens += u64::from(record.request.definition.target_output_tokens);
                requests += 1;
            });
        }
        let numerator = decoded_tokens
            .checked_add(u64::from(config.retract_decode_steps) * requests)
            .expect("post-retraction ratio numerator overflow");
        self.current_new_token_ratio = (numerator as f64 / (max_new_tokens + 1) as f64).min(1.0);
    }

    fn select_length_retraction<K: ChunkedPrefillKv>(
        &self,
        kv_store: &K,
        context: &WorkerContext,
        partition: u16,
    ) -> Option<RequestId> {
        let mut candidates = Vec::new();
        kv_store.visit_decode_states(partition, |request, _, _| candidates.push(request));
        let store = context.requests.borrow();
        candidates
            .into_iter()
            .enumerate()
            .min_by_key(|(index, request)| {
                let record = &store[*request];
                let input_tokens = record
                    .request
                    .definition
                    .prompt_tokens
                    .checked_add(record.request.definition.session.declared_prefix_tokens())
                    .expect("retraction input length overflow");
                (
                    record.progress.output_tokens_emitted,
                    Reverse(input_tokens),
                    Reverse(*index),
                )
            })
            .map(|(_, request)| request)
    }

    /// vLLM's FCFS preemption: the request that most recently entered the
    /// running set goes first.
    fn select_fcfs_retraction<K: ChunkedPrefillKv>(
        &self,
        kv_store: &K,
        partition: u16,
    ) -> Option<RequestId> {
        let mut victim = None;
        kv_store.visit_decode_states(partition, |request, _, _| {
            let order = self.admission_order[&request];
            if victim.is_none_or(|(best, _)| order > best) {
                victim = Some((order, request));
            }
        });
        victim.map(|(_, request)| request)
    }

    fn requeue_retracted<K: ChunkedPrefillKv>(
        &mut self,
        kv_store: &K,
        context: &WorkerContext,
        partition: u16,
        request: RequestId,
        retraction_policy: DecodeRetractionPolicy,
        now: Time,
    ) {
        let (
            reprocessed_input_tokens,
            remaining_output_tokens,
            session_input,
            conversation_start_time,
        ) = {
            let mut store = context.requests.borrow_mut();
            let record = &mut store[request];
            record.record_retraction();
            context.stamp_stage(record, now, UnifiedStage::Pending as u16);
            (
                record
                    .request
                    .definition
                    .prompt_tokens
                    .checked_add(record.progress.output_tokens_emitted)
                    .expect("reprocessed input length overflow"),
                record
                    .request
                    .definition
                    .target_output_tokens
                    .saturating_sub(record.progress.output_tokens_emitted),
                record.request.definition.session,
                record
                    .request
                    .definition
                    .session
                    .session_start_or(record.request.core.arrival_time),
            )
        };
        let resident_prefix_tokens =
            kv_store.resident_prefix_tokens(reprocessed_input_tokens, session_input);
        let candidate = self.enqueue_sequence.freeze_retracted(
            request,
            reprocessed_input_tokens,
            remaining_output_tokens,
            session_input,
            conversation_start_time,
            resident_prefix_tokens,
        );
        self.admission_order.remove(&request);
        let (policy, policy_context) = &mut self.partition_policies[partition as usize];
        match retraction_policy {
            DecodeRetractionPolicy::Length => policy.push(candidate, policy_context),
            DecodeRetractionPolicy::Fcfs => policy.push_front(candidate, policy_context),
        }
    }

    /// Retract decodes until every partition fits its next decode step, each
    /// request advancing up to `step_tokens`. Returns whether it retracted.
    fn prepare_decode<K: ChunkedPrefillKv>(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        now: Time,
        config: BoundedFutureKvAdmissionConfig,
        step_tokens: u32,
    ) -> bool {
        let mut retracted_any = false;
        for partition in 0..kv_store.num_partitions() as u16 {
            while kv_store.has_live_decode(partition)
                && kv_store.prepare_next_decode(partition, config.page_size, step_tokens, now) > 0
            {
                let live_count = kv_store.live_decode_count(partition);
                assert!(
                    live_count > 1,
                    "bounded-future decode cannot fit its last request; abort lifecycle is required"
                );
                let request = match config.retraction_policy {
                    DecodeRetractionPolicy::Length => {
                        self.select_length_retraction(kv_store, context, partition)
                    }
                    DecodeRetractionPolicy::Fcfs => {
                        self.select_fcfs_retraction(kv_store, partition)
                    }
                }
                .expect("decode shortfall requires a retraction candidate");
                kv_store.release(request, partition);
                self.requeue_retracted(
                    kv_store,
                    context,
                    partition,
                    request,
                    config.retraction_policy,
                    now,
                );
                retracted_any = true;
            }
        }
        if retracted_any {
            self.update_ratio_after_retraction(kv_store, context, config);
        } else {
            self.decay_new_token_ratio(config);
        }
        retracted_any
    }

    #[cfg(test)]
    pub(crate) fn current_new_token_ratio(&self) -> f64 {
        self.current_new_token_ratio
    }
}

impl<P, D, K> IterAdmission<K> for ChunkedPrefillAdmission<P, D>
where
    P: PendingOrderPolicy,
    D: DecodeCompletion<K>,
    K: ChunkedPrefillKv,
{
    type Msg = WorkerMsgCommon;
    type Event = WorkerEventCommon;

    fn accept_message(&mut self, kv_store: &mut K, msg: Self::Msg, context: &WorkerContext) {
        let WorkerMsgCommon::Request(request) = msg;
        debug_assert_eq!(self.partition_policies.len(), kv_store.num_partitions());
        let (fresh_prompt_tokens, remaining_output_tokens, session_input, conversation_start_time) = {
            let mut store = context.requests.borrow_mut();
            let record = &mut store[request];
            let arrival_time = record.request.core.arrival_time;
            context.stamp_stage(record, arrival_time, UnifiedStage::Pending as u16);
            // The async shell delivers right after a decision, which queues
            // the request at that boundary, or to an idle engine on arrival
            // (`queued_time` left unset means the arrival).
            if self.async_scheduling && self.last_decision_boundary > arrival_time {
                record.telemetry.queued_time = Some(self.last_decision_boundary);
            }
            (
                record.request.definition.prompt_tokens,
                record.request.definition.target_output_tokens,
                record.request.definition.session,
                record
                    .request
                    .definition
                    .session
                    .session_start_or(arrival_time),
            )
        };
        let candidate = self.enqueue_sequence.freeze(
            request,
            fresh_prompt_tokens,
            remaining_output_tokens,
            session_input,
            conversation_start_time,
            kv_store.resident_prefix_tokens(fresh_prompt_tokens, session_input),
        );
        // Placement is frozen when the request enters the worker. Retained KV
        // has hard affinity; cold requests use the same generic balance policy
        // as ordinary local admission.
        let partition = self.choose_partition(kv_store, session_input);
        let (policy, policy_context) = &mut self.partition_policies[partition];
        debug_assert!(!policy.contains(request));
        policy.push(candidate, policy_context);
    }

    fn form_batch(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        batch_plan: &mut IterBatchPlan,
        now: Time,
    ) -> bool {
        let num_partitions = kv_store.num_partitions();
        debug_assert_eq!(self.partition_policies.len(), num_partitions);
        // When vLLM made this decision: the launch of the step that just ended
        // under async scheduling, else now.
        let decision_time = if self.async_scheduling {
            self.pipelined_decision_time.unwrap_or(now)
        } else {
            now
        };
        self.started_prefills.resize_with(num_partitions, Vec::new);
        self.loading_prefills.resize_with(num_partitions, Vec::new);
        self.trailing_decode_rows
            .resize_with(num_partitions, Vec::new);
        // Host copies reported done by the time this decision is made.
        kv_store.settle_host_transfers(decision_time);
        batch_plan.reset_decode_participation(num_partitions, true);
        for (partition, trailing) in self.trailing_decode_rows.iter().enumerate() {
            if !trailing.is_empty() {
                batch_plan
                    .set_partition_trailing_decode_rows(partition as u16, trailing.len() as u32);
            }
        }
        let has_live_decode = |kv_store: &K| {
            (0..num_partitions as u16).any(|partition| kv_store.has_live_decode(partition))
        };
        let mixes_prefill_with_decode = self.batch_policy == BatchPolicy::Mix;
        let bounded_config = self.bounded_config();
        let query_width = self.decode_completion.query_tokens_per_request();
        // A mixed engine (vLLM) schedules its running decodes first, each
        // claiming KV for its next step, and admits no waiting request in a step
        // that had to preempt. A separate-prefill engine (SGLang) checks decode
        // headroom only on the decode-only steps it runs, below.
        let incremental = self.incremental();
        let mut retracted_before_admission = match bounded_config {
            Some(config) if mixes_prefill_with_decode && has_live_decode(kv_store) => {
                self.prepare_decode(kv_store, context, now, config, query_width)
            }
            _ => false,
        };
        // vLLM v1 allocates each running decode's step first, in running
        // order, and preempts on a shortfall.
        if incremental && has_live_decode(kv_store) {
            retracted_before_admission |=
                self.allocate_decode_steps(kv_store, context, now, query_width);
        }
        let had_decode = has_live_decode(kv_store);
        let current_new_token_ratio = self.current_new_token_ratio;
        // Every scheduled request, decode or prefill chunk, also spends the
        // drafter's slots.
        let drafting_slots = self.drafting_slots;
        let chunk_end_quantum = self.chunk_end_quantum;
        let eagle_block_drop = self.eagle_block_drop;
        let partial_tail_unit = self.partial_tail_unit;
        let max_batch_tokens = self.max_batch_tokens;
        let decode_budget = query_width + drafting_slots;
        // One short prompt can create a whole verify window next iteration.
        // Reserve those future query slots before admitting its prefill; an
        // unfinished started prompt keeps its reservation across iterations.
        let mut future_decode_slots = (query_width > 1).then(|| {
            (0..num_partitions)
                .map(|partition| {
                    let started = self.started_prefills[partition]
                        .iter()
                        .chain(&self.loading_prefills[partition])
                        .filter(|candidate| candidate.remaining_output_tokens > 1)
                        .count() as u32;
                    (self.max_batch_tokens / decode_budget)
                        .saturating_sub(kv_store.live_decode_count(partition as u16))
                        .saturating_sub(started)
                })
                .collect::<Vec<_>>()
        });
        let mut remaining_budgets: Vec<u32> = (0..num_partitions as u16)
            .map(|partition| {
                if mixes_prefill_with_decode {
                    // Decode spends the shared cap in query rows, not requests:
                    // a speculating engine submits a whole verify window per
                    // resident request, so a mixed iteration must leave room
                    // for all of it.
                    self.max_batch_tokens.saturating_sub(
                        kv_store
                            .live_decode_count(partition)
                            .saturating_mul(decode_budget),
                    )
                } else {
                    self.max_batch_tokens
                }
            })
            .collect();

        // Started prompts keep partition-local priority, in start order, until
        // their prompt is complete, matching one EngineCore scheduler queue per
        // DP partition. One that cannot run this iteration keeps its place.
        for partition_index in 0..num_partitions {
            if incremental {
                retracted_before_admission |= self.schedule_started_prefills_incremental(
                    kv_store,
                    context,
                    partition_index,
                    &mut remaining_budgets[partition_index],
                    now,
                );
                continue;
            }
            let mut started_prefills = std::mem::take(&mut self.started_prefills[partition_index]);
            started_prefills.retain(|candidate| {
                let resolved_prefill = kv_store.resolved_prefill_context(candidate.request_id);
                let chunk_tokens = next_chunk_tokens(
                    resolved_prefill,
                    remaining_budgets[partition_index].saturating_sub(drafting_slots),
                    chunk_end_quantum,
                    eagle_block_drop,
                    partial_tail_unit,
                    max_batch_tokens,
                );
                if chunk_tokens == 0 {
                    return true;
                }
                kv_store.schedule_prefill_chunk(
                    candidate.request_id,
                    partition_index as u16,
                    chunk_tokens,
                );
                remaining_budgets[partition_index] -= chunk_tokens + drafting_slots;
                chunk_tokens < resolved_prefill.remaining_prefill_tokens()
            });
            self.started_prefills[partition_index] = started_prefills;
        }

        for partition_index in 0..num_partitions {
            let partition = partition_index as u16;
            let running_before_admission = self.running_count(kv_store, partition_index);
            let mut admitted = 0u32;
            // Prompts whose host load this decision sees done run first, as
            // vLLM drains its KV-holding waiting requests first; one still
            // copying is skipped and keeps its place.
            let mut loading_index = 0;
            while loading_index < self.loading_prefills[partition_index].len()
                && !retracted_before_admission
                && remaining_budgets[partition_index] > drafting_slots
            {
                if self
                    .max_running_requests
                    .is_some_and(|max_running| running_before_admission + admitted >= max_running)
                {
                    break;
                }
                let candidate = self.loading_prefills[partition_index][loading_index];
                if kv_store
                    .host_load_ready_at(candidate.request_id)
                    .is_some_and(|ready| ready > decision_time)
                {
                    loading_index += 1;
                    continue;
                }
                let resolved_prefill = kv_store.resolved_prefill_context(candidate.request_id);
                let chunk_tokens = next_chunk_tokens(
                    resolved_prefill,
                    remaining_budgets[partition_index] - drafting_slots,
                    chunk_end_quantum,
                    eagle_block_drop,
                    partial_tail_unit,
                    max_batch_tokens,
                );
                if chunk_tokens == 0 {
                    break;
                }
                // A waiting request that cannot be allocated stops the
                // waiting loop; it preempts nobody.
                if incremental
                    && !kv_store.allocate_incremental(
                        candidate.request_id,
                        partition,
                        u64::from(resolved_prefill.active_chunk().0 + chunk_tokens),
                        now,
                    )
                {
                    break;
                }
                self.loading_prefills[partition_index].remove(loading_index);
                self.admission_order
                    .insert(candidate.request_id, self.next_admission);
                self.next_admission += 1;
                {
                    let mut store = context.requests.borrow_mut();
                    context.stamp_stage(
                        &mut store[candidate.request_id],
                        decision_time,
                        UnifiedStage::Prefill as u16,
                    );
                }
                kv_store.schedule_prefill_chunk(candidate.request_id, partition, chunk_tokens);
                remaining_budgets[partition_index] -= chunk_tokens + drafting_slots;
                admitted += 1;
                if chunk_tokens < resolved_prefill.remaining_prefill_tokens() {
                    self.started_prefills[partition_index].push(candidate);
                }
            }
            while !retracted_before_admission && remaining_budgets[partition_index] > drafting_slots
            {
                if self
                    .max_running_requests
                    .is_some_and(|max_running| running_before_admission + admitted >= max_running)
                {
                    break;
                }
                let (policy, policy_context) = &mut self.partition_policies[partition_index];
                policy.refresh_head(&mut |candidate| {
                    kv_store.resident_prefix_tokens(
                        candidate.fresh_prompt_tokens,
                        candidate.session_input,
                    )
                });
                let Some(candidate) = policy.peek() else {
                    break;
                };
                if candidate.remaining_output_tokens > 1
                    && future_decode_slots
                        .as_ref()
                        .is_some_and(|slots| slots[partition_index] == 0)
                {
                    break;
                }
                let resolved_prefill = kv_store.preview_prefill_context(
                    partition,
                    candidate.fresh_prompt_tokens,
                    candidate.session_input,
                );
                // A prompt resuming from the host is admitted to wait for its
                // load and runs nothing now (vLLM schedules no tokens for it).
                let loads_from_host = resolved_prefill.host_loaded_prefix_tokens() > 0;
                // A fresh prompt whose first chunk cannot end on a checkpoint
                // boundary within the budget stops admission (vLLM breaks its
                // waiting loop here rather than skipping ahead).
                let chunk_tokens = if loads_from_host {
                    0
                } else {
                    next_chunk_tokens(
                        resolved_prefill,
                        remaining_budgets[partition_index] - drafting_slots,
                        chunk_end_quantum,
                        eagle_block_drop,
                        partial_tail_unit,
                        max_batch_tokens,
                    )
                };
                if chunk_tokens == 0 && !loads_from_host {
                    break;
                }
                let footprint = match bounded_config {
                    None => kv_store.footprint(
                        candidate.request_id,
                        resolved_prefill.post_prefill_context_tokens(),
                        candidate.remaining_output_tokens,
                    ),
                    Some(config) => kv_store.bounded_future_footprint(
                        candidate.request_id,
                        resolved_prefill.post_prefill_context_tokens(),
                        candidate.remaining_output_tokens,
                        config.max_future_tokens,
                        config.page_size,
                    ),
                };
                let fits = match bounded_config {
                    None if incremental => kv_store.incremental_prompt_fits(
                        partition,
                        resolved_prefill.post_prefill_context_tokens(),
                        resolved_prefill.active_chunk().0 + chunk_tokens,
                    ),
                    None => kv_store.fits(partition, &footprint),
                    Some(config) => kv_store.fits_bounded_future(
                        partition,
                        &footprint,
                        config.max_future_tokens,
                        current_new_token_ratio,
                        if mixes_prefill_with_decode {
                            query_width
                        } else {
                            0
                        },
                    ),
                };
                if !fits {
                    break;
                }
                let popped = policy.pop(policy_context);
                debug_assert_eq!(popped, Some(candidate));
                if !loads_from_host {
                    self.admission_order
                        .insert(candidate.request_id, self.next_admission);
                    self.next_admission += 1;
                }
                if candidate.remaining_output_tokens > 1 {
                    if let Some(slots) = future_decode_slots.as_mut() {
                        slots[partition_index] -= 1;
                    }
                }
                kv_store.reserve_chunked_prefill_context(
                    candidate.request_id,
                    partition,
                    resolved_prefill,
                    footprint,
                    now,
                );
                if !loads_from_host {
                    admitted += 1;
                }
                if incremental {
                    let allocated = kv_store.allocate_incremental(
                        candidate.request_id,
                        partition,
                        u64::from(resolved_prefill.active_chunk().0 + chunk_tokens),
                        now,
                    );
                    assert!(
                        allocated,
                        "an admitted prompt's first chunk fits the blocks its admission checked"
                    );
                }
                {
                    let mut store = context.requests.borrow_mut();
                    store.mark_admitted(candidate.request_id);
                    let record = &mut store[candidate.request_id];
                    if candidate.retracted {
                        record.begin_reprocessed_prefill(resolved_prefill.resident_prefix_tokens());
                    } else {
                        record.record_prefix_cache_hit_tokens(
                            resolved_prefill.resident_prefix_tokens(),
                        );
                    }
                    if !loads_from_host {
                        context.stamp_stage(record, decision_time, UnifiedStage::Prefill as u16);
                    }
                }
                assert!(
                    self.prefill_episodes
                        .insert(candidate.request_id, candidate.retracted)
                        .is_none(),
                    "request entered two simultaneous prefill episodes"
                );
                if loads_from_host {
                    self.loading_prefills[partition_index].push(candidate);
                    continue;
                }
                kv_store.schedule_prefill_chunk(candidate.request_id, partition, chunk_tokens);
                remaining_budgets[partition_index] -= chunk_tokens + drafting_slots;
                if chunk_tokens < resolved_prefill.remaining_prefill_tokens() {
                    self.started_prefills[partition_index].push(candidate);
                }
            }
        }

        let mut had_prefill = false;
        for partition in 0..num_partitions as u16 {
            let partition_has_prefill = kv_store.has_prefill_admit(partition);
            had_prefill |= partition_has_prefill;
            if !mixes_prefill_with_decode && partition_has_prefill {
                batch_plan.set_partition_runs_decode(partition, false);
            }
        }

        let had_decode = match bounded_config {
            Some(config) if !mixes_prefill_with_decode && !had_prefill && had_decode => {
                self.prepare_decode(kv_store, context, now, config, query_width);
                has_live_decode(kv_store)
            }
            _ => had_decode,
        };

        let has_batch = had_decode || had_prefill;
        if !has_batch
            && self
                .partition_policies
                .iter()
                .all(|(policy, _)| policy.len() == 0)
            && self.started_prefills.iter().all(Vec::is_empty)
        {
            // SGLang resets its tracker only after running, chunked, and
            // waiting queues are all empty (`Scheduler::on_idle`). A blocked
            // waiting request is therefore deliberately not a reset boundary.
            if let Some(config) = bounded_config {
                self.current_new_token_ratio = config.initial_new_token_ratio;
            }
        }
        if has_batch && query_width > 1 {
            let mut store = context.requests.borrow_mut();
            for partition in 0..kv_store.num_partitions() as u16 {
                kv_store.visit_prefill_admits(partition, |request| {
                    let observation = store[request]
                        .telemetry
                        .speculative
                        .get_or_insert_with(Default::default);
                    observation.query_width = query_width;
                    observation.pending_prefill =
                        Some(kv_store.resolved_prefill_context(request).active_chunk());
                });
                if batch_plan.partition_runs_decode(partition) {
                    kv_store.visit_decode_members(partition, |request, resident| {
                        let record = &mut store[request];
                        // A trailing row's request already completed; this
                        // iteration is not one of its verify rounds.
                        if record.is_complete() {
                            return;
                        }
                        let observation = record
                            .telemetry
                            .speculative
                            .get_or_insert_with(Default::default);
                        observation.query_width = query_width;
                        observation.pending_decode = Some(resident as u64);
                    });
                }
            }
        }
        if self.async_scheduling {
            self.last_decision_boundary = now;
            self.pipelined_decision_time = has_batch.then_some(now);
            // The decision has been made: the KV it could not see is free for
            // the next one.
            for (request, partition) in self.deferred_releases.drain(..) {
                kv_store.release_retaining_prefix(request, partition, now);
            }
        }
        has_batch
    }

    fn complete_iteration(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        batch_plan: &IterBatchPlan,
        events: &mut Vec<Self::Event>,
        now: Time,
    ) {
        for partition in 0..kv_store.num_partitions() as u16 {
            let mut completed = Vec::new();
            self.finished_by_accepted_drafts.clear();
            if batch_plan.partition_runs_decode(partition) {
                // Trailing rows ran their billed verify this iteration; they
                // retire nothing, so they leave before any decode advances.
                if let Some(trailing) = self.trailing_decode_rows.get_mut(partition as usize) {
                    for request in trailing.drain(..) {
                        self.admission_order.remove(&request);
                        kv_store.release_retaining_prefix(request, partition, now);
                    }
                }
                self.decode_completion.complete_decodes(
                    kv_store,
                    partition,
                    context,
                    &mut completed,
                    &mut self.finished_by_accepted_drafts,
                    now,
                );
            }

            let mut prefills = Vec::new();
            kv_store.visit_prefill_admits(partition, |request| prefills.push(request));
            for request in prefills {
                let reprocessed = *self
                    .prefill_episodes
                    .get(&request)
                    .expect("scheduled prefill requires episode metadata");
                let chunk_tokens = kv_store.resolved_prefill_context(request).active_chunk().1;
                kv_store.complete_prefill_chunk(request);
                let finished = kv_store
                    .resolved_prefill_context(request)
                    .remaining_prefill_tokens()
                    == 0;
                {
                    let mut store = context.requests.borrow_mut();
                    let record = &mut store[request];
                    if reprocessed {
                        record.record_reprocessed_prefill_tokens(chunk_tokens);
                    } else {
                        record.progress.prefill_tokens_processed = record
                            .progress
                            .prefill_tokens_processed
                            .checked_add(chunk_tokens)
                            .expect("prefill progress overflows u32");
                    }
                    let query_width = self.decode_completion.query_tokens_per_request();
                    if query_width > 1 {
                        let observation = record
                            .telemetry
                            .speculative
                            .get_or_insert_with(Default::default);
                        observation.query_width = query_width;
                        observation.prefill_chunks += 1;
                        observation.pending_prefill = None;
                    }
                    if finished {
                        if reprocessed {
                            record.complete_reprocessed_prefill();
                            record.record_token(now, context.log_tokens());
                        } else {
                            record.record_first_token(now, context.log_tokens());
                        }
                        context.stamp_stage(
                            record,
                            now,
                            if record.is_complete() {
                                UnifiedStage::Done
                            } else {
                                UnifiedStage::Decode
                            } as u16,
                        );
                    }
                }
                if finished {
                    self.prefill_episodes.remove(&request);
                    let remaining_output_tokens = {
                        let store = context.requests.borrow();
                        let record = &store[request];
                        record
                            .request
                            .definition
                            .target_output_tokens
                            .saturating_sub(record.progress.output_tokens_emitted)
                    };
                    kv_store.finish_chunked_prefill(
                        request,
                        partition,
                        remaining_output_tokens,
                        now,
                    );
                    if context.requests.borrow()[request].is_complete() {
                        completed.push(request);
                    }
                }
            }
            kv_store.clear_prefill_admits(partition);

            for request in completed {
                if self.async_scheduling && self.finished_by_accepted_drafts.contains(&request) {
                    // The async scheduler already issued this request's next
                    // row: it stays resident for one trailing verify.
                    kv_store.hold_trailing_decode_row(request, partition);
                    self.trailing_decode_rows[partition as usize].push(request);
                } else if self.async_scheduling {
                    // No row next step, but the decision for it was made
                    // before this completion: its KV stays held until then.
                    self.admission_order.remove(&request);
                    self.deferred_releases.push((request, partition));
                } else {
                    self.admission_order.remove(&request);
                    kv_store.release_retaining_prefix(request, partition, now);
                }
                events.push(WorkerEventCommon::RequestComplete {
                    worker: context.id,
                    req: request,
                });
            }
            kv_store.sample_submit(partition, now);
        }
    }

    fn queued_requests(&self) -> u32 {
        self.partition_policies
            .iter()
            .map(|(policy, _)| policy.len() as u32)
            .sum()
    }

    fn cancel_pending(&mut self, request: RequestId) -> bool {
        for (policy, _) in &mut self.partition_policies {
            if policy.remove(request).is_some() {
                return true;
            }
        }
        for started_prefills in self
            .started_prefills
            .iter_mut()
            .chain(self.loading_prefills.iter_mut())
        {
            let started_count = started_prefills.len();
            started_prefills.retain(|candidate| candidate.request_id != request);
            if started_prefills.len() != started_count {
                self.prefill_episodes.remove(&request);
            }
        }
        false
    }
}

/// The next chunk of `resolved_prefill` that `budget` tokens can carry, ending
/// on a `chunk_end_quantum` boundary when one is set. `0` means the request
/// cannot run this iteration.
fn next_chunk_tokens(
    resolved_prefill: ResolvedPrefillContext,
    budget: u32,
    chunk_end_quantum: Option<u32>,
    eagle_block_drop: bool,
    partial_tail_unit: Option<u32>,
    max_chunk_tokens: u32,
) -> u32 {
    let remaining = resolved_prefill.remaining_prefill_tokens();
    let chunk_tokens = remaining.min(budget);
    match chunk_end_quantum {
        None => chunk_tokens,
        Some(quantum) => {
            let start = resolved_prefill.active_chunk().0;
            let chunk_tokens = checkpoint_aligned_chunk_tokens(
                start,
                start + remaining,
                chunk_tokens,
                quantum,
                eagle_block_drop,
                max_chunk_tokens,
            );
            match partial_tail_unit {
                None => chunk_tokens,
                Some(unit) => partial_tail_chunk_tokens(
                    start,
                    start + remaining,
                    chunk_tokens,
                    quantum,
                    unit,
                    eagle_block_drop,
                ),
            }
        }
    }
}

/// The partial-tail stop of vLLM's `_mamba_block_aligned_split` for a prefix
/// match `unit` finer than `quantum` (`mamba_partial_cache_hit`): a chunk
/// starting at `start` and already clipped to `chunk_tokens` also stops at the
/// prompt's last whole unit, one unit lower under `eagle_block_drop`, when that
/// position lies past the prompt's last cacheable block boundary. Its state is
/// the finer resumable checkpoint `PrefixCache::with_match_unit` retains.
/// Every stop is a `min` over positions inside the chunk, so applying this one
/// after [`checkpoint_aligned_chunk_tokens`] is the engine's single `min`.
fn partial_tail_chunk_tokens(
    start: u32,
    prefill_end: u32,
    chunk_tokens: u32,
    quantum: u32,
    unit: u32,
    eagle_block_drop: bool,
) -> u32 {
    let mut last_boundary = prefill_end / quantum * quantum;
    let mut tail = prefill_end / unit * unit;
    if eagle_block_drop {
        last_boundary = last_boundary.saturating_sub(quantum);
        tail = tail.saturating_sub(unit);
    }
    let end = start + chunk_tokens;
    if last_boundary < tail && tail < prefill_end && start < tail && tail < end {
        tail - start
    } else {
        chunk_tokens
    }
}

/// vLLM's `Scheduler._mamba_block_aligned_split` for a prompt whose next
/// unprocessed context position is `start` and whose prefill ends at
/// `prefill_end`: clip a `chunk_tokens`-long chunk so the recurrent state it
/// leaves behind sits on a `quantum` boundary and can be checkpointed.
///
/// - A non-final chunk ends on the last boundary it reaches. When `quantum`
///   exceeds the chunk cap no boundary may fit, so the chunk runs sub-block
///   instead and re-aligns at the next boundary.
/// - A chunk starting mid-block stops at the next boundary.
/// - No chunk runs past the prompt's last boundary, so its state is cached
///   before the unaligned tail. Under `eagle_block_drop` that boundary backs
///   off one quantum (`last_cache_position - block_size`).
///
/// The fork's partial-tail hash stop is [`partial_tail_chunk_tokens`]; its
/// shared-prefix junction stop and in-chunk prefill checkpoints are not
/// modeled.
fn checkpoint_aligned_chunk_tokens(
    start: u32,
    prefill_end: u32,
    chunk_tokens: u32,
    quantum: u32,
    eagle_block_drop: bool,
    max_chunk_tokens: u32,
) -> u32 {
    if chunk_tokens == 0 || start >= prefill_end {
        return chunk_tokens;
    }
    let mut end = start + chunk_tokens;
    if end < prefill_end {
        let aligned_end = end / quantum * quantum;
        if aligned_end > start || quantum <= max_chunk_tokens {
            end = aligned_end;
        }
    }
    let next_boundary = (start / quantum + 1) * quantum;
    let mut last_boundary = prefill_end / quantum * quantum;
    if eagle_block_drop {
        last_boundary = last_boundary.saturating_sub(quantum);
    }
    let stops = [
        (start % quantum != 0).then_some(next_boundary),
        Some(last_boundary),
    ];
    let end = stops
        .into_iter()
        .flatten()
        .filter(|&stop| start < stop && stop < end)
        .min()
        .unwrap_or(end);
    end.saturating_sub(start)
}

#[cfg(test)]
mod tests {
    use super::{checkpoint_aligned_chunk_tokens, partial_tail_chunk_tokens};

    const BLOCK: u32 = 2_176;

    #[test]
    fn a_non_final_chunk_ends_on_the_last_checkpoint_boundary_it_reaches() {
        // GLM-5.3-Flash at an 8192-token cap: 3 x 2176, as the trace shows.
        assert_eq!(
            checkpoint_aligned_chunk_tokens(0, 40_000, 8_192, BLOCK, false, 8_192),
            6_528
        );
        assert_eq!(
            checkpoint_aligned_chunk_tokens(6_528, 40_000, 8_192, BLOCK, false, 8_192),
            6_528
        );
        // Budget shared with decode still floors to a boundary.
        assert_eq!(
            checkpoint_aligned_chunk_tokens(0, 40_000, 3_837, BLOCK, false, 8_192),
            2_176
        );
    }

    #[test]
    fn a_budget_below_one_block_cannot_start_an_aligned_chunk() {
        assert_eq!(
            checkpoint_aligned_chunk_tokens(0, 40_000, 1_664, BLOCK, false, 8_192),
            0
        );
    }

    #[test]
    fn no_chunk_runs_past_the_prompt_last_boundary() {
        // The last 4532 tokens cross 43520 = 20 x 2176; the tail runs separately.
        let prefill_end = 43_700;
        assert_eq!(
            checkpoint_aligned_chunk_tokens(39_168, prefill_end, 4_532, BLOCK, false, 8_192),
            4_352
        );
        assert_eq!(
            checkpoint_aligned_chunk_tokens(43_520, prefill_end, 180, BLOCK, false, 8_192),
            180
        );
    }

    #[test]
    fn a_final_chunk_that_crosses_no_boundary_is_not_clipped() {
        assert_eq!(
            checkpoint_aligned_chunk_tokens(0, 1_801, 1_801, BLOCK, false, 8_192),
            1_801
        );
    }

    #[test]
    fn a_block_wider_than_the_cap_runs_sub_block_then_realigns() {
        // GLM-5.3-Flash at a 2048-token cap: 0 -> 2034 -> 2176, as the trace shows.
        assert_eq!(
            checkpoint_aligned_chunk_tokens(0, 5_000, 2_034, BLOCK, false, 2_048),
            2_034
        );
        assert_eq!(
            checkpoint_aligned_chunk_tokens(2_034, 5_000, 2_048, BLOCK, false, 2_048),
            142
        );
        assert_eq!(
            checkpoint_aligned_chunk_tokens(2_176, 5_000, 2_048, BLOCK, false, 2_048),
            2_048
        );
    }

    #[test]
    fn eagle_block_drop_backs_the_last_chunk_end_off_one_block() {
        // GLM-5.3-Flash H200 MTP 5: hybrid block 1152, 4096-token prompt, 8192 cap.
        // vLLM (`use_eagle_block_drop`) stops the cacheable chunk at 2304, one
        // block before the prompt's last boundary 3456.
        assert_eq!(
            checkpoint_aligned_chunk_tokens(0, 4_096, 4_096, 1_152, false, 8_192),
            3_456
        );
        assert_eq!(
            checkpoint_aligned_chunk_tokens(0, 4_096, 4_096, 1_152, true, 8_192),
            2_304
        );
        // The tail after the stop runs as one chunk.
        assert_eq!(
            checkpoint_aligned_chunk_tokens(2_304, 4_096, 1_792, 1_152, true, 8_192),
            1_792
        );
        // A prompt shorter than two blocks has no cacheable stop left.
        assert_eq!(
            checkpoint_aligned_chunk_tokens(0, 2_048, 2_048, 1_152, true, 8_192),
            2_048
        );
    }

    #[test]
    fn a_finer_prefix_match_unit_stops_at_the_partial_tail_boundary() {
        // --prefix-match-unit 64, EAGLE block drop: a 4096-token prompt stops
        // at 2304 as before, then at floor(4096 / 64) * 64 - 64 = 4032.
        let chunk = |start: u32, chunk_tokens: u32| {
            let aligned =
                checkpoint_aligned_chunk_tokens(start, 4_096, chunk_tokens, 1_152, true, 8_192);
            partial_tail_chunk_tokens(start, 4_096, aligned, 1_152, 64, true)
        };
        assert_eq!(chunk(0, 4_096), 2_304);
        assert_eq!(chunk(2_304, 1_792), 1_728);
        assert_eq!(chunk(4_032, 64), 64);
        // A turn resumed mid-block at 19904 of a 21200-token prompt: the
        // block stop, then the tail stop at 21120, then the last 80 tokens.
        let chunk = |start: u32, chunk_tokens: u32| {
            let aligned =
                checkpoint_aligned_chunk_tokens(start, 21_200, chunk_tokens, 1_152, true, 8_192);
            partial_tail_chunk_tokens(start, 21_200, aligned, 1_152, 64, true)
        };
        assert_eq!(chunk(19_904, 1_296), 832);
        assert_eq!(chunk(20_736, 464), 384);
        assert_eq!(chunk(21_120, 80), 80);
        // Without the drop, a tail on the last block boundary adds no stop.
        assert_eq!(
            partial_tail_chunk_tokens(0, 2_310, 2_310, 1_152, 64, false),
            2_310
        );
    }
}
