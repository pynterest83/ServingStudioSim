//! Speculating chunked-prefill worker recipe.
//!
//! Same composition as `build_chunked_prefill_worker` on every axis but two: the
//! lifecycle carries `SpeculativeDecodeCompletion` instead of the single-token
//! one, and execution lowers a verify batch through `SpeculativeIterExecution`.
//! Both differences are the same underlying fact — a resident decode moves by a
//! whole accepted chain per iteration rather than one token — split across the
//! axis that owns advancement and the axis that owns model input.

use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::SpeculativeUnifiedModel;
use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::log::PrefixCacheLogger;
use crate::worker::admission::{
    ChunkedPrefillAdmission, LoadBalance, PendingOrder, SpeculativeDecodeCompletion,
};
use crate::worker::execution::SpeculativeIterExecution;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::kv::FullAttnKv;
use crate::worker::types::WorkerConfig;

use super::iter_batch_worker::{IterBatchWorker, SpeculativeWorker};
use crate::worker::workers::iter_build_essentials::{
    full_attention_token_capacity, prepare_iter_build_essentials,
};

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_speculative_worker<M: SpeculativeUnifiedModel>(
    id: WorkerId,
    pool_tag: &'static str,
    model: Arc<M>,
    requests: SharedRequests,
    config: WorkerConfig,
    cost_log_dir: Option<PathBuf>,
    pool: PoolId,
    gpu_name: &str,
    cluster: SharedGpuCluster,
) -> SpeculativeWorker<M> {
    let max_batch_tokens = config
        .max_batch_tokens
        .expect("speculative worker requires max_batch_tokens");
    let draft_tokens = config.speculative_draft_tokens;
    assert!(
        draft_tokens > 0,
        "speculative worker requires speculative_draft_tokens > 0; \
         zero drafts is the chunked-prefill worker"
    );
    let drafting_slots = model.drafting_slots_per_request();
    assert!(
        draft_tokens
            .checked_add(1)
            .and_then(|width| width.checked_add(drafting_slots))
            .is_some_and(|width| width <= max_batch_tokens),
        "speculative verify width must fit max_batch_tokens"
    );
    let prefix_cache_logger = PrefixCacheLogger::open_opt(cost_log_dir.as_deref(), pool_tag, id);
    let num_partitions = model.num_attn_dp_groups().max(1) as usize;
    let kv_capacity = full_attention_token_capacity(
        model.num_attn_shards(),
        model.total_kv_bytes_per_token(),
        &config,
    );
    let prefix_cache = config.prefix_cache.resolve_tokens(
        kv_capacity,
        model.total_kv_bytes_per_token(),
        model.num_attn_shards(),
    );
    let essentials = prepare_iter_build_essentials(
        id,
        pool_tag,
        requests,
        &config,
        cost_log_dir,
        pool,
        gpu_name,
        &cluster,
        model.gpus_per_replica(),
        model.cost_log_manifest(),
        kv_capacity,
        num_partitions,
    );
    let kv_store = FullAttnKv::with_prefix_cache(
        num_partitions,
        essentials.kv_capacity,
        prefix_cache,
        essentials.sampler,
        prefix_cache_logger,
    );
    let balance = if num_partitions == 1 {
        LoadBalance::Single
    } else {
        LoadBalance::RoundRobin { next: 0 }
    };
    let admission = ChunkedPrefillAdmission::with_decode_completion(
        (0..num_partitions)
            .map(|_| (PendingOrder::new(config.pending_order), ()))
            .collect(),
        max_batch_tokens,
        config.batch_policy,
        config.kv_admission,
        balance,
        SpeculativeDecodeCompletion::new(
            draft_tokens,
            config.speculative_acceptance_seed.unwrap_or(0),
        ),
        drafting_slots,
    );
    // vLLM's async scheduling has a lifecycle half (the trailing verify row
    // and the QUEUED stamp) and a cadence half (the output lag); both or
    // neither.
    let admission = if config.async_scheduling {
        admission.with_async_scheduling()
    } else {
        admission
    };
    let worker = IterBatchWorker::from_components(
        essentials.context,
        kv_store,
        admission,
        SpeculativeIterExecution::new(model, essentials.cost, draft_tokens),
    );
    if config.async_scheduling {
        worker.with_async_scheduling()
    } else {
        worker
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::arch::contract::SpeculativeArchInput;
    use crate::common::{AcceptanceProfile, DecodingStrategy, RequestId, Time};
    use crate::test_helpers::{lm, shared_with, test_cluster};
    use crate::timing::LeafMetrics;
    use crate::worker::config::{BoundedFutureKvAdmissionConfig, KvAdmissionConfig};
    use crate::worker::iter_worker::IterWorker;
    use crate::worker::types::{WorkerEventCommon, WorkerMsgCommon};
    use crate::worker::DecodeRetractionPolicy;

    const DRAFT_TOKENS: u32 = 5;
    const PROMPT_TOKENS: u32 = 8;
    const OUTPUT_TOKENS: u32 = 12;

    /// Fixed 1 ms per iteration, mirroring `FakeModel` on the ordinary face, so
    /// a 1 ms tick step drives exactly one iteration and a test can count them.
    struct FakeSpeculativeModel;

    impl SpeculativeUnifiedModel for FakeSpeculativeModel {
        fn eval_speculative_iter(
            &self,
            batch: &SpeculativeArchInput,
            slots: &mut Vec<LeafMetrics>,
            _scratch: &mut Vec<LeafMetrics>,
        ) -> LeafMetrics {
            slots.clear();
            assert_eq!(batch.draft_tokens, DRAFT_TOKENS);
            for group in &batch.groups {
                assert_eq!(
                    group.total_kv_len,
                    group
                        .decode_requests
                        .iter()
                        .map(|request| request.kv_len - request.query_len)
                        .sum::<u32>(),
                    "real KV admission stores computed context, excluding pending queries",
                );
            }
            lm(1.0)
        }

        fn total_kv_bytes_per_token(&self) -> u64 {
            1024
        }

        fn max_model_len(&self) -> u32 {
            8192
        }

        fn gpus_per_replica(&self) -> u16 {
            1
        }
    }

    fn config(kv_admission: KvAdmissionConfig) -> WorkerConfig {
        WorkerConfig {
            attn_kv_bytes: 1_000_000_000,
            max_batch_tokens: Some(2048),
            kv_admission,
            speculative_draft_tokens: DRAFT_TOKENS,
            speculative_acceptance_seed: Some(7),
            ..WorkerConfig::default()
        }
    }

    fn build(
        config: WorkerConfig,
        requests: SharedRequests,
    ) -> SpeculativeWorker<FakeSpeculativeModel> {
        build_speculative_worker(
            WorkerId(0),
            "main",
            Arc::new(FakeSpeculativeModel),
            requests,
            config,
            None,
            PoolId(0),
            "test-gpu",
            test_cluster(),
        )
    }

    /// One request at `accept_rate`, run to completion. Returns the number of
    /// 1 ms iterations it took.
    fn iterations_to_complete(accept_rate: f32) -> u64 {
        let store = shared_with(&[(0, PROMPT_TOKENS, OUTPUT_TOKENS)]);
        store.borrow_mut()[RequestId(0)].request.definition.decoding =
            DecodingStrategy::Speculative {
                accept_rate: AcceptanceProfile::Uniform(accept_rate),
            };
        let mut worker = build(config(KvAdmissionConfig::FullFootprint), store);
        worker.enqueue(WorkerMsgCommon::Request(RequestId(0)));

        let mut events = Vec::new();
        for step in 0..200u64 {
            worker.tick(Time::from_ms(step as f64), &mut events);
            if events.contains(&WorkerEventCommon::RequestComplete {
                worker: WorkerId(0),
                req: RequestId(0),
            }) {
                return step;
            }
        }
        panic!("the request never completed");
    }

    #[test]
    fn the_recipe_composes_a_worker_the_shell_accepts() {
        // Compile-time proof of the production tuple: the `IterWorker` bound is
        // what rejects a mismatched KV / lifecycle / execution combination.
        fn assert_iter_worker<W: IterWorker>() {}
        assert_iter_worker::<SpeculativeWorker<FakeSpeculativeModel>>();
    }

    #[test]
    fn a_fully_accepting_engine_finishes_in_a_fraction_of_the_decode_steps() {
        // The defect this catches: wiring `SingleTokenDecodeCompletion` into the
        // speculative recipe. The worker would still run and still emit the
        // right token count — it would just silently model no speedup at all.
        let accepting = iterations_to_complete(1.0);
        let rejecting = iterations_to_complete(0.0);
        assert!(
            accepting * 2 < rejecting,
            "a k={DRAFT_TOKENS} chain retires up to {} tokens per verify, but the \
             accepting run took {accepting} iterations against {rejecting}",
            DRAFT_TOKENS + 1
        );
    }

    #[test]
    fn every_acceptance_rate_emits_exactly_the_requested_output_length() {
        // Acceptance changes how fast tokens retire, never how many: an
        // over-long chain past `target_output_tokens` must be truncated.
        for accept_rate in [0.0, 0.5, 1.0] {
            let store = shared_with(&[(0, PROMPT_TOKENS, OUTPUT_TOKENS)]);
            store.borrow_mut()[RequestId(0)].request.definition.decoding =
                DecodingStrategy::Speculative {
                    accept_rate: AcceptanceProfile::Uniform(accept_rate),
                };
            let mut worker = build(config(KvAdmissionConfig::FullFootprint), store.clone());
            worker.enqueue(WorkerMsgCommon::Request(RequestId(0)));
            for step in 0..200u64 {
                worker.tick(Time::from_ms(step as f64), &mut Vec::new());
            }

            let store = store.borrow();
            let record = &store[RequestId(0)];
            assert!(record.lifecycle.completed, "accept_rate={accept_rate}");
            let observation = record.telemetry.speculative.as_ref().unwrap();
            assert_eq!(observation.query_width, 6);
            assert_eq!(observation.prefill_chunks, 1);
            assert_eq!(observation.emitted_tokens, 11);
            assert_eq!(observation.pending_decode, None);
            assert_eq!(observation.pending_prefill, None);
            if accept_rate == 0.0 {
                assert_eq!(observation.decode_rounds, 11);
                assert_eq!(observation.resident_kv_sum, 143);
            } else if accept_rate == 1.0 {
                assert_eq!(observation.decode_rounds, 2);
                assert_eq!(observation.resident_kv_sum, 22);
            }
            assert_eq!(
                record.progress.output_tokens_emitted, OUTPUT_TOKENS,
                "accept_rate={accept_rate}"
            );
        }
    }

    #[test]
    fn bounded_future_admission_retracts_before_a_verify_overflows_kv() {
        // The defect this catches: sizing a speculating engine's decode headroom
        // at one token per request. A verify advances up to k+1 positions, so
        // the store would overfill silently and retract only afterwards.
        use std::sync::atomic::{AtomicU32, Ordering};
        const CAPACITY: u64 = 76;
        const OUTPUT: u32 = 24;
        struct CapacityModel {
            most_decodes: AtomicU32,
        }
        impl SpeculativeUnifiedModel for CapacityModel {
            fn eval_speculative_iter(
                &self,
                batch: &SpeculativeArchInput,
                slots: &mut Vec<LeafMetrics>,
                scratch: &mut Vec<LeafMetrics>,
            ) -> LeafMetrics {
                for group in &batch.groups {
                    self.most_decodes
                        .fetch_max(group.decode_requests.len() as u32, Ordering::Relaxed);
                }
                FakeSpeculativeModel.eval_speculative_iter(batch, slots, scratch)
            }
            fn total_kv_bytes_per_token(&self) -> u64 {
                1024
            }
            fn max_model_len(&self) -> u32 {
                8192
            }
            fn gpus_per_replica(&self) -> u16 {
                1
            }
        }
        let rows: Vec<_> = (0..4).map(|id| (id, PROMPT_TOKENS, OUTPUT)).collect();
        let requests = shared_with(&rows);
        for id in 0..4 {
            requests.borrow_mut()[RequestId(id)]
                .request
                .definition
                .decoding = DecodingStrategy::Speculative {
                accept_rate: AcceptanceProfile::Uniform(1.0),
            };
        }
        let model = Arc::new(CapacityModel {
            most_decodes: AtomicU32::new(0),
        });
        let mut config = config(KvAdmissionConfig::BoundedFuture(
            BoundedFutureKvAdmissionConfig {
                page_size: 1,
                max_future_tokens: 8,
                initial_new_token_ratio: 0.0,
                minimum_new_token_ratio: 0.0,
                new_token_ratio_decay_steps: 1,
                retract_decode_steps: 1,
                retraction_policy: DecodeRetractionPolicy::Length,
            },
        ));
        config.attn_kv_bytes = CAPACITY * 1024;
        let mut worker = build_speculative_worker(
            WorkerId(0),
            "test",
            model.clone(),
            requests.clone(),
            config,
            None,
            PoolId(0),
            "test-gpu",
            test_cluster(),
        );
        for id in 0..4 {
            worker.enqueue(WorkerMsgCommon::Request(RequestId(id)));
        }
        for tick in 0..400 {
            // `FullAttnKv::advance` asserts the store never overfills.
            worker.tick(Time::from_ms(tick as f64), &mut Vec::new());
        }

        let requests = requests.borrow();
        for id in 0..4 {
            let record = &requests[RequestId(id)];
            assert!(record.lifecycle.completed, "request {id}");
            assert_eq!(
                record.progress.output_tokens_emitted, OUTPUT,
                "request {id}"
            );
        }
        assert!(
            (0..4).any(|id| requests[RequestId(id)].telemetry.retraction_count > 0),
            "the pressure this test builds must force a retraction"
        );
        // Full-footprint would hold two (2 x 32 of 76 tokens); bounded-future
        // admits on the near future and decodes more at once.
        assert!(model.most_decodes.load(Ordering::Relaxed) > 2);
    }

    /// vLLM async-scheduling rules. Every iteration costs 1 ms and the tests
    /// tick at 1 ms, so while the engine stays busy iteration `i` (1-based) is
    /// formed at tick `i - 1` and completes at tick `i`.
    mod async_scheduling {
        use std::sync::Mutex;

        use super::*;

        /// One iteration as the model saw it.
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        struct Iteration {
            decode_rows: u32,
            prefill_tokens: u32,
        }

        fn iteration(decode_rows: u32, prefill_tokens: u32) -> Iteration {
            Iteration {
                decode_rows,
                prefill_tokens,
            }
        }

        /// 1 ms per iteration; records each iteration's composition.
        #[derive(Default)]
        struct RecordingModel {
            iterations: Mutex<Vec<Iteration>>,
        }

        impl SpeculativeUnifiedModel for RecordingModel {
            fn eval_speculative_iter(
                &self,
                batch: &SpeculativeArchInput,
                slots: &mut Vec<LeafMetrics>,
                scratch: &mut Vec<LeafMetrics>,
            ) -> LeafMetrics {
                self.iterations.lock().unwrap().push(Iteration {
                    decode_rows: batch
                        .groups
                        .iter()
                        .map(|group| group.decode_requests.len() as u32)
                        .sum(),
                    prefill_tokens: batch.groups.iter().map(|group| group.prefill_tokens).sum(),
                });
                FakeSpeculativeModel.eval_speculative_iter(batch, slots, scratch)
            }
            fn total_kv_bytes_per_token(&self) -> u64 {
                1024
            }
            fn max_model_len(&self) -> u32 {
                8192
            }
            fn gpus_per_replica(&self) -> u16 {
                1
            }
        }

        /// One request: output length, uniform acceptance, and, for a
        /// closed-loop replacement, the request whose completion releases it.
        #[derive(Clone, Copy)]
        struct Spec {
            output_tokens: u32,
            accept_rate: f32,
            replaces: Option<u32>,
        }

        fn spec(output_tokens: u32, accept_rate: f32) -> Spec {
            Spec {
                output_tokens,
                accept_rate,
                replaces: None,
            }
        }

        fn replacing(predecessor: u32, output_tokens: u32, accept_rate: f32) -> Spec {
            Spec {
                replaces: Some(predecessor),
                ..spec(output_tokens, accept_rate)
            }
        }

        struct Run {
            iterations: Vec<Iteration>,
            /// Tick whose `tick` call emitted each request's `RequestComplete`.
            completed_at: Vec<usize>,
            /// Active (KV-resident) requests the worker reported after each
            /// tick.
            active_after_tick: Vec<u32>,
            /// Tick of the `tick` call that formed each iteration (1-based
            /// iteration `i` at index `i - 1`).
            formed_at: Vec<usize>,
            requests: SharedRequests,
        }

        impl Run {
            /// Iteration (1-based) in which `request`'s prefill ran.
            fn prefilled_at(&self, request: u32) -> usize {
                let admitted = self.requests.borrow()[RequestId(request)]
                    .telemetry
                    .speculative
                    .is_some();
                assert!(admitted, "request {request} never ran");
                // Every prompt in these scenarios fits one chunk, so the k-th
                // prefill-bearing iteration admitted the k-th prompt group.
                self.iterations
                    .iter()
                    .enumerate()
                    .filter(|(_, iteration)| iteration.prefill_tokens > 0)
                    .map(|(index, _)| index + 1)
                    .last()
                    .unwrap()
            }

            /// Client-visible output token times (ms).
            fn output_times(&self, request: u32) -> Vec<f64> {
                self.requests.borrow()[RequestId(request)]
                    .telemetry
                    .output_times
                    .iter()
                    .map(|time| time.as_ms())
                    .collect()
            }

            fn first_output_ms(&self, request: u32) -> f64 {
                self.requests.borrow()[RequestId(request)]
                    .telemetry
                    .first_output_time
                    .unwrap()
                    .as_ms()
            }

            fn queued_time(&self, request: u32) -> Option<Time> {
                self.requests.borrow()[RequestId(request)]
                    .telemetry
                    .queued_time
            }
        }

        fn run(async_scheduling: bool, specs: &[Spec]) -> Run {
            let rows: Vec<_> = specs
                .iter()
                .enumerate()
                .map(|(id, spec)| (id as u32, PROMPT_TOKENS, spec.output_tokens))
                .collect();
            let requests = shared_with(&rows);
            for (id, spec) in specs.iter().enumerate() {
                requests.borrow_mut()[RequestId(id as u32)]
                    .request
                    .definition
                    .decoding = DecodingStrategy::Speculative {
                    accept_rate: AcceptanceProfile::Uniform(spec.accept_rate),
                };
            }
            let model = Arc::new(RecordingModel::default());
            let mut config = config(KvAdmissionConfig::FullFootprint);
            config.async_scheduling = async_scheduling;
            config.log_output_token_times = true;
            let mut worker = build_speculative_worker(
                WorkerId(0),
                "test",
                model.clone(),
                requests.clone(),
                config,
                None,
                PoolId(0),
                "test-gpu",
                test_cluster(),
            );
            for (id, spec) in specs.iter().enumerate() {
                if spec.replaces.is_none() {
                    worker.enqueue(WorkerMsgCommon::Request(RequestId(id as u32)));
                }
            }
            let mut completed_at = vec![0; specs.len()];
            let mut active_after_tick = Vec::new();
            let mut formed_at = Vec::new();
            let mut events = Vec::new();
            for tick in 0..100u64 {
                worker.tick(Time::from_ms(tick as f64), &mut events);
                active_after_tick.push(worker.status().active_requests);
                let formed = model.iterations.lock().unwrap().len();
                formed_at.resize(formed, tick as usize);
                for event in events.drain(..) {
                    let WorkerEventCommon::RequestComplete { req, .. } = event;
                    completed_at[req.0 as usize] = tick as usize;
                    // L6 delivers the replacement after this tick, while the
                    // iteration the tick just formed (if any) runs; it
                    // arrives at the completion instant.
                    for (id, spec) in specs.iter().enumerate() {
                        if spec.replaces == Some(req.0) {
                            requests.borrow_mut()[RequestId(id as u32)]
                                .request
                                .core
                                .arrival_time = Time::from_ms(tick as f64);
                            worker.enqueue(WorkerMsgCommon::Request(RequestId(id as u32)));
                        }
                    }
                }
            }
            for (id, spec) in specs.iter().enumerate() {
                let record = &requests.borrow()[RequestId(id as u32)];
                assert!(record.lifecycle.completed, "request {id}");
                assert_eq!(record.progress.output_tokens_emitted, spec.output_tokens);
                assert_eq!(
                    record.telemetry.output_times.len(),
                    spec.output_tokens as usize
                );
            }
            let iterations = model.iterations.lock().unwrap().clone();
            Run {
                iterations,
                completed_at,
                active_after_tick,
                formed_at,
                requests,
            }
        }

        #[test]
        fn a_step_output_is_visible_at_the_end_of_the_next_step() {
            // Request 0 keeps the engine busy. Request 1 prefills in
            // iteration 1 and rejects every draft, retiring one token in each
            // of iterations 2..4. vLLM processes each step's output only after
            // it launched the next step.
            let specs = [spec(64, 0.0), spec(4, 0.0)];
            let on = run(true, &specs);
            let off = run(false, &specs);
            assert_eq!(off.output_times(1), [1.0, 2.0, 3.0, 4.0]);
            assert_eq!(off.completed_at[1], 4);
            assert_eq!(on.output_times(1), [2.0, 3.0, 4.0, 5.0]);
            assert_eq!(on.first_output_ms(1), 2.0, "TTFT moves by one step");
            assert_eq!(on.completed_at[1], 5, "so does the completion");
            // The lag is visibility only: the schedule and KV are unchanged.
            assert_eq!(on.iterations, off.iterations);
            assert_eq!(on.active_after_tick, off.active_after_tick);
        }

        #[test]
        fn an_idle_engine_shows_a_step_output_at_its_own_end() {
            // One request, every draft rejected: iterations 1..4. Iteration
            // 4's output has no next step to wait for.
            let on = run(true, &[spec(4, 0.0)]);
            let off = run(false, &[spec(4, 0.0)]);
            assert_eq!(off.output_times(0), [1.0, 2.0, 3.0, 4.0]);
            assert_eq!(on.output_times(0), [2.0, 3.0, 4.0, 4.0]);
            assert_eq!(on.completed_at, [4]);
            assert_eq!(on.completed_at, off.completed_at);
            assert_eq!(on.iterations, off.iterations);
        }

        #[test]
        fn a_closed_loop_replacement_enters_at_n_plus_3_through_the_lag() {
            // Request 0 keeps the engine busy; request 1 finishes at N = 2 by
            // accepting drafts; request 2 replaces it.
            let specs = [spec(64, 0.0), spec(4, 1.0), replacing(1, 2, 0.0)];
            let on = run(true, &specs);
            // N+1 runs the trailing row, so N's completion becomes visible at
            // its end, and the replacement arrives while N+2 runs.
            assert_eq!(on.iterations[2], iteration(2, 0), "N+1 trailing row");
            assert_eq!(on.completed_at[1], 2 + 1);
            assert_eq!(on.iterations[3], iteration(1, 0), "N+2 without it");
            assert_eq!(on.prefilled_at(2), 2 + 3);
            // The engine picks it up when N+2 ends and schedules it in that
            // same decision: no admission hold.
            let queued = on.queued_time(2).unwrap();
            assert_eq!(queued, Time::from_ms(4.0));
            assert_eq!(on.formed_at[on.prefilled_at(2) - 1], 4);
            let off = run(false, &specs);
            assert_eq!(off.completed_at[1], 2);
            assert_eq!(off.prefilled_at(2), 2 + 2);
            assert_eq!(off.queued_time(2), None);
        }

        #[test]
        fn a_closed_loop_replacement_enters_at_n_plus_2_into_an_otherwise_idle_engine() {
            // Request 0 alone finishes at N = 2 by accepting drafts. Its
            // trailing row runs at N+1, whose end shows the completion; the
            // engine then idles, so the replacement is queued on arrival and
            // runs next.
            let on = run(true, &[spec(4, 1.0), replacing(0, 2, 0.0)]);
            assert_eq!(on.iterations[2], iteration(1, 0));
            assert_eq!(on.completed_at[0], 3);
            assert_eq!(on.prefilled_at(1), 2 + 2);
            assert_eq!(on.queued_time(1), None, "queued at its arrival");
            // With no trailing row the engine idles after N already, shows the
            // completion at once, and the replacement starts as without the
            // option.
            let on = run(true, &[spec(2, 1.0), replacing(0, 2, 0.0)]);
            let off = run(false, &[spec(2, 1.0), replacing(0, 2, 0.0)]);
            assert_eq!(on.iterations, off.iterations);
            assert_eq!(on.completed_at, off.completed_at);
            assert_eq!(on.prefilled_at(1), off.prefilled_at(1));
        }

        #[test]
        fn a_request_finished_by_accepted_drafts_runs_exactly_one_trailing_row() {
            // Prefill emits 1 of 4; round 2 starts with 3 left and accepts them
            // all, so only the drafts finished it. The async scheduler had
            // already issued a third row.
            let on = run(true, &[spec(4, 1.0)]);
            assert_eq!(
                on.iterations,
                [
                    iteration(0, PROMPT_TOKENS),
                    iteration(1, 0),
                    iteration(1, 0)
                ]
            );
            let off = run(false, &[spec(4, 1.0)]);
            assert_eq!(
                off.iterations,
                [iteration(0, PROMPT_TOKENS), iteration(1, 0)]
            );
            // The trailing row is the step after N, so it carries N's output.
            assert_eq!(on.completed_at, [3]);
            assert_eq!(on.output_times(0), [2.0, 3.0, 3.0, 3.0]);
            assert_eq!(off.completed_at, [2]);
            // KV stays resident through the trailing row, then is released.
            assert_eq!(on.active_after_tick[2..4], [1, 0]);
            assert_eq!(off.active_after_tick[2], 0);
            // The trailing row emits nothing and is not a verify round.
            let requests = on.requests.borrow();
            let observation = requests[RequestId(0)]
                .telemetry
                .speculative
                .as_ref()
                .unwrap();
            assert_eq!(observation.decode_rounds, 1);
            assert_eq!(observation.emitted_tokens, 3);
            assert_eq!(observation.pending_decode, None);
        }

        #[test]
        fn a_request_with_at_most_one_token_left_runs_no_trailing_row() {
            // Two tokens: prefill emits 1 and round 2 starts with 1 left, so it
            // is sure to finish. Four tokens rejecting every draft: the final
            // round also starts with 1 left.
            for (output_tokens, accept_rate, rounds) in [(2, 1.0, 1), (4, 0.0, 3)] {
                let on = run(true, &[spec(output_tokens, accept_rate)]);
                let off = run(false, &[spec(output_tokens, accept_rate)]);
                assert_eq!(on.iterations, off.iterations);
                assert_eq!(on.iterations.len(), 1 + rounds);
                assert_eq!(on.completed_at, off.completed_at);
                assert_eq!(on.active_after_tick[1 + rounds], 0);
            }
        }

        #[test]
        fn queued_is_the_next_step_boundary_and_a_kv_wait_stays_in_server_ttft() {
            // KV holds one 8 + 8 token request at a time. Request 1 arrives at
            // 0.5 ms, while iteration 1 prefills request 0.
            let requests = shared_with(&[(0, PROMPT_TOKENS, 8), (1, PROMPT_TOKENS, 8)]);
            for id in 0..2 {
                let mut store = requests.borrow_mut();
                store[RequestId(id)].request.definition.decoding = DecodingStrategy::Speculative {
                    accept_rate: AcceptanceProfile::Uniform(0.0),
                };
            }
            requests.borrow_mut()[RequestId(1)]
                .request
                .core
                .arrival_time = Time::from_ms(0.5);
            let model = Arc::new(RecordingModel::default());
            let mut config = config(KvAdmissionConfig::FullFootprint);
            config.async_scheduling = true;
            config.attn_kv_bytes = 24 * 1024;
            let mut worker = build_speculative_worker(
                WorkerId(0),
                "test",
                model.clone(),
                requests.clone(),
                config,
                None,
                PoolId(0),
                "test-gpu",
                test_cluster(),
            );
            worker.enqueue(WorkerMsgCommon::Request(RequestId(0)));
            let mut events = Vec::new();
            worker.tick(Time::ZERO, &mut events);
            worker.enqueue(WorkerMsgCommon::Request(RequestId(1)));
            for tick in 1..40u64 {
                worker.tick(Time::from_ms(tick as f64), &mut events);
            }
            // Request 0 decodes in iterations 2..8 and frees its KV at the
            // end of 8; request 1 prefills in 9, its first token shows at the
            // end of 10.
            let iterations = model.iterations.lock().unwrap().clone();
            assert_eq!(iterations[8], iteration(0, PROMPT_TOKENS));
            let store = requests.borrow();
            let record = &store[RequestId(1)];
            assert_eq!(record.telemetry.queued_time, Some(Time::from_ms(1.0)));
            assert_eq!(
                record.telemetry.first_output_time,
                Some(Time::from_ms(10.0))
            );
            // Server TTFT (10 - 1 ms) keeps the KV wait from 1 ms to the
            // first schedule at 8 ms; client TTFT also keeps the half step
            // before pickup.
            assert!(record.lifecycle.completed);
            // A request that arrived at an idle engine is queued on arrival.
            assert_eq!(store[RequestId(0)].telemetry.queued_time, None);
        }

        #[test]
        fn the_option_off_keeps_the_synchronous_schedule() {
            // Hand-derived synchronous schedule: request 0 decodes rounds 2..8;
            // request 1 finishes at 2 and frees its row at once; its
            // replacement joins iteration 4 and decodes once.
            let specs = [spec(8, 0.0), spec(4, 1.0), replacing(1, 2, 0.0)];
            let off = run(false, &specs);
            assert_eq!(
                off.iterations,
                [
                    iteration(0, 2 * PROMPT_TOKENS),
                    iteration(2, 0),
                    iteration(1, 0),
                    iteration(1, PROMPT_TOKENS),
                    iteration(2, 0),
                    iteration(1, 0),
                    iteration(1, 0),
                    iteration(1, 0),
                ]
            );
            assert_eq!(off.completed_at, [8, 2, 5]);
            // Tokens are visible when their iteration ends, and the queued
            // time is the arrival.
            assert_eq!(off.output_times(1), [1.0, 2.0, 2.0, 2.0]);
            assert_eq!(off.output_times(2), [4.0, 5.0]);
            assert!((0..3).all(|request| off.queued_time(request).is_none()));
            let default_config = WorkerConfig::default();
            assert!(!default_config.async_scheduling);
        }
    }

    #[test]
    #[should_panic(expected = "speculative_draft_tokens > 0")]
    fn a_worker_that_drafts_nothing_is_not_a_speculative_worker() {
        let mut config = config(KvAdmissionConfig::FullFootprint);
        config.speculative_draft_tokens = 0;
        build(config, shared_with(&[]));
    }

    #[test]
    #[should_panic(expected = "verify width must fit max_batch_tokens")]
    fn a_batch_must_fit_at_least_one_verify_window() {
        let mut config = config(KvAdmissionConfig::FullFootprint);
        config.max_batch_tokens = Some(DRAFT_TOKENS);
        build(config, shared_with(&[]));
    }

    #[test]
    fn drafting_slots_are_charged_to_every_scheduled_request() {
        // The defect this catches: budgeting a parallel drafter's requests at
        // the verify width alone. vLLM charges DFlash K more slots per scheduled
        // request, so at 33 tokens and k=5 it runs 3 requests per step, not 5.
        use std::sync::atomic::{AtomicU32, Ordering};
        const BUDGET: u32 = 33;
        struct DraftSlotModel {
            drafting_slots: u32,
            most_requests: AtomicU32,
        }
        impl SpeculativeUnifiedModel for DraftSlotModel {
            fn eval_speculative_iter(
                &self,
                batch: &SpeculativeArchInput,
                slots: &mut Vec<LeafMetrics>,
                scratch: &mut Vec<LeafMetrics>,
            ) -> LeafMetrics {
                for group in &batch.groups {
                    let requests = group.request_count();
                    assert!(group.batch_tokens + requests * self.drafting_slots <= BUDGET);
                    self.most_requests.fetch_max(requests, Ordering::Relaxed);
                }
                FakeSpeculativeModel.eval_speculative_iter(batch, slots, scratch)
            }
            fn total_kv_bytes_per_token(&self) -> u64 {
                1024
            }
            fn max_model_len(&self) -> u32 {
                8192
            }
            fn gpus_per_replica(&self) -> u16 {
                1
            }
            fn drafting_slots_per_request(&self) -> u32 {
                self.drafting_slots
            }
        }
        for (drafting_slots, expected_requests) in [(0, 5), (DRAFT_TOKENS, 3)] {
            let rows: Vec<_> = (0..8).map(|id| (id, 1, OUTPUT_TOKENS)).collect();
            let requests = shared_with(&rows);
            for id in 0..8 {
                requests.borrow_mut()[RequestId(id)]
                    .request
                    .definition
                    .decoding = DecodingStrategy::Speculative {
                    accept_rate: AcceptanceProfile::Uniform(0.0),
                };
            }
            let model = Arc::new(DraftSlotModel {
                drafting_slots,
                most_requests: AtomicU32::new(0),
            });
            let mut config = config(KvAdmissionConfig::FullFootprint);
            config.max_batch_tokens = Some(BUDGET);
            let mut worker = build_speculative_worker(
                WorkerId(0),
                "test",
                model.clone(),
                requests.clone(),
                config,
                None,
                PoolId(0),
                "test-gpu",
                test_cluster(),
            );
            for id in 0..8 {
                worker.enqueue(WorkerMsgCommon::Request(RequestId(id)));
            }
            for tick in 0..200 {
                worker.tick(Time::from_ms(tick as f64), &mut Vec::new());
            }
            for id in 0..8 {
                assert!(requests.borrow()[RequestId(id)].lifecycle.completed);
            }
            assert_eq!(
                model.most_requests.load(Ordering::Relaxed),
                expected_requests,
                "drafting_slots={drafting_slots}"
            );
        }
    }

    #[test]
    fn short_prefills_cannot_overfill_the_following_verify_batch() {
        struct BoundedBatchModel;
        impl SpeculativeUnifiedModel for BoundedBatchModel {
            fn eval_speculative_iter(
                &self,
                batch: &SpeculativeArchInput,
                slots: &mut Vec<LeafMetrics>,
                scratch: &mut Vec<LeafMetrics>,
            ) -> LeafMetrics {
                assert!(batch.groups.iter().all(|group| group.batch_tokens <= 8));
                FakeSpeculativeModel.eval_speculative_iter(batch, slots, scratch)
            }
            fn total_kv_bytes_per_token(&self) -> u64 {
                1024
            }
            fn max_model_len(&self) -> u32 {
                8192
            }
            fn gpus_per_replica(&self) -> u16 {
                1
            }
        }
        for (batch_policy, prompt_tokens) in [
            (crate::worker::config::BatchPolicy::Mix, 1),
            (crate::worker::config::BatchPolicy::Mix, 9),
            (
                crate::worker::config::BatchPolicy::SeparatePrefillPriority,
                1,
            ),
            (
                crate::worker::config::BatchPolicy::SeparatePrefillPriority,
                9,
            ),
        ] {
            let rows: Vec<_> = (0..8)
                .map(|id| (id, prompt_tokens, if id % 3 == 0 { 1 } else { 8 }))
                .collect();
            let requests = shared_with(&rows);
            for id in 0..8 {
                requests.borrow_mut()[RequestId(id)]
                    .request
                    .definition
                    .decoding = DecodingStrategy::Speculative {
                    accept_rate: AcceptanceProfile::Uniform(0.0),
                };
            }
            let mut config = config(KvAdmissionConfig::FullFootprint);
            config.max_batch_tokens = Some(8);
            config.batch_policy = batch_policy;
            let mut worker = build_speculative_worker(
                WorkerId(0),
                "test",
                Arc::new(BoundedBatchModel),
                requests.clone(),
                config,
                None,
                PoolId(0),
                "test-gpu",
                test_cluster(),
            );
            for id in 0..8 {
                worker.enqueue(WorkerMsgCommon::Request(RequestId(id)));
            }
            for tick in 0..100 {
                worker.tick(Time::from_ms(tick as f64), &mut Vec::new());
            }
            for id in 0..8 {
                assert!(requests.borrow()[RequestId(id)].lifecycle.completed);
            }
        }
    }
}
