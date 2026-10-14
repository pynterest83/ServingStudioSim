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
use crate::worker::config::KvAdmissionConfig;
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
    assert!(
        !matches!(config.kv_admission, KvAdmissionConfig::Incremental),
        "incremental KV admission is modeled only for the hybrid stores"
    );
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
        )
        .with_position_bands(config.speculative_acceptance_position_bands)
        .with_request_spread(config.speculative_acceptance_request_spread),
        drafting_slots,
    );
    let admission = match config.max_running_requests {
        Some(max_running) => admission.with_max_running_requests(max_running),
        None => admission,
    };
    // vLLM's async scheduling has a lifecycle half (decisions from the state
    // at the previous launch, the trailing verify row, the QUEUED stamp) and a
    // cadence half (between-step message pickup, the first-input step, output
    // processing); both or neither.
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
        worker.with_async_scheduling(config.async_output_processing)
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

    /// vLLM async-scheduling rules. Every iteration costs exactly 1 ms and the
    /// tests tick every 0.25 ms, so iteration ends land on ticks. An arrival
    /// is enqueued before the tick at its time; a closed-loop replacement is
    /// enqueued after the tick that emitted its predecessor's completion, at
    /// that time, as L6 does.
    mod async_scheduling {
        use std::sync::Mutex;

        use super::*;
        use crate::common::UnifiedStage;

        const TICK_MS: f64 = 0.25;
        const TICKS: u64 = 400;

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

        /// One request: prompt and output length, uniform acceptance, its
        /// arrival, and, for a closed-loop replacement, the request whose
        /// completion releases it.
        #[derive(Clone, Copy)]
        struct Spec {
            prompt_tokens: u32,
            output_tokens: u32,
            accept_rate: f32,
            arrival_ms: f64,
            replaces: Option<u32>,
        }

        fn spec(output_tokens: u32, accept_rate: f32) -> Spec {
            Spec {
                prompt_tokens: PROMPT_TOKENS,
                output_tokens,
                accept_rate,
                arrival_ms: 0.0,
                replaces: None,
            }
        }

        fn arriving(arrival_ms: f64, output_tokens: u32, accept_rate: f32) -> Spec {
            Spec {
                arrival_ms,
                ..spec(output_tokens, accept_rate)
            }
        }

        fn replacing(predecessor: u32, output_tokens: u32, accept_rate: f32) -> Spec {
            Spec {
                replaces: Some(predecessor),
                ..spec(output_tokens, accept_rate)
            }
        }

        #[derive(Clone, Copy)]
        struct Engine {
            async_scheduling: bool,
            output_processing_ms: f64,
            /// KV capacity in tokens (`None`: plenty).
            kv_tokens: Option<u64>,
        }

        const SYNC: Engine = Engine {
            async_scheduling: false,
            output_processing_ms: 0.0,
            kv_tokens: None,
        };
        const ASYNC: Engine = Engine {
            async_scheduling: true,
            ..SYNC
        };

        struct Run {
            iterations: Vec<Iteration>,
            /// Time (ms) at which each iteration was formed and launched.
            launched_ms: Vec<f64>,
            /// Time (ms) of the tick that emitted each `RequestComplete`.
            completed_ms: Vec<f64>,
            /// KV-resident requests the worker reported after each tick.
            active_after: Vec<(f64, u32)>,
            requests: SharedRequests,
        }

        impl Run {
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

            fn queued_ms(&self, request: u32) -> Option<f64> {
                self.requests.borrow()[RequestId(request)]
                    .telemetry
                    .queued_time
                    .map(Time::as_ms)
            }

            /// Server TTFT as `request_slo` writes it.
            fn server_ttft_ms(&self, request: u32) -> f64 {
                let arrival = self.requests.borrow()[RequestId(request)]
                    .request
                    .core
                    .arrival_time
                    .as_ms();
                self.first_output_ms(request) - self.queued_ms(request).unwrap_or(arrival)
            }

            /// The first-scheduled stamp (stage 1, `Prefill`).
            fn first_scheduled_ms(&self, request: u32) -> f64 {
                self.requests.borrow()[RequestId(request)]
                    .lifecycle
                    .stage_log
                    .iter()
                    .find(|stage| stage.code == UnifiedStage::Prefill as u16)
                    .expect("the request was scheduled")
                    .time
                    .as_ms()
            }

            fn active_after(&self, ms: f64) -> u32 {
                self.active_after
                    .iter()
                    .find(|(time, _)| *time == ms)
                    .expect("a tick at that time")
                    .1
            }
        }

        fn run(engine: Engine, specs: &[Spec]) -> Run {
            let rows: Vec<_> = specs
                .iter()
                .enumerate()
                .map(|(id, spec)| (id as u32, spec.prompt_tokens, spec.output_tokens))
                .collect();
            let requests = shared_with(&rows);
            for (id, spec) in specs.iter().enumerate() {
                let mut store = requests.borrow_mut();
                let record = &mut store[RequestId(id as u32)];
                record.request.definition.decoding = DecodingStrategy::Speculative {
                    accept_rate: AcceptanceProfile::Uniform(spec.accept_rate),
                };
                record.request.core.arrival_time = Time::from_ms(spec.arrival_ms);
            }
            let model = Arc::new(RecordingModel::default());
            let mut config = config(KvAdmissionConfig::FullFootprint);
            config.async_scheduling = engine.async_scheduling;
            config.async_output_processing = Time::from_ms(engine.output_processing_ms);
            config.log_output_token_times = true;
            if let Some(kv_tokens) = engine.kv_tokens {
                config.attn_kv_bytes = kv_tokens * 1024;
            }
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
            let mut enqueued = vec![false; specs.len()];
            let mut completed_ms = vec![f64::NAN; specs.len()];
            let mut launched_ms = Vec::new();
            let mut active_after = Vec::new();
            let mut events = Vec::new();
            for tick in 0..TICKS {
                let now_ms = tick as f64 * TICK_MS;
                // Arrivals due by now, in trace order.
                for (id, spec) in specs.iter().enumerate() {
                    if spec.replaces.is_none() && !enqueued[id] && spec.arrival_ms <= now_ms {
                        enqueued[id] = true;
                        worker.enqueue(WorkerMsgCommon::Request(RequestId(id as u32)));
                    }
                }
                worker.tick(Time::from_ms(now_ms), &mut events);
                active_after.push((now_ms, worker.status().active_requests));
                let launched = model.iterations.lock().unwrap().len();
                launched_ms.resize(launched, now_ms);
                for event in events.drain(..) {
                    let WorkerEventCommon::RequestComplete { req, .. } = event;
                    completed_ms[req.0 as usize] = now_ms;
                    for (id, spec) in specs.iter().enumerate() {
                        if spec.replaces == Some(req.0) {
                            requests.borrow_mut()[RequestId(id as u32)]
                                .request
                                .core
                                .arrival_time = Time::from_ms(now_ms);
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
                launched_ms,
                completed_ms,
                active_after,
                requests,
            }
        }

        #[test]
        fn a_step_output_is_visible_after_output_processing() {
            // Request 0 keeps the engine busy. Request 1, queued at the end of
            // step 1, prefills in step 3 and retires one token in each of
            // steps 4..6 (every draft rejected). Step k's output is visible at
            // the end of k plus the processing delay, not at the end of k+1.
            let specs = [spec(64, 0.0), spec(4, 0.0)];
            let processed = run(
                Engine {
                    output_processing_ms: 0.5,
                    ..ASYNC
                },
                &specs,
            );
            assert_eq!(processed.output_times(1), [3.5, 4.5, 5.5, 6.5]);
            assert_eq!(processed.completed_ms[1], 6.5, "the completion too");
            assert_eq!(processed.server_ttft_ms(1), 3.5 - 1.0);
            let at_step_end = run(ASYNC, &specs);
            assert_eq!(at_step_end.output_times(1), [3.0, 4.0, 5.0, 6.0]);
            assert_eq!(at_step_end.completed_ms[1], 6.0);
            // Processing is visibility only: the schedule and KV are unchanged.
            assert_eq!(processed.iterations, at_step_end.iterations);
            assert_eq!(processed.launched_ms, at_step_end.launched_ms);
            assert_eq!(processed.active_after, at_step_end.active_after);
            // Synchronous: both prefill in step 1.
            let off = run(SYNC, &specs);
            assert_eq!(off.output_times(1), [1.0, 2.0, 3.0, 4.0]);
            assert_eq!(off.completed_ms[1], 4.0);
        }

        #[test]
        fn a_request_queued_during_step_k_first_runs_in_k_plus_2() {
            // Request 1 arrives at 1.5 ms, while step 2 runs. Step 3 was formed
            // when step 2 launched, so request 1 is queued at the end of 2, is
            // scheduled by the decision made at 3's launch, and runs in 4.
            let specs = [spec(64, 0.0), arriving(1.5, 4, 0.0)];
            let on = run(ASYNC, &specs);
            assert_eq!(on.iterations[2], iteration(1, 0), "step 3 without it");
            assert_eq!(on.iterations[3], iteration(1, PROMPT_TOKENS));
            assert_eq!(on.launched_ms[3], 3.0);
            assert_eq!(on.queued_ms(1), Some(2.0));
            // First scheduled when the decision was made: no queue wait.
            assert_eq!(on.first_scheduled_ms(1), 2.0);
            assert_eq!(on.first_output_ms(1), 4.0);
            assert_eq!(on.server_ttft_ms(1), 2.0);
            // Synchronous: picked up on arrival, runs in step 3.
            let off = run(SYNC, &specs);
            assert_eq!(off.iterations[2], iteration(1, PROMPT_TOKENS));
            assert_eq!(off.queued_ms(1), None);
            assert_eq!(off.first_scheduled_ms(1), 2.0);
            assert_eq!(off.first_output_ms(1), 3.0);
        }

        #[test]
        fn an_idle_engine_runs_its_first_input_alone_and_the_burst_from_step_3() {
            // Four requests at the same instant. The idle engine launches the
            // first in trace order at once and forms step 2 from that state;
            // the rest are queued when step 1 ends and run in step 3.
            let specs = [spec(8, 0.0); 4];
            let on = run(ASYNC, &specs);
            assert_eq!(
                on.iterations[..3],
                [
                    iteration(0, PROMPT_TOKENS),
                    iteration(1, 0),
                    iteration(1, 3 * PROMPT_TOKENS)
                ]
            );
            assert_eq!(on.queued_ms(0), None, "queued at its arrival");
            assert_eq!(on.first_output_ms(0), 1.0);
            for request in 1..4 {
                assert_eq!(on.queued_ms(request), Some(1.0), "request {request}");
                assert_eq!(on.first_scheduled_ms(request), 1.0);
                assert_eq!(on.first_output_ms(request), 3.0);
            }
            let off = run(SYNC, &specs);
            assert_eq!(off.iterations[0], iteration(0, 4 * PROMPT_TOKENS));
        }

        #[test]
        fn the_idle_engine_rule_applies_again_later_in_the_run() {
            // Request 0 finishes at step 2 (sure to finish, no trailing row)
            // and the engine idles. Two requests arrive together at 4 ms: the
            // first runs alone at once, the second is queued when that step
            // ends and runs two steps later.
            let specs = [spec(2, 1.0), arriving(4.0, 8, 0.0), arriving(4.0, 8, 0.0)];
            let on = run(ASYNC, &specs);
            assert_eq!(
                on.iterations[..5],
                [
                    iteration(0, PROMPT_TOKENS),
                    iteration(1, 0),
                    iteration(0, PROMPT_TOKENS),
                    iteration(1, 0),
                    iteration(1, PROMPT_TOKENS)
                ]
            );
            assert_eq!(on.launched_ms[2..5], [4.0, 5.0, 6.0]);
            assert_eq!(on.queued_ms(1), None);
            assert_eq!(on.first_scheduled_ms(1), 4.0);
            assert_eq!(on.queued_ms(2), Some(5.0));
            assert_eq!(on.first_scheduled_ms(2), 5.0);
            assert_eq!(on.first_output_ms(2), 7.0);
        }

        #[test]
        fn a_request_finished_by_accepted_drafts_runs_exactly_one_trailing_row() {
            // Prefill emits 1 of 4; round 2 starts with 3 left and accepts them
            // all, so only the drafts finished it. Step 3 was formed when step
            // 2 launched, before that output: it carries a wasted row.
            let on = run(ASYNC, &[spec(4, 1.0)]);
            assert_eq!(
                on.iterations,
                [
                    iteration(0, PROMPT_TOKENS),
                    iteration(1, 0),
                    iteration(1, 0)
                ]
            );
            let off = run(SYNC, &[spec(4, 1.0)]);
            assert_eq!(
                off.iterations,
                [iteration(0, PROMPT_TOKENS), iteration(1, 0)]
            );
            // The output is visible at the end of the step that computed it.
            assert_eq!(on.completed_ms, [2.0]);
            assert_eq!(on.output_times(0), [1.0, 2.0, 2.0, 2.0]);
            assert_eq!(off.completed_ms, [2.0]);
            // KV stays resident through the trailing row, then is released.
            assert_eq!(on.active_after(2.0), 1);
            assert_eq!(on.active_after(3.0), 0);
            assert_eq!(off.active_after(2.0), 0);
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
                let on = run(ASYNC, &[spec(output_tokens, accept_rate)]);
                let off = run(SYNC, &[spec(output_tokens, accept_rate)]);
                assert_eq!(on.iterations, off.iterations);
                assert_eq!(on.iterations.len(), 1 + rounds);
                assert_eq!(on.completed_ms, off.completed_ms);
                assert_eq!(on.active_after((1 + rounds) as f64), 0);
            }
        }

        #[test]
        fn a_closed_loop_replacement_enters_at_n_plus_3() {
            // Request 0 keeps the engine busy; request 1 prefills in step 3
            // and finishes at N = 4 by accepting drafts; request 2 replaces it.
            let specs = [spec(64, 0.0), spec(4, 1.0), replacing(1, 2, 0.0)];
            let on = run(ASYNC, &specs);
            assert_eq!(on.iterations[2], iteration(1, PROMPT_TOKENS));
            assert_eq!(on.completed_ms[1], 4.0, "visible at the end of N");
            // The replacement arrives while N+1 runs (with the trailing row),
            // is queued at its end, and runs in N+3.
            assert_eq!(on.iterations[4], iteration(2, 0), "N+1 trailing row");
            assert_eq!(on.iterations[5], iteration(1, 0), "N+2 without it");
            assert_eq!(on.iterations[6], iteration(1, PROMPT_TOKENS), "N+3");
            assert_eq!(on.queued_ms(2), Some(5.0));
            assert_eq!(on.first_scheduled_ms(2), 5.0);
            assert_eq!(on.first_output_ms(2), 7.0);
        }

        #[test]
        fn a_closed_loop_replacement_enters_at_n_plus_2_into_an_otherwise_idle_engine() {
            // Request 0 alone finishes at N = 2 by accepting drafts. The
            // replacement arrives while N+1 runs the trailing row; the
            // decision formed at N+1's launch is empty, so the engine waits for
            // N+1, picks it up and runs it next.
            let on = run(ASYNC, &[spec(4, 1.0), replacing(0, 2, 0.0)]);
            assert_eq!(on.iterations[2], iteration(1, 0));
            assert_eq!(on.iterations[3], iteration(0, PROMPT_TOKENS));
            assert_eq!(on.launched_ms[3], 3.0);
            assert_eq!(on.queued_ms(1), Some(3.0));
            assert_eq!(on.first_scheduled_ms(1), 3.0);
            // With no trailing row the engine idles after N already and picks
            // the replacement up on arrival, as without the option.
            let on = run(ASYNC, &[spec(2, 1.0), replacing(0, 2, 0.0)]);
            let off = run(SYNC, &[spec(2, 1.0), replacing(0, 2, 0.0)]);
            assert_eq!(on.iterations, off.iterations);
            assert_eq!(on.launched_ms, off.launched_ms);
            assert_eq!(on.completed_ms, off.completed_ms);
            assert_eq!(on.queued_ms(1), None);
        }

        #[test]
        fn a_kv_wait_stays_in_server_ttft_and_freed_kv_serves_the_step_after_next() {
            // Request 0 (8 + 64) keeps the engine busy. Requests 1 and 2 (64 +
            // 4 each) are queued at the end of step 1; KV holds request 0 and
            // one of them, so request 1 runs in step 3 and request 2 waits.
            // Request 1 finishes at 6, but the decision for 7 was made at 6's
            // launch, with its KV still held: request 2 is scheduled by the
            // decision at 7's launch and runs in 8.
            const LONG_PROMPT: u32 = 64;
            let waiting = Spec {
                prompt_tokens: LONG_PROMPT,
                ..spec(4, 0.0)
            };
            let specs = [spec(64, 0.0), waiting, waiting];
            let kv_tokens = Some(144);
            let on = run(Engine { kv_tokens, ..ASYNC }, &specs);
            assert_eq!(on.iterations[2], iteration(1, LONG_PROMPT));
            assert_eq!(on.completed_ms[1], 6.0);
            assert_eq!(on.iterations[6], iteration(1, 0), "freed KV not yet seen");
            assert_eq!(on.iterations[7], iteration(1, LONG_PROMPT));
            assert_eq!(on.queued_ms(2), Some(1.0));
            assert_eq!(on.first_scheduled_ms(2), 6.0, "a 5 ms KV wait");
            assert_eq!(on.first_output_ms(2), 8.0);
            assert_eq!(on.server_ttft_ms(2), 7.0, "the KV wait counts");
            // Synchronous: requests 0 and 1 start together, and request 2 runs
            // as soon as request 1's KV is freed.
            let off = run(Engine { kv_tokens, ..SYNC }, &specs);
            assert_eq!(off.iterations[0], iteration(0, PROMPT_TOKENS + LONG_PROMPT));
            assert_eq!(off.completed_ms[1], 4.0);
            assert_eq!(off.iterations[4], iteration(1, LONG_PROMPT));
            assert_eq!(off.first_scheduled_ms(2), 4.0);
        }

        #[test]
        fn the_option_off_keeps_the_synchronous_schedule() {
            // Hand-derived synchronous schedule: request 0 decodes rounds 2..8;
            // request 1 finishes at 2 and frees its row at once; its
            // replacement joins iteration 4 and decodes once.
            let specs = [spec(8, 0.0), spec(4, 1.0), replacing(1, 2, 0.0)];
            let off = run(SYNC, &specs);
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
            assert_eq!(off.launched_ms, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
            assert_eq!(off.completed_ms, [8.0, 2.0, 5.0]);
            // Tokens are visible when their iteration ends, the queued time is
            // the arrival, and first scheduled is the decision at the end of
            // the previous iteration.
            assert_eq!(off.output_times(1), [1.0, 2.0, 2.0, 2.0]);
            assert_eq!(off.output_times(2), [4.0, 5.0]);
            assert!((0..3).all(|request| off.queued_ms(request).is_none()));
            assert_eq!(off.first_scheduled_ms(2), 3.0);
            let default_config = WorkerConfig::default();
            assert!(!default_config.async_scheduling);
            assert_eq!(default_config.async_output_processing, Time::ZERO);
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
