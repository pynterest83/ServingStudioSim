//! Speculating chunked-prefill worker recipe for a hybrid recurrent +
//! full-attention arch.
//!
//! `build_speculative_worker` on every axis but KV, the same way
//! `build_hybrid_chunked_prefill_worker` relates to the chunked-prefill
//! recipe: the store is `HybridGdnKv`, so each request's recurrent state shares
//! one capacity with the per-token attention KV. Under speculation the model's
//! per-request state already counts one state copy per draft position (vLLM's
//! Mamba `align` mode keeps `2 + num_speculative_tokens` pages per group).
//! With prefix caching on, non-final prefill chunks end on the arch's
//! checkpoint interval, as in the non-speculating hybrid recipe. A retained
//! prefix snapshot is charged the arch's `recurrent_checkpoint_bytes`, which
//! can be smaller than the state a running request pins. Under
//! `eagle_block_drop` the store also retains and resumes only vLLM's EAGLE
//! replay boundaries of each prompt (`HybridGdnKv::with_eagle_replay_boundaries`).
//! A `prefix_match_unit_tokens` finer than the checkpoint interval (vLLM's
//! `--prefix-match-unit`) adds each prompt's partial-tail boundary at that unit,
//! both as a retained resume point (`HybridGdnKv::with_prefix_match_unit`) and
//! as a chunk stop (`ChunkedPrefillAdmission::with_prefix_match_unit`).

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
use crate::worker::kv::{HybridGdnKv, PrefixCacheConfig};
use crate::worker::types::WorkerConfig;

use super::iter_batch_worker::{HybridSpeculativeWorker, IterBatchWorker};
use crate::worker::workers::iter_build_essentials::{
    full_attention_token_capacity, prepare_iter_build_essentials,
};

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_hybrid_speculative_worker<M: SpeculativeUnifiedModel>(
    id: WorkerId,
    pool_tag: &'static str,
    model: Arc<M>,
    requests: SharedRequests,
    config: WorkerConfig,
    cost_log_dir: Option<PathBuf>,
    pool: PoolId,
    gpu_name: &str,
    cluster: SharedGpuCluster,
) -> HybridSpeculativeWorker<M> {
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
    // Whole-model quantities in one token space, as in the hybrid chunked recipe.
    let state_tokens_per_request = model
        .recurrent_state_bytes_per_request()
        .div_ceil(model.total_kv_bytes_per_token().max(1));
    let checkpoint_charge_tokens = model
        .recurrent_checkpoint_bytes()
        .div_ceil(model.total_kv_bytes_per_token().max(1));
    let checkpoint_interval_tokens = config
        .ssm_checkpoint_interval_tokens
        .unwrap_or_else(|| model.recurrent_checkpoint_interval_tokens());
    let chunk_end_quantum = (checkpoint_interval_tokens > 0
        && !matches!(config.prefix_cache, PrefixCacheConfig::Disabled))
    .then_some(checkpoint_interval_tokens);
    tracing::info!(
        worker = id.0,
        pool_tag,
        draft_tokens,
        state_tokens_per_request,
        checkpoint_charge_tokens,
        checkpoint_interval_tokens,
        chunk_end_quantum,
        max_batch_tokens,
        kv_capacity_tokens = kv_capacity,
        "hybrid speculative: recurrent state shares the attention capacity"
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
    let kv_store = HybridGdnKv::new(
        num_partitions,
        essentials.kv_capacity,
        state_tokens_per_request,
        checkpoint_interval_tokens,
        checkpoint_charge_tokens,
        prefix_cache,
        essentials.sampler,
        prefix_cache_logger,
    );
    // The same vLLM constraint the chunk split below reproduces also decides
    // which prompt boundaries a completed request leaves resumable.
    let eagle_replay_boundaries = chunk_end_quantum.is_some() && config.eagle_block_drop;
    // vLLM requires the unit to divide every group's block size and the
    // indexer's 4-token key pool granularity.
    let prefix_match_unit = config.prefix_match_unit_tokens.inspect(|&unit| {
        assert!(
            eagle_replay_boundaries,
            "prefix_match_unit_tokens is modeled only under eagle_block_drop with prefix caching"
        );
        assert!(
            unit > 0 && unit.is_multiple_of(4) && checkpoint_interval_tokens.is_multiple_of(unit),
            "prefix_match_unit_tokens={unit} must be a multiple of 4 dividing the \
             recurrent checkpoint interval {checkpoint_interval_tokens}"
        );
    });
    let kv_store = match (eagle_replay_boundaries, prefix_match_unit) {
        (true, Some(unit)) => kv_store
            .with_eagle_replay_boundaries()
            .with_prefix_match_unit(unit),
        (true, None) => kv_store.with_eagle_replay_boundaries(),
        (false, _) => kv_store,
    };
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
    let admission = match chunk_end_quantum {
        Some(quantum) if config.eagle_block_drop => {
            let admission = admission
                .with_chunk_end_quantum(quantum)
                .with_eagle_block_drop();
            match prefix_match_unit {
                Some(unit) => admission.with_prefix_match_unit(unit),
                None => admission,
            }
        }
        Some(quantum) => admission.with_chunk_end_quantum(quantum),
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
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::*;

    use crate::arch::contract::SpeculativeArchInput;
    use crate::common::{AcceptanceProfile, DecodingStrategy, RequestId, SessionInput, Time};
    use crate::test_helpers::{lm, shared_with, test_cluster};
    use crate::timing::LeafMetrics;
    use crate::worker::config::KvAdmissionConfig;
    use crate::worker::iter_worker::IterWorker;
    use crate::worker::kv::PrefixCachePolicy;
    use crate::worker::types::WorkerMsgCommon;

    const DRAFT_TOKENS: u32 = 5;
    const PROMPT_TOKENS: u32 = 8;
    const OUTPUT_TOKENS: u32 = 12;
    const KV_BYTES_PER_TOKEN: u64 = 1024;
    /// 100 tokens' worth of pinned recurrent state per request.
    const STATE_TOKENS: u64 = 100;

    /// The hybrid model's checkpoint interval, in tokens.
    const CHECKPOINT_INTERVAL: u32 = 16;

    /// 1 ms per iteration; records the most requests one iteration carried.
    struct HybridModel {
        state_tokens: u64,
        /// A retained checkpoint's size when it differs from the state.
        checkpoint_tokens: Option<u64>,
        most_requests: AtomicU32,
    }

    impl SpeculativeUnifiedModel for HybridModel {
        fn eval_speculative_iter(
            &self,
            batch: &SpeculativeArchInput,
            slots: &mut Vec<LeafMetrics>,
            _scratch: &mut Vec<LeafMetrics>,
        ) -> LeafMetrics {
            slots.clear();
            assert_eq!(batch.draft_tokens, DRAFT_TOKENS);
            for group in &batch.groups {
                self.most_requests
                    .fetch_max(group.request_count(), Ordering::Relaxed);
            }
            lm(1.0)
        }

        fn total_kv_bytes_per_token(&self) -> u64 {
            KV_BYTES_PER_TOKEN
        }

        fn max_model_len(&self) -> u32 {
            8192
        }

        fn gpus_per_replica(&self) -> u16 {
            1
        }

        fn recurrent_state_bytes_per_request(&self) -> u64 {
            self.state_tokens * KV_BYTES_PER_TOKEN
        }

        fn recurrent_checkpoint_interval_tokens(&self) -> u32 {
            CHECKPOINT_INTERVAL
        }

        fn recurrent_checkpoint_bytes(&self) -> u64 {
            self.checkpoint_tokens.unwrap_or(self.state_tokens) * KV_BYTES_PER_TOKEN
        }
    }

    /// Runs `count` fully-accepting requests to completion on a 260-token pool
    /// and returns the most requests any iteration carried.
    fn most_concurrent_requests(state_tokens: u64, count: u32) -> u32 {
        let rows: Vec<_> = (0..count)
            .map(|id| (id, PROMPT_TOKENS, OUTPUT_TOKENS))
            .collect();
        let requests = shared_with(&rows);
        for id in 0..count {
            requests.borrow_mut()[RequestId(id)]
                .request
                .definition
                .decoding = DecodingStrategy::Speculative {
                accept_rate: AcceptanceProfile::Uniform(1.0),
            };
        }
        let model = Arc::new(HybridModel {
            state_tokens,
            checkpoint_tokens: None,
            most_requests: AtomicU32::new(0),
        });
        let config = WorkerConfig {
            attn_kv_bytes: 260 * KV_BYTES_PER_TOKEN,
            max_batch_tokens: Some(2048),
            kv_admission: KvAdmissionConfig::FullFootprint,
            speculative_draft_tokens: DRAFT_TOKENS,
            speculative_acceptance_seed: Some(7),
            ..WorkerConfig::default()
        };
        let mut worker = build_hybrid_speculative_worker(
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
        for id in 0..count {
            worker.enqueue(WorkerMsgCommon::Request(RequestId(id)));
        }
        for tick in 0..400 {
            worker.tick(Time::from_ms(tick as f64), &mut Vec::new());
        }
        let requests = requests.borrow();
        for id in 0..count {
            let record = &requests[RequestId(id)];
            assert!(record.lifecycle.completed, "request {id}");
            assert_eq!(record.progress.output_tokens_emitted, OUTPUT_TOKENS);
        }
        model.most_requests.load(Ordering::Relaxed)
    }

    #[test]
    fn the_recipe_composes_a_worker_the_shell_accepts() {
        fn assert_iter_worker<W: IterWorker>() {}
        assert_iter_worker::<HybridSpeculativeWorker<HybridModel>>();
    }

    #[test]
    fn pinned_recurrent_state_limits_concurrency() {
        // The defect this catches: costing a speculating hybrid on the
        // full-attention store. Each request needs 20 KV tokens plus 100 of
        // pinned state, so a 260-token pool holds two at once; ignoring the
        // state would admit all six.
        assert_eq!(most_concurrent_requests(STATE_TOKENS, 6), 2);
        assert_eq!(most_concurrent_requests(0, 6), 6);
    }

    /// Runs a two-turn session through the recipe and returns the second
    /// turn's `(prefix-cache hit, prefill tokens processed)`. Turn one sends a
    /// 40-token prompt and decodes 12; turn two declares those 40 tokens as its
    /// prefix and adds 10 fresh ones.
    fn second_turn(
        eagle_block_drop: bool,
        checkpoint_tokens: Option<u64>,
        max_retained_tokens: Option<u64>,
        prefix_match_unit_tokens: Option<u32>,
    ) -> (u32, u32) {
        let requests = shared_with(&[(0, 40, OUTPUT_TOKENS), (1, 10, OUTPUT_TOKENS)]);
        {
            let mut requests = requests.borrow_mut();
            for (id, declared_prefix_tokens) in [(0, 0), (1, 40)] {
                let definition = &mut requests[RequestId(id)].request.definition;
                definition.decoding = DecodingStrategy::Speculative {
                    accept_rate: AcceptanceProfile::Uniform(1.0),
                };
                definition.session = SessionInput::Session {
                    session_id: 7,
                    session_start_time: Time::ZERO,
                    declared_prefix_tokens,
                };
            }
        }
        let model = Arc::new(HybridModel {
            state_tokens: STATE_TOKENS,
            checkpoint_tokens,
            most_requests: AtomicU32::new(0),
        });
        let config = WorkerConfig {
            attn_kv_bytes: 100_000 * KV_BYTES_PER_TOKEN,
            max_batch_tokens: Some(2048),
            kv_admission: KvAdmissionConfig::FullFootprint,
            speculative_draft_tokens: DRAFT_TOKENS,
            speculative_acceptance_seed: Some(7),
            eagle_block_drop,
            prefix_match_unit_tokens,
            prefix_cache: PrefixCacheConfig::Opportunistic {
                policy: PrefixCachePolicy::Lru,
                max_retained_bytes: max_retained_tokens.map(|tokens| tokens * KV_BYTES_PER_TOKEN),
            },
            ..WorkerConfig::default()
        };
        let mut worker = build_hybrid_speculative_worker(
            WorkerId(0),
            "test",
            model,
            requests.clone(),
            config,
            None,
            PoolId(0),
            "test-gpu",
            test_cluster(),
        );
        for (id, ticks) in [(0u32, 0..200), (1, 200..400)] {
            worker.enqueue(WorkerMsgCommon::Request(RequestId(id)));
            for tick in ticks {
                worker.tick(Time::from_ms(tick as f64), &mut Vec::new());
            }
            assert!(
                requests.borrow()[RequestId(id)].lifecycle.completed,
                "turn {id}"
            );
        }
        let requests = requests.borrow();
        let second = &requests[RequestId(1)];
        (
            second.telemetry.prefix_cache_hit_tokens.expect("resolved"),
            second.progress.prefill_tokens_processed,
        )
    }

    #[test]
    fn eagle_block_drop_resumes_one_block_below_the_previous_prompt() {
        // P = 40 with 16-token blocks: the replay boundaries are both
        // floor(40 / 16) * 16 - 16 = 16. Turn two reaches all 40 declared
        // tokens, drops one block from the 32 it matches, and resumes at 16.
        assert_eq!(second_turn(true, None, None, None), (16, 50 - 16));
        // Without the drop the store resumes from the decoded context, capped
        // at the declared prefix's last whole block.
        assert_eq!(second_turn(false, None, None, None), (32, 50 - 32));
    }

    #[test]
    fn a_retained_boundary_is_charged_the_arch_checkpoint_not_the_running_state() {
        // The entry keeps 32 attention tokens plus one checkpoint. A 60-token
        // retained-prefix ceiling holds it at a 10-token checkpoint, but not
        // at the 100-token state a running request pins.
        assert_eq!(second_turn(true, Some(10), Some(60), None), (16, 50 - 16));
        assert_eq!(second_turn(true, None, Some(60), None), (0, 50));
    }

    #[test]
    fn a_finer_prefix_match_unit_resumes_one_unit_below_the_previous_prompt() {
        // P = 40, 16-token blocks, a 4-token unit: turn two reaches 40 tokens,
        // drops one unit, and resumes at floor(40 / 4) * 4 - 4 = 36 rather
        // than the block boundary 16.
        assert_eq!(second_turn(true, None, None, Some(4)), (36, 50 - 36));
        // A unit equal to the block is the default rule.
        assert_eq!(
            second_turn(true, None, None, Some(CHECKPOINT_INTERVAL)),
            (16, 50 - 16)
        );
    }

    #[test]
    #[should_panic(expected = "must be a multiple of 4 dividing")]
    fn a_prefix_match_unit_must_divide_the_checkpoint_interval() {
        second_turn(true, None, None, Some(12));
    }
}
