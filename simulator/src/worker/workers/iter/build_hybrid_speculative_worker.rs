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
//! checkpoint interval, as in the non-speculating hybrid recipe.

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
    let admission = match chunk_end_quantum {
        Some(quantum) => admission.with_chunk_end_quantum(quantum),
        None => admission,
    };
    IterBatchWorker::from_components(
        essentials.context,
        kv_store,
        admission,
        SpeculativeIterExecution::new(model, essentials.cost, draft_tokens),
    )
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::*;

    use crate::arch::contract::SpeculativeArchInput;
    use crate::common::{AcceptanceProfile, DecodingStrategy, RequestId, Time};
    use crate::test_helpers::{lm, shared_with, test_cluster};
    use crate::timing::LeafMetrics;
    use crate::worker::config::KvAdmissionConfig;
    use crate::worker::iter_worker::IterWorker;
    use crate::worker::types::WorkerMsgCommon;

    const DRAFT_TOKENS: u32 = 5;
    const PROMPT_TOKENS: u32 = 8;
    const OUTPUT_TOKENS: u32 = 12;
    const KV_BYTES_PER_TOKEN: u64 = 1024;
    /// 100 tokens' worth of pinned recurrent state per request.
    const STATE_TOKENS: u64 = 100;

    /// 1 ms per iteration; records the most requests one iteration carried.
    struct HybridModel {
        state_tokens: u64,
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
            16
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
}
