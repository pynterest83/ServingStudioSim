//! `unified` deployment — one worker type runs the whole model per iteration
//! (co-located attention + FFN), in one homogeneous `main` pool.
//!
//! `build` reads the structured `UnifiedConfig`: a single pool (`main`) with one
//! homogeneous group. The arch is selected by its explicit tag (NO `tp_size`
//! dispatch; see the L4/L6 design). Wired arms are
//! `llama3_dense` + `barebone`, `llama3_dense_tp` + `barebone`,
//! `llama3_dp_attn_tp_ffn` + `hp_unified`, and `qwen3_moe_dp_attn_ep_ffn` +
//! `hp_unified`, `glm52_vllm_dsa_moe` + `hp_unified`, and
//! `glm52_vllm_nvfp4_dsa_moe` + either `hp_unified` or `chunked_prefill`,
//! `glm52_vllm_nvfp4_dsa_moe_speculative` + `speculative`, and
//! `glm52_sglang_nvfp4_tp_dsa_moe` + `chunked_prefill`, and `deepseek_v4_vllm`
//! / `deepseek_v41_vllm` + either `hp_unified` or `chunked_prefill`. GLM's TP1 local
//! attention uses one independent KV/input partition per EP rank; the worker
//! shells and mutable request/KV lifecycle are unchanged. Each wired arm
//! monomorphizes its concrete model/worker pair and
//! erases to `Box<dyn Flow>` — the single `dyn` point (the cost path is
//! `dyn`-free, L4 §4.1).
//!
//! The speculative pair is the one arm whose model does not implement
//! `IterwiseUnifiedModel` at all — it implements `SpeculativeUnifiedModel`
//! instead — which is why neither `assemble_flow` nor `UnifiedWorkerFactory`
//! names an L4 contract on `M`. The recipe each arm passes already does.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{bail, ensure};

use crate::arch::build as arch_build;
use crate::arch::contract::IterwiseUnifiedModel;
use crate::arch::contract::SpeculativeUnifiedModel;
use crate::arch::{Glm52MtpMode, IterArchSel, KernelTimeScaleRule, WhatIfKernelTimeScaled};
use crate::common::{PoolId, SharedRequests};
use crate::deployment::UnifiedConfig;
use crate::orchestrator::common::WorkerBuildFn;
use crate::orchestrator::{
    DpPlacementPolicy, Flow, PlacementPolicy, SimpleDpConfig, SimpleDpFlow, SimpleDpPoolConfig,
    UnifiedWorkerFactory,
};
use crate::timing::PerfApiBridge;
use crate::worker::config::RecurrentCheckpointRelease;
use crate::worker::{
    build_barebone_worker, build_chunked_prefill_worker, build_hp_worker,
    build_hybrid_chunked_prefill_worker, build_hybrid_speculative_worker,
    build_qwen36_hybrid_worker, build_speculative_worker, resolve_prefix_cache_config, BatchPolicy,
    IterWorker, IterWorkerSel, KvAdmissionConfig, PendingOrderKind, PrefixCacheMode,
    PrefixCachePolicy, WorkerConfig,
};

use super::Deployment;

pub struct UnifiedDeployment;

impl Deployment for UnifiedDeployment {
    const NAME: &'static str = "unified";
    type Config = UnifiedConfig;

    fn build(
        cfg: &UnifiedConfig,
        bridge: &PerfApiBridge,
        store: SharedRequests,
    ) -> anyhow::Result<Box<dyn Flow>> {
        // Single homogeneous group only (heterogeneous `groups` is parse-only).
        let pool = &cfg.pools.main;
        ensure!(
            pool.groups.len() == 1,
            "unified: only a single homogeneous group is supported (got {})",
            pool.groups.len()
        );
        let g = &pool.groups[0];

        // Model dims load from JSON (arch-independent, §4); `sim_num_layers` /
        // `num_layers` truncation must apply BEFORE build_configs. Loaded per-arm
        // because dense and MoE archs deserialize different field sets — dense
        // archs read `ModelCfg`, the MoE arch reads `MoeModelCfg`.
        let model_spec = g.arch.model();

        // L5 worker env: `attn_kv_bytes` sizes the worker's KV partitions, and
        // `log_output_token_times` controls request_slo detail logging. The
        // worker *type* is matched against the arch in the arms below.
        let (
            attn_gpu_memory_gb,
            gpu_time_multiplier,
            max_batch_tokens,
            pending_order,
            prefix_cache_mode,
            prefix_cache_policy,
            prefix_cache_max_gpu_memory_gb,
            batch_policy,
            kv_admission,
        ) = match &g.worker {
            IterWorkerSel::Barebone {
                attn_gpu_memory_gb,
                gpu_time_multiplier,
                max_batch_tokens,
                pending_order,
                prefix_cache_mode,
                prefix_cache_policy,
                prefix_cache_max_gpu_memory_gb,
                // Read by `ssm_checkpoint_interval_tokens` and
                // `prefill_gpu_time_multiplier` below, so the two arms keep the
                // same bindings.
                ..
            }
            | IterWorkerSel::HpUnified {
                attn_gpu_memory_gb,
                gpu_time_multiplier,
                max_batch_tokens,
                pending_order,
                prefix_cache_mode,
                prefix_cache_policy,
                prefix_cache_max_gpu_memory_gb,
                ..
            } => (
                *attn_gpu_memory_gb,
                *gpu_time_multiplier,
                *max_batch_tokens,
                *pending_order,
                *prefix_cache_mode,
                *prefix_cache_policy,
                *prefix_cache_max_gpu_memory_gb,
                BatchPolicy::Mix,
                KvAdmissionConfig::FullFootprint,
            ),
            IterWorkerSel::ChunkedPrefill {
                attn_gpu_memory_gb,
                max_batch_tokens,
                batch_policy,
                kv_admission,
                gpu_time_multiplier,
                // Read by `prefill_gpu_time_multiplier` below.
                ..
            } => (
                *attn_gpu_memory_gb,
                *gpu_time_multiplier,
                Some(*max_batch_tokens),
                PendingOrderKind::Fifo,
                PrefixCacheMode::Opportunistic,
                PrefixCachePolicy::Lru,
                None,
                *batch_policy,
                kv_admission.resolve()?,
            ),
            IterWorkerSel::Speculative {
                attn_gpu_memory_gb,
                max_batch_tokens,
                batch_policy,
                kv_admission,
                gpu_time_multiplier,
                // Read by the two helpers below so this arm keeps the same
                // bindings as its siblings. See `ssm_checkpoint_interval_tokens`.
                ..
            } => (
                *attn_gpu_memory_gb,
                *gpu_time_multiplier,
                Some(*max_batch_tokens),
                PendingOrderKind::Fifo,
                PrefixCacheMode::Opportunistic,
                PrefixCachePolicy::Lru,
                None,
                *batch_policy,
                kv_admission.resolve()?,
            ),
            IterWorkerSel::PdPrefill { .. } | IterWorkerSel::PdDecode { .. } => {
                bail!("unified: pd_prefill / pd_decode workers belong to the `pd` deployment")
            }
        };
        let prefix_cache = resolve_prefix_cache_config(
            "unified",
            prefix_cache_mode,
            prefix_cache_policy,
            prefix_cache_max_gpu_memory_gb,
            attn_gpu_memory_gb,
        )?;
        let worker_config = WorkerConfig {
            attn_kv_bytes: (attn_gpu_memory_gb * 1e9) as u64,
            log_output_token_times: cfg.io.log_output_token_times,
            log_stage_transitions: cfg.io.log_stage_transitions,
            kv_log_stride: cfg.io.kv_log_stride,
            gpu_time_multiplier,
            prefill_gpu_time_multiplier: prefill_gpu_time_multiplier(&g.worker)?,
            host_time: host_time_model(&g.worker)?,
            max_batch_tokens,
            pending_order,
            batch_policy,
            kv_admission,
            prefix_cache,
            ssm_checkpoint_interval_tokens: ssm_checkpoint_interval_tokens(&g.worker),
            speculative_draft_tokens: speculative_draft_tokens(&g.worker),
            speculative_acceptance_seed: speculative_acceptance_seed(&g.worker),
            eagle_block_drop: eagle_block_drop(&g.worker),
            prefix_match_unit_tokens: prefix_match_unit_tokens(&g.worker)?,
            async_scheduling: async_scheduling(&g.worker),
            async_output_processing: async_output_processing(&g.worker)?,
            max_running_requests: max_running_requests(&g.worker),
            checkpoint_release: checkpoint_release(&g.worker),
            kv_offload: kv_offload(&g.worker)?,
            ..WorkerConfig::default()
        };

        let dp_cfg = SimpleDpConfig {
            dp_pool: SimpleDpPoolConfig {
                pool: PoolId(0),
                num_workers: g.replicas,
                placement: placement_into(pool.placement),
            },
        };
        let log_dir: Option<PathBuf> = Some(cfg.io.log_dir.clone());
        let gpu_name = g.gpu.clone();

        // Scope the single pool `main` over the whole arch match: one call
        // activates its backend overrides (run) and tags it for the enumerate walk
        // (emit). Every arm's `arch_build::*` threads the same `bridge`; the guard
        // restores both on drop (end of `build`, or an early `?`).
        let _scope = bridge.with_backend_overrides("main", cfg.backends.get("main"));

        // Arch selected by explicit tag; each arm builds its concrete model via
        // the shared `arch::build` builders (the same ones the `pd` deployment and
        // the offline `timing-predict` path use), validates the paired worker
        // tag, and erases via assemble_flow. dense / dense_tp run on the
        // single-group barebone worker; the DP-attn / MoE archs run on the
        // multi-group hp_unified worker (one KV partition state per DP shard).
        match &g.arch {
            // Hybrid KV: 30 GDN layers of per-request recurrent state and 10
            // GQA layers of per-token KV live in one attention budget, so only
            // the KV axis differs from the arms below.
            IterArchSel::Qwen36Local {
                routing,
                routing_seed,
                expert_popularity_file,
                ..
            } => {
                ensure_hybrid_worker("Qwen3.6 local", &g.worker)?;
                let model = Arc::new(arch_build::qwen36_local(
                    model_spec,
                    *routing,
                    *routing_seed,
                    expert_popularity_file.as_deref(),
                    &gpu_name,
                    MODEL_NAME,
                    bridge,
                )?);
                assemble_hybrid_flow(
                    "Qwen3.6 local",
                    model,
                    store,
                    worker_config,
                    log_dir,
                    gpu_name,
                    dp_cfg,
                    &g.worker,
                )
            }
            // Hybrid KV like Qwen3.6: 34 KDA layers of per-request recurrent
            // state (fp32 SSM + conv window) and 11 DSA layers of per-token MLA
            // latent + kpool index cache share one attention budget. Only the
            // hybrid recipes charge both; the full-attention chunked and
            // hp_unified recipes would leave the KDA state invisible.
            IterArchSel::Glm53FlashVllmFp8KdaDsaMoe {
                tp_size,
                max_model_len,
                routing,
                routing_seed,
                expert_popularity_file,
                token_corpus_file,
                cudagraph_capture_sizes,
                ..
            } => {
                ensure_hybrid_worker("GLM-5.3-Flash vLLM FP8", &g.worker)?;
                let model = Arc::new(arch_build::glm53_flash_vllm_fp8_kda_dsa_moe(
                    model_spec,
                    *tp_size,
                    *max_model_len,
                    *routing,
                    *routing_seed,
                    expert_popularity_file.as_deref(),
                    token_corpus_file.as_deref(),
                    cudagraph_capture_sizes,
                    &gpu_name,
                    MODEL_NAME,
                    bridge,
                )?);
                assemble_hybrid_flow(
                    "GLM-5.3-Flash vLLM FP8",
                    model,
                    store,
                    worker_config,
                    log_dir,
                    gpu_name,
                    dp_cfg,
                    &g.worker,
                )
            }
            // Hybrid KV as for the B200 arch above, under speculation: the
            // per-request KDA state (one page per draft position per group)
            // shares the attention budget with the MLA latent, so the
            // speculative lifecycle runs over `HybridGdnKv`.
            IterArchSel::Glm53FlashVllmFp8KdaDsaMoeSpeculative {
                tp_size,
                max_model_len,
                routing,
                routing_seed,
                draft_tokens,
                expert_popularity_file,
                token_corpus_file,
                cudagraph_capture_sizes,
                sparse_mla_backend,
                moe_backend,
                ..
            } => {
                ensure_speculative(&g.worker, *draft_tokens)?;
                let model = arch_build::glm53_flash_vllm_fp8_kda_dsa_moe_speculative_model(
                    model_spec,
                    *tp_size,
                    *max_model_len,
                    *routing,
                    *routing_seed,
                    expert_popularity_file.as_deref(),
                    token_corpus_file.as_deref(),
                    cudagraph_capture_sizes,
                    *draft_tokens,
                    *sparse_mla_backend,
                    *moe_backend,
                    &gpu_name,
                    MODEL_NAME,
                    bridge,
                )?;
                assemble_speculative_flow(
                    model,
                    what_if_kernel_time_scale(&g.worker),
                    store,
                    worker_config,
                    log_dir,
                    gpu_name,
                    dp_cfg,
                    build_hybrid_speculative_worker,
                    build_hybrid_speculative_worker,
                )
            }
            IterArchSel::Llama3Dense { .. } => {
                ensure_barebone(&g.worker)?;
                let model = Arc::new(arch_build::dense(
                    model_spec, &gpu_name, MODEL_NAME, bridge,
                )?);
                Ok(assemble_flow(
                    model,
                    store,
                    worker_config,
                    log_dir,
                    gpu_name,
                    dp_cfg,
                    build_barebone_worker,
                ))
            }
            IterArchSel::Llama3DenseTp { tp_size, .. } => {
                ensure_barebone(&g.worker)?;
                let model = Arc::new(arch_build::dense_tp(
                    model_spec, *tp_size, &gpu_name, MODEL_NAME, bridge,
                )?);
                Ok(assemble_flow(
                    model,
                    store,
                    worker_config,
                    log_dir,
                    gpu_name,
                    dp_cfg,
                    build_barebone_worker,
                ))
            }
            IterArchSel::Llama3DpAttnTpFfn {
                attn_tp_size,
                ffn_tp_size,
                ..
            } => {
                ensure_hp_unified(&g.worker)?;
                let model = Arc::new(arch_build::dp_attn_tp_ffn(
                    model_spec,
                    *attn_tp_size,
                    *ffn_tp_size,
                    &gpu_name,
                    MODEL_NAME,
                    bridge,
                )?);
                Ok(assemble_flow(
                    model,
                    store,
                    worker_config,
                    log_dir,
                    gpu_name,
                    dp_cfg,
                    build_hp_worker,
                ))
            }
            IterArchSel::Qwen3MoeDpAttnEpFfn {
                attn_tp_size,
                ep_size,
                hp_size,
                nvl_num_gpu,
                routing,
                routing_seed,
                expert_popularity_file,
                ..
            } => {
                ensure_hp_unified(&g.worker)?;
                let model = Arc::new(arch_build::qwen3_moe(
                    model_spec,
                    *attn_tp_size,
                    *ep_size,
                    *hp_size,
                    *nvl_num_gpu,
                    *routing,
                    *routing_seed,
                    expert_popularity_file.as_deref(),
                    &gpu_name,
                    MODEL_NAME,
                    bridge,
                )?);
                Ok(assemble_flow(
                    model,
                    store,
                    worker_config,
                    log_dir,
                    gpu_name,
                    dp_cfg,
                    build_hp_worker,
                ))
            }
            IterArchSel::Qwen3MoeFp8DpAttnEpFfn {
                attn_tp_size,
                ep_size,
                hp_size,
                nvl_num_gpu,
                routing,
                routing_seed,
                expert_popularity_file,
                ..
            } => {
                ensure_hp_unified(&g.worker)?;
                let model = Arc::new(arch_build::qwen3_moe_fp8(
                    model_spec,
                    *attn_tp_size,
                    *ep_size,
                    *hp_size,
                    *nvl_num_gpu,
                    *routing,
                    *routing_seed,
                    expert_popularity_file.as_deref(),
                    &gpu_name,
                    MODEL_NAME,
                    bridge,
                )?);
                Ok(assemble_flow(
                    model,
                    store,
                    worker_config,
                    log_dir,
                    gpu_name,
                    dp_cfg,
                    build_hp_worker,
                ))
            }
            IterArchSel::Qwen3VllmMoeDpAttnEpFfn {
                attn_tp_size,
                ep_size,
                hp_size,
                nvl_num_gpu,
                routing,
                routing_seed,
                expert_popularity_file,
                ..
            } => {
                ensure_hp_unified(&g.worker)?;
                let model = Arc::new(arch_build::qwen3_vllm_moe(
                    model_spec,
                    *attn_tp_size,
                    *ep_size,
                    *hp_size,
                    *nvl_num_gpu,
                    *routing,
                    *routing_seed,
                    expert_popularity_file.as_deref(),
                    &gpu_name,
                    MODEL_NAME,
                    bridge,
                )?);
                Ok(assemble_flow(
                    model,
                    store,
                    worker_config,
                    log_dir,
                    gpu_name,
                    dp_cfg,
                    build_hp_worker,
                ))
            }
            arch @ (IterArchSel::DeepseekV4Vllm {
                routing,
                routing_seed,
                expert_popularity_file,
                ..
            }
            | IterArchSel::DeepseekV4VllmSerialStreams {
                routing,
                routing_seed,
                expert_popularity_file,
                ..
            }) => {
                ensure_hp_or_chunked_worker("DeepSeek-V4", &g.worker)?;
                let serialize_streams =
                    matches!(arch, IterArchSel::DeepseekV4VllmSerialStreams { .. });
                let model = Arc::new(arch_build::deepseek_v4_vllm(
                    model_spec,
                    *routing,
                    *routing_seed,
                    expert_popularity_file.as_deref(),
                    serialize_streams,
                    chunk_tokens(&g.worker),
                    &gpu_name,
                    MODEL_NAME,
                    bridge,
                )?);
                assemble_hp_or_chunked_flow(
                    "DeepSeek-V4",
                    model,
                    store,
                    worker_config,
                    log_dir,
                    gpu_name,
                    dp_cfg,
                    &g.worker,
                )
            }
            arch @ (IterArchSel::DeepseekV41Vllm {
                routing,
                routing_seed,
                expert_popularity_file,
                token_corpus_file,
                decoder_swa_bounded_replay,
                max_model_len,
                ..
            }
            | IterArchSel::DeepseekV41VllmSerialStreams {
                routing,
                routing_seed,
                expert_popularity_file,
                token_corpus_file,
                decoder_swa_bounded_replay,
                max_model_len,
                ..
            }) => {
                // Pure TP4 attention (one attention group, four replicated KV
                // shards) like the GLM-5.2 SGLang TP4 arch. The worker's
                // `FullAttnKv` sizes on the arch's per-token compressed + index
                // pages; the sliding-window cache is a per-request constant the
                // arch leaves out (see `total_kv_bytes_per_token`).
                ensure_hp_or_chunked_worker("DeepSeek-V4.1", &g.worker)?;
                let serialize_streams =
                    matches!(arch, IterArchSel::DeepseekV41VllmSerialStreams { .. });
                let model = Arc::new(arch_build::deepseek_v41_vllm(
                    model_spec,
                    *routing,
                    *routing_seed,
                    expert_popularity_file.as_deref(),
                    token_corpus_file.as_deref(),
                    serialize_streams,
                    *decoder_swa_bounded_replay,
                    *max_model_len,
                    &gpu_name,
                    MODEL_NAME,
                    bridge,
                )?);
                assemble_hp_or_chunked_flow(
                    "DeepSeek-V4.1",
                    model,
                    store,
                    worker_config,
                    log_dir,
                    gpu_name,
                    dp_cfg,
                    &g.worker,
                )
            }
            IterArchSel::Glm52VllmDsaMoe {
                ep_size,
                nvl_num_gpu,
                routing,
                routing_seed,
                mtp_mode,
                expert_popularity_file,
                ..
            } => {
                ensure_hp_unified(&g.worker)?;
                let model = Arc::new(arch_build::glm52_vllm_dsa_moe(
                    model_spec,
                    *ep_size,
                    *nvl_num_gpu,
                    *routing,
                    *routing_seed,
                    *mtp_mode,
                    expert_popularity_file.as_deref(),
                    &gpu_name,
                    MODEL_NAME,
                    bridge,
                )?);
                Ok(assemble_flow(
                    model,
                    store,
                    worker_config,
                    log_dir,
                    gpu_name,
                    dp_cfg,
                    build_hp_worker,
                ))
            }
            IterArchSel::Glm52VllmNvfp4DsaMoe {
                ep_size,
                nvl_num_gpu,
                max_model_len,
                routing,
                routing_seed,
                mtp_mode,
                expert_popularity_file,
                token_corpus_file,
                ..
            } => {
                ensure_hp_or_chunked_worker("GLM-4.5 NVFP4", &g.worker)?;
                let model = Arc::new(arch_build::glm52_vllm_nvfp4_dsa_moe(
                    model_spec,
                    *ep_size,
                    *nvl_num_gpu,
                    *max_model_len,
                    *routing,
                    *routing_seed,
                    *mtp_mode,
                    expert_popularity_file.as_deref(),
                    token_corpus_file.as_deref(),
                    &gpu_name,
                    MODEL_NAME,
                    bridge,
                )?);
                assemble_hp_or_chunked_flow(
                    "GLM-4.5 NVFP4",
                    model,
                    store,
                    worker_config,
                    log_dir,
                    gpu_name,
                    dp_cfg,
                    &g.worker,
                )
            }
            IterArchSel::Glm52VllmNvfp4DsaMoeSpeculative {
                ep_size,
                nvl_num_gpu,
                max_model_len,
                routing,
                routing_seed,
                mtp_mode,
                draft_tokens,
                expert_popularity_file,
                token_corpus_file,
                ..
            } => {
                ensure_speculative(&g.worker, *draft_tokens)?;
                ensure!(
                    *mtp_mode != Glm52MtpMode::Off,
                    "unified: a speculative GLM must run its MTP layer; mtp_mode=off \
                     leaves it with nothing to draft with"
                );
                let model = arch_build::glm52_vllm_nvfp4_dsa_moe_speculative(
                    model_spec,
                    *ep_size,
                    *nvl_num_gpu,
                    *max_model_len,
                    *routing,
                    *routing_seed,
                    *mtp_mode,
                    expert_popularity_file.as_deref(),
                    token_corpus_file.as_deref(),
                    *draft_tokens,
                    &gpu_name,
                    MODEL_NAME,
                    bridge,
                )?;
                assemble_speculative_flow(
                    model,
                    what_if_kernel_time_scale(&g.worker),
                    store,
                    worker_config,
                    log_dir,
                    gpu_name,
                    dp_cfg,
                    build_speculative_worker,
                    build_speculative_worker,
                )
            }
            IterArchSel::Glm53VllmNvfp4DsaMoeDflash2 {
                ep_size,
                nvl_num_gpu,
                max_model_len,
                routing,
                routing_seed,
                draft_tokens,
                draft_sliding_window,
                expert_popularity_file,
                token_corpus_file,
                ..
            } => {
                ensure_speculative(&g.worker, *draft_tokens)?;
                let model = arch_build::glm53_vllm_nvfp4_dsa_moe_dflash2(
                    model_spec,
                    *ep_size,
                    *nvl_num_gpu,
                    *max_model_len,
                    *routing,
                    *routing_seed,
                    expert_popularity_file.as_deref(),
                    token_corpus_file.as_deref(),
                    *draft_tokens,
                    *draft_sliding_window,
                    &gpu_name,
                    MODEL_NAME,
                    bridge,
                )?;
                assemble_speculative_flow(
                    model,
                    what_if_kernel_time_scale(&g.worker),
                    store,
                    worker_config,
                    log_dir,
                    gpu_name,
                    dp_cfg,
                    build_speculative_worker,
                    build_speculative_worker,
                )
            }
            IterArchSel::Glm52SglangNvfp4TpDsaMoe {
                tp_size,
                max_model_len,
                routing,
                routing_seed,
                mtp_mode,
                expert_popularity_file,
                token_corpus_file,
                ..
            } => {
                ensure_hp_or_chunked_worker("GLM-5.2 SGLang NVFP4 pure TP", &g.worker)?;
                let model = Arc::new(arch_build::glm52_sglang_nvfp4_tp_dsa_moe(
                    model_spec,
                    *tp_size,
                    *max_model_len,
                    *routing,
                    *routing_seed,
                    *mtp_mode,
                    expert_popularity_file.as_deref(),
                    token_corpus_file.as_deref(),
                    &gpu_name,
                    MODEL_NAME,
                    bridge,
                )?);
                assemble_hp_or_chunked_flow(
                    "GLM-5.2 SGLang NVFP4 pure TP",
                    model,
                    store,
                    worker_config,
                    log_dir,
                    gpu_name,
                    dp_cfg,
                    &g.worker,
                )
            }
        }
    }
}

/// The model's dotted-leaf prefix for this deployment (e.g. `unified.embedding`).
const MODEL_NAME: &str = "unified";

/// The dense / dense_tp archs run on the single-group barebone worker.
fn ensure_barebone(worker: &IterWorkerSel) -> anyhow::Result<()> {
    match worker {
        IterWorkerSel::Barebone { .. } => Ok(()),
        other => bail!("unified: this arch requires worker `barebone`, got {other:?}"),
    }
}

/// Hybrid-KV archs run on a hybrid-store recipe under either lifecycle:
/// barebone admission or hard-capped chunked prefill.
fn ensure_hybrid_worker(arch_name: &str, worker: &IterWorkerSel) -> anyhow::Result<()> {
    match worker {
        IterWorkerSel::Barebone { .. } | IterWorkerSel::ChunkedPrefill { .. } => Ok(()),
        other => bail!(
            "unified: {arch_name} requires worker `barebone` or `chunked_prefill`, got {other:?}"
        ),
    }
}

fn assemble_hybrid_flow<M>(
    arch_name: &str,
    model: Arc<M>,
    store: SharedRequests,
    worker_config: WorkerConfig,
    log_dir: Option<PathBuf>,
    gpu_name: String,
    dp_cfg: SimpleDpConfig,
    worker: &IterWorkerSel,
) -> anyhow::Result<Box<dyn Flow>>
where
    M: IterwiseUnifiedModel,
{
    match worker {
        IterWorkerSel::Barebone { .. } => Ok(assemble_flow(
            model,
            store,
            worker_config,
            log_dir,
            gpu_name,
            dp_cfg,
            build_qwen36_hybrid_worker,
        )),
        IterWorkerSel::ChunkedPrefill { .. } => Ok(assemble_flow(
            model,
            store,
            worker_config,
            log_dir,
            gpu_name,
            dp_cfg,
            build_hybrid_chunked_prefill_worker,
        )),
        other => bail!("unified: unsupported {arch_name} worker {other:?}"),
    }
}

/// The hybrid KV recipe's optional snapshot-interval override. Only `barebone`
/// carries it; every other selector leaves the arch's own alignment in force.
fn ssm_checkpoint_interval_tokens(worker: &IterWorkerSel) -> Option<u32> {
    match worker {
        IterWorkerSel::Barebone {
            ssm_checkpoint_interval_tokens,
            ..
        } => *ssm_checkpoint_interval_tokens,
        _ => None,
    }
}

/// Prefill-iteration multiplier of any co-located selector. The PD selectors do
/// not carry it: every PD prefill iteration schedules prefill and no PD decode
/// iteration does, so `gpu_time_multiplier` already expresses either stage.
fn prefill_gpu_time_multiplier(worker: &IterWorkerSel) -> anyhow::Result<Option<f64>> {
    let multiplier = match worker {
        IterWorkerSel::Barebone {
            prefill_gpu_time_multiplier,
            ..
        }
        | IterWorkerSel::HpUnified {
            prefill_gpu_time_multiplier,
            ..
        }
        | IterWorkerSel::ChunkedPrefill {
            prefill_gpu_time_multiplier,
            ..
        }
        | IterWorkerSel::Speculative {
            prefill_gpu_time_multiplier,
            ..
        } => *prefill_gpu_time_multiplier,
        IterWorkerSel::PdPrefill { .. } | IterWorkerSel::PdDecode { .. } => None,
    };
    if let Some(multiplier) = multiplier {
        ensure!(
            multiplier.is_finite() && multiplier >= 1.0,
            "unified: prefill_gpu_time_multiplier must be a finite value >= 1.0 (got {multiplier})"
        );
    }
    Ok(multiplier)
}

/// Host-time model, carried by the `speculative` selector only: the GLM-5.3
/// H200 MTP arch is the one that reports its graph mode. Other selectors reject
/// the knobs through the launcher's unknown-key guard.
fn host_time_model(
    worker: &IterWorkerSel,
) -> anyhow::Result<Option<crate::worker::config::HostTimeModel>> {
    match worker {
        IterWorkerSel::Speculative { host_time, .. } => host_time.resolve(),
        _ => Ok(None),
    }
}

/// vLLM's eagle block-drop chunk split, carried by the `speculative` selector only.
fn eagle_block_drop(worker: &IterWorkerSel) -> bool {
    match worker {
        IterWorkerSel::Speculative {
            eagle_block_drop, ..
        } => *eagle_block_drop,
        _ => false,
    }
}

/// vLLM's `--prefix-match-unit`, carried by the `speculative` selector only.
/// A unit is a positive multiple of 4 (vLLM's index-pool granularity) and only
/// takes effect under `eagle_block_drop`; whether it divides the recurrent
/// checkpoint interval is checked by the recipe, which knows that interval.
fn prefix_match_unit_tokens(worker: &IterWorkerSel) -> anyhow::Result<Option<u32>> {
    match worker {
        IterWorkerSel::Speculative {
            prefix_match_unit_tokens: Some(unit),
            eagle_block_drop,
            ..
        } => {
            ensure!(
                *unit > 0 && unit.is_multiple_of(4),
                "unified: prefix_match_unit_tokens must be a positive multiple of 4, got {unit}"
            );
            ensure!(
                *eagle_block_drop,
                "unified: prefix_match_unit_tokens is modeled only with eagle_block_drop: true"
            );
            Ok(Some(*unit))
        }
        _ => Ok(None),
    }
}

/// vLLM's async scheduling, carried by the `speculative` selector only.
/// vLLM's `max_num_seqs`, from the two chunked selectors; None elsewhere.
fn max_running_requests(worker: &IterWorkerSel) -> Option<u32> {
    match worker {
        IterWorkerSel::ChunkedPrefill {
            max_running_requests,
            ..
        }
        | IterWorkerSel::Speculative {
            max_running_requests,
            ..
        } => *max_running_requests,
        _ => None,
    }
}

/// The speculative selector's `recurrent_checkpoint_release`.
fn checkpoint_release(worker: &IterWorkerSel) -> RecurrentCheckpointRelease {
    match worker {
        IterWorkerSel::Speculative {
            recurrent_checkpoint_release,
            ..
        } => *recurrent_checkpoint_release,
        _ => RecurrentCheckpointRelease::Completion,
    }
}

fn async_scheduling(worker: &IterWorkerSel) -> bool {
    match worker {
        IterWorkerSel::Speculative {
            async_scheduling, ..
        } => *async_scheduling,
        _ => false,
    }
}

/// vLLM's async output-processing delay, carried by the `speculative` selector
/// only and meaningful only with `async_scheduling`.
fn async_output_processing(worker: &IterWorkerSel) -> anyhow::Result<crate::common::Time> {
    match worker {
        IterWorkerSel::Speculative {
            async_scheduling,
            async_output_processing_ms,
            ..
        } => {
            let delay_ms = *async_output_processing_ms;
            ensure!(
                delay_ms.is_finite() && delay_ms >= 0.0,
                "unified: async_output_processing_ms must be finite and >= 0, got {delay_ms}"
            );
            ensure!(
                *async_scheduling || delay_ms == 0.0,
                "unified: async_output_processing_ms={delay_ms} needs async_scheduling: true"
            );
            Ok(crate::common::Time::from_ms(delay_ms))
        }
        _ => Ok(crate::common::Time::ZERO),
    }
}

/// vLLM's native CPU KV offload, carried by the `speculative` selector only:
/// the host tier's total pinned bytes and copy rates. `None` without
/// `kv_offload_cpu_gb_total`. Whether the arch and prefix-cache setup support
/// it is checked by the recipe.
fn kv_offload(
    worker: &IterWorkerSel,
) -> anyhow::Result<Option<crate::worker::kv::HostOffloadConfig>> {
    let IterWorkerSel::Speculative {
        kv_offload_cpu_gb_total: Some(gib_total),
        kv_offload_h2d_gbps,
        kv_offload_d2h_gbps,
        kv_offload_transfer_latency_us,
        ..
    } = worker
    else {
        return Ok(None);
    };
    ensure!(
        gib_total.is_finite() && *gib_total > 0.0,
        "unified: kv_offload_cpu_gb_total must be finite and > 0, got {gib_total}"
    );
    for (name, rate) in [
        ("kv_offload_h2d_gbps", *kv_offload_h2d_gbps),
        ("kv_offload_d2h_gbps", *kv_offload_d2h_gbps),
    ] {
        ensure!(
            rate.is_finite() && rate > 0.0,
            "unified: {name} must be finite and > 0, got {rate}"
        );
    }
    let latency_us = *kv_offload_transfer_latency_us;
    ensure!(
        latency_us.is_finite() && latency_us >= 0.0,
        "unified: kv_offload_transfer_latency_us must be finite and >= 0, got {latency_us}"
    );
    Ok(Some(crate::worker::kv::HostOffloadConfig {
        // vLLM: `kv_offloading_size * (1 << 30)`.
        cpu_bytes_total: (gib_total * (1u64 << 30) as f64) as u64,
        h2d_bytes_per_s: kv_offload_h2d_gbps * 1e9,
        d2h_bytes_per_s: kv_offload_d2h_gbps * 1e9,
        transfer_latency: crate::common::Time::from_ns((latency_us * 1e3).round() as u64),
    }))
}

/// Draft candidate count, carried by the `speculative` selector only. Every other
/// worker leaves it at `0`, which is what marks it as not speculating.
fn speculative_draft_tokens(worker: &IterWorkerSel) -> u32 {
    match worker {
        IterWorkerSel::Speculative { draft_tokens, .. } => *draft_tokens,
        _ => 0,
    }
}

/// Which deterministic acceptance stream the speculative worker draws from.
fn speculative_acceptance_seed(worker: &IterWorkerSel) -> Option<u64> {
    match worker {
        IterWorkerSel::Speculative {
            acceptance_seed, ..
        } => *acceptance_seed,
        _ => None,
    }
}

/// The speculative GLM arch pairs only with the speculative worker: it builds a
/// different model type, so no other recipe can even name it.
fn ensure_speculative(worker: &IterWorkerSel, arch_draft_tokens: u32) -> anyhow::Result<()> {
    match worker {
        IterWorkerSel::Speculative { draft_tokens, .. } => {
            ensure!(
                *draft_tokens == arch_draft_tokens,
                "unified: worker draft_tokens={draft_tokens} must match the arch's \
                 draft_tokens={arch_draft_tokens} — the width selects a profiled \
                 kernel shape, so the two cannot differ"
            );
            Ok(())
        }
        other => {
            bail!("unified: this speculative arch requires worker `speculative`, got {other:?}")
        }
    }
}

/// DP-attention and MoE archs run on the multi-group hp_unified worker.
fn ensure_hp_unified(worker: &IterWorkerSel) -> anyhow::Result<()> {
    match worker {
        IterWorkerSel::HpUnified { .. } => Ok(()),
        other => {
            bail!("unified: this DP-attention/MoE arch requires worker `hp_unified`, got {other:?}")
        }
    }
}

/// The most tokens one step of `worker` runs: its chunk under chunked
/// prefill; None for a worker that runs each prompt whole.
fn chunk_tokens(worker: &IterWorkerSel) -> Option<u32> {
    match worker {
        IterWorkerSel::ChunkedPrefill {
            max_batch_tokens, ..
        } => Some(*max_batch_tokens),
        _ => None,
    }
}

/// Whole-iteration arches with a captured chunked-prefill runtime may use the
/// ordinary HP admission recipe or the dedicated hard-cap/chunking recipe.
fn ensure_hp_or_chunked_worker(arch_name: &str, worker: &IterWorkerSel) -> anyhow::Result<()> {
    match worker {
        IterWorkerSel::HpUnified { .. } | IterWorkerSel::ChunkedPrefill { .. } => Ok(()),
        other => bail!(
            "unified: {arch_name} requires worker `hp_unified` or `chunked_prefill`, got {other:?}"
        ),
    }
}

fn assemble_hp_or_chunked_flow<M>(
    arch_name: &str,
    model: Arc<M>,
    store: SharedRequests,
    worker_config: WorkerConfig,
    log_dir: Option<PathBuf>,
    gpu_name: String,
    dp_cfg: SimpleDpConfig,
    worker: &IterWorkerSel,
) -> anyhow::Result<Box<dyn Flow>>
where
    M: IterwiseUnifiedModel,
{
    match worker {
        IterWorkerSel::HpUnified { .. } => Ok(assemble_flow(
            model,
            store,
            worker_config,
            log_dir,
            gpu_name,
            dp_cfg,
            build_hp_worker,
        )),
        IterWorkerSel::ChunkedPrefill { .. } => Ok(assemble_flow(
            model,
            store,
            worker_config,
            log_dir,
            gpu_name,
            dp_cfg,
            build_chunked_prefill_worker,
        )),
        other => bail!("unified: unsupported {arch_name} worker {other:?}"),
    }
}

/// The speculative selector's exploration-only kernel-time counterfactual;
/// empty for every other worker and whenever it is unset.
fn what_if_kernel_time_scale(worker: &IterWorkerSel) -> &[KernelTimeScaleRule] {
    match worker {
        IterWorkerSel::Speculative {
            what_if_kernel_time_scale,
            ..
        } => what_if_kernel_time_scale,
        _ => &[],
    }
}

/// [`assemble_flow`] for a speculative arm. With no `what_if_kernel_time_scale`
/// rules the worker drives the arch model itself, so the default path is the
/// measured model unchanged; with rules it drives the model wrapped in
/// [`WhatIfKernelTimeScaled`], and a rule that fails against the model's slots
/// fails the build. The arm passes its recipe twice because each branch
/// instantiates it at a different model type.
#[allow(clippy::too_many_arguments)]
fn assemble_speculative_flow<M, W, Ws>(
    model: M,
    what_if: &[KernelTimeScaleRule],
    store: SharedRequests,
    worker_config: WorkerConfig,
    log_dir: Option<PathBuf>,
    gpu_name: String,
    dp_cfg: SimpleDpConfig,
    build_fn: WorkerBuildFn<M, W>,
    build_what_if_fn: WorkerBuildFn<WhatIfKernelTimeScaled<M>, Ws>,
) -> anyhow::Result<Box<dyn Flow>>
where
    M: SpeculativeUnifiedModel,
    W: IterWorker<Event = crate::worker::WorkerEventCommon> + 'static,
    W::Msg: From<crate::common::RequestId>,
    Ws: IterWorker<Event = crate::worker::WorkerEventCommon> + 'static,
    Ws::Msg: From<crate::common::RequestId>,
{
    if what_if.is_empty() {
        return Ok(assemble_flow(
            Arc::new(model),
            store,
            worker_config,
            log_dir,
            gpu_name,
            dp_cfg,
            build_fn,
        ));
    }
    let model = WhatIfKernelTimeScaled::new(model, what_if).map_err(anyhow::Error::msg)?;
    Ok(assemble_flow(
        Arc::new(model),
        store,
        worker_config,
        log_dir,
        gpu_name,
        dp_cfg,
        build_what_if_fn,
    ))
}

/// Wrap a built iter-wise model in a `simple_dp` flow. Generic over the concrete
/// model `M` and worker `W`; `build_fn` is the chosen worker's `new`. The returned
/// `Box<dyn Flow>` is the only `dyn` erasure point.
fn assemble_flow<M, W>(
    model: Arc<M>,
    store: SharedRequests,
    worker_config: WorkerConfig,
    log_dir: Option<PathBuf>,
    gpu_name: String,
    dp_cfg: SimpleDpConfig,
    build_fn: WorkerBuildFn<M, W>,
) -> Box<dyn Flow>
where
    // No L4 bound on `M`: `build_fn` already states the model contract its
    // recipe needs, and the speculative recipe's model implements
    // `SpeculativeUnifiedModel` rather than `IterwiseUnifiedModel`.
    W: IterWorker<Event = crate::worker::WorkerEventCommon> + 'static,
    W::Msg: From<crate::common::RequestId>,
{
    let factory = UnifiedWorkerFactory::new(
        model,
        store,
        worker_config,
        log_dir,
        gpu_name,
        "main",
        build_fn,
    );
    Box::new(SimpleDpFlow::new(dp_cfg, factory))
}

fn placement_into(p: PlacementPolicy) -> DpPlacementPolicy {
    match p {
        PlacementPolicy::LeastQueued => DpPlacementPolicy::LeastQueued,
        PlacementPolicy::RoundRobin => DpPlacementPolicy::RoundRobin,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::{
        Glm52VllmDsaMoeModel, Glm52VllmNvfp4DsaMoeModel, Glm52VllmNvfp4DsaMoeSpeculativeModel,
        Glm53FlashVllmModel, Qwen36LocalModel,
    };
    use crate::common::RequestId;
    use crate::worker::{ChunkedPrefillWorker, HpUnifiedWorker, WorkerEventCommon};

    fn hp_worker() -> IterWorkerSel {
        IterWorkerSel::HpUnified {
            attn_gpu_memory_gb: 80.0,
            gpu_time_multiplier: 1.0,
            max_batch_tokens: None,
            pending_order: crate::worker::PendingOrderKind::SessionStart,
            prefix_cache_mode: crate::worker::PrefixCacheMode::Opportunistic,
            prefix_cache_policy: crate::worker::PrefixCachePolicy::Lru,
            prefix_cache_max_gpu_memory_gb: None,
            prefill_gpu_time_multiplier: None,
        }
    }

    fn barebone_worker() -> IterWorkerSel {
        IterWorkerSel::Barebone {
            attn_gpu_memory_gb: 80.0,
            gpu_time_multiplier: 1.0,
            max_batch_tokens: None,
            pending_order: crate::worker::PendingOrderKind::SessionStart,
            prefix_cache_mode: crate::worker::PrefixCacheMode::Opportunistic,
            prefix_cache_policy: crate::worker::PrefixCachePolicy::Lru,
            prefix_cache_max_gpu_memory_gb: None,
            ssm_checkpoint_interval_tokens: None,
            prefill_gpu_time_multiplier: None,
        }
    }

    fn chunked_prefill_worker() -> IterWorkerSel {
        IterWorkerSel::ChunkedPrefill {
            attn_gpu_memory_gb: 120.0,
            max_batch_tokens: 2048,
            batch_policy: BatchPolicy::Mix,
            kv_admission: crate::worker::config::KvAdmissionSpec::default(),
            gpu_time_multiplier: 1.0,
            prefill_gpu_time_multiplier: None,
            max_running_requests: None,
        }
    }

    fn assert_iter_worker_contract<W>()
    where
        W: IterWorker<Event = WorkerEventCommon> + 'static,
        W::Msg: From<RequestId>,
    {
    }

    #[test]
    fn glm52_hp_unified_pair_satisfies_the_iter_worker_contract() {
        assert_iter_worker_contract::<HpUnifiedWorker<Glm52VllmDsaMoeModel>>();
        ensure_hp_unified(&hp_worker()).expect("GLM accepts hp_unified");
    }

    #[test]
    fn glm52_nvfp4_chunked_prefill_pair_satisfies_the_iter_worker_contract() {
        assert_iter_worker_contract::<ChunkedPrefillWorker<Glm52VllmNvfp4DsaMoeModel>>();
        ensure_hp_or_chunked_worker("GLM-4.5 NVFP4", &chunked_prefill_worker())
            .expect("GLM NVFP4 accepts the dedicated chunked-prefill worker");
    }

    /// Qwen3.6 local keeps the `barebone` preset tag — cadence, admission and
    /// execution really are barebone — but composes the hybrid KV store, so the
    /// contract is asserted against `Qwen36HybridWorker`, not `BareboneWorker`.
    #[test]
    fn qwen36_local_barebone_pair_satisfies_the_iter_worker_contract() {
        assert_iter_worker_contract::<crate::worker::Qwen36HybridWorker<Qwen36LocalModel>>();
        ensure_barebone(&barebone_worker()).expect("Qwen3.6 local accepts barebone");
        let error = ensure_barebone(&hp_worker()).unwrap_err().to_string();
        assert!(error.contains("requires worker `barebone`"));
    }

    /// Both hybrid archs take the hybrid store under either lifecycle.
    #[test]
    fn hybrid_archs_pair_with_barebone_or_chunked_prefill() {
        assert_iter_worker_contract::<crate::worker::Qwen36HybridWorker<Glm53FlashVllmModel>>();
        assert_iter_worker_contract::<crate::worker::HybridChunkedPrefillWorker<Glm53FlashVllmModel>>(
        );
        assert_iter_worker_contract::<crate::worker::HybridChunkedPrefillWorker<Qwen36LocalModel>>(
        );
        for worker in [barebone_worker(), chunked_prefill_worker()] {
            ensure_hybrid_worker("GLM-5.3-Flash", &worker).expect("hybrid lifecycle");
        }
        let error = ensure_hybrid_worker("GLM-5.3-Flash", &hp_worker())
            .unwrap_err()
            .to_string();
        assert!(error.contains("requires worker `barebone` or `chunked_prefill`"));
    }

    #[test]
    fn the_snapshot_interval_override_is_barebone_only_and_defaults_to_the_arch() {
        assert_eq!(ssm_checkpoint_interval_tokens(&barebone_worker()), None);
        assert_eq!(ssm_checkpoint_interval_tokens(&hp_worker()), None);
        let IterWorkerSel::Barebone {
            attn_gpu_memory_gb,
            gpu_time_multiplier,
            max_batch_tokens,
            pending_order,
            prefix_cache_mode,
            prefix_cache_policy,
            prefix_cache_max_gpu_memory_gb,
            ..
        } = barebone_worker()
        else {
            unreachable!()
        };
        let overridden = IterWorkerSel::Barebone {
            attn_gpu_memory_gb,
            gpu_time_multiplier,
            max_batch_tokens,
            pending_order,
            prefix_cache_mode,
            prefix_cache_policy,
            prefix_cache_max_gpu_memory_gb,
            ssm_checkpoint_interval_tokens: Some(528),
            prefill_gpu_time_multiplier: None,
        };
        assert_eq!(ssm_checkpoint_interval_tokens(&overridden), Some(528));
    }

    #[test]
    fn every_co_located_selector_carries_a_prefill_multiplier_of_at_least_one() {
        let with = |mut worker: IterWorkerSel, multiplier: f64| {
            match &mut worker {
                IterWorkerSel::Barebone {
                    prefill_gpu_time_multiplier,
                    ..
                }
                | IterWorkerSel::HpUnified {
                    prefill_gpu_time_multiplier,
                    ..
                }
                | IterWorkerSel::ChunkedPrefill {
                    prefill_gpu_time_multiplier,
                    ..
                }
                | IterWorkerSel::Speculative {
                    prefill_gpu_time_multiplier,
                    ..
                } => *prefill_gpu_time_multiplier = Some(multiplier),
                other => panic!("not a co-located selector: {other:?}"),
            }
            worker
        };
        for worker in [
            barebone_worker(),
            hp_worker(),
            chunked_prefill_worker(),
            speculative_worker(3),
        ] {
            assert_eq!(prefill_gpu_time_multiplier(&worker).unwrap(), None);
            assert_eq!(
                prefill_gpu_time_multiplier(&with(worker.clone(), 1.2)).unwrap(),
                Some(1.2)
            );
            let error = prefill_gpu_time_multiplier(&with(worker, 0.9))
                .unwrap_err()
                .to_string();
            assert!(error.contains("prefill_gpu_time_multiplier must be a finite value >= 1.0"));
        }
    }

    #[test]
    fn glm52_rejects_non_hp_iter_workers_with_an_actionable_error() {
        let error = ensure_hp_unified(&barebone_worker())
            .unwrap_err()
            .to_string();
        assert!(error.contains("DP-attention/MoE arch requires worker `hp_unified`"));
        assert!(error.contains("Barebone"));

        let pd_prefill = IterWorkerSel::PdPrefill {
            attn_gpu_memory_gb: 80.0,
            gpu_time_multiplier: 1.0,
            prefix_cache_mode: crate::worker::PrefixCacheMode::Opportunistic,
            prefix_cache_policy: crate::worker::PrefixCachePolicy::Lru,
            prefix_cache_max_gpu_memory_gb: None,
        };
        let error = ensure_hp_unified(&pd_prefill).unwrap_err().to_string();
        assert!(error.contains("worker `hp_unified`"));
        assert!(error.contains("PdPrefill"));
    }

    #[test]
    fn glm52_unified_config_preserves_ep8_hp_pairing() {
        let yaml = r#"
deployment: unified
workload: { trace_files: ["trace/smoke.csv"], input_file_format: text-generation-independent, arrival_mode: trace_timed, duration_ms: 1000.0, run_to_end: true, request_rate: 1.0 }
io: { log_dir: "logs/test", log_level: info, quiet: true, force_cache_build: false, log_output_token_times: false }
pools:
  main:
    placement: least-queued
    groups:
      - gpu: "NVIDIA H200"
        replicas: 1
        arch:
          type: glm52_vllm_dsa_moe
          model_config: model/config/glm52.json
          fp8: false
          ep_size: 8
          nvl_num_gpu: 8
          routing: uniform
          mtp_mode: off
        worker:
          type: hp_unified
          attn_gpu_memory_gb: 80.0
          gpu_time_multiplier: 1.0
"#;
        let cfg: crate::deployment::RunConfig =
            serde_yaml::from_str(yaml).expect("GLM hp_unified config parses");
        let crate::deployment::RunConfig::Unified(cfg) = cfg else {
            panic!("expected unified config")
        };
        let group = &cfg.pools.main.groups[0];
        let IterArchSel::Glm52VllmDsaMoe {
            ep_size,
            nvl_num_gpu,
            mtp_mode,
            ..
        } = &group.arch
        else {
            panic!("expected GLM arch")
        };
        assert_eq!((*ep_size, *nvl_num_gpu), (8, 8));
        assert_eq!(*mtp_mode, crate::arch::Glm52MtpMode::Off);
        ensure_hp_unified(&group.worker).expect("GLM hp_unified pairing is accepted");
    }

    fn speculative_worker(draft_tokens: u32) -> IterWorkerSel {
        IterWorkerSel::Speculative {
            attn_gpu_memory_gb: 180.0,
            max_batch_tokens: 8192,
            draft_tokens,
            acceptance_seed: Some(7),
            batch_policy: BatchPolicy::Mix,
            kv_admission: Default::default(),
            gpu_time_multiplier: 1.0,
            prefill_gpu_time_multiplier: None,
            host_time: Default::default(),
            eagle_block_drop: false,
            prefix_match_unit_tokens: None,
            async_scheduling: false,
            async_output_processing_ms: 0.0,
            kv_offload_cpu_gb_total: None,
            kv_offload_h2d_gbps: 37.7,
            kv_offload_d2h_gbps: 24.3,
            kv_offload_transfer_latency_us: 7.0,
            what_if_kernel_time_scale: Vec::new(),
            max_running_requests: None,
            recurrent_checkpoint_release: Default::default(),
        }
    }

    #[test]
    fn kv_offload_is_off_unless_sized_and_validates_its_rates() {
        assert_eq!(kv_offload(&speculative_worker(5)).unwrap(), None);
        assert_eq!(kv_offload(&barebone_worker()).unwrap(), None);
        let sized = |gib: f64, h2d: f64, latency_us: f64| {
            let mut worker = speculative_worker(5);
            if let IterWorkerSel::Speculative {
                kv_offload_cpu_gb_total,
                kv_offload_h2d_gbps,
                kv_offload_transfer_latency_us,
                ..
            } = &mut worker
            {
                *kv_offload_cpu_gb_total = Some(gib);
                *kv_offload_h2d_gbps = h2d;
                *kv_offload_transfer_latency_us = latency_us;
            }
            kv_offload(&worker)
        };
        let config = sized(256.0, 37.7, 7.0).unwrap().unwrap();
        assert_eq!(config.cpu_bytes_total, 256 << 30);
        assert_eq!(config.h2d_bytes_per_s, 37.7e9);
        assert_eq!(config.d2h_bytes_per_s, 24.3e9);
        assert_eq!(config.transfer_latency, crate::common::Time::from_us(7));
        assert!(sized(0.0, 37.7, 7.0).is_err());
        assert!(sized(128.0, 0.0, 7.0).is_err());
        assert!(sized(128.0, 37.7, -1.0).is_err());
    }

    /// The smallest speculative model the recipe accepts: a routed-MoE down
    /// GEMM after its input quant.
    struct TwoSlotSpeculativeModel;

    impl SpeculativeUnifiedModel for TwoSlotSpeculativeModel {
        fn eval_speculative_iter(
            &self,
            _batch: &crate::arch::SpeculativeArchInput,
            _slots: &mut Vec<crate::timing::LeafMetrics>,
            _scratch: &mut Vec<crate::timing::LeafMetrics>,
        ) -> crate::timing::LeafMetrics {
            crate::timing::LeafMetrics::ZERO
        }

        fn cost_log_manifest(&self) -> crate::timing::CostManifest {
            let mut builder = crate::timing::CostTreeBuilder::new();
            let quant = builder.leaf(
                "unified.moe.routed.down_quant",
                "k",
                serde_json::Value::Null,
            );
            let down = builder.leaf("unified.moe.routed.down", "k", serde_json::Value::Null);
            builder
                .finish(crate::timing::CostNode::Sum(vec![quant, down]))
                .manifest()
        }

        fn total_kv_bytes_per_token(&self) -> u64 {
            1
        }

        fn max_model_len(&self) -> u32 {
            4096
        }

        fn gpus_per_replica(&self) -> u16 {
            1
        }
    }

    #[test]
    fn the_what_if_scale_rides_the_speculative_selector_into_the_model_build() {
        let worker: IterWorkerSel = serde_yaml::from_str(
            "type: speculative\nattn_gpu_memory_gb: 80.0\nmax_batch_tokens: 8192\n\
             draft_tokens: 3\nwhat_if_kernel_time_scale:\n\
             - {match: moe.routed.down, factor: 0.8}\n",
        )
        .expect("parse speculative worker");
        let rules = what_if_kernel_time_scale(&worker).to_vec();
        assert_eq!(
            rules,
            vec![KernelTimeScaleRule {
                pattern: "moe.routed.down".to_string(),
                factor: 0.8,
            }]
        );
        assert!(what_if_kernel_time_scale(&speculative_worker(3)).is_empty());
        assert!(what_if_kernel_time_scale(&barebone_worker()).is_empty());

        let assemble = |rules: &[KernelTimeScaleRule]| {
            assemble_speculative_flow(
                TwoSlotSpeculativeModel,
                rules,
                crate::test_helpers::shared_with(&[]),
                WorkerConfig {
                    attn_kv_bytes: 1_000_000,
                    max_batch_tokens: Some(64),
                    speculative_draft_tokens: 3,
                    ..WorkerConfig::default()
                },
                None,
                "NVIDIA H200".to_string(),
                SimpleDpConfig {
                    dp_pool: SimpleDpPoolConfig {
                        pool: PoolId(0),
                        num_workers: 1,
                        placement: DpPlacementPolicy::LeastQueued,
                    },
                },
                build_speculative_worker,
                build_speculative_worker,
            )
        };
        assemble(&[]).expect("no rules runs the measured model");
        assemble(&rules).expect("a rule that scales a slot builds");
        let error = assemble(&[KernelTimeScaleRule {
            pattern: "moe.routed.gate_up".to_string(),
            factor: 0.9,
        }])
        .err()
        .expect("a rule that scales no slot fails the build")
        .to_string();
        assert!(error.contains("\"moe.routed.gate_up\""), "{error}");
        assert!(error.contains("hits no kernel slot"), "{error}");
        assert_iter_worker_contract::<
            crate::worker::HybridSpeculativeWorker<
                WhatIfKernelTimeScaled<crate::arch::Glm53FlashSpecModel>,
            >,
        >();
    }

    #[test]
    fn eagle_block_drop_reaches_only_the_speculative_selector() {
        assert!(!eagle_block_drop(&speculative_worker(5)));
        let mut reproduce_vllm = speculative_worker(5);
        if let IterWorkerSel::Speculative {
            eagle_block_drop, ..
        } = &mut reproduce_vllm
        {
            *eagle_block_drop = true;
        }
        assert!(eagle_block_drop(&reproduce_vllm));
        assert!(!eagle_block_drop(&barebone_worker()));
    }

    #[test]
    fn a_prefix_match_unit_is_a_multiple_of_four_under_the_eagle_block_drop() {
        let with_unit = |unit: Option<u32>, block_drop: bool| {
            let mut worker = speculative_worker(5);
            if let IterWorkerSel::Speculative {
                eagle_block_drop,
                prefix_match_unit_tokens,
                ..
            } = &mut worker
            {
                *eagle_block_drop = block_drop;
                *prefix_match_unit_tokens = unit;
            }
            prefix_match_unit_tokens(&worker)
        };
        assert_eq!(with_unit(None, false).unwrap(), None);
        assert_eq!(with_unit(None, true).unwrap(), None);
        assert_eq!(with_unit(Some(64), true).unwrap(), Some(64));
        for unit in [0, 66] {
            let error = with_unit(Some(unit), true).unwrap_err().to_string();
            assert!(error.contains("positive multiple of 4"), "{error}");
        }
        let error = with_unit(Some(64), false).unwrap_err().to_string();
        assert!(error.contains("eagle_block_drop"), "{error}");
        assert_eq!(prefix_match_unit_tokens(&barebone_worker()).unwrap(), None);
    }

    #[test]
    fn async_scheduling_reaches_only_the_speculative_selector() {
        assert!(!async_scheduling(&speculative_worker(5)));
        let mut reproduce_vllm = speculative_worker(5);
        if let IterWorkerSel::Speculative {
            async_scheduling, ..
        } = &mut reproduce_vllm
        {
            *async_scheduling = true;
        }
        assert!(async_scheduling(&reproduce_vllm));
        assert!(!eagle_block_drop(&reproduce_vllm));
        assert!(!async_scheduling(&barebone_worker()));
    }

    #[test]
    fn the_output_processing_delay_needs_async_scheduling() {
        let with_delay = |async_on: bool, delay_ms: f64| {
            let mut worker = speculative_worker(5);
            if let IterWorkerSel::Speculative {
                async_scheduling,
                async_output_processing_ms,
                ..
            } = &mut worker
            {
                *async_scheduling = async_on;
                *async_output_processing_ms = delay_ms;
            }
            async_output_processing(&worker)
        };
        assert_eq!(
            with_delay(true, 0.4).unwrap(),
            crate::common::Time::from_ms(0.4)
        );
        assert_eq!(with_delay(false, 0.0).unwrap(), crate::common::Time::ZERO);
        assert!(with_delay(false, 0.4).is_err());
        assert!(with_delay(true, -1.0).is_err());
        assert!(with_delay(true, f64::NAN).is_err());
        assert_eq!(
            async_output_processing(&barebone_worker()).unwrap(),
            crate::common::Time::ZERO
        );
    }

    #[test]
    fn the_speculative_pair_satisfies_the_iter_worker_contract() {
        assert_iter_worker_contract::<
            crate::worker::SpeculativeWorker<Glm52VllmNvfp4DsaMoeSpeculativeModel>,
        >();
        ensure_speculative(&speculative_worker(3), 3).expect("matching widths pair");
    }

    #[test]
    fn a_verify_width_that_disagrees_with_the_arch_is_rejected_before_anything_is_built() {
        // The defect this catches: the arch compiles a tree for k=3 while the
        // worker submits k=5 rows. The model would reject the batch at runtime,
        // deep inside an iteration, instead of at configuration time.
        let error = ensure_speculative(&speculative_worker(5), 3)
            .unwrap_err()
            .to_string();
        assert!(error.contains("draft_tokens=5"));
        assert!(error.contains("draft_tokens=3"));

        let error = ensure_speculative(&chunked_prefill_worker(), 3)
            .unwrap_err()
            .to_string();
        assert!(error.contains("requires worker `speculative`"));
    }

    #[test]
    fn a_speculative_preset_parses_into_the_paired_arch_and_worker() {
        // Both selectors are new serde surfaces. This is the shape a preset
        // author actually writes, so a rename or a missing `serde(default)`
        // shows up here rather than at run time.
        let yaml = r#"
deployment: unified
workload: { trace_files: ["trace/smoke.csv"], input_file_format: text-generation-independent, arrival_mode: trace_timed, duration_ms: 1000.0, run_to_end: true, request_rate: 1.0 }
io: { log_dir: "logs/test", log_level: info, quiet: true, force_cache_build: false, log_output_token_times: false }
pools:
  main:
    placement: least-queued
    groups:
      - gpu: "NVIDIA B200"
        replicas: 1
        arch:
          type: glm52_vllm_nvfp4_dsa_moe_speculative
          model_config: model/config/glm52.json
          fp8: false
          ep_size: 4
          nvl_num_gpu: 4
          max_model_len: 8192
          draft_tokens: 3
        worker:
          type: speculative
          attn_gpu_memory_gb: 180.0
          max_batch_tokens: 8192
          draft_tokens: 3
"#;
        let cfg: crate::deployment::RunConfig =
            serde_yaml::from_str(yaml).expect("speculative config parses");
        let crate::deployment::RunConfig::Unified(cfg) = cfg else {
            panic!("expected unified config")
        };
        let group = &cfg.pools.main.groups[0];
        let IterArchSel::Glm52VllmNvfp4DsaMoeSpeculative {
            draft_tokens,
            mtp_mode,
            max_model_len,
            ..
        } = &group.arch
        else {
            panic!("expected the speculative GLM arch")
        };
        assert_eq!((*draft_tokens, *max_model_len), (3, 8192));
        // A speculative arch has to run its drafter, so unlike the ordinary
        // selector it does not default `mtp_mode` to `off`.
        assert_eq!(*mtp_mode, Glm52MtpMode::IndexShare);
        ensure_speculative(&group.worker, *draft_tokens).expect("the pair is accepted");
        assert_eq!(speculative_draft_tokens(&group.worker), 3);
        assert_eq!(speculative_acceptance_seed(&group.worker), None);
    }

    #[test]
    fn only_the_speculative_selector_carries_a_verify_width_or_an_acceptance_seed() {
        // Every other worker must read as "not speculating"; a nonzero default
        // would silently turn an ordinary recipe's assertion into a live width.
        assert_eq!(speculative_draft_tokens(&speculative_worker(5)), 5);
        assert_eq!(speculative_acceptance_seed(&speculative_worker(5)), Some(7));
        for worker in [barebone_worker(), hp_worker(), chunked_prefill_worker()] {
            assert_eq!(speculative_draft_tokens(&worker), 0);
            assert_eq!(speculative_acceptance_seed(&worker), None);
        }
    }
}
