//! GLM-5.3-Flash (`Glm5NextForConditionalGeneration`) FP8 block checkpoint on
//! H200 as upstream vLLM v0.31 serves it in production: TP4 without expert
//! parallelism, BF16 KV cache, and MTP speculative decoding with
//! `draft_tokens` candidates per request.
//!
//! Every forward is the B200 arch's hybrid stack (34 KDA + 11 DSA layers,
//! 3 dense + 42 MoE FFNs, 4-wide mHC, `glm53_flash_common`) costed with the
//! H200 launches measured in the case-03 capture
//! (`logs/20261007_0_glm53_flash_h200_campaign`):
//!
//! - routed MoE: router GEMM once, grouped top-k, Triton FP8 `fused_moe` with
//!   all 288 experts on every rank and the intermediate sharded by TP. At
//!   most 256 tokens the shared expert runs on the main stream while router
//!   and routed experts run on vLLM's auxiliary stream (`Parallel`);
//! - KDA: verify rows take conv update + recurrent KDA, prefill rows the
//!   chunked core;
//! - DSA: the kpool indexer over `K+1` rows per request (flattened to
//!   single-token rows on SM90, where DeepGEMM has no `next_n = 6`),
//!   dense-MHA prefill or sparse MQA through `FLASH_ATTN_MLA_SPARSE`, BF16 latent cache;
//! - every TP all-reduce is FlashInfer's up to its sm90 fusion workspace and
//!   a plain all-reduce above it (vLLM's FLASHINFER > CUSTOM > SYMM_MEM >
//!   PYNCCL dispatch; custom and symm_mem have no L1 backend yet).
//!
//! The draft stage is the MTP layer (DSA + MoE, no mHC): step 0 forwards every
//! target row and builds the indexer state; the remaining `K - 1` steps
//! forward one row per request, share the step-0 indices
//! (`index_share_for_mtp_iteration`) and rebuild attention metadata between
//! steps (fused multi-step draft is unsupported for these backends). Each
//! step ends in the vocab-parallel head: row select, `lm_head`, logits
//! all-gather and argmax.
//!
//! ```text
//! verify   prologue, 45 hybrid layers (Scale-folded groups), epilogue
//! draft    mtp.step_0 [all target rows, indexer]
//!          mtp.recurrent x (K-1) [one row per request, index shared]
//! ```
//!
//! KV accounting follows vLLM's hybrid allocator (`kv_cache_utils.py`): one
//! page holds 1152 attention tokens of the 12 MLA layers (11 + MTP), and
//! each request pins `kda_groups x (2 + K) + 1` pages of recurrent state
//! (Mamba `align` mode: two state pages plus one per speculative token per
//! KDA group, plus the kpool tail block).

use anyhow::Result;

use crate::arch::contract::{SpeculativeArchInput, SpeculativeUnifiedModel};
use crate::arch::glm53_flash_common::{
    atomic, graph_padded_tokens, layer_groups, push, AttnKind, Boundary, FfnKind,
};
pub use crate::arch::glm53_flash_common::{Glm53FlashLayerGroup, Glm53FlashModelCfg};
use crate::common::Fabric;
use crate::op::mhc::{MhcTerminalPostConfig, MhcTerminalPostInput, MhcTerminalPostOp};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::expert_demand::ExpertDemand;
use crate::timing::kernels::all_reduce_fusion::fused_all_reduce_refusal;
use crate::timing::kernels::{
    AllReduceFusionKernel, AllReduceFusionKernelConfig, AllReduceFusionKernelInput,
    AllReduceFusionSpec, AllReduceKernel, AllReduceKernelConfig, AllReduceKernelInput,
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput,
    MhcFusedPostPreRmsNormKernel, MhcPreRmsNormKernel, MhcRmsNormKernelConfig,
    MhcRmsNormKernelInput, ResidualRmsNormKernel, ResidualRmsNormKernelConfig,
    ResidualRmsNormKernelInput, RmsNormKernel, RmsNormKernelConfig, RmsNormKernelInput,
    SingleGemmKernel, SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, Dim, Evaluator, FlatCostNode,
    LeafMetrics, PerfApiBridge, SlotInput,
};
use crate::worklet::glm53_fp8_mlp_local::COLUMN_MAJOR_SCALE_FORMAT;
use crate::worklet::{
    Glm53DsaSpecAttnLocalWorklet, Glm53DsaSpecAttnLocalWorkletConfig,
    Glm53DsaSpecAttnLocalWorkletInput, Glm53DsaSpecAttnLocalWorkletResolved,
    Glm53Fp8MlpLocalWorklet, Glm53Fp8MlpLocalWorkletConfig, Glm53Fp8MlpLocalWorkletInput,
    Glm53Fp8MlpLocalWorkletResolved, Glm53KdaSpecAttnLocalWorklet,
    Glm53KdaSpecAttnLocalWorkletConfig, Glm53KdaSpecAttnLocalWorkletInput,
    Glm53KdaSpecAttnLocalWorkletResolved, Glm53TritonMoeRouterLocalWorklet,
    Glm53TritonMoeRouterLocalWorkletConfig, Glm53TritonMoeRouterLocalWorkletInput,
    Glm53TritonMoeRouterLocalWorkletResolved, Glm53TritonMoeTpLocalWorklet,
    Glm53TritonMoeTpLocalWorkletConfig, Glm53TritonMoeTpLocalWorkletInput,
    Glm53TritonMoeTpLocalWorkletResolved,
};

pub const ARCH_KIND: &str = "glm53_flash_vllm_fp8_kda_dsa_moe_speculative";
const ACTIVATION_DTYPE: DType = DType::Bf16;
const KV_CACHE_DTYPE: DType = DType::Bf16;
/// Sparse page-table width: `round_up(index_topk + index_kpool - 1, 128)`.
const SELECTED_K: u32 = 2176;
/// FLASH_ATTN_MLA_SPARSE kernel block size ("Setting kv cache block size to
/// 64" in the H200 server log).
const CACHE_BLOCK_SIZE: u32 = 64;
/// vLLM's hybrid page for this deployment ("attention block size 1152
/// tokens" in the H200 server log): the attention tokens one page holds, the
/// KDA checkpoint interval, and the prefix-cache reuse quantum.
pub const HYBRID_BLOCK_SIZE: u32 = 1152;
/// Mamba `align` cache mode keeps two state pages per KDA group per request,
/// plus one per speculative token (`MambaSpec.max_memory_usage_bytes`).
const ALIGN_STATE_PAGES: u32 = 2;
/// The kpool tail group's one block per request.
const KPOOL_TAIL_PAGES: u32 = 1;
/// vLLM runs the shared expert on an aux stream at or below this batch size.
const SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD: u32 = 256;
/// `Parallel{overlap}` of the main-stream shared expert against the
/// aux-stream router + routed experts. B200's measured value (union /
/// isolated max 1.09-1.13); provisional on H200 until kernel-align.
const SHARED_EXPERTS_STREAM_OVERLAP: f32 = 0.9;
/// Draft-metadata rebuild and input preparation between MTP steps: about
/// ten small launches per step (`_compute_slot_mappings`, paged-MQA
/// metadata, token/request maps, `_update_draft_inputs`).
const DRAFT_STEP_GLUE_LAUNCHES: u32 = 10;
const DRAFT_STEP_GLUE_BYTES_PER_ROW: u32 = 64;

const BF16_GEMM_BACKENDS: &[&str] = &["torch_linear_vllm"];
const FP8_GEMM_BACKENDS: &[&str] = &["deepgemm"];
const FP8_QUANT_BACKENDS: &[&str] = &["vllm_cuda"];
const ROUTER_GEMM_BACKENDS: &[&str] = &["torch_cublas"];
const FP32_GEMM_BACKENDS: &[&str] = &["torch_cublas"];
const ELEMENTWISE_BACKENDS: &[&str] = &["triton"];
const MHC_BACKENDS: &[&str] = &["vllm_tilelang"];
const RMS_NORM_BACKENDS: &[&str] = &["vllm_cuda"];
const RESIDUAL_NORM_BACKENDS: &[&str] = &["vllm_cuda"];
/// Stand-ins until FlashKDA and the verify-width axis land (plan M4).
const KDA_PREFILL_BACKENDS: &[&str] = &["vllm_triton"];
const KDA_DECODE_BACKENDS: &[&str] = &["vllm_triton"];
const CONV_BACKENDS: &[&str] = &["vllm_triton"];
const QKV_NORM_BACKENDS: &[&str] = &["vllm_triton"];
const Q_ABSORB_BACKENDS: &[&str] = &["torch_mla_q_absorb_no_rope"];
const V_UP_BACKENDS: &[&str] = &["torch_mla_v_up_unpadded"];
const MQA_LOGITS_BACKENDS: &[&str] = &["deepgemm_fp8"];
const TOPK_BACKENDS: &[&str] = &["vllm_cuda"];
const MQA_LOGITS_PREFILL_BACKENDS: &[&str] = &["deepgemm_fp8"];
const TOPK_PREFILL_BACKENDS: &[&str] = &["vllm_cuda"];
const MLA_APPEND_BACKENDS: &[&str] = &["vllm_cuda"];
const FUSED_MOE_BACKENDS: &[&str] = &["vllm_triton"];
const MOE_ALIGN_BACKENDS: &[&str] = &["vllm_cuda"];
const MOE_SUM_BACKENDS: &[&str] = &["vllm_cuda"];
const ALL_REDUCE_FUSION_BACKENDS: &[&str] = &["flashinfer_trtllm"];
const ALL_REDUCE_BACKENDS: &[&str] = &["nccl", "nvshmem"];

/// Deployment layout: one TP rank group, no EP, MTP drafting.
#[derive(Clone, Debug)]
pub struct Glm53FlashSpecParallel {
    pub tp_size: u16,
    pub max_model_len: u32,
    pub gpu_name: String,
    /// vLLM's captured CUDA-graph sizes; empty runs eager (no padding).
    pub cudagraph_capture_sizes: Vec<u32>,
    pub draft_tokens: u32,
}

/// The routing demand of each MoE forward kind: the body and the MTP step 0
/// see verify blocks of `K + 1` rows, the later MTP steps single rows.
#[derive(Clone, Debug)]
pub struct Glm53FlashSpecDemand {
    pub body: ExpertDemand,
    pub mtp_step0: ExpertDemand,
    pub mtp_recurrent: ExpertDemand,
}

/// Every configuration the model is built from.
#[derive(Clone, Debug)]
pub struct Glm53FlashSpecConfigs {
    pub model: Glm53FlashModelCfg,
    pub parallel: Glm53FlashSpecParallel,
    pub groups: Vec<Glm53FlashLayerGroup>,
    pub kda: Glm53KdaSpecAttnLocalWorkletConfig,
    pub dsa: Glm53DsaSpecAttnLocalWorkletConfig,
    pub dense_ffn: Glm53Fp8MlpLocalWorkletConfig,
    pub shared_expert: Glm53Fp8MlpLocalWorkletConfig,
    pub router: Glm53TritonMoeRouterLocalWorkletConfig,
    pub routed_body: Glm53TritonMoeTpLocalWorkletConfig,
    pub routed_mtp_step0: Glm53TritonMoeTpLocalWorkletConfig,
    pub routed_mtp_recurrent: Glm53TritonMoeTpLocalWorkletConfig,
    pub embedding: ElementwiseKernelConfig,
    pub hc_expand: ElementwiseKernelConfig,
    pub mhc: MhcRmsNormKernelConfig,
    pub all_reduce_fusion: AllReduceFusionKernelConfig,
    pub all_reduce: AllReduceKernelConfig,
    /// Shared + routed combine.
    pub moe_combine_glue: ElementwiseKernelConfig,
    pub hc_contract_mean: ElementwiseKernelConfig,
    pub final_norm: RmsNormKernelConfig,
    pub lm_head: SingleGemmKernelConfig,
    pub mtp_eh_norm: ElementwiseKernelConfig,
    pub mtp_eh_proj: SingleGemmKernelConfig,
    pub mtp_residual_norm: ResidualRmsNormKernelConfig,
    pub mtp_row_select: ElementwiseKernelConfig,
    pub mtp_logits_gather: ElementwiseKernelConfig,
    pub mtp_argmax: ElementwiseKernelConfig,
    pub mtp_step_glue: ElementwiseKernelConfig,
}

pub fn build_configs(
    model: &Glm53FlashModelCfg,
    parallel: &Glm53FlashSpecParallel,
    demand: &Glm53FlashSpecDemand,
) -> std::result::Result<Glm53FlashSpecConfigs, BuildError> {
    let tp = u32::from(parallel.tp_size);
    if parallel.draft_tokens == 0 {
        return Err(fit_failed(
            "draft_tokens must be positive for an MTP drafter",
        ));
    }
    if let Some(reason) =
        fused_all_reduce_refusal(ALL_REDUCE_FUSION_BACKENDS, &parallel.gpu_name, tp)
    {
        return Err(fit_failed(format!("tp_size {tp}: {reason}")));
    }
    let divide = |what: &str, value: u32| -> std::result::Result<u32, BuildError> {
        if tp == 0 || value % tp != 0 {
            return Err(fit_failed(format!("{what} {value} must divide by TP {tp}")));
        }
        Ok(value / tp)
    };
    if model.index_topk + model.index_kpool - 1 > SELECTED_K {
        return Err(fit_failed(
            "pooled indexer window exceeds the 2176-wide page table",
        ));
    }
    let gpu = parallel.gpu_name.clone();
    let hidden_bytes = model.hidden * ACTIVATION_DTYPE.size_bytes();
    let stream_bytes = model.hc_mult * hidden_bytes;
    let vocab_per_rank = divide("vocab_size", model.vocab_size)?;
    let ew = |input: u32, output: u32| ElementwiseKernelConfig {
        backends: ELEMENTWISE_BACKENDS.to_vec(),
        gpu_name: gpu.clone(),
        input_bytes_per_token: input.into(),
        output_bytes_per_token: output.into(),
    };
    let mlp = |intermediate: u32| Glm53Fp8MlpLocalWorkletConfig {
        hidden: model.hidden.into(),
        intermediate: intermediate.into(),
        activation_dtype: ACTIVATION_DTYPE,
        gpu_name: gpu.clone(),
        quant_backends: FP8_QUANT_BACKENDS.to_vec(),
        fp8_gemm_backends: FP8_GEMM_BACKENDS.to_vec(),
        elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
        quant_scale_format: COLUMN_MAJOR_SCALE_FORMAT,
    };
    let intermediate_per_rank = divide("moe_intermediate_size", model.moe_intermediate_size)?;
    let routed = |expert_demand: &ExpertDemand| Glm53TritonMoeTpLocalWorkletConfig {
        hidden: model.hidden.into(),
        intermediate_per_rank: intermediate_per_rank.into(),
        num_experts: model.n_routed_experts.into(),
        top_k: model.num_experts_per_tok,
        activation_dtype: ACTIVATION_DTYPE,
        gpu_name: gpu.clone(),
        quant_backends: FP8_QUANT_BACKENDS.to_vec(),
        fused_moe_backends: FUSED_MOE_BACKENDS.to_vec(),
        align_backends: MOE_ALIGN_BACKENDS.to_vec(),
        sum_backends: MOE_SUM_BACKENDS.to_vec(),
        elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
        expert_demand: expert_demand.clone(),
    };
    let fp32 = DType::Fp32.size_bytes();
    Ok(Glm53FlashSpecConfigs {
        groups: layer_groups(model),
        kda: Glm53KdaSpecAttnLocalWorkletConfig {
            hidden: model.hidden.into(),
            num_heads: divide("KDA heads", model.kda_num_heads)?.into(),
            head_dim: model.kda_head_dim.into(),
            gate_rank: model.kda_head_dim,
            conv_kernel_size: model.short_conv_kernel_size.into(),
            activation_dtype: ACTIVATION_DTYPE,
            gpu_name: gpu.clone(),
            bf16_gemm_backends: BF16_GEMM_BACKENDS.to_vec(),
            conv_backends: CONV_BACKENDS.to_vec(),
            prefill_core_backends: KDA_PREFILL_BACKENDS.to_vec(),
            decode_core_backends: KDA_DECODE_BACKENDS.to_vec(),
            elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
        },
        dsa: Glm53DsaSpecAttnLocalWorkletConfig {
            hidden: model.hidden.into(),
            num_heads: divide("attention heads", model.num_attention_heads)?.into(),
            q_lora_rank: model.q_lora_rank.into(),
            kv_lora_rank: model.kv_lora_rank.into(),
            qk_nope_head_dim: model.qk_nope_head_dim.into(),
            v_head_dim: model.v_head_dim.into(),
            index_num_heads: model.index_n_heads.into(),
            index_head_dim: model.index_head_dim.into(),
            index_topk: model.index_topk,
            index_kpool: model.index_kpool,
            selected_k: SELECTED_K,
            max_model_len: parallel.max_model_len,
            cache_block_size: CACHE_BLOCK_SIZE,
            rms_eps: model.rms_norm_eps,
            verify_width: parallel.draft_tokens + 1,
            kv_cache_dtype: KV_CACHE_DTYPE,
            gpu_name: gpu.clone(),
            bf16_gemm_backends: BF16_GEMM_BACKENDS.to_vec(),
            fp32_gemm_backends: FP32_GEMM_BACKENDS.to_vec(),
            qkv_norm_backends: QKV_NORM_BACKENDS.to_vec(),
            mla_bmm_q_absorb_backends: Q_ABSORB_BACKENDS.to_vec(),
            mla_bmm_v_up_backends: V_UP_BACKENDS.to_vec(),
            mqa_logits_backends: MQA_LOGITS_BACKENDS.to_vec(),
            topk_backends: TOPK_BACKENDS.to_vec(),
            mqa_logits_prefill_backends: MQA_LOGITS_PREFILL_BACKENDS.to_vec(),
            topk_prefill_backends: TOPK_PREFILL_BACKENDS.to_vec(),
            mla_cache_append_backends: MLA_APPEND_BACKENDS.to_vec(),
            elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
        },
        dense_ffn: mlp(divide("intermediate_size", model.intermediate_size)?),
        shared_expert: mlp(divide(
            "shared expert intermediate",
            model.moe_intermediate_size * model.n_shared_experts,
        )?),
        router: Glm53TritonMoeRouterLocalWorkletConfig {
            hidden: model.hidden.into(),
            num_experts: model.n_routed_experts.into(),
            top_k: model.num_experts_per_tok,
            activation_dtype: ACTIVATION_DTYPE,
            gpu_name: gpu.clone(),
            gemm_backends: ROUTER_GEMM_BACKENDS.to_vec(),
            elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
        },
        routed_body: routed(&demand.body),
        routed_mtp_step0: routed(&demand.mtp_step0),
        routed_mtp_recurrent: routed(&demand.mtp_recurrent),
        // Row gather: reads and writes one hidden row per token.
        embedding: ew(hidden_bytes, hidden_bytes),
        hc_expand: ew(hidden_bytes, stream_bytes),
        mhc: MhcRmsNormKernelConfig {
            backends: MHC_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            hidden_size: model.hidden.into(),
            hc_mult: model.hc_mult,
            hidden_dtype: ACTIVATION_DTYPE,
        },
        all_reduce_fusion: AllReduceFusionKernelConfig {
            backends: ALL_REDUCE_FUSION_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            num_gpus: tp,
            hidden_dim: model.hidden,
            dtype: ACTIVATION_DTYPE,
            fabric: Fabric::Nvlink,
        },
        all_reduce: AllReduceKernelConfig {
            backends: ALL_REDUCE_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            num_gpus: tp,
            fabric: Fabric::Nvlink,
        },
        moe_combine_glue: ew(2 * hidden_bytes, hidden_bytes),
        hc_contract_mean: ew(stream_bytes, hidden_bytes),
        final_norm: RmsNormKernelConfig {
            backends: RMS_NORM_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            hidden: model.hidden.into(),
            dtype: ACTIVATION_DTYPE,
        },
        lm_head: SingleGemmKernelConfig {
            backends: BF16_GEMM_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            n: Dim::param("vocab_per_rank", vocab_per_rank),
            k: model.hidden.into(),
            dtype: ACTIVATION_DTYPE,
        },
        // enorm(embeds) + hnorm(prev hidden), concatenated: two hidden rows
        // in, one 2H row out.
        mtp_eh_norm: ew(2 * hidden_bytes, 2 * hidden_bytes),
        // `nn.Linear(2H, H)`: replicated on every rank.
        mtp_eh_proj: SingleGemmKernelConfig {
            backends: BF16_GEMM_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            n: model.hidden.into(),
            k: (2 * model.hidden).into(),
            dtype: ACTIVATION_DTYPE,
        },
        mtp_residual_norm: ResidualRmsNormKernelConfig {
            backends: RESIDUAL_NORM_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            hidden: model.hidden.into(),
            dtype: ACTIVATION_DTYPE,
        },
        // Placeholder: one hidden row gathered per sampled request.
        mtp_row_select: ew(hidden_bytes, hidden_bytes),
        // Placeholder for `ncclAllGather` of the fp32 vocab shard: each rank
        // contributes its shard and receives the full row.
        mtp_logits_gather: ew(vocab_per_rank * fp32, model.vocab_size * fp32),
        // Placeholder: argmax over one full fp32 logits row.
        mtp_argmax: ew(model.vocab_size * fp32, 8),
        mtp_step_glue: ew(DRAFT_STEP_GLUE_BYTES_PER_ROW, DRAFT_STEP_GLUE_BYTES_PER_ROW),
        model: model.clone(),
        parallel: parallel.clone(),
    })
}

#[derive(Clone, Debug)]
pub struct Glm53FlashSpecResolved {
    pub raw_cfg: Glm53FlashSpecConfigs,
    pub kda: Glm53KdaSpecAttnLocalWorkletResolved,
    pub dsa: Glm53DsaSpecAttnLocalWorkletResolved,
    pub dense_ffn: Glm53Fp8MlpLocalWorkletResolved,
    pub shared_expert: Glm53Fp8MlpLocalWorkletResolved,
    pub router: Glm53TritonMoeRouterLocalWorkletResolved,
    pub routed_body: Glm53TritonMoeTpLocalWorkletResolved,
    pub routed_mtp_step0: Glm53TritonMoeTpLocalWorkletResolved,
    pub routed_mtp_recurrent: Glm53TritonMoeTpLocalWorkletResolved,
}

pub fn resolve_configs(cfgs: &Glm53FlashSpecConfigs) -> Glm53FlashSpecResolved {
    Glm53FlashSpecResolved {
        kda: Glm53KdaSpecAttnLocalWorklet::resolve_config(&cfgs.kda),
        dsa: Glm53DsaSpecAttnLocalWorklet::resolve_config(&cfgs.dsa),
        dense_ffn: Glm53Fp8MlpLocalWorklet::resolve_config(&cfgs.dense_ffn),
        shared_expert: Glm53Fp8MlpLocalWorklet::resolve_config(&cfgs.shared_expert),
        router: Glm53TritonMoeRouterLocalWorklet::resolve_config(&cfgs.router),
        routed_body: Glm53TritonMoeTpLocalWorklet::resolve_config(&cfgs.routed_body),
        routed_mtp_step0: Glm53TritonMoeTpLocalWorklet::resolve_config(&cfgs.routed_mtp_step0),
        routed_mtp_recurrent: Glm53TritonMoeTpLocalWorklet::resolve_config(
            &cfgs.routed_mtp_recurrent,
        ),
        raw_cfg: cfgs.clone(),
    }
}

/// Per-GPU bytes one cached token occupies in an attention page: the BF16
/// MLA latent (no rope part) plus the kpool index cache (one 128-byte FP8
/// key and a 4-byte scale per 4-token pool), over the 11 DSA layers and the
/// MTP layer. The latent is replicated on every TP rank.
pub fn kv_bytes_per_token_per_gpu(model: &Glm53FlashModelCfg) -> u64 {
    let mla_layers = u64::from(model.num_dsa_layers()) + 1;
    let latent = u64::from(model.kv_lora_rank) * KV_CACHE_DTYPE.size_bytes() as u64;
    let index = u64::from(model.index_head_dim + 4) / u64::from(model.index_kpool);
    mla_layers * (latent + index)
}

/// Pages of recurrent state vLLM pins per request: KDA layers are packed into
/// groups the size of the attention group (12 layers), each holding
/// `2 + draft_tokens` pages in `align` mode, plus the kpool tail block.
pub fn state_pages_per_request(model: &Glm53FlashModelCfg, draft_tokens: u32) -> u32 {
    let attention_layers = model.num_dsa_layers() + 1;
    let kda_groups = (model.kda_layers.len() as u32).div_ceil(attention_layers);
    kda_groups * (ALIGN_STATE_PAGES + draft_tokens) + KPOOL_TAIL_PAGES
}

/// One TP all-reduce: FlashInfer up to its fusion workspace, a plain
/// all-reduce above it. Both leaves are compiled; one is filled per call.
struct TpAllReduce {
    fusion: Op<AllReduceFusionKernel>,
    fallback: Op<AllReduceKernel>,
    max_fused_tokens: u32,
    bytes_per_token: u64,
}

impl TpAllReduce {
    fn build(
        prefix: &str,
        suffix: &str,
        cfg: &Glm53FlashSpecConfigs,
        bridge: &PerfApiBridge,
    ) -> std::result::Result<Self, BuildError> {
        Ok(Self {
            fusion: atomic(
                prefix,
                &format!("{suffix}.fusion"),
                cfg.all_reduce_fusion.clone(),
                AllReduceFusionKernel::build,
                bridge,
            )?,
            fallback: atomic(
                prefix,
                &format!("{suffix}.fallback"),
                cfg.all_reduce.clone(),
                AllReduceKernel::build,
                bridge,
            )?,
            max_fused_tokens: AllReduceFusionSpec::max_fused_tokens(&cfg.all_reduce_fusion),
            bytes_per_token: u64::from(cfg.model.hidden) * ACTIVATION_DTYPE.size_bytes() as u64,
        })
    }

    fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Sum(vec![
            self.fusion.compile(builder),
            self.fallback.compile(builder),
        ])
    }

    fn eval(&self, num_tokens: u32, ev: &mut Evaluator) {
        let fused = num_tokens <= self.max_fused_tokens;
        let fusion_input = AllReduceFusionKernelInput { num_tokens };
        let fallback_input = AllReduceKernelInput {
            message_size_bytes: u64::from(num_tokens) * self.bytes_per_token,
        };
        let fusion = if fused {
            self.fusion.kernel.eval(&fusion_input)
        } else {
            LeafMetrics::ZERO
        };
        ev.push(fusion, || fusion_input.clone().into());
        let fallback = if fused {
            LeafMetrics::ZERO
        } else {
            self.fallback.kernel.eval(&fallback_input)
        };
        ev.push(fallback, || fallback_input.clone().into());
    }
}

/// Router + routed experts (aux stream at decode sizes) and the shared expert
/// (main stream), then the combine.
struct MoeBlock {
    name: String,
    router: Glm53TritonMoeRouterLocalWorklet,
    shared_expert: Glm53Fp8MlpLocalWorklet,
    routed: Glm53TritonMoeTpLocalWorklet,
    combine_glue: Op<ElementwiseKernel>,
}

impl MoeBlock {
    fn build(
        name: String,
        resolved: &Glm53FlashSpecResolved,
        routed: &Glm53TritonMoeTpLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> std::result::Result<Self, BuildError> {
        let m = name.as_str();
        Ok(Self {
            router: Glm53TritonMoeRouterLocalWorklet::build(
                format!("{m}.router"),
                resolved.router.clone(),
                bridge,
            )?,
            shared_expert: Glm53Fp8MlpLocalWorklet::build(
                format!("{m}.shared_expert"),
                resolved.shared_expert.clone(),
                bridge,
            )?,
            routed: Glm53TritonMoeTpLocalWorklet::build(
                format!("{m}.routed"),
                routed.clone(),
                bridge,
            )?,
            combine_glue: atomic(
                m,
                "combine_glue",
                resolved.raw_cfg.moe_combine_glue.clone(),
                ElementwiseKernel::build,
                bridge,
            )?,
            name,
        })
    }

    /// The aux-stream pair and the serial copies; one of the two is filled
    /// per call (INV-1 fixes the shape).
    fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let concurrent_shared = self.shared_expert.compile(builder);
        let concurrent_router = self.router.compile(builder);
        let concurrent_routed = self.routed.compile(builder);
        let serial = vec![
            self.router.compile(builder),
            self.shared_expert.compile(builder),
            self.routed.compile(builder),
        ];
        let combine_glue = self.combine_glue.compile(builder);
        CostNode::Labeled {
            label: format!(
                "{} (MoE) [TP; all experts per rank; shared on main stream, router+routed on \
                 aux stream at T<={}]",
                self.name, SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
            ),
            child: Box::new(CostNode::Sum(vec![
                CostNode::Parallel {
                    overlap: SHARED_EXPERTS_STREAM_OVERLAP,
                    children: vec![
                        concurrent_shared,
                        CostNode::Sum(vec![concurrent_router, concurrent_routed]),
                    ],
                },
                CostNode::Sum(serial),
                combine_glue,
            ])),
        }
    }

    fn eval(&self, num_tokens: u32, ev: &mut Evaluator) {
        let overlapped = num_tokens <= SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD;
        let (concurrent, serial) = if overlapped {
            (num_tokens, 0)
        } else {
            (0, num_tokens)
        };
        let shared = Glm53Fp8MlpLocalWorkletInput { num_tokens };
        self.shared_expert.eval_or_zero(&shared, !overlapped, ev);
        self.router.eval(
            &Glm53TritonMoeRouterLocalWorkletInput {
                num_tokens: concurrent,
            },
            ev,
        );
        self.routed.eval(
            &Glm53TritonMoeTpLocalWorkletInput {
                num_tokens: concurrent,
            },
            ev,
        );
        self.router.eval(
            &Glm53TritonMoeRouterLocalWorkletInput { num_tokens: serial },
            ev,
        );
        self.shared_expert.eval_or_zero(&shared, overlapped, ev);
        self.routed.eval(
            &Glm53TritonMoeTpLocalWorkletInput { num_tokens: serial },
            ev,
        );
        push(
            &self.combine_glue,
            ElementwiseKernelInput { num_tokens },
            ev,
        );
    }
}

enum Attention {
    Kda(Glm53KdaSpecAttnLocalWorklet),
    Dsa(Glm53DsaSpecAttnLocalWorklet),
}

enum Ffn {
    Dense(Glm53Fp8MlpLocalWorklet),
    Moe(MoeBlock),
}

struct LayerGroup {
    label: String,
    layers: Vec<u32>,
    attn_boundary: Boundary,
    attn: Attention,
    attn_all_reduce: TpAllReduce,
    ffn_boundary: Op<MhcFusedPostPreRmsNormKernel>,
    ffn: Ffn,
    ffn_all_reduce: TpAllReduce,
}

impl LayerGroup {
    fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let body = CostNode::Sum(vec![
            self.attn_boundary.compile(builder),
            match &self.attn {
                Attention::Kda(worklet) => worklet.compile(builder),
                Attention::Dsa(worklet) => worklet.compile(builder),
            },
            self.attn_all_reduce.compile(builder),
            self.ffn_boundary.compile(builder),
            match &self.ffn {
                Ffn::Dense(worklet) => worklet.compile(builder),
                Ffn::Moe(block) => block.compile(builder),
            },
            self.ffn_all_reduce.compile(builder),
        ]);
        CostNode::Labeled {
            label: format!("{} layers {:?}", self.label, self.layers),
            child: Box::new(CostNode::Scale {
                n: self.layers.len() as u32,
                child: Box::new(body),
            }),
        }
    }

    fn eval(&self, batch: &NormalizedBatch, tokens: u32, ev: &mut Evaluator) {
        self.attn_boundary.eval(tokens, ev);
        match &self.attn {
            Attention::Kda(worklet) => worklet.eval(&batch.kda, ev),
            Attention::Dsa(worklet) => worklet.eval(&batch.dsa, ev),
        }
        self.attn_all_reduce.eval(tokens, ev);
        push(
            &self.ffn_boundary,
            MhcRmsNormKernelInput { num_tokens: tokens },
            ev,
        );
        match &self.ffn {
            Ffn::Dense(worklet) => {
                worklet.eval(&Glm53Fp8MlpLocalWorkletInput { num_tokens: tokens }, ev)
            }
            Ffn::Moe(block) => block.eval(tokens, ev),
        }
        self.ffn_all_reduce.eval(tokens, ev);
    }
}

/// One forward through the MTP layer and its sampling head.
struct MtpPass {
    step_glue: Op<ElementwiseKernel>,
    embedding: Op<ElementwiseKernel>,
    embedding_all_reduce: TpAllReduce,
    eh_norm: Op<ElementwiseKernel>,
    eh_proj: Op<SingleGemmKernel>,
    input_norm: Op<RmsNormKernel>,
    attn: Glm53DsaSpecAttnLocalWorklet,
    attn_all_reduce: TpAllReduce,
    post_attn_norm: Op<ResidualRmsNormKernel>,
    moe: MoeBlock,
    ffn_all_reduce: TpAllReduce,
    head_norm: Op<ResidualRmsNormKernel>,
    row_select: Op<ElementwiseKernel>,
    lm_head: Op<SingleGemmKernel>,
    logits_gather: Op<ElementwiseKernel>,
    argmax: Op<ElementwiseKernel>,
}

/// The shapes one MTP pass runs at.
struct MtpPassInput<'a> {
    /// Rows the layer forwards (padded where the drafter replays a graph).
    rows: u32,
    attn: &'a Glm53DsaSpecAttnLocalWorkletInput,
    /// Requests the head samples one candidate for.
    sample_rows: u32,
}

impl MtpPass {
    fn build(
        name: &str,
        resolved: &Glm53FlashSpecResolved,
        routed: &Glm53TritonMoeTpLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> std::result::Result<Self, BuildError> {
        let cfg = &resolved.raw_cfg;
        Ok(Self {
            step_glue: atomic(
                name,
                "step_glue",
                cfg.mtp_step_glue.clone(),
                ElementwiseKernel::build,
                bridge,
            )?,
            embedding: atomic(
                name,
                "embedding",
                cfg.embedding.clone(),
                ElementwiseKernel::build,
                bridge,
            )?,
            embedding_all_reduce: TpAllReduce::build(name, "embedding_all_reduce", cfg, bridge)?,
            eh_norm: atomic(
                name,
                "eh_norm",
                cfg.mtp_eh_norm.clone(),
                ElementwiseKernel::build,
                bridge,
            )?,
            eh_proj: atomic(
                name,
                "eh_proj",
                cfg.mtp_eh_proj.clone(),
                SingleGemmKernel::build,
                bridge,
            )?,
            input_norm: atomic(
                name,
                "input_norm",
                cfg.final_norm.clone(),
                RmsNormKernel::build,
                bridge,
            )?,
            attn: Glm53DsaSpecAttnLocalWorklet::build(
                format!("{name}.dsa"),
                resolved.dsa.clone(),
                bridge,
            )?,
            attn_all_reduce: TpAllReduce::build(name, "attn_all_reduce", cfg, bridge)?,
            post_attn_norm: atomic(
                name,
                "post_attn_norm",
                cfg.mtp_residual_norm.clone(),
                ResidualRmsNormKernel::build,
                bridge,
            )?,
            moe: MoeBlock::build(format!("{name}.moe"), resolved, routed, bridge)?,
            ffn_all_reduce: TpAllReduce::build(name, "ffn_all_reduce", cfg, bridge)?,
            head_norm: atomic(
                name,
                "head_norm",
                cfg.mtp_residual_norm.clone(),
                ResidualRmsNormKernel::build,
                bridge,
            )?,
            row_select: atomic(
                name,
                "row_select",
                cfg.mtp_row_select.clone(),
                ElementwiseKernel::build,
                bridge,
            )?,
            lm_head: atomic(
                name,
                "lm_head",
                cfg.lm_head.clone(),
                SingleGemmKernel::build,
                bridge,
            )?,
            logits_gather: atomic(
                name,
                "logits_gather",
                cfg.mtp_logits_gather.clone(),
                ElementwiseKernel::build,
                bridge,
            )?,
            argmax: atomic(
                name,
                "argmax",
                cfg.mtp_argmax.clone(),
                ElementwiseKernel::build,
                bridge,
            )?,
        })
    }

    fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Sum(vec![
            CostNode::Scale {
                n: DRAFT_STEP_GLUE_LAUNCHES,
                child: Box::new(self.step_glue.compile(builder)),
            },
            self.embedding.compile(builder),
            self.embedding_all_reduce.compile(builder),
            self.eh_norm.compile(builder),
            self.eh_proj.compile(builder),
            self.input_norm.compile(builder),
            self.attn.compile(builder),
            self.attn_all_reduce.compile(builder),
            self.post_attn_norm.compile(builder),
            self.moe.compile(builder),
            self.ffn_all_reduce.compile(builder),
            self.head_norm.compile(builder),
            self.row_select.compile(builder),
            self.lm_head.compile(builder),
            self.logits_gather.compile(builder),
            self.argmax.compile(builder),
        ])
    }

    fn eval(&self, input: &MtpPassInput<'_>, ev: &mut Evaluator) {
        let rows = input.rows;
        let sample = input.sample_rows;
        let ew = |num_tokens| ElementwiseKernelInput { num_tokens };
        push(&self.step_glue, ew(sample), ev);
        push(&self.embedding, ew(rows), ev);
        self.embedding_all_reduce.eval(rows, ev);
        push(&self.eh_norm, ew(rows), ev);
        push(&self.eh_proj, SingleGemmKernelInput { m: rows }, ev);
        push(&self.input_norm, RmsNormKernelInput { m: rows }, ev);
        self.attn.eval(input.attn, ev);
        self.attn_all_reduce.eval(rows, ev);
        push(
            &self.post_attn_norm,
            ResidualRmsNormKernelInput { m: rows },
            ev,
        );
        self.moe.eval(rows, ev);
        self.ffn_all_reduce.eval(rows, ev);
        push(&self.head_norm, ResidualRmsNormKernelInput { m: rows }, ev);
        push(&self.row_select, ew(sample), ev);
        push(&self.lm_head, SingleGemmKernelInput { m: sample }, ev);
        push(&self.logits_gather, ew(sample), ev);
        push(&self.argmax, ew(sample), ev);
    }
}

pub struct Glm53FlashSpecModel {
    pub name: String,
    pub tp_size: u16,
    pub max_model_len: u32,
    pub draft_tokens: u32,
    cudagraph_capture_sizes: Vec<u32>,
    embedding: Op<ElementwiseKernel>,
    embedding_all_reduce: TpAllReduce,
    hc_expand: Op<ElementwiseKernel>,
    groups: Vec<LayerGroup>,
    terminal_post: MhcTerminalPostOp,
    hc_contract_mean: Op<ElementwiseKernel>,
    final_norm: Op<RmsNormKernel>,
    lm_head: Op<SingleGemmKernel>,
    mtp_step0: MtpPass,
    /// Draft steps 1..K, folded: same launch graph, one row per request.
    mtp_recurrent: Option<MtpPass>,
    total_kv_bytes_per_token: u64,
    recurrent_state_bytes_per_request: u64,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

pub fn build(
    name: String,
    resolved: Glm53FlashSpecResolved,
    bridge: &PerfApiBridge,
) -> std::result::Result<Glm53FlashSpecModel, BuildError> {
    let cfg = &resolved.raw_cfg;
    let n = name.as_str();
    let mut groups = Vec::with_capacity(cfg.groups.len());
    for group in &cfg.groups {
        let prefix = format!("{n}.{}", group.label);
        let p = prefix.as_str();
        let attn_boundary = if group.opens_stream {
            Boundary::Pre(atomic(
                p,
                "attn_mhc_pre",
                cfg.mhc.clone(),
                MhcPreRmsNormKernel::build,
                bridge,
            )?)
        } else {
            Boundary::Fused(atomic(
                p,
                "attn_mhc_post_pre",
                cfg.mhc.clone(),
                MhcFusedPostPreRmsNormKernel::build,
                bridge,
            )?)
        };
        let attn = match group.attn {
            AttnKind::Kda => Attention::Kda(Glm53KdaSpecAttnLocalWorklet::build(
                format!("{p}.kda"),
                resolved.kda.clone(),
                bridge,
            )?),
            AttnKind::Dsa => Attention::Dsa(Glm53DsaSpecAttnLocalWorklet::build(
                format!("{p}.dsa"),
                resolved.dsa.clone(),
                bridge,
            )?),
        };
        let ffn = match group.ffn {
            FfnKind::Dense => Ffn::Dense(Glm53Fp8MlpLocalWorklet::build(
                format!("{p}.dense_ffn"),
                resolved.dense_ffn.clone(),
                bridge,
            )?),
            FfnKind::Moe => Ffn::Moe(MoeBlock::build(
                format!("{p}.moe"),
                &resolved,
                &resolved.routed_body,
                bridge,
            )?),
        };
        groups.push(LayerGroup {
            label: group.label.clone(),
            layers: group.layers.clone(),
            attn_boundary,
            attn,
            attn_all_reduce: TpAllReduce::build(p, "attn_all_reduce", cfg, bridge)?,
            ffn_boundary: atomic(
                p,
                "ffn_mhc_post_pre",
                cfg.mhc.clone(),
                MhcFusedPostPreRmsNormKernel::build,
                bridge,
            )?,
            ffn,
            ffn_all_reduce: TpAllReduce::build(p, "ffn_all_reduce", cfg, bridge)?,
        });
    }
    let model_cfg = &cfg.model;
    let tp = u64::from(cfg.parallel.tp_size);
    let draft_tokens = cfg.parallel.draft_tokens;
    let per_gpu_token = kv_bytes_per_token_per_gpu(model_cfg);
    let page_bytes_per_gpu = u64::from(HYBRID_BLOCK_SIZE) * per_gpu_token;
    let total_kv_bytes_per_token = tp * per_gpu_token;
    let recurrent_state_bytes_per_request =
        tp * u64::from(state_pages_per_request(model_cfg, draft_tokens)) * page_bytes_per_gpu;
    let mtp_step0 = MtpPass::build(
        &format!("{n}.mtp.step_0"),
        &resolved,
        &resolved.routed_mtp_step0,
        bridge,
    )?;
    let mtp_recurrent = if draft_tokens > 1 {
        Some(MtpPass::build(
            &format!("{n}.mtp.recurrent"),
            &resolved,
            &resolved.routed_mtp_recurrent,
            bridge,
        )?)
    } else {
        None
    };
    let mut model = Glm53FlashSpecModel {
        embedding: atomic(
            n,
            "embedding",
            cfg.embedding.clone(),
            ElementwiseKernel::build,
            bridge,
        )?,
        embedding_all_reduce: TpAllReduce::build(n, "embedding_all_reduce", cfg, bridge)?,
        hc_expand: atomic(
            n,
            "hc_expand",
            cfg.hc_expand.clone(),
            ElementwiseKernel::build,
            bridge,
        )?,
        groups,
        terminal_post: MhcTerminalPostOp::build(
            format!("{n}.final_mhc_post"),
            MhcTerminalPostConfig {
                mhc: cfg.mhc.clone(),
                pre_backends: MHC_BACKENDS.to_vec(),
                fused_backends: MHC_BACKENDS.to_vec(),
            },
            bridge,
        )?,
        hc_contract_mean: atomic(
            n,
            "hc_contract_mean",
            cfg.hc_contract_mean.clone(),
            ElementwiseKernel::build,
            bridge,
        )?,
        final_norm: atomic(
            n,
            "final_norm",
            cfg.final_norm.clone(),
            RmsNormKernel::build,
            bridge,
        )?,
        lm_head: atomic(
            n,
            "lm_head",
            cfg.lm_head.clone(),
            SingleGemmKernel::build,
            bridge,
        )?,
        mtp_step0,
        mtp_recurrent,
        tp_size: cfg.parallel.tp_size,
        max_model_len: cfg.parallel.max_model_len,
        draft_tokens,
        cudagraph_capture_sizes: {
            let mut sizes = cfg.parallel.cudagraph_capture_sizes.clone();
            sizes.sort_unstable();
            sizes.dedup();
            sizes
        },
        total_kv_bytes_per_token,
        recurrent_state_bytes_per_request,
        cost_flat: Vec::new(),
        n_slots: 0,
        name,
    };
    let tree = model.cost_tree();
    model.cost_flat = tree.flatten();
    model.n_slots = tree.n_slots();
    Ok(model)
}

impl Glm53FlashSpecModel {
    pub fn cost_tree(&self) -> CostTree {
        let mut builder = CostTreeBuilder::new();
        let mut target = vec![
            self.embedding.compile(&mut builder),
            self.embedding_all_reduce.compile(&mut builder),
            self.hc_expand.compile(&mut builder),
        ];
        for group in &self.groups {
            target.push(group.compile(&mut builder));
        }
        target.extend([
            self.terminal_post.compile(&mut builder),
            self.hc_contract_mean.compile(&mut builder),
            self.final_norm.compile(&mut builder),
            self.lm_head.compile(&mut builder),
        ]);
        let mut draft = vec![CostNode::Labeled {
            label: format!(
                "{}.mtp.step_0 [all target rows; builds the index state]",
                self.name
            ),
            child: Box::new(self.mtp_step0.compile(&mut builder)),
        }];
        if let Some(recurrent) = &self.mtp_recurrent {
            draft.push(CostNode::Labeled {
                label: format!(
                    "{}.mtp.recurrent [one row per request; index shared; {} steps]",
                    self.name,
                    self.draft_tokens - 1
                ),
                child: Box::new(CostNode::Scale {
                    n: self.draft_tokens - 1,
                    child: Box::new(recurrent.compile(&mut builder)),
                }),
            });
        }
        let root = CostNode::Labeled {
            label: format!(
                "{} (Glm53FlashSpecModel) [TP{} no EP; MTP K={}; timing_context<={}]",
                self.name, self.tp_size, self.draft_tokens, self.max_model_len
            ),
            child: Box::new(CostNode::Sum(vec![
                CostNode::Labeled {
                    label: format!("{}.verify [target forward, K+1 rows per decode]", self.name),
                    child: Box::new(CostNode::Sum(target)),
                },
                CostNode::Labeled {
                    label: format!(
                        "{}.mtp [MTP layer {} draft steps]",
                        self.name, self.draft_tokens
                    ),
                    child: Box::new(CostNode::Sum(draft)),
                },
            ])),
        };
        builder.finish(root)
    }

    fn eval_into(&self, input: &SpeculativeArchInput, ev: &mut Evaluator) {
        let batch = normalize_speculative_input(input, self.draft_tokens, self.max_model_len)
            .unwrap_or_else(|reason| panic!("invalid Glm53FlashSpecModel input: {reason}"));
        let tokens = graph_padded_tokens(&self.cudagraph_capture_sizes, batch.total_tokens);
        let ew = |num_tokens| ElementwiseKernelInput { num_tokens };
        push(&self.embedding, ew(tokens), ev);
        self.embedding_all_reduce.eval(tokens, ev);
        push(&self.hc_expand, ew(tokens), ev);
        for group in &self.groups {
            group.eval(&batch, tokens, ev);
        }
        self.terminal_post
            .eval(&MhcTerminalPostInput { num_tokens: tokens }, ev);
        push(&self.hc_contract_mean, ew(tokens), ev);
        push(&self.final_norm, RmsNormKernelInput { m: tokens }, ev);
        push(
            &self.lm_head,
            SingleGemmKernelInput {
                m: batch.logits_rows,
            },
            ev,
        );
        // Step 0 forwards every target row so the MTP layer fills its own
        // attention state; only the request endpoints are sampled.
        self.mtp_step0.eval(
            &MtpPassInput {
                rows: tokens,
                attn: &batch.dsa,
                sample_rows: batch.request_count,
            },
            ev,
        );
        if let Some(recurrent) = &self.mtp_recurrent {
            // The fold bills one evaluation `K - 1` times, at the mean context
            // advance of steps 1..K-1 (as the GLM-5.2 MTP stage).
            let steps = self.draft_tokens - 1;
            let advance = (steps + 1).div_ceil(2);
            let contexts: Vec<u32> = batch
                .endpoint_contexts
                .iter()
                .map(|&context| context.saturating_add(advance).min(self.max_model_len))
                .collect();
            let endpoints = Glm53DsaSpecAttnLocalWorkletInput {
                prefill_chunk_pairs: Vec::new(),
                decode_kv_lens: contexts,
                decode_width: 1,
                skip_indexer: true,
            };
            let rows = graph_padded_tokens(&self.cudagraph_capture_sizes, batch.request_count);
            recurrent.eval(
                &MtpPassInput {
                    rows,
                    attn: &endpoints,
                    sample_rows: batch.request_count,
                },
                ev,
            );
        }
    }

    fn eval_checked(
        &self,
        batch: &SpeculativeArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: Option<&mut Vec<SlotInput>>,
    ) -> LeafMetrics {
        slots.clear();
        slots.resize(self.n_slots, LeafMetrics::ZERO);
        let mut evaluator = match inputs {
            Some(inputs) => Evaluator::with_inputs(slots, inputs),
            None => Evaluator::new(slots),
        };
        self.eval_into(batch, &mut evaluator);
        assert_eq!(
            evaluator.filled(),
            self.n_slots,
            "eval must fill every compiled slot"
        );
        drop(evaluator);
        CostTree::aggregate(&self.cost_flat, slots, scratch)
    }
}

impl SpeculativeUnifiedModel for Glm53FlashSpecModel {
    fn check_input(&self, batch: &SpeculativeArchInput) -> Result<(), String> {
        normalize_speculative_input(batch, self.draft_tokens, self.max_model_len).map(|_| ())
    }

    fn eval_speculative_iter(
        &self,
        batch: &SpeculativeArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics {
        self.eval_checked(batch, slots, scratch, None)
    }

    fn eval_speculative_iter_with_inputs(
        &self,
        batch: &SpeculativeArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        let total = self.eval_checked(batch, slots, scratch, Some(inputs));
        assert_eq!(
            inputs.len(),
            self.n_slots,
            "slot inputs must align with compiled slots"
        );
        total
    }

    fn cost_log_manifest(&self) -> CostManifest {
        self.cost_tree().manifest()
    }

    fn total_kv_bytes_per_token(&self) -> u64 {
        self.total_kv_bytes_per_token
    }

    fn max_model_len(&self) -> u32 {
        self.max_model_len
    }

    fn gpus_per_replica(&self) -> u16 {
        self.tp_size
    }

    fn num_attn_shards(&self) -> u16 {
        self.tp_size
    }

    /// The 22 pinned state pages (all ranks): KDA groups x (2 + K) + tail.
    fn recurrent_state_bytes_per_request(&self) -> u64 {
        self.recurrent_state_bytes_per_request
    }

    /// The hybrid page: KDA checkpoint interval and prefix-reuse quantum.
    fn recurrent_checkpoint_interval_tokens(&self) -> u32 {
        HYBRID_BLOCK_SIZE
    }
}

/// One verify forward, as every layer sees it (one TP group).
struct NormalizedBatch {
    total_tokens: u32,
    request_count: u32,
    /// Rows the target head scores: every verify row plus one per prefill.
    logits_rows: u32,
    /// Each request's context after this step, for the draft endpoints.
    endpoint_contexts: Vec<u32>,
    kda: Glm53KdaSpecAttnLocalWorkletInput,
    dsa: Glm53DsaSpecAttnLocalWorkletInput,
}

fn normalize_speculative_input(
    input: &SpeculativeArchInput,
    draft_tokens: u32,
    max_model_len: u32,
) -> std::result::Result<NormalizedBatch, String> {
    if input.draft_tokens != draft_tokens {
        return Err(format!(
            "input draft_tokens {} must match model draft_tokens {draft_tokens}",
            input.draft_tokens
        ));
    }
    let [group] = input.groups.as_slice() else {
        return Err(format!(
            "expected exactly one tensor-parallel attention group, got {}",
            input.groups.len()
        ));
    };
    let verify_width = draft_tokens + 1;
    let mut prefill_tokens = 0_u32;
    let mut endpoint_contexts =
        Vec::with_capacity(group.prefill_chunk_pairs.len() + group.decode_requests.len());
    for (request, &(prefix, append)) in group.prefill_chunk_pairs.iter().enumerate() {
        if append == 0 {
            return Err(format!("prefill request {request} append must be nonzero"));
        }
        let context = prefix
            .checked_add(append)
            .ok_or_else(|| format!("prefill request {request} context overflows u32"))?;
        if context > max_model_len {
            return Err(format!(
                "prefill request {request} context {context} exceeds max_model_len {max_model_len}"
            ));
        }
        prefill_tokens = prefill_tokens
            .checked_add(append)
            .ok_or("prefill token sum overflows u32")?;
        endpoint_contexts.push(context);
    }
    if group.prefill_tokens != prefill_tokens {
        return Err(format!(
            "prefill_tokens {} must equal append sum {prefill_tokens}",
            group.prefill_tokens
        ));
    }
    let mut decode_tokens = 0_u32;
    let mut decode_kv_lens = Vec::with_capacity(group.decode_requests.len());
    for (request, decode) in group.decode_requests.iter().enumerate() {
        // vLLM schedules whole verify groups; acceptance truncates the output
        // after the compute, not the compute.
        if decode.query_len != verify_width {
            return Err(format!(
                "decode request {request} query_len {} must equal verify width {verify_width}",
                decode.query_len
            ));
        }
        if !(verify_width..=max_model_len).contains(&decode.kv_len) {
            return Err(format!(
                "decode request {request} kv_len {} must be in {verify_width}..={max_model_len}",
                decode.kv_len
            ));
        }
        decode_tokens += decode.query_len;
        decode_kv_lens.push(decode.kv_len);
        endpoint_contexts.push(decode.kv_len);
    }
    if group.decode_tokens != decode_tokens {
        return Err(format!(
            "decode_tokens {} must equal query_len sum {decode_tokens}",
            group.decode_tokens
        ));
    }
    let total_tokens = prefill_tokens
        .checked_add(decode_tokens)
        .ok_or("token sum overflows u32")?;
    if group.batch_tokens != total_tokens {
        return Err(format!(
            "batch_tokens {} must equal prefill+decode {total_tokens}",
            group.batch_tokens
        ));
    }
    if total_tokens == 0 {
        return Err("an iteration must carry at least one token".into());
    }
    let prefill_requests = group.prefill_chunk_pairs.len() as u32;
    let decode_requests = group.decode_requests.len() as u32;
    Ok(NormalizedBatch {
        total_tokens,
        request_count: prefill_requests + decode_requests,
        logits_rows: prefill_requests + decode_tokens,
        endpoint_contexts,
        kda: Glm53KdaSpecAttnLocalWorkletInput {
            prefill_sequence_lengths: group
                .prefill_chunk_pairs
                .iter()
                .map(|&(_, append)| append)
                .collect(),
            verify_requests: decode_requests,
            verify_width,
        },
        dsa: Glm53DsaSpecAttnLocalWorkletInput {
            prefill_chunk_pairs: group.prefill_chunk_pairs.clone(),
            decode_kv_lens,
            decode_width: verify_width,
            skip_indexer: false,
        },
    })
}

fn fit_failed(reason: impl Into<String>) -> BuildError {
    BuildError::FitFailed {
        kind: ARCH_KIND,
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::arch::contract::{SpeculativeArchGroupInput, SpeculativeDecodeInput};
    use crate::timing::routing::RoutingDistribution;

    fn model_cfg() -> Glm53FlashModelCfg {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("model/config/glm53_flash.json");
        Glm53FlashModelCfg::from_json(&path).unwrap()
    }

    fn configs() -> Glm53FlashSpecConfigs {
        let uniform = ExpertDemand::popularity(&RoutingDistribution::uniform(288), 43);
        build_configs(
            &model_cfg(),
            &Glm53FlashSpecParallel {
                tp_size: 4,
                max_model_len: 8192,
                gpu_name: "NVIDIA H200".into(),
                cudagraph_capture_sizes: vec![1, 2, 4, 6, 8, 12, 16, 24, 32, 512],
                draft_tokens: 5,
            },
            &Glm53FlashSpecDemand {
                body: uniform.clone(),
                mtp_step0: uniform.clone(),
                mtp_recurrent: uniform,
            },
        )
        .unwrap()
    }

    /// The H200 server logs: a 42.79 GiB per-GPU pool held 3144 pages of 1152
    /// tokens, and 858,521 "tokens" = 104.8 requests x 8192 because every
    /// request also pins 22 state pages; at 37.34 GiB, 2744 pages and
    /// 2.941 x 1,048,576 = 3,083,914.
    #[test]
    fn kv_and_state_pages_reproduce_the_vllm_capacity_lines() {
        let model = model_cfg();
        assert_eq!(kv_bytes_per_token_per_gpu(&model), 12 * (1024 + 33));
        assert_eq!(state_pages_per_request(&model, 5), 22);
        assert_eq!(state_pages_per_request(&model, 0), 7);
        let page = u64::from(HYBRID_BLOCK_SIZE) * kv_bytes_per_token_per_gpu(&model);
        // The logs print the pool rounded to 0.01 GiB, so round the page count.
        let pages = |gib: f64| (gib * f64::from(1u32 << 30) / page as f64).round() as u64;
        assert_eq!(pages(42.79), 3144);
        assert_eq!(pages(37.34), 2744);
        let concurrency = |pages: u64, max_len: u32| {
            pages as f64 / f64::from(max_len.div_ceil(HYBRID_BLOCK_SIZE) + 22)
        };
        assert_eq!((concurrency(3144, 8192) * 8192.0) as u64, 858_521);
        assert_eq!(
            (concurrency(2744, 1 << 20) * f64::from(1u32 << 20)) as u64,
            3_083_914
        );
    }

    #[test]
    fn configs_take_the_h200_tp_only_realization() {
        let c = configs();
        let r = resolve_configs(&c);
        assert_eq!(r.routed_body.gate_up.num_local_experts.get(), 288);
        assert_eq!(r.routed_body.down.k.get(), 512);
        // H200 MTP 5: the decode indexer is flattened to next_n = 1.
        assert!(r.dsa.flatten_decode_indexer);
        assert_eq!(r.dsa.mqa_logits_decode.next_n, 1);
        assert_eq!(r.dsa.mla_cache_append.kv_dtype, DType::Bf16);
        assert_eq!(
            r.shared_expert.gate_up_quant.scale_format,
            COLUMN_MAJOR_SCALE_FORMAT
        );
        assert_eq!((c.mtp_eh_proj.n.get(), c.mtp_eh_proj.k.get()), (4096, 8192));
        assert_eq!(c.lm_head.n.get(), 154_880 / 4);
    }

    fn verify_input(decodes: &[u32], prefills: &[(u32, u32)]) -> SpeculativeArchInput {
        let decode_requests: Vec<_> = decodes
            .iter()
            .map(|&kv_len| SpeculativeDecodeInput {
                kv_len,
                query_len: 6,
            })
            .collect();
        let prefill_tokens: u32 = prefills.iter().map(|&(_, append)| append).sum();
        let decode_tokens = 6 * decodes.len() as u32;
        SpeculativeArchInput {
            draft_tokens: 5,
            groups: vec![SpeculativeArchGroupInput {
                batch_tokens: prefill_tokens + decode_tokens,
                prefill_tokens,
                decode_tokens,
                prefill_chunk_pairs: prefills.to_vec(),
                decode_requests,
                total_kv_len: 0,
            }],
            tokens_per_source_rank: Vec::new(),
        }
    }

    #[test]
    fn normalization_counts_verify_rows_and_endpoints() {
        let batch = normalize_speculative_input(&verify_input(&[1006, 300], &[(0, 2048)]), 5, 8192)
            .unwrap();
        assert_eq!(batch.total_tokens, 2048 + 12);
        assert_eq!(batch.request_count, 3);
        assert_eq!(batch.logits_rows, 1 + 12);
        assert_eq!(batch.endpoint_contexts, vec![2048, 1006, 300]);
        assert_eq!(batch.kda.verify_requests, 2);
        let mut wrong = verify_input(&[1006], &[]);
        wrong.draft_tokens = 4;
        assert!(normalize_speculative_input(&wrong, 5, 8192).is_err());
        let mut ragged = verify_input(&[1006], &[]);
        ragged.groups[0].decode_requests[0].query_len = 3;
        ragged.groups[0].decode_tokens = 3;
        ragged.groups[0].batch_tokens = 3;
        assert!(normalize_speculative_input(&ragged, 5, 8192).is_err());
    }
}
