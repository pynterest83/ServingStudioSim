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
//! - KDA: verify rows take conv update + recurrent KDA as one call over the
//!   verifying requests at `query_len = K + 1`, prefill rows FlashKDA's
//!   chunked core;
//! - DSA: the kpool indexer over `K+1` rows per request (flattened to
//!   single-token rows on SM90, where DeepGEMM has no `next_n = 6`),
//!   dense-MHA prefill or sparse MQA through `FLASH_ATTN_MLA_SPARSE` (or,
//!   with `sparse_mla_backend: flashinfer_mla_sparse_sm90`, vLLM's
//!   `FLASHINFER_MLA_SPARSE_SM90`), BF16 latent cache (FP8 with
//!   `mla_kv_cache_dtype: fp8`, below);
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
//! page holds one hybrid block of attention tokens of the 12 MLA layers
//! (11 + MTP), and each request pins `kda_groups x (2 + K) + 1` pages of
//! recurrent state (Mamba `align` mode: two state pages plus one per
//! speculative token per KDA group, plus the kpool tail block). The block is
//! the smallest multiple of 128 tokens whose MLA latent page holds one KDA
//! state page ([`hybrid_block_size`]): 1152 tokens for the production BF16
//! latent and FP32 recurrent state.
//!
//! `mla_kv_cache_dtype: fp8` and `kda_state_dtype: bf16` change the latent
//! bytes per token, the KDA state page, and through them the hybrid block, the
//! page bytes, and every quantity the worker derives from the block (state and
//! checkpoint charges, the prefix reuse and chunk-end quantum).
//!
//! `mla_kv_cache_dtype: fp8` is vLLM's `--kv-cache-dtype fp8`, which on H200
//! only `FLASHINFER_MLA_SPARSE_SM90` serves (`sparse_mla_backend:
//! flashinfer_mla_sparse_sm90` is required). Its DSA layers are costed on the
//! FP8 rows: the e4m3 latent append, the FP8-cache FlashInfer call
//! (`vllm_flashinfer_sparse_sm90_fp8`) for every MQA row, and the dense
//! path's one-byte context gather. `mla_fp8_prefill_bf16_workspace: true`
//! adds the vLLM fork's `VLLM_FP8_KV_PREFILL_BF16_WORKSPACE=1`: prefill rows
//! of a sparse-MQA step run on a BF16 plan over a per-layer dequantized
//! workspace (`glm53_dsa_spec_attn_local`); it does nothing with a BF16 cache,
//! as in vLLM. `mla_fp8_flashinfer_b1: true` is the fork's
//! `VLLM_FLASHINFER_INCLUDE_OVERLAY` with the B1 FlashInfer headers: the rows
//! on the FP8 call take `vllm_flashinfer_sparse_sm90_fp8_b1` (same plan and
//! grid; only the kernel time differs). Also a no-op with a BF16 cache. The KDA kernels keep the FP32-state rows: the BF16 KDA state
//! traffic is not profiled on H200.
//!
//! `moe_flashinfer_cutlass_min_tokens: N` prices every routed MoE layer (body
//! and MTP) of at least `N` tokens as vLLM's `--moe-backend flashinfer_cutlass`
//! (FlashInfer `cutlass_fused_moe`, DeepSeek FP8 block scales, SM90) instead
//! of the Triton launches: `1` is that backend everywhere, a larger `N` the
//! per-shape hybrid (Triton under `moe_backend` below `N`).

use anyhow::Result;

use crate::arch::contract::{GraphMode, SpeculativeArchInput, SpeculativeUnifiedModel};
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
use crate::timing::kernels::dsa_sparse_mla_attention::flashinfer_sm90_clusters;
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
    FusedMoeBackend, Glm53DsaSpecAttnLocalWorklet, Glm53DsaSpecAttnLocalWorkletConfig,
    Glm53DsaSpecAttnLocalWorkletInput, Glm53DsaSpecAttnLocalWorkletResolved,
    Glm53Fp8MlpLocalWorklet, Glm53Fp8MlpLocalWorkletConfig, Glm53Fp8MlpLocalWorkletInput,
    Glm53Fp8MlpLocalWorkletResolved, Glm53KdaSpecAttnLocalWorklet,
    Glm53KdaSpecAttnLocalWorkletConfig, Glm53KdaSpecAttnLocalWorkletInput,
    Glm53KdaSpecAttnLocalWorkletResolved, Glm53TritonMoeRouterLocalWorklet,
    Glm53TritonMoeRouterLocalWorkletConfig, Glm53TritonMoeRouterLocalWorkletInput,
    Glm53TritonMoeRouterLocalWorkletResolved, Glm53TritonMoeTpLocalWorklet,
    Glm53TritonMoeTpLocalWorkletConfig, Glm53TritonMoeTpLocalWorkletInput,
    Glm53TritonMoeTpLocalWorkletResolved, SparseMlaBackend,
};

pub const ARCH_KIND: &str = "glm53_flash_vllm_fp8_kda_dsa_moe_speculative";
const ACTIVATION_DTYPE: DType = DType::Bf16;
/// Sparse page-table width: `round_up(index_topk + index_kpool - 1, 128)`.
const SELECTED_K: u32 = 2176;
/// FLASH_ATTN_MLA_SPARSE kernel block size ("Setting kv cache block size to
/// 64" in the H200 server log).
const CACHE_BLOCK_SIZE: u32 = 64;
/// Hybrid MLA + Mamba block alignment (`Platform._align_hybrid_block_size`):
/// TRTLLM/FlashInfer MLA decode kernels need the block a multiple of 128.
const MLA_HYBRID_BLOCK_ALIGNMENT: u32 = 128;
/// The smallest DeepGEMM paged-MQA pool page (`PAGED_MQA_PAGE_SIZES`): the
/// kpool indexer needs the block a multiple of `index_kpool` of them on SM90.
const PAGED_MQA_MIN_POOL_PAGE: u32 = 32;
/// Mamba `align` cache mode sizes the pool for two state pages per KDA group
/// per request, plus one per speculative token
/// (`MambaSpec.max_memory_usage_bytes`): the running state, the prefill
/// checkpoint, and the speculative scratch.
const ALIGN_STATE_PAGES: u32 = 2;
/// Of those, a decoding request holds only the running state and the
/// speculative scratch: `MambaManager.remove_skipped_blocks` frees the prefill
/// checkpoint once the request computes past it.
const ALIGN_RESIDENT_STATE_PAGES: u32 = 1;
/// The kpool tail group's one block per request.
const KPOOL_TAIL_PAGES: u32 = 1;
/// State blocks one retained prefix-cache boundary keeps per KDA group: in
/// `align` mode a Mamba hit needs exactly the single state block ending on the
/// boundary (`MambaManager.reachable_block_mask`).
const CHECKPOINT_STATE_PAGES_PER_GROUP: u32 = 1;
/// vLLM runs the shared expert on an aux stream at or below this batch size.
const SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD: u32 = 256;
/// `Parallel{overlap}` of the main-stream shared expert against the
/// aux-stream router + routed experts, as H200 case-03 kernel-align implies it
/// (simulated isolated branch / measured on-path branch + shared expert, per
/// decode iteration): body layers p10/p50/p90 0.976 / 0.995 / 1.014, so the
/// visible union is only ~1.025x the routed branch. B200's 0.9 (union 1.09-1.13)
/// over-charged every H200 decode MoE by 11%
/// (logs/20261008_3_h200_moe_decode in wt-glm53-h200-kda).
const SHARED_EXPERTS_STREAM_OVERLAP: f32 = 0.98;
/// The same for the MTP layer's MoE: p10/p50/p90 0.952 / 0.964 / 0.975.
const MTP_SHARED_EXPERTS_STREAM_OVERLAP: f32 = 0.965;
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
/// FlashKDA: `_resolve_kda_prefill_backend` "auto" on Hopper (bf16, head_dim
/// 128, bounded gate); the capture runs `_flash_kda_fwd_*`.
const KDA_PREFILL_BACKENDS: &[&str] = &["vllm_flashkda"];
const KDA_DECODE_BACKENDS: &[&str] = &["vllm_triton"];
const CONV_BACKENDS: &[&str] = &["vllm_triton"];
const QKV_NORM_BACKENDS: &[&str] = &["vllm_triton"];
const Q_ABSORB_BACKENDS: &[&str] = &["torch_mla_q_absorb_no_rope"];
const V_UP_BACKENDS: &[&str] = &["torch_mla_v_up_unpadded"];
const MQA_LOGITS_BACKENDS: &[&str] = &["deepgemm_fp8"];
/// vLLM v0.31's `SparseIndexerTopk("auto")`, the deployment's default
/// `sparse_indexer_topk_backend`: `cooperative_topk` up to 64 rows, then
/// `persistent_topk`'s FilteredTopK, as the H200 capture shows.
const TOPK_BACKENDS: &[&str] = &["vllm_auto"];
const MQA_LOGITS_PREFILL_BACKENDS: &[&str] = &["deepgemm_fp8"];
const TOPK_PREFILL_BACKENDS: &[&str] = &["vllm_cuda"];
const MLA_APPEND_BACKENDS: &[&str] = &["vllm_cuda"];
/// `gather_and_maybe_dequant_cache` of the FP8-cache prefill workspace.
const MLA_GATHER_BACKENDS: &[&str] = &["vllm_cuda"];
/// The H200 deployment's `max_num_batched_tokens` (chunk_size 8192): one
/// scheduler step's new prefill tokens.
const MAX_NUM_BATCHED_TOKENS: u32 = 8192;
/// The vLLM fork's `_PREFILL_WS_MAX_CONTEXT`: the default prefill workspace
/// holds one context up to this long plus a step of new tokens.
const PREFILL_WS_MAX_CONTEXT: u32 = 262_144;
/// vLLM's `FlashAttnPrefillBackend` FA3 varlen calls of dense-MHA prefill.
const DENSE_PREFILL_ATTENTION_BACKENDS: &[&str] = &["vllm_fa3"];
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
    /// vLLM's sparse-MLA backend for the MQA rows: `FLASH_ATTN_MLA_SPARSE`
    /// (the production recipe without nvcc on PATH) unless set.
    pub sparse_mla_backend: SparseMlaBackend,
    /// Which `vllm_fused_moe` rows price the Triton MoE launches: vLLM's
    /// default configs unless a tuned config folder's backend is selected.
    pub moe_backend: FusedMoeBackend,
    /// MLA latent cache element type: sizes the attention page and the hybrid
    /// block, and selects the DSA layers' latent-cache kernel rows.
    pub mla_kv_cache_dtype: MlaKvCacheDtype,
    /// The vLLM fork's `VLLM_FP8_KV_PREFILL_BF16_WORKSPACE`: with an FP8
    /// latent, sparse-MQA prefill rows run on a BF16 plan over a dequantized
    /// workspace. No effect with a BF16 latent.
    pub mla_fp8_prefill_bf16_workspace: bool,
    /// The fork's B1 FlashInfer header overlay: FP8-call rows priced on
    /// `vllm_flashinfer_sparse_sm90_fp8_b1`. No effect with a BF16 latent.
    pub mla_fp8_flashinfer_b1: bool,
    /// Routed MoE layers of at least this many tokens run FlashInfer's CUTLASS
    /// FP8 block-scale fused MoE (`--moe-backend flashinfer_cutlass`); `None`
    /// keeps the Triton launches everywhere.
    pub moe_flashinfer_cutlass_min_tokens: Option<u32>,
    /// KDA recurrent state element type: sizes the state page and the hybrid
    /// block, not the kernel rows.
    pub kda_state_dtype: KdaStateDtype,
}

/// vLLM `--kv-cache-dtype` for the MLA latent. `fp8` (e4m3, one per-tensor
/// scale) is what `FLASHINFER_MLA_SPARSE_SM90` accepts on H200; the rope-free
/// latent then takes one byte per element. The indexer key cache is FP8
/// either way.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlaKvCacheDtype {
    /// The production `auto` cache: the model's BF16.
    #[default]
    Bf16,
    Fp8,
}

impl MlaKvCacheDtype {
    pub fn size_bytes(self) -> u64 {
        match self {
            Self::Bf16 => 2,
            Self::Fp8 => 1,
        }
    }

    /// The latent's kernel element type.
    pub fn dtype(self) -> DType {
        match self {
            Self::Bf16 => DType::Bf16,
            Self::Fp8 => DType::Fp8E4m3,
        }
    }
}

/// vLLM `--mamba-ssm-cache-dtype` for the KDA recurrent state (the conv state
/// stays in the activation dtype). `bf16` halves the recurrent state; the
/// served KDA kernels need a fork change to run on it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KdaStateDtype {
    /// The production `auto` state: FP32.
    #[default]
    Fp32,
    Bf16,
}

impl KdaStateDtype {
    pub fn size_bytes(self) -> u64 {
        match self {
            Self::Fp32 => 4,
            Self::Bf16 => 2,
        }
    }
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
    if parallel.sparse_mla_backend == SparseMlaBackend::FlashinferMlaSparseSm90
        && flashinfer_sm90_clusters(&parallel.gpu_name).is_none()
    {
        return Err(fit_failed(format!(
            "flashinfer_mla_sparse_sm90 is costed only on NVIDIA H200 (its planner waves), got {}",
            parallel.gpu_name
        )));
    }
    let kv_cache_dtype = parallel.mla_kv_cache_dtype.dtype();
    if parallel
        .sparse_mla_backend
        .l1_backend(kv_cache_dtype)
        .is_none()
    {
        return Err(fit_failed(format!(
            "sparse_mla_backend {:?} does not serve mla_kv_cache_dtype {}: vLLM's \
             FLASH_ATTN_MLA_SPARSE takes only BF16/FP16 caches; set \
             sparse_mla_backend: flashinfer_mla_sparse_sm90",
            parallel.sparse_mla_backend,
            kv_cache_dtype.as_str()
        )));
    }
    // vLLM `prefill_workspace_tokens`: 0 (off) without an FP8 cache.
    let prefill_workspace_tokens =
        if parallel.mla_fp8_prefill_bf16_workspace && kv_cache_dtype == DType::Fp8E4m3 {
            parallel.max_model_len.min(PREFILL_WS_MAX_CONTEXT) + MAX_NUM_BATCHED_TOKENS
        } else {
            0
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
        fused_moe_backends: vec![parallel.moe_backend.l1_backend()],
        align_backends: MOE_ALIGN_BACKENDS.to_vec(),
        sum_backends: MOE_SUM_BACKENDS.to_vec(),
        elementwise_backends: ELEMENTWISE_BACKENDS.to_vec(),
        expert_demand: expert_demand.clone(),
        flashinfer_cutlass_min_tokens: parallel.moe_flashinfer_cutlass_min_tokens,
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
            verify_width: parallel.draft_tokens + 1,
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
            kv_cache_dtype,
            prefill_workspace_tokens,
            fp8_flashinfer_b1: parallel.mla_fp8_flashinfer_b1 && kv_cache_dtype == DType::Fp8E4m3,
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
            mla_cache_gather_backends: MLA_GATHER_BACKENDS.to_vec(),
            sparse_mla_backend: parallel.sparse_mla_backend,
            dense_prefill_attention_backends: DENSE_PREFILL_ATTENTION_BACKENDS.to_vec(),
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
        // Placeholder: argmax over one full fp32 logits row (a `reduce_kernel`,
        // ~16.5 us per call in the H200 case-03 capture). The bytes are split
        // evenly between read and write: the triton elementwise L1 spreads its
        // grid over output bytes, so a read-the-row, write-8-bytes shape
        // measures at ~1 GB/s (28 ms for 32 rows) instead of a reduction's
        // read bandwidth.
        mtp_argmax: ew(model.vocab_size * fp32 / 2, model.vocab_size * fp32 / 2),
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

/// Per-GPU bytes one cached token occupies in an attention page: the MLA
/// latent (no rope part) plus the kpool index cache (one 128-byte FP8 key and
/// a 4-byte scale per 4-token pool), over the 11 DSA layers and the MTP
/// layer. The latent is replicated on every TP rank.
pub fn kv_bytes_per_token_per_gpu(model: &Glm53FlashModelCfg, mla: MlaKvCacheDtype) -> u64 {
    let mla_layers = u64::from(model.num_dsa_layers()) + 1;
    let index = u64::from(model.index_head_dim + 4) / u64::from(model.index_kpool);
    mla_layers * (mla_latent_bytes_per_token(model, mla) + index)
}

/// One MLA layer's latent bytes per token: the attention page vLLM sizes the
/// hybrid block against (`MLAAttentionSpec(block_size=1).page_size_bytes`).
fn mla_latent_bytes_per_token(model: &Glm53FlashModelCfg, mla: MlaKvCacheDtype) -> u64 {
    u64::from(model.kv_lora_rank) * mla.size_bytes()
}

/// One KDA layer's state page on one rank (`MambaSpec.page_size_bytes`): the
/// `heads/TP x head_dim x head_dim` recurrent state plus the activation-dtype
/// conv state over q, k and v (`kda_state_shape`, which keeps
/// `conv_kernel - 1 + draft_tokens` rows under speculation).
pub fn kda_state_page_bytes_per_gpu(
    model: &Glm53FlashModelCfg,
    tp_size: u16,
    draft_tokens: u32,
    state: KdaStateDtype,
) -> u64 {
    let heads = u64::from(model.kda_num_heads / u32::from(tp_size.max(1)));
    let head_dim = u64::from(model.kda_head_dim);
    let recurrent = heads * head_dim * head_dim * state.size_bytes();
    let conv_rows = u64::from(model.short_conv_kernel_size - 1 + draft_tokens);
    let conv = 3 * heads * head_dim * conv_rows * ACTIVATION_DTYPE.size_bytes() as u64;
    recurrent + conv
}

/// vLLM's hybrid block (`Platform._align_hybrid_block_size`, no `all`-mode
/// chunk alignment): the fewest 128-token units whose one-layer latent page
/// holds a KDA state page, rounded up to the kpool indexer's pool pages. The
/// state page is then padded to the attention page, so a state page costs one
/// attention page. This is the attention tokens one page holds, the KDA
/// checkpoint interval, and the prefix-cache reuse quantum: 1152 for the
/// production BF16 latent and FP32 state ("attention block size 1152 tokens"
/// in the H200 server log).
pub fn hybrid_block_size(
    model: &Glm53FlashModelCfg,
    tp_size: u16,
    draft_tokens: u32,
    mla: MlaKvCacheDtype,
    state: KdaStateDtype,
) -> u32 {
    let mamba_page = kda_state_page_bytes_per_gpu(model, tp_size, draft_tokens, state);
    let align = u64::from(MLA_HYBRID_BLOCK_ALIGNMENT);
    let block = align * mamba_page.div_ceil(align * mla_latent_bytes_per_token(model, mla));
    let indexer_align = u64::from(model.index_kpool * PAGED_MQA_MIN_POOL_PAGE);
    let block = indexer_align * block.div_ceil(indexer_align);
    u32::try_from(block).expect("hybrid block fits u32")
}

/// Whole hybrid pages a per-GPU pool of `pool_bytes_per_gpu` holds (vLLM's
/// `num_blocks`, before the reserved null block).
pub fn pool_pages(pool_bytes_per_gpu: u64, block: u32, per_gpu_token_bytes: u64) -> u64 {
    pool_bytes_per_gpu / (u64::from(block) * per_gpu_token_bytes)
}

/// Pages of recurrent state vLLM's startup sizer reserves per request: KDA
/// layers are packed into groups the size of the attention group (12 layers),
/// each sized for `2 + draft_tokens` pages in `align` mode, plus the kpool tail
/// block. This is the "maximum concurrency" figure in the server log.
pub fn state_pages_per_request(model: &Glm53FlashModelCfg, draft_tokens: u32) -> u32 {
    kda_groups(model) * (ALIGN_STATE_PAGES + draft_tokens) + KPOOL_TAIL_PAGES
}

/// Pages of recurrent state a decoding request actually holds:
/// `1 + draft_tokens` per KDA group plus the kpool tail. The worker charges
/// this, so admission sees the pool vLLM's scheduler sees. The prefill
/// checkpoint page each group holds until the request computes past it is not
/// charged.
pub fn resident_state_pages_per_request(model: &Glm53FlashModelCfg, draft_tokens: u32) -> u32 {
    kda_groups(model) * (ALIGN_RESIDENT_STATE_PAGES + draft_tokens) + KPOOL_TAIL_PAGES
}

/// Pages one retained prefix-cache checkpoint occupies: one state block per
/// KDA group. The kpool tail group is not prefix-cacheable
/// (`KpoolTailSpec.prefix_cacheable = False`), and the attention KV of the
/// prefix itself is per-token, already counted as retained context.
pub fn checkpoint_state_pages(model: &Glm53FlashModelCfg) -> u32 {
    kda_groups(model) * CHECKPOINT_STATE_PAGES_PER_GROUP
}

fn kda_groups(model: &Glm53FlashModelCfg) -> u32 {
    let attention_layers = model.num_dsa_layers() + 1;
    (model.kda_layers.len() as u32).div_ceil(attention_layers)
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
    /// `Parallel{overlap}` of the shared expert and the router + routed branch.
    shared_overlap: f32,
}

impl MoeBlock {
    fn build(
        name: String,
        resolved: &Glm53FlashSpecResolved,
        routed: &Glm53TritonMoeTpLocalWorkletResolved,
        shared_overlap: f32,
        bridge: &PerfApiBridge,
    ) -> std::result::Result<Self, BuildError> {
        let m = name.as_str();
        Ok(Self {
            shared_overlap,
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
                    overlap: self.shared_overlap,
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
            moe: MoeBlock::build(
                format!("{name}.moe"),
                resolved,
                routed,
                MTP_SHARED_EXPERTS_STREAM_OVERLAP,
                bridge,
            )?,
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
    graph_dispatch: GraphDispatch,
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
    /// [`hybrid_block_size`] for the configured cache dtypes.
    hybrid_block_size: u32,
    recurrent_state_bytes_per_request: u64,
    recurrent_checkpoint_bytes: u64,
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
                SHARED_EXPERTS_STREAM_OVERLAP,
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
    let mla_kv_cache_dtype = cfg.parallel.mla_kv_cache_dtype;
    let per_gpu_token = kv_bytes_per_token_per_gpu(model_cfg, mla_kv_cache_dtype);
    let hybrid_block_size = hybrid_block_size(
        model_cfg,
        cfg.parallel.tp_size,
        draft_tokens,
        mla_kv_cache_dtype,
        cfg.parallel.kda_state_dtype,
    );
    let page_bytes_per_gpu = u64::from(hybrid_block_size) * per_gpu_token;
    let total_kv_bytes_per_token = tp * per_gpu_token;
    let recurrent_state_bytes_per_request = tp
        * u64::from(resident_state_pages_per_request(model_cfg, draft_tokens))
        * page_bytes_per_gpu;
    let recurrent_checkpoint_bytes =
        tp * u64::from(checkpoint_state_pages(model_cfg)) * page_bytes_per_gpu;
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
        graph_dispatch: GraphDispatch::new(&cfg.parallel.cudagraph_capture_sizes, draft_tokens + 1),
        total_kv_bytes_per_token,
        hybrid_block_size,
        recurrent_state_bytes_per_request,
        recurrent_checkpoint_bytes,
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
        let mut batch = normalize_speculative_input(input, self.draft_tokens, self.max_model_len)
            .unwrap_or_else(|reason| panic!("invalid Glm53FlashSpecModel input: {reason}"));
        let tokens = graph_padded_tokens(&self.cudagraph_capture_sizes, batch.total_tokens);
        // Attention takes the FULL-graph path only when the whole forward
        // replays one; the same rule sets the iteration's host floor.
        batch.dsa.full_cuda_graph = self.graph_dispatch.mode(
            batch.total_tokens,
            !batch.dsa.prefill_chunk_pairs.is_empty(),
        ) == GraphMode::FullGraph;
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
                full_cuda_graph: batch.dsa.full_cuda_graph,
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

    fn graph_mode(&self, batch: &SpeculativeArchInput) -> Option<GraphMode> {
        let total_tokens = batch.groups.iter().map(|group| group.batch_tokens).sum();
        let has_prefill = batch
            .groups
            .iter()
            .any(|group| !group.prefill_chunk_pairs.is_empty());
        Some(self.graph_dispatch.mode(total_tokens, has_prefill))
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

    /// The 19 resident state pages (all ranks): KDA groups x (1 + K) + tail.
    fn recurrent_state_bytes_per_request(&self) -> u64 {
        self.recurrent_state_bytes_per_request
    }

    /// The hybrid block: KDA checkpoint interval and prefix-reuse quantum.
    fn recurrent_checkpoint_interval_tokens(&self) -> u32 {
        self.hybrid_block_size
    }

    /// The 3 pages one retained boundary keeps (one per KDA group), not the 19
    /// a decoding request pins.
    fn recurrent_checkpoint_bytes(&self) -> u64 {
        self.recurrent_checkpoint_bytes
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
            // Set by the caller, which knows the CUDA-graph capture sizes.
            full_cuda_graph: false,
        },
    })
}

/// Which [`GraphMode`] vLLM v0.31's model runner V2 dispatches one verify
/// forward to under `FULL_AND_PIECEWISE` with MTP
/// (`CudaGraphManager._init_candidates` / `dispatch` in
/// `vllm/v1/worker/gpu/cudagraph_utils.py`, batch facts from
/// `GPUModelRunner.execute_model` in `vllm/v1/worker/gpu/model_runner.py`):
///
/// - a decode-only batch is uniform (every request submits `K + 1` rows) and
///   replays a FULL graph when a captured decode size covers it. Those sizes
///   are the capture sizes rounded up to a multiple of `K + 1` that stay within
///   the largest capture size: with sizes up to 512 and `K + 1 = 6` the largest
///   is 498, not 512;
/// - any other batch within the largest capture size replays PIECEWISE graphs
///   (attention eager);
/// - a larger batch, or every batch when nothing is captured, runs eagerly.
///
/// The MTP drafter's step 0 dispatches through the same rule on the same rows
/// (`AutoRegressiveSpeculator`'s prefill manager); its later steps replay FULL
/// single-row graphs whatever the target did. The host floor this mode keys is
/// measured over the whole engine step, so one mode describes the iteration.
///
/// Not modeled: the `max_num_seqs * (K + 1)` cap on decode graphs (no batch
/// larger than `max_num_seqs` requests is ever scheduled) and microbatching.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GraphDispatch {
    /// Largest capture size; 0 when nothing is captured.
    piecewise_max_tokens: u32,
    /// Largest FULL decode graph, in tokens; 0 when none is captured.
    full_graph_max_tokens: u32,
}

impl GraphDispatch {
    fn new(capture_sizes: &[u32], verify_width: u32) -> Self {
        let piecewise_max_tokens = capture_sizes.iter().copied().max().unwrap_or(0);
        let full_graph_max_tokens = capture_sizes
            .iter()
            .map(|&size| size.div_ceil(verify_width) * verify_width)
            .filter(|&rounded| rounded <= piecewise_max_tokens)
            .max()
            .unwrap_or(0);
        Self {
            piecewise_max_tokens,
            full_graph_max_tokens,
        }
    }

    fn mode(self, total_tokens: u32, has_prefill: bool) -> GraphMode {
        if !has_prefill && total_tokens <= self.full_graph_max_tokens {
            GraphMode::FullGraph
        } else if total_tokens <= self.piecewise_max_tokens {
            GraphMode::Piecewise
        } else {
            GraphMode::Eager
        }
    }
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
        configs_with_moe(FusedMoeBackend::default())
    }

    fn uniform_demand() -> Glm53FlashSpecDemand {
        let uniform = ExpertDemand::popularity(&RoutingDistribution::uniform(288), 43);
        Glm53FlashSpecDemand {
            body: uniform.clone(),
            mtp_step0: uniform.clone(),
            mtp_recurrent: uniform,
        }
    }

    fn configs_with_moe(moe_backend: FusedMoeBackend) -> Glm53FlashSpecConfigs {
        let uniform = ExpertDemand::popularity(&RoutingDistribution::uniform(288), 43);
        build_configs(
            &model_cfg(),
            &Glm53FlashSpecParallel {
                tp_size: 4,
                max_model_len: 8192,
                gpu_name: "NVIDIA H200".into(),
                cudagraph_capture_sizes: vec![1, 2, 4, 6, 8, 12, 16, 24, 32, 512],
                draft_tokens: 5,
                sparse_mla_backend: SparseMlaBackend::default(),
                moe_backend,
                mla_kv_cache_dtype: MlaKvCacheDtype::default(),
                mla_fp8_prefill_bf16_workspace: false,
                mla_fp8_flashinfer_b1: false,
                moe_flashinfer_cutlass_min_tokens: None,
                kda_state_dtype: KdaStateDtype::default(),
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
    /// tokens, and 858,521 "tokens" = 104.8 requests x 8192 because the sizer
    /// reserves 22 state pages per request; at 37.34 GiB, 2744 pages and
    /// 2.941 x 1,048,576 = 3,083,914.
    #[test]
    fn kv_and_state_pages_reproduce_the_vllm_capacity_lines() {
        let model = model_cfg();
        let block = hybrid_block_size(&model, 4, 5, MlaKvCacheDtype::Bf16, KdaStateDtype::Fp32);
        assert_eq!(block, 1152);
        let per_gpu_token = kv_bytes_per_token_per_gpu(&model, MlaKvCacheDtype::Bf16);
        assert_eq!(per_gpu_token, 12 * (1024 + 33));
        assert_eq!(state_pages_per_request(&model, 5), 22);
        assert_eq!(state_pages_per_request(&model, 0), 7);
        // Case 10 (saturated at 256 clients, no preemption) peaks at 3142 of
        // 3144 pages with 19 state pages per running request and would
        // overflow the pool with 20.
        assert_eq!(resident_state_pages_per_request(&model, 5), 19);
        // A retained prefix-cache boundary keeps one state block per KDA
        // group (34 KDA layers in groups of 12): 3 pages, whatever K is.
        assert_eq!(checkpoint_state_pages(&model), 3);
        let page = u64::from(block) * per_gpu_token;
        // The logs print the pool rounded to 0.01 GiB, so round the page count.
        let pages = |gib: f64| (gib * f64::from(1u32 << 30) / page as f64).round() as u64;
        assert_eq!(pages(42.79), 3144);
        assert_eq!(pages(37.34), 2744);
        let concurrency =
            |pages: u64, max_len: u32| pages as f64 / f64::from(max_len.div_ceil(block) + 22);
        assert_eq!((concurrency(3144, 8192) * 8192.0) as u64, 858_521);
        assert_eq!(
            (concurrency(2744, 1 << 20) * f64::from(1u32 << 20)) as u64,
            3_083_914
        );
    }

    /// The KDA state page per layer per rank at TP4 and MTP 5: 16 heads of
    /// 128x128 recurrent state plus the BF16 conv state over 3 x 2048 channels
    /// and 3 + 5 rows.
    #[test]
    fn the_kda_state_page_follows_the_vllm_mamba_spec() {
        let model = model_cfg();
        assert_eq!(
            kda_state_page_bytes_per_gpu(&model, 4, 5, KdaStateDtype::Fp32),
            1_146_880
        );
        assert_eq!(
            kda_state_page_bytes_per_gpu(&model, 4, 5, KdaStateDtype::Bf16),
            622_592
        );
    }

    /// `128 x ceil(state page / (128 x latent bytes))`: halving the latent
    /// doubles the block; halving the recurrent state as well brings it back
    /// to 1280.
    #[test]
    fn the_hybrid_block_derives_from_the_cache_dtypes() {
        let model = model_cfg();
        let block = |mla, state| hybrid_block_size(&model, 4, 5, mla, state);
        assert_eq!(block(MlaKvCacheDtype::Bf16, KdaStateDtype::Fp32), 1152);
        assert_eq!(block(MlaKvCacheDtype::Fp8, KdaStateDtype::Fp32), 2304);
        assert_eq!(block(MlaKvCacheDtype::Fp8, KdaStateDtype::Bf16), 1280);
        assert_eq!(block(MlaKvCacheDtype::Bf16, KdaStateDtype::Bf16), 640);
        // The state page always fits the padded attention page.
        for mla in [MlaKvCacheDtype::Bf16, MlaKvCacheDtype::Fp8] {
            for state in [KdaStateDtype::Fp32, KdaStateDtype::Bf16] {
                assert!(
                    u64::from(block(mla, state)) * mla_latent_bytes_per_token(&model, mla)
                        >= kda_state_page_bytes_per_gpu(&model, 4, 5, state)
                );
            }
        }
    }

    /// The 45.13 GB per-GPU pool of the agentic passes: 3088 pages (3087
    /// past vLLM's null block) of 1152 BF16 tokens; an FP8 latent nearly
    /// halves the per-token bytes but doubles the page, so the pool holds
    /// 2995 pages of 2304 tokens, and 5391 of 1280 with a BF16 KDA state.
    #[test]
    fn the_pool_pages_follow_the_derived_block() {
        let model = model_cfg();
        let pool = (45.13e9_f64) as u64;
        let pages = |mla, state| {
            let block = hybrid_block_size(&model, 4, 5, mla, state);
            let per_gpu_token = kv_bytes_per_token_per_gpu(&model, mla);
            (block, per_gpu_token, pool_pages(pool, block, per_gpu_token))
        };
        assert_eq!(
            pages(MlaKvCacheDtype::Bf16, KdaStateDtype::Fp32),
            (1152, 12_684, 3088)
        );
        assert_eq!(
            pages(MlaKvCacheDtype::Fp8, KdaStateDtype::Fp32),
            (2304, 6_540, 2995)
        );
        assert_eq!(
            pages(MlaKvCacheDtype::Fp8, KdaStateDtype::Bf16),
            (1280, 6_540, 5391)
        );
    }

    /// What the worker reads from the built model: the per-token bytes, the
    /// checkpoint interval (prefix reuse, chunk end, EAGLE drop quantum), and
    /// the 19 resident / 3 checkpoint state pages, which in the worker's token
    /// space are 19 and 3 blocks whatever the dtypes.
    #[test]
    fn the_built_model_exposes_the_derived_block_to_the_worker() {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        for (mla, state, block, per_gpu_token) in [
            (
                MlaKvCacheDtype::Bf16,
                KdaStateDtype::Fp32,
                1152u64,
                12_684u64,
            ),
            (MlaKvCacheDtype::Fp8, KdaStateDtype::Fp32, 2304, 6_540),
            (MlaKvCacheDtype::Fp8, KdaStateDtype::Bf16, 1280, 6_540),
        ] {
            let mut parallel = configs().parallel;
            parallel.mla_kv_cache_dtype = mla;
            parallel.kda_state_dtype = state;
            // vLLM serves an FP8 latent only through FLASHINFER_MLA_SPARSE_SM90.
            parallel.sparse_mla_backend = SparseMlaBackend::FlashinferMlaSparseSm90;
            let c = build_configs(&model_cfg(), &parallel, &uniform_demand()).unwrap();
            let model = build("unified".into(), resolve_configs(&c), &bridge).unwrap();
            let total = 4 * per_gpu_token;
            assert_eq!(model.total_kv_bytes_per_token(), total);
            assert_eq!(
                u64::from(model.recurrent_checkpoint_interval_tokens()),
                block
            );
            assert_eq!(
                model.recurrent_state_bytes_per_request(),
                19 * block * total
            );
            assert_eq!(model.recurrent_checkpoint_bytes(), 3 * block * total);
        }
    }

    #[test]
    fn configs_take_the_h200_tp_only_realization() {
        let c = configs();
        let r = resolve_configs(&c);
        assert_eq!(r.routed_body.gate_up.num_local_experts.get(), 288);
        assert_eq!(r.routed_body.down.k.get(), 512);
        // KDA verify leaves run at query_len K + 1 = 6; prefill is FlashKDA.
        assert_eq!(r.kda.recurrent_verify.query_len, 6);
        assert_eq!(r.kda.conv_verify.query_len, 6);
        assert_eq!(r.kda.chunk_prefill.backends, ["vllm_flashkda"]);
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

    /// The defect this catches: a tuned-config deployment priced from the
    /// default `vllm_triton` rows, or the tuned rows leaking into the default
    /// presets. Every routed MoE launch (body and both MTP steps, gate/up and
    /// down) follows the selector; nothing else changes backend.
    #[test]
    fn the_moe_backend_selects_which_fused_moe_rows_price_every_routed_launch() {
        for (moe_backend, expected) in [
            (FusedMoeBackend::VllmTriton, "vllm_triton"),
            (
                FusedMoeBackend::VllmTritonTuned212d200c,
                "vllm_triton_tuned_212d200c",
            ),
        ] {
            let c = configs_with_moe(moe_backend);
            for routed in [&c.routed_body, &c.routed_mtp_step0, &c.routed_mtp_recurrent] {
                assert_eq!(routed.fused_moe_backends, [expected]);
            }
            let r = resolve_configs(&c);
            assert_eq!(r.routed_body.gate_up.backends, [expected]);
            assert_eq!(r.routed_body.down.backends, [expected]);
            assert_eq!(
                r.routed_body.align.backends,
                configs().routed_body.align_backends
            );
        }
        assert_eq!(FusedMoeBackend::default(), FusedMoeBackend::VllmTriton);
    }

    /// Existing presets omit the field and keep the default rows; the tuned
    /// spelling is the L1 backend name.
    #[test]
    fn the_moe_backend_parses_from_a_preset_and_defaults_to_vllm_triton() {
        use crate::arch::config::IterArchSel;
        let parse = |extra: &[(&str, &str)]| {
            let mut arch = serde_json::json!({
                "type": ARCH_KIND,
                "model_config": "model/config/glm53_flash.json",
                "fp8": true,
            });
            for (key, value) in extra {
                arch[*key] = serde_json::Value::from(*value);
            }
            match serde_json::from_value::<IterArchSel>(arch).unwrap() {
                IterArchSel::Glm53FlashVllmFp8KdaDsaMoeSpeculative { moe_backend, .. } => {
                    moe_backend
                }
                _ => unreachable!(),
            }
        };
        assert_eq!(parse(&[]), FusedMoeBackend::VllmTriton);
        assert_eq!(
            parse(&[("moe_backend", "vllm_triton_tuned_212d200c")]).l1_backend(),
            "vllm_triton_tuned_212d200c"
        );
    }

    /// Existing presets omit the cache dtypes and keep the production BF16
    /// latent and FP32 state.
    #[test]
    fn the_cache_dtypes_parse_from_a_preset_and_default_to_production() {
        use crate::arch::config::IterArchSel;
        let parse = |extra: &[(&str, &str)]| {
            let mut arch = serde_json::json!({
                "type": ARCH_KIND,
                "model_config": "model/config/glm53_flash.json",
                "fp8": true,
            });
            for (key, value) in extra {
                arch[*key] = serde_json::Value::from(*value);
            }
            match serde_json::from_value::<IterArchSel>(arch).unwrap() {
                IterArchSel::Glm53FlashVllmFp8KdaDsaMoeSpeculative {
                    mla_kv_cache_dtype,
                    kda_state_dtype,
                    ..
                } => (mla_kv_cache_dtype, kda_state_dtype),
                _ => unreachable!(),
            }
        };
        assert_eq!(parse(&[]), (MlaKvCacheDtype::Bf16, KdaStateDtype::Fp32));
        assert_eq!(
            parse(&[("mla_kv_cache_dtype", "fp8"), ("kda_state_dtype", "bf16")]),
            (MlaKvCacheDtype::Fp8, KdaStateDtype::Bf16)
        );
    }

    /// Catches an FP8 latent priced on the BF16 rows (the pre-FP8 behavior),
    /// the FA3 backend accepted for a cache vLLM refuses it, or the workspace
    /// option changing a BF16 deployment. The workspace capacity is vLLM's
    /// default `min(max_model_len, 262144) + max_num_batched_tokens`.
    #[test]
    fn the_latent_dtype_selects_the_dsa_rows_and_the_workspace_option_needs_fp8() {
        let with = |mla, backend, workspace: bool, max_model_len: u32| {
            let mut parallel = configs().parallel;
            parallel.mla_kv_cache_dtype = mla;
            parallel.sparse_mla_backend = backend;
            parallel.mla_fp8_prefill_bf16_workspace = workspace;
            parallel.max_model_len = max_model_len;
            build_configs(&model_cfg(), &parallel, &uniform_demand())
        };
        let fi = SparseMlaBackend::FlashinferMlaSparseSm90;
        let fa3 = SparseMlaBackend::FlashAttnMlaSparse;
        let bf16 = with(MlaKvCacheDtype::Bf16, fi, false, 8192).unwrap();
        assert_eq!(bf16.dsa.kv_cache_dtype, DType::Bf16);
        // vLLM's env flag is a no-op on a BF16 cache.
        let bf16_ws = with(MlaKvCacheDtype::Bf16, fi, true, 8192).unwrap();
        assert_eq!(bf16_ws.dsa.prefill_workspace_tokens, 0);
        let fp8 = with(MlaKvCacheDtype::Fp8, fi, false, 262_144).unwrap();
        assert_eq!(fp8.dsa.kv_cache_dtype, DType::Fp8E4m3);
        assert_eq!(fp8.dsa.prefill_workspace_tokens, 0);
        let r = resolve_configs(&fp8);
        assert_eq!(
            r.dsa.sparse_attention.backends,
            ["vllm_flashinfer_sparse_sm90_fp8"]
        );
        assert_eq!(r.dsa.mla_cache_append.kv_dtype, DType::Fp8E4m3);
        let ws = with(MlaKvCacheDtype::Fp8, fi, true, 262_144).unwrap();
        assert_eq!(ws.dsa.prefill_workspace_tokens, 262_144 + 8192);
        assert_eq!(
            with(MlaKvCacheDtype::Fp8, fi, true, 8192)
                .unwrap()
                .dsa
                .prefill_workspace_tokens,
            8192 + 8192
        );
        assert!(resolve_configs(&ws).dsa.prefill_workspace.is_some());
        assert!(with(MlaKvCacheDtype::Fp8, fa3, false, 8192).is_err());
    }

    /// Catches the B1 option pricing anything but the FP8 call on the B1 rows:
    /// it must swap only that backend, keep the BF16 workspace rows on the
    /// stock BF16 call, and change nothing on a BF16 latent.
    #[test]
    fn the_b1_option_swaps_only_the_fp8_call_rows() {
        let with = |mla, workspace: bool, b1: bool| {
            let mut parallel = configs().parallel;
            parallel.mla_kv_cache_dtype = mla;
            parallel.sparse_mla_backend = SparseMlaBackend::FlashinferMlaSparseSm90;
            parallel.mla_fp8_prefill_bf16_workspace = workspace;
            parallel.mla_fp8_flashinfer_b1 = b1;
            parallel.max_model_len = 262_144;
            resolve_configs(&build_configs(&model_cfg(), &parallel, &uniform_demand()).unwrap())
        };
        let stock = with(MlaKvCacheDtype::Fp8, true, false);
        let b1 = with(MlaKvCacheDtype::Fp8, true, true);
        assert_eq!(
            b1.dsa.sparse_attention.backends,
            ["vllm_flashinfer_sparse_sm90_fp8_b1"]
        );
        let mut same = b1.dsa.sparse_attention.clone();
        same.backends = stock.dsa.sparse_attention.backends.clone();
        assert_eq!(same, stock.dsa.sparse_attention);
        let ws = |r: &Glm53FlashSpecResolved| {
            r.dsa
                .prefill_workspace
                .as_ref()
                .unwrap()
                .attention
                .backends
                .clone()
        };
        assert_eq!(ws(&b1), ["vllm_flashinfer_sparse_sm90_bf16"]);
        assert_eq!(ws(&b1), ws(&stock));
        assert_eq!(
            with(MlaKvCacheDtype::Bf16, false, true)
                .dsa
                .sparse_attention,
            with(MlaKvCacheDtype::Bf16, false, false)
                .dsa
                .sparse_attention
        );
    }

    /// Existing presets omit the workspace option and keep it off.
    #[test]
    fn the_prefill_workspace_option_parses_and_defaults_off() {
        use crate::arch::config::IterArchSel;
        let parse = |extra: serde_json::Value| {
            let mut arch = serde_json::json!({
                "type": ARCH_KIND,
                "model_config": "model/config/glm53_flash.json",
                "fp8": true,
            });
            arch.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            match serde_json::from_value::<IterArchSel>(arch).unwrap() {
                IterArchSel::Glm53FlashVllmFp8KdaDsaMoeSpeculative {
                    mla_fp8_prefill_bf16_workspace,
                    ..
                } => mla_fp8_prefill_bf16_workspace,
                _ => unreachable!(),
            }
        };
        assert!(!parse(serde_json::json!({})));
        assert!(parse(serde_json::json!({
            "mla_kv_cache_dtype": "fp8",
            "mla_fp8_prefill_bf16_workspace": true,
        })));
    }

    /// Existing presets omit the B1 option and keep it off.
    #[test]
    fn the_b1_option_parses_and_defaults_off() {
        use crate::arch::config::IterArchSel;
        let parse = |extra: serde_json::Value| {
            let mut arch = serde_json::json!({
                "type": ARCH_KIND,
                "model_config": "model/config/glm53_flash.json",
                "fp8": true,
            });
            arch.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            match serde_json::from_value::<IterArchSel>(arch).unwrap() {
                IterArchSel::Glm53FlashVllmFp8KdaDsaMoeSpeculative {
                    mla_fp8_flashinfer_b1,
                    ..
                } => mla_fp8_flashinfer_b1,
                _ => unreachable!(),
            }
        };
        assert!(!parse(serde_json::json!({})));
        assert!(parse(serde_json::json!({
            "mla_kv_cache_dtype": "fp8",
            "mla_fp8_flashinfer_b1": true,
        })));
    }

    /// Catches the CUTLASS option leaking into the default tree or missing a
    /// routed callable: unset, no routed worklet gets a CUTLASS leaf; set, the
    /// body and both MTP callables carry the same threshold.
    #[test]
    fn the_cutlass_option_reaches_every_routed_callable_and_defaults_off() {
        let off = resolve_configs(&configs());
        for routed in [
            &off.routed_body,
            &off.routed_mtp_step0,
            &off.routed_mtp_recurrent,
        ] {
            assert!(routed.cutlass.is_none());
        }
        let mut parallel = configs().parallel;
        parallel.moe_flashinfer_cutlass_min_tokens = Some(2048);
        let on =
            resolve_configs(&build_configs(&model_cfg(), &parallel, &uniform_demand()).unwrap());
        for routed in [
            &on.routed_body,
            &on.routed_mtp_step0,
            &on.routed_mtp_recurrent,
        ] {
            assert_eq!(routed.raw_cfg.flashinfer_cutlass_min_tokens, Some(2048));
            assert_eq!(
                routed.cutlass.as_ref().unwrap().backends,
                ["flashinfer_cutlass_fp8_block_sm90"]
            );
        }
    }

    /// Existing presets omit the CUTLASS threshold and keep the Triton tree.
    #[test]
    fn the_cutlass_threshold_parses_and_defaults_off() {
        use crate::arch::config::IterArchSel;
        let parse = |extra: serde_json::Value| {
            let mut arch = serde_json::json!({
                "type": ARCH_KIND,
                "model_config": "model/config/glm53_flash.json",
                "fp8": true,
            });
            arch.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            match serde_json::from_value::<IterArchSel>(arch).unwrap() {
                IterArchSel::Glm53FlashVllmFp8KdaDsaMoeSpeculative {
                    moe_flashinfer_cutlass_min_tokens,
                    ..
                } => moe_flashinfer_cutlass_min_tokens,
                _ => unreachable!(),
            }
        };
        assert_eq!(parse(serde_json::json!({})), None);
        assert_eq!(
            parse(serde_json::json!({"moe_flashinfer_cutlass_min_tokens": 1024})),
            Some(1024)
        );
    }

    /// The case-03 capture sizes: MTP 5 adds 6 and 12 to vLLM's default list.
    fn h200_capture_sizes() -> Vec<u32> {
        let mut sizes = vec![1, 2, 4, 6, 8, 12, 16];
        sizes.extend((24..=256).step_by(8));
        sizes.extend((272..=512).step_by(16));
        sizes
    }

    #[test]
    fn graph_mode_follows_the_v2_runner_dispatch_at_the_capture_boundaries() {
        let dispatch = GraphDispatch::new(&h200_capture_sizes(), 6);
        assert_eq!(h200_capture_sizes().len(), 53, "the 53 PIECEWISE captures");
        assert_eq!(dispatch.piecewise_max_tokens, 512);
        // 512 rounds up to 516 > 512, so 496 -> 498 is the largest FULL graph.
        assert_eq!(dispatch.full_graph_max_tokens, 498);

        // Mixed batches: PIECEWISE through the largest capture size, then eager.
        assert_eq!(dispatch.mode(512, true), GraphMode::Piecewise);
        assert_eq!(dispatch.mode(513, true), GraphMode::Eager);
        assert_eq!(dispatch.mode(6, true), GraphMode::Piecewise);
        // Decode-only: FULL up to 83 requests, PIECEWISE for 84-85, then eager.
        assert_eq!(dispatch.mode(6, false), GraphMode::FullGraph);
        assert_eq!(dispatch.mode(498, false), GraphMode::FullGraph);
        assert_eq!(dispatch.mode(504, false), GraphMode::Piecewise);
        assert_eq!(dispatch.mode(510, false), GraphMode::Piecewise);
        assert_eq!(dispatch.mode(516, false), GraphMode::Eager);
    }

    #[test]
    fn the_full_graph_count_matches_the_captured_decode_graphs() {
        // The H200 server log captured 48 FULL target graphs: one per distinct
        // capture size rounded up to K + 1 = 6 within 512.
        let sizes = h200_capture_sizes();
        let mut full: Vec<u32> = sizes
            .iter()
            .map(|size| size.div_ceil(6) * 6)
            .filter(|&rounded| rounded <= 512)
            .collect();
        full.dedup();
        assert_eq!(full.len(), 48);
        assert_eq!(full.last(), Some(&498));
    }

    #[test]
    fn nothing_captured_runs_every_batch_eagerly() {
        let dispatch = GraphDispatch::new(&[], 6);
        assert_eq!(dispatch.mode(6, false), GraphMode::Eager);
        assert_eq!(dispatch.mode(1, true), GraphMode::Eager);
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
