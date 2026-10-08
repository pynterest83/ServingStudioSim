//! GLM-5.3-Flash DeepSeek-sparse-attention (DSA) on one TP rank, as vLLM
//! v0.31 runs it on H200 with a BF16 KV cache, `FLASH_ATTN_MLA_SPARSE`, and MTP
//! speculative decoding.
//!
//! The projections and the pooled (kpool) indexer launch as on B200
//! (`glm53_dsa_attn_local`), with two differences: verify rows reach the
//! decode indexer `K+1` per request, and an MTP draft step after the first
//! shares the step-0 indices and skips the whole indexer.
//!
//! The decode indexer takes the `K+1` rows natively (`next_n = K+1`) only where
//! DeepGEMM's paged MQA logits supports that width (`native_next_n_supported`:
//! SM90 runs `next_n` in {1, 2, 4}, SM100 any). Elsewhere vLLM flattens the
//! batch to one single-token row per query (`_use_flattening` in
//! `v1/attention/backends/mla/indexer.py`): logits and top-k run at
//! `next_n = 1` over `B * (K+1)` rows. The H200 MTP-5 capture shows exactly
//! that (`sm90_fp8_paged_mqa_logits<1, ...>`).
//!
//! The MLA part follows `sparse_mla_attention.py` / `flashattn_mla_sparse.py`:
//!
//! - prefill rows run dense MHA when every prefill context fits in
//!   `index_topk` (`_use_dense_mha_prefill`; the masked-MHA path is sm100
//!   only): the `kv_b` up-projection, then FlashAttention varlen causal;
//! - verify rows (and prefill rows of a longer batch) run sparse MQA:
//!   `W_UK` absorb, the index remap, the zero fill of FA3's 64-wide rope-free
//!   query (`q_nope.new_zeros(..., 64)` in `_run_mqa_kernel`), FlashAttention
//!   over the top-k block table, then `W_UV`.
//!
//! The sparse FlashAttention is ONE `forward_mqa` call per layer over every
//! MQA row, decode rows first (vLLM only splits it per decode / prefill under
//! HiSparse, which this deployment does not enable). It is priced on the
//! `dsa_sparse_mla_attention` L1 kind, backend `vllm_fa3_sparse_bf16`, whose
//! measured call is `prepare_varlen_num_blocks` + `FlashAttnFwdSm90` +
//! `FlashAttnFwdCombine`, the combine only at 992 rows or fewer
//! (`FA3_COMBINE_MAX_QUERIES`). The call's rows are collapsed onto the kind's
//! `PooledUniformFull` encoding with the same total of valid slots (see
//! `sparse_attention_input`).
//!
//! The BF16 latent is appended with `concat_and_cache_mla<bf16, bf16>`; there is
//! no query FP8 quant and no query concat (both FP8-cache artifacts on B200).
//!
//! First-cut placeholders, labeled until their L1 lands (plan M4): the dense
//! FlashAttention call and the index remap are byte-sized elementwise leaves;
//! the small indexer kernels keep the B200 byte formulas and
//! effective-bandwidth factors (provisional on H200). The glue launch counts
//! are H200-measured per forward kind.

use crate::common::gpu::compute_capability;
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    BatchedGemmKernel, BatchedGemmKernelConfig, BatchedGemmKernelInput, DsaMqaLogitsPrefillKernel,
    DsaMqaLogitsPrefillKernelConfig, DsaMqaLogitsPrefillKernelInput, DsaPagedMqaLogitsDecodeKernel,
    DsaPagedMqaLogitsDecodeKernelConfig, DsaPagedMqaLogitsDecodeKernelInput,
    DsaPersistentTopkDecodeKernel, DsaPersistentTopkDecodeKernelConfig,
    DsaPersistentTopkDecodeKernelInput, DsaSparseMlaAttentionKernel,
    DsaSparseMlaAttentionKernelConfig, DsaSparseMlaAttentionKernelInput, DsaTopkPrefillKernel,
    DsaTopkPrefillKernelConfig, DsaTopkPrefillKernelInput, ElementwiseKernel,
    ElementwiseKernelConfig, ElementwiseKernelInput, GemmFp32OutputKernel,
    GemmFp32OutputKernelConfig, GemmFp32OutputKernelInput, MlaCacheAppendKernel,
    MlaCacheAppendKernelConfig, MlaCacheAppendKernelInput, QKvRmsNormKernel,
    QKvRmsNormKernelConfig, QKvRmsNormKernelInput, SingleGemmKernel, SingleGemmKernelConfig,
    SingleGemmKernelInput, ValidCountsPattern,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

use super::glm53_common::{atomic, elementwise, push_or_zero, repeated};
use super::glm53_dsa_attn_local::integer_sqrt;

/// Small copy / fill / index launches per DSA layer, counted in the H200
/// case-03 capture (logs/20261007_0_glm53_flash_h200_campaign): every
/// forward runs one after the index remap; an indexer-running forward adds
/// seven around the indexer (k copy, kpool score, top-k output copies, pool
/// expansion); a prefill-bearing forward adds twelve more (arange, mask and
/// gather plumbing of the prefill indexer path). Decode: 1 + 7 = 8,
/// mixed: 8 + 12 = 20, an index-sharing draft step: 1.
const GLUE_LAUNCHES: u32 = 1;
const INDEXER_GLUE_LAUNCHES: u32 = 7;
const GLUE_BYTES_PER_TOKEN: u32 = 64;
const PREFILL_GLUE_LAUNCHES: u32 = 12;
/// Hadamard-128 + quant effective-bandwidth factor (B200; provisional here).
const FWHT_EFFECTIVE_FACTOR: u32 = 5;
/// `k[idx]` and `gate_score[idx]` gathers ahead of the prefill kpool write.
const PREFILL_GATHER_LAUNCHES: u32 = 2;
/// FA3's q/k width on vLLM's rope-free path: `q` is zeros of this width and
/// `k` the first this-many latent columns (`_FA3_QV_HEAD_DIM`).
const FA3_QK_HEAD_DIM: u32 = 64;
/// The synthetic index pattern the pooled sparse-attention rows were measured
/// with (as the B200 pooled decode).
const SPARSE_INDEX_DISTRIBUTION: &str = "unique_scattered_pages";

#[derive(Clone, Debug)]
pub struct Glm53DsaSpecAttnLocalWorkletConfig {
    pub hidden: Dim,
    /// MLA query heads on this rank.
    pub num_heads: Dim,
    pub q_lora_rank: Dim,
    pub kv_lora_rank: Dim,
    pub qk_nope_head_dim: Dim,
    pub v_head_dim: Dim,
    pub index_num_heads: Dim,
    pub index_head_dim: Dim,
    pub index_topk: u32,
    pub index_kpool: u32,
    /// Sparse page-table width, `round_up(index_topk + index_kpool - 1, 128)`.
    pub selected_k: u32,
    pub max_model_len: u32,
    pub cache_block_size: u32,
    pub rms_eps: f64,
    /// Rows each verifying request brings to the decode indexer (`K + 1`).
    pub verify_width: u32,
    pub kv_cache_dtype: DType,
    pub gpu_name: String,
    pub bf16_gemm_backends: Vec<&'static str>,
    pub fp32_gemm_backends: Vec<&'static str>,
    pub qkv_norm_backends: Vec<&'static str>,
    pub mla_bmm_q_absorb_backends: Vec<&'static str>,
    pub mla_bmm_v_up_backends: Vec<&'static str>,
    pub mqa_logits_backends: Vec<&'static str>,
    pub topk_backends: Vec<&'static str>,
    pub mqa_logits_prefill_backends: Vec<&'static str>,
    pub topk_prefill_backends: Vec<&'static str>,
    pub mla_cache_append_backends: Vec<&'static str>,
    pub sparse_attention_backends: Vec<&'static str>,
    pub elementwise_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct Glm53DsaSpecAttnLocalWorkletResolved {
    pub raw_cfg: Glm53DsaSpecAttnLocalWorkletConfig,
    pub fused_qkv_a: SingleGemmKernelConfig,
    pub q_kv_norm: QKvRmsNormKernelConfig,
    pub q_b: SingleGemmKernelConfig,
    pub index_wq_b: SingleGemmKernelConfig,
    pub index_wk_weights: SingleGemmKernelConfig,
    pub index_head_weights: GemmFp32OutputKernelConfig,
    pub index_k_norm: ElementwiseKernelConfig,
    pub index_q_fwht_quant: ElementwiseKernelConfig,
    pub index_weight_scale: ElementwiseKernelConfig,
    pub kpool_gate_score: SingleGemmKernelConfig,
    pub prefill_gather: ElementwiseKernelConfig,
    pub kpool_decode_update: ElementwiseKernelConfig,
    pub kpool_prefill_write: ElementwiseKernelConfig,
    pub kpool_tail_seed: ElementwiseKernelConfig,
    pub mqa_logits_decode: DsaPagedMqaLogitsDecodeKernelConfig,
    pub topk_decode: DsaPersistentTopkDecodeKernelConfig,
    /// The decode indexer runs one single-token row per verify row.
    pub flatten_decode_indexer: bool,
    pub mqa_logits_prefill: DsaMqaLogitsPrefillKernelConfig,
    pub topk_prefill: DsaTopkPrefillKernelConfig,
    pub expand_pools: ElementwiseKernelConfig,
    pub mla_cache_append: MlaCacheAppendKernelConfig,
    pub kv_b_proj: SingleGemmKernelConfig,
    pub dense_prefill_attention: ElementwiseKernelConfig,
    pub q_absorb: BatchedGemmKernelConfig,
    pub index_remap: ElementwiseKernelConfig,
    pub q_rope_zero_fill: ElementwiseKernelConfig,
    pub sparse_attention: DsaSparseMlaAttentionKernelConfig,
    pub v_up: BatchedGemmKernelConfig,
    pub o_proj: SingleGemmKernelConfig,
    pub glue: ElementwiseKernelConfig,
    pub indexer_glue: ElementwiseKernelConfig,
    pub prefill_glue: ElementwiseKernelConfig,
}

/// One forward's DSA work on this rank.
#[derive(Clone, Debug, Default)]
pub struct Glm53DsaSpecAttnLocalWorkletInput {
    /// `(prefix, append)` per prefill request.
    pub prefill_chunk_pairs: Vec<(u32, u32)>,
    /// KV length per decoding request after this step's rows are appended.
    pub decode_kv_lens: Vec<u32>,
    /// Rows per decoding request: `K + 1` in a verify forward, 1 in an MTP
    /// draft step after the first.
    pub decode_width: u32,
    /// MTP draft steps after the first reuse the step-0 indices.
    pub skip_indexer: bool,
}

pub struct Glm53DsaSpecAttnLocalWorklet {
    pub name: String,
    pub fused_qkv_a: Op<SingleGemmKernel>,
    pub q_kv_norm: Op<QKvRmsNormKernel>,
    pub q_b: Op<SingleGemmKernel>,
    pub index_wq_b: Op<SingleGemmKernel>,
    pub index_wk_weights: Op<SingleGemmKernel>,
    pub index_head_weights: Op<GemmFp32OutputKernel>,
    pub index_k_norm: Op<ElementwiseKernel>,
    pub index_q_fwht_quant: Op<ElementwiseKernel>,
    pub index_weight_scale: Op<ElementwiseKernel>,
    pub kpool_gate_score: Op<SingleGemmKernel>,
    pub prefill_gather: Op<ElementwiseKernel>,
    pub kpool_decode_update: Op<ElementwiseKernel>,
    pub kpool_prefill_write: Op<ElementwiseKernel>,
    pub kpool_tail_seed: Op<ElementwiseKernel>,
    pub mqa_logits_decode: Op<DsaPagedMqaLogitsDecodeKernel>,
    pub topk_decode: Op<DsaPersistentTopkDecodeKernel>,
    pub mqa_logits_prefill: Op<DsaMqaLogitsPrefillKernel>,
    pub topk_prefill: Op<DsaTopkPrefillKernel>,
    pub expand_pools: Op<ElementwiseKernel>,
    pub mla_cache_append: Op<MlaCacheAppendKernel>,
    pub kv_b_proj: Op<SingleGemmKernel>,
    pub dense_prefill_attention: Op<ElementwiseKernel>,
    pub q_absorb: Op<BatchedGemmKernel>,
    pub index_remap: Op<ElementwiseKernel>,
    pub q_rope_zero_fill: Op<ElementwiseKernel>,
    pub sparse_attention: Op<DsaSparseMlaAttentionKernel>,
    pub v_up: Op<BatchedGemmKernel>,
    pub o_proj: Op<SingleGemmKernel>,
    pub glue: Op<ElementwiseKernel>,
    pub indexer_glue: Op<ElementwiseKernel>,
    pub prefill_glue: Op<ElementwiseKernel>,
    resolved: Glm53DsaSpecAttnLocalWorkletResolved,
}

impl Glm53DsaSpecAttnLocalWorklet {
    pub fn resolve_config(
        cfg: &Glm53DsaSpecAttnLocalWorkletConfig,
    ) -> Glm53DsaSpecAttnLocalWorkletResolved {
        assert!(cfg.verify_width > 0, "verify_width must be positive");
        let bf16 = DType::Bf16;
        let heads = cfg.num_heads.get();
        let latent = cfg.kv_lora_rank.get();
        let nope = cfg.qk_nope_head_dim.get();
        let v_dim = cfg.v_head_dim.get();
        let index_heads = cfg.index_num_heads.get();
        let index_dim = cfg.index_head_dim.get();
        let kpool = cfg.index_kpool;
        let gemm = |n: u32, k: Dim| SingleGemmKernelConfig {
            backends: cfg.bf16_gemm_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            n: n.into(),
            k,
            dtype: bf16,
        };
        let ew = |input: u32, output: u32| {
            elementwise(&cfg.elementwise_backends, &cfg.gpu_name, input, output)
        };
        let bmm = |backends: &[&'static str], n: Dim, k: Dim| BatchedGemmKernelConfig {
            backends: backends.to_vec(),
            gpu_name: cfg.gpu_name.clone(),
            num_batches: cfg.num_heads.clone(),
            n,
            k,
            dtype: bf16,
        };
        let max_model_len = Dim::param("max_model_len", cfg.max_model_len);
        let window = cfg.index_topk + kpool - 1;
        let flatten_decode_indexer =
            !native_decode_next_n_supported(&cfg.gpu_name, cfg.verify_width);
        let indexer_next_n = if flatten_decode_indexer {
            1
        } else {
            cfg.verify_width
        };
        Glm53DsaSpecAttnLocalWorkletResolved {
            fused_qkv_a: gemm(cfg.q_lora_rank.get() + latent, cfg.hidden.clone()),
            q_kv_norm: QKvRmsNormKernelConfig {
                backends: cfg.qkv_norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                q_dim: cfg.q_lora_rank.clone(),
                kv_dim: cfg.kv_lora_rank.clone(),
                rms_eps_bits: cfg.rms_eps.to_bits(),
                dtype: bf16,
            },
            q_b: gemm(heads * nope, cfg.q_lora_rank.clone()),
            index_wq_b: gemm(index_heads * index_dim, cfg.q_lora_rank.clone()),
            index_wk_weights: gemm(index_dim + index_heads, cfg.hidden.clone()),
            index_head_weights: GemmFp32OutputKernelConfig {
                backends: cfg.fp32_gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.index_num_heads.clone(),
                k: cfg.hidden.clone(),
                input_dtype: DType::Fp32,
            },
            index_k_norm: ew(2 * index_dim, 2 * index_dim),
            index_q_fwht_quant: ew(
                FWHT_EFFECTIVE_FACTOR * 2 * index_dim * index_heads,
                FWHT_EFFECTIVE_FACTOR * (index_dim + 4) * index_heads,
            ),
            index_weight_scale: ew(2 * 4 * index_heads, 4 * index_heads),
            kpool_gate_score: gemm(index_dim, cfg.hidden.clone()),
            prefill_gather: ew(kpool * 2 * index_dim, kpool * 2 * index_dim),
            kpool_decode_update: ew(524 + 1536 / kpool, 512 + (index_dim + 4) / kpool),
            kpool_prefill_write: ew(9 + 4 * index_dim, (index_dim + 4) / kpool),
            kpool_tail_seed: ew(8, 4),
            mqa_logits_decode: DsaPagedMqaLogitsDecodeKernelConfig {
                backends: cfg.mqa_logits_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                next_n: indexer_next_n,
                max_model_len: max_model_len.clone(),
                num_heads: cfg.index_num_heads.clone(),
                head_dim: cfg.index_head_dim.clone(),
                block_size: cfg.cache_block_size,
                q_dtype: DType::Fp8E4m3,
                cache_dtype: DType::Fp8E4m3,
                scale_dtype: DType::Fp32,
                weight_dtype: DType::Fp32,
                output_dtype: DType::Fp32,
                context_mode: "uniform".into(),
                page_mapping: "unique_scattered".into(),
                cache_format: "page_planar_fp8_fp32_scale".into(),
                clean_logits: false,
            },
            topk_decode: DsaPersistentTopkDecodeKernelConfig {
                backends: cfg.topk_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                next_n: indexer_next_n,
                max_model_len: max_model_len.clone(),
                top_k: cfg.index_topk / kpool,
                logits_row_stride: max_model_len,
                logits_dtype: DType::Fp32,
                index_dtype: "int32".into(),
                context_mode: "uniform".into(),
            },
            flatten_decode_indexer,
            mqa_logits_prefill: DsaMqaLogitsPrefillKernelConfig {
                backends: cfg.mqa_logits_prefill_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_sequences: 1,
                num_heads: cfg.index_num_heads.clone(),
                head_dim: cfg.index_head_dim.clone(),
                q_dtype: DType::Fp8E4m3,
                k_dtype: DType::Fp8E4m3,
                k_scale_dtype: DType::Fp32,
                weight_dtype: DType::Fp32,
                output_dtype: DType::Fp32,
                span_mode: "single_causal_tail".into(),
                clean_logits: false,
            },
            topk_prefill: DsaTopkPrefillKernelConfig {
                backends: cfg.topk_prefill_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_sequences: 1,
                top_k: cfg.index_topk / kpool,
                logits_dtype: DType::Fp32,
                index_dtype: "int32".into(),
                span_mode: "single_causal_tail".into(),
            },
            expand_pools: ew(8 * cfg.index_topk / kpool + 4, 4 * window),
            mla_cache_append: MlaCacheAppendKernelConfig {
                backends: cfg.mla_cache_append_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                kv_lora_rank: cfg.kv_lora_rank.clone(),
                rope_dim: Dim::param("qk_rope_head_dim", 0),
                block_size: cfg.cache_block_size,
                input_dtype: bf16,
                kv_dtype: cfg.kv_cache_dtype,
                cache_format: "plain".into(),
            },
            // Dense-MHA prefill: k_nope and v for every prefill row.
            kv_b_proj: gemm(heads * (nope + v_dim), cfg.kv_lora_rank.clone()),
            // Placeholder: q, k, v in and o out once per prefill row.
            dense_prefill_attention: ew(heads * (2 * nope + v_dim) * 2, heads * v_dim * 2),
            q_absorb: bmm(
                &cfg.mla_bmm_q_absorb_backends,
                cfg.kv_lora_rank.clone(),
                cfg.qk_nope_head_dim.clone(),
            ),
            // Placeholder: one int32 index row in, one global-slot row out.
            index_remap: ew(4 * cfg.selected_k, 4 * cfg.selected_k),
            // `q_nope.new_zeros(Q, H, 64)`: written once per MQA row, nothing read.
            q_rope_zero_fill: ew(0, heads * FA3_QK_HEAD_DIM * bf16.size_bytes()),
            sparse_attention: DsaSparseMlaAttentionKernelConfig {
                backends: cfg.sparse_attention_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_heads: cfg.num_heads.clone(),
                num_kv_heads: Dim::param("num_kv_heads", 1),
                selected_k: cfg.selected_k,
                latent_dim: cfg.kv_lora_rank.clone(),
                rope_dim: Dim::param("qk_rope_head_dim", 0),
                value_dim: cfg.kv_lora_rank.clone(),
                softmax_scale_denominator: integer_sqrt(nope),
                q_dtype: bf16,
                cache_dtype: cfg.kv_cache_dtype,
                index_dtype: "int32".into(),
                output_dtype: bf16,
                valid_counts_pattern: ValidCountsPattern::PooledUniformFull {
                    index_topk: cfg.index_topk,
                    index_kpool: kpool,
                },
                index_distribution: SPARSE_INDEX_DISTRIBUTION.into(),
                cache_layout: "token_major_mqa_bf16_latent".into(),
            },
            v_up: bmm(
                &cfg.mla_bmm_v_up_backends,
                cfg.v_head_dim.clone(),
                cfg.kv_lora_rank.clone(),
            ),
            o_proj: gemm(cfg.hidden.get(), (heads * v_dim).into()),
            glue: ew(GLUE_BYTES_PER_TOKEN, GLUE_BYTES_PER_TOKEN),
            indexer_glue: ew(GLUE_BYTES_PER_TOKEN, GLUE_BYTES_PER_TOKEN),
            prefill_glue: ew(4 * cfg.selected_k, 4 * cfg.selected_k),
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: Glm53DsaSpecAttnLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let r = resolved.clone();
        let n = name.as_str();
        let ew = |suffix: &str, config| atomic(n, suffix, config, ElementwiseKernel::build, bridge);
        let gemm =
            |suffix: &str, config| atomic(n, suffix, config, SingleGemmKernel::build, bridge);
        Ok(Self {
            fused_qkv_a: gemm("fused_qkv_a", r.fused_qkv_a)?,
            q_kv_norm: atomic(
                n,
                "q_kv_a_norm",
                r.q_kv_norm,
                QKvRmsNormKernel::build,
                bridge,
            )?,
            q_b: gemm("q_b_proj", r.q_b)?,
            index_wq_b: gemm("indexer.wq_b", r.index_wq_b)?,
            index_wk_weights: gemm("indexer.wk_weights", r.index_wk_weights)?,
            index_head_weights: atomic(
                n,
                "indexer.head_weights",
                r.index_head_weights,
                GemmFp32OutputKernel::build,
                bridge,
            )?,
            index_k_norm: ew("indexer.k_norm", r.index_k_norm)?,
            index_q_fwht_quant: ew("indexer.q_fwht_quant", r.index_q_fwht_quant)?,
            index_weight_scale: ew("indexer.weight_scale", r.index_weight_scale)?,
            kpool_gate_score: gemm("indexer.kpool_gate_score", r.kpool_gate_score)?,
            prefill_gather: ew("indexer.prefill_gather", r.prefill_gather)?,
            kpool_decode_update: ew("indexer.kpool_decode_update", r.kpool_decode_update)?,
            kpool_prefill_write: ew("indexer.kpool_prefill_write", r.kpool_prefill_write)?,
            kpool_tail_seed: ew("indexer.kpool_tail_seed", r.kpool_tail_seed)?,
            mqa_logits_decode: atomic(
                n,
                "indexer.mqa_logits_decode",
                r.mqa_logits_decode,
                DsaPagedMqaLogitsDecodeKernel::build,
                bridge,
            )?,
            topk_decode: atomic(
                n,
                "indexer.topk_decode",
                r.topk_decode,
                DsaPersistentTopkDecodeKernel::build,
                bridge,
            )?,
            mqa_logits_prefill: atomic(
                n,
                "indexer.mqa_logits_prefill",
                r.mqa_logits_prefill,
                DsaMqaLogitsPrefillKernel::build,
                bridge,
            )?,
            topk_prefill: atomic(
                n,
                "indexer.topk_prefill",
                r.topk_prefill,
                DsaTopkPrefillKernel::build,
                bridge,
            )?,
            expand_pools: ew("indexer.expand_pools", r.expand_pools)?,
            mla_cache_append: atomic(
                n,
                "sparse_mla.mla_cache_append",
                r.mla_cache_append,
                MlaCacheAppendKernel::build,
                bridge,
            )?,
            kv_b_proj: gemm("dense_prefill.kv_b_proj", r.kv_b_proj)?,
            dense_prefill_attention: ew("dense_prefill.attention", r.dense_prefill_attention)?,
            q_absorb: atomic(n, "q_absorb", r.q_absorb, BatchedGemmKernel::build, bridge)?,
            index_remap: ew("sparse_mla.index_remap", r.index_remap)?,
            q_rope_zero_fill: ew("sparse_mla.q_rope_zero_fill", r.q_rope_zero_fill)?,
            sparse_attention: atomic(
                n,
                "sparse_mla.attention",
                r.sparse_attention,
                DsaSparseMlaAttentionKernel::build,
                bridge,
            )?,
            v_up: atomic(n, "v_up", r.v_up, BatchedGemmKernel::build, bridge)?,
            o_proj: gemm("o_proj", r.o_proj)?,
            glue: ew("glue", r.glue)?,
            indexer_glue: ew("indexer_glue", r.indexer_glue)?,
            prefill_glue: ew("prefill_glue", r.prefill_glue)?,
            name,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let cfg = &self.resolved.raw_cfg;
        CostNode::Labeled {
            label: format!(
                "{} (Glm53DsaSpecAttnLocalWorklet) [TP rank; H={}, index heads={} replicated, \
                 kpool top-{} x {}, verify next_n={}, {} KV]",
                self.name,
                cfg.num_heads,
                cfg.index_num_heads,
                cfg.index_topk,
                cfg.index_kpool,
                cfg.verify_width,
                cfg.kv_cache_dtype.as_str()
            ),
            child: Box::new(CostNode::Sum(vec![
                self.fused_qkv_a.compile(builder),
                self.q_kv_norm.compile(builder),
                self.q_b.compile(builder),
                self.index_wq_b.compile(builder),
                self.index_wk_weights.compile(builder),
                self.index_head_weights.compile(builder),
                self.index_k_norm.compile(builder),
                self.index_q_fwht_quant.compile(builder),
                self.index_weight_scale.compile(builder),
                self.kpool_gate_score.compile(builder),
                repeated(&self.prefill_gather, PREFILL_GATHER_LAUNCHES, builder),
                self.kpool_decode_update.compile(builder),
                self.kpool_prefill_write.compile(builder),
                self.kpool_tail_seed.compile(builder),
                self.mqa_logits_decode.compile(builder),
                self.topk_decode.compile(builder),
                self.mqa_logits_prefill.compile(builder),
                self.topk_prefill.compile(builder),
                self.expand_pools.compile(builder),
                self.mla_cache_append.compile(builder),
                self.kv_b_proj.compile(builder),
                self.dense_prefill_attention.compile(builder),
                self.q_absorb.compile(builder),
                self.index_remap.compile(builder),
                self.q_rope_zero_fill.compile(builder),
                self.sparse_attention.compile(builder),
                self.v_up.compile(builder),
                self.o_proj.compile(builder),
                repeated(&self.glue, GLUE_LAUNCHES, builder),
                repeated(&self.indexer_glue, INDEXER_GLUE_LAUNCHES, builder),
                repeated(&self.prefill_glue, PREFILL_GLUE_LAUNCHES, builder),
            ])),
        }
    }

    pub fn eval(&self, input: &Glm53DsaSpecAttnLocalWorkletInput, ev: &mut Evaluator) {
        let cfg = &self.resolved.raw_cfg;
        let w = derive_work(input, cfg)
            .unwrap_or_else(|reason| panic!("invalid Glm53DsaSpecAttnLocalWorkletInput: {reason}"));
        let rows = SingleGemmKernelInput { m: w.total_tokens };
        let tokens = |num_tokens| ElementwiseKernelInput { num_tokens };
        let all = tokens(w.total_tokens);
        let no_prefill = w.prefill_tokens == 0;
        let skip = input.skip_indexer;
        let no_decode_indexer = skip || w.decode_requests == 0;
        let indexer_decode_batch = if self.resolved.flatten_decode_indexer {
            w.decode_rows
        } else {
            w.decode_requests
        };
        push_or_zero(&self.fused_qkv_a, rows.clone(), false, ev);
        push_or_zero(
            &self.q_kv_norm,
            QKvRmsNormKernelInput {
                num_tokens: w.total_tokens,
            },
            false,
            ev,
        );
        push_or_zero(&self.q_b, rows.clone(), false, ev);
        push_or_zero(&self.index_wq_b, rows.clone(), skip, ev);
        push_or_zero(&self.index_wk_weights, rows.clone(), skip, ev);
        push_or_zero(
            &self.index_head_weights,
            GemmFp32OutputKernelInput { m: w.total_tokens },
            skip,
            ev,
        );
        push_or_zero(&self.index_k_norm, all.clone(), skip, ev);
        push_or_zero(&self.index_q_fwht_quant, all.clone(), skip, ev);
        push_or_zero(&self.index_weight_scale, all.clone(), skip, ev);
        push_or_zero(&self.kpool_gate_score, rows.clone(), skip, ev);
        push_or_zero(
            &self.prefill_gather,
            tokens(w.prefill_tokens),
            skip || no_prefill,
            ev,
        );
        push_or_zero(
            &self.kpool_decode_update,
            tokens(w.decode_rows),
            no_decode_indexer,
            ev,
        );
        push_or_zero(
            &self.kpool_prefill_write,
            tokens(w.prefill_tokens),
            skip || no_prefill,
            ev,
        );
        push_or_zero(
            &self.kpool_tail_seed,
            tokens(w.prefill_tokens),
            skip || no_prefill,
            ev,
        );
        push_or_zero(
            &self.mqa_logits_decode,
            DsaPagedMqaLogitsDecodeKernelInput {
                batch_size: indexer_decode_batch,
                context_len: w.decode_pools,
            },
            no_decode_indexer,
            ev,
        );
        push_or_zero(
            &self.topk_decode,
            DsaPersistentTopkDecodeKernelInput {
                batch_size: indexer_decode_batch,
                context_len: w.decode_pools,
            },
            no_decode_indexer,
            ev,
        );
        let skip_prefill_indexer = skip || w.prefill_indexer_rows == 0;
        push_or_zero(
            &self.mqa_logits_prefill,
            DsaMqaLogitsPrefillKernelInput {
                num_queries: w.prefill_indexer_rows,
                num_keys: w.prefill_indexer_keys,
            },
            skip_prefill_indexer,
            ev,
        );
        push_or_zero(
            &self.topk_prefill,
            DsaTopkPrefillKernelInput {
                num_queries: w.prefill_indexer_rows,
                num_keys: w.prefill_indexer_keys,
            },
            skip_prefill_indexer,
            ev,
        );
        let expand_rows = w.decode_rows + w.prefill_indexer_rows;
        push_or_zero(
            &self.expand_pools,
            tokens(expand_rows),
            skip || expand_rows == 0,
            ev,
        );
        push_or_zero(
            &self.mla_cache_append,
            MlaCacheAppendKernelInput {
                num_tokens: w.total_tokens,
            },
            false,
            ev,
        );
        let dense = w.dense_prefill_rows;
        push_or_zero(
            &self.kv_b_proj,
            SingleGemmKernelInput { m: dense },
            dense == 0,
            ev,
        );
        push_or_zero(&self.dense_prefill_attention, tokens(dense), dense == 0, ev);
        let mqa = w.mqa_rows;
        push_or_zero(
            &self.q_absorb,
            BatchedGemmKernelInput { m: mqa },
            mqa == 0,
            ev,
        );
        push_or_zero(&self.index_remap, tokens(mqa), mqa == 0, ev);
        push_or_zero(&self.q_rope_zero_fill, tokens(mqa), mqa == 0, ev);
        push_or_zero(
            &self.sparse_attention,
            DsaSparseMlaAttentionKernelInput {
                num_queries: mqa,
                num_cache_tokens: w.sparse_cache_tokens,
            },
            mqa == 0,
            ev,
        );
        push_or_zero(&self.v_up, BatchedGemmKernelInput { m: mqa }, mqa == 0, ev);
        push_or_zero(&self.o_proj, rows, false, ev);
        push_or_zero(&self.glue, all.clone(), false, ev);
        push_or_zero(&self.indexer_glue, all.clone(), skip, ev);
        push_or_zero(&self.prefill_glue, all, skip || no_prefill, ev);
    }
}

struct Work {
    total_tokens: u32,
    prefill_tokens: u32,
    decode_requests: u32,
    decode_rows: u32,
    /// `ceil(max decode context / index_kpool)`.
    decode_pools: u32,
    /// As on B200: every prefill row once any prefill context exceeds
    /// `index_topk`, else 0.
    prefill_indexer_rows: u32,
    prefill_indexer_keys: u32,
    /// Prefill rows on the dense-MHA path (every prefill context fits in
    /// `index_topk`), else 0.
    dense_prefill_rows: u32,
    /// Rows on the sparse-MQA path: decode rows, plus the prefill rows of a
    /// batch too long for dense MHA.
    mqa_rows: u32,
    /// The pooled-uniform context the one FA3 call over `mqa_rows` is priced
    /// at (`sparse_attention_input`); 0 without MQA rows.
    sparse_cache_tokens: u32,
}

fn derive_work(
    input: &Glm53DsaSpecAttnLocalWorkletInput,
    cfg: &Glm53DsaSpecAttnLocalWorkletConfig,
) -> Result<Work, String> {
    let (index_topk, index_kpool) = (cfg.index_topk, cfg.index_kpool);
    let mut prefill_tokens = 0_u32;
    let mut max_prefill_context = 0_u32;
    let mut pooled_pairs = 0_u64;
    for (index, &(prefix, append)) in input.prefill_chunk_pairs.iter().enumerate() {
        if append == 0 {
            return Err(format!("prefill request {index} append must be positive"));
        }
        let context = prefix
            .checked_add(append)
            .ok_or("prefill context overflows u32")?;
        prefill_tokens = prefill_tokens
            .checked_add(append)
            .ok_or("prefill token sum overflows u32")?;
        max_prefill_context = max_prefill_context.max(context);
        pooled_pairs +=
            pool_prefix_sum(context, index_kpool) - pool_prefix_sum(prefix, index_kpool);
    }
    let long_prefill = max_prefill_context > index_topk;
    let (prefill_indexer_rows, prefill_indexer_keys) = if long_prefill {
        let rows = u64::from(prefill_tokens);
        let keys = (2 * pooled_pairs + rows * (rows - 1)).div_ceil(2 * rows);
        let keys =
            u32::try_from(keys.max(rows)).map_err(|_| "prefill indexer keys overflow u32")?;
        (prefill_tokens, keys)
    } else {
        (0, 0)
    };
    if input.decode_kv_lens.contains(&0) {
        return Err("decode KV lengths must be positive".into());
    }
    let decode_requests = input.decode_kv_lens.len() as u32;
    if decode_requests > 0 && input.decode_width == 0 {
        return Err("decode_width must be positive".into());
    }
    if decode_requests > 0 && !input.skip_indexer && input.decode_width != cfg.verify_width {
        return Err(format!(
            "an indexer-running forward must decode {} rows per request, got {}",
            cfg.verify_width, input.decode_width
        ));
    }
    let decode_rows = decode_requests
        .checked_mul(input.decode_width)
        .ok_or("decode rows overflow u32")?;
    let max_decode = input.decode_kv_lens.iter().copied().max().unwrap_or(0);
    let total_tokens = prefill_tokens
        .checked_add(decode_rows)
        .ok_or("token sum overflows u32")?;
    if total_tokens == 0 {
        return Err("an iteration must carry at least one token".into());
    }
    let dense_prefill_rows = if long_prefill { 0 } else { prefill_tokens };
    // Every MQA row's own context: verify row `i` of a `width`-row request
    // sees `kv_len - (width - 1 - i)` tokens; a long batch's prefill rows walk
    // their causal spans.
    let width = input.decode_width;
    let decode_contexts = input.decode_kv_lens.iter().flat_map(|&kv_len| {
        (0..width).map(move |row| kv_len.saturating_sub(width - 1 - row).max(1))
    });
    let prefill_contexts = input
        .prefill_chunk_pairs
        .iter()
        .filter(|_| long_prefill)
        .flat_map(|&(prefix, append)| prefix + 1..=prefix + append);
    let sparse_cache_tokens = sparse_attention_context(
        decode_contexts.chain(prefill_contexts),
        index_topk,
        index_kpool,
    );
    Ok(Work {
        total_tokens,
        prefill_tokens,
        decode_requests,
        decode_rows,
        decode_pools: max_decode.div_ceil(index_kpool),
        prefill_indexer_rows,
        prefill_indexer_keys,
        dense_prefill_rows,
        mqa_rows: total_tokens - dense_prefill_rows,
        sparse_cache_tokens,
    })
}

/// The `num_cache_tokens` that prices one FA3 call over these row contexts on
/// the kind's `PooledUniformFull` encoding.
///
/// A row at context `n` reads `min(n, index_topk + n mod index_kpool)` slots
/// (the kpool law). The encoding gives every row the same context `S`, reading
/// `min(S, index_topk + row mod index_kpool)`. Below `index_topk` that is `S`
/// itself, so `S` is the rows' mean count, which keeps the call's total valid
/// slots. Once the mean reaches `index_topk` the rows are saturated, and the
/// call's largest context gives the saturated phase mix and gather footprint.
/// Returns 0 for no rows.
fn sparse_attention_context(
    contexts: impl Iterator<Item = u32>,
    index_topk: u32,
    index_kpool: u32,
) -> u32 {
    let (mut rows, mut slots, mut max_context) = (0_u64, 0_u64, 0_u32);
    for context in contexts {
        rows += 1;
        slots += u64::from(context.min(index_topk + context % index_kpool));
        max_context = max_context.max(context);
    }
    if rows == 0 {
        return 0;
    }
    let mean = (2 * slots + rows) / (2 * rows);
    if mean < u64::from(index_topk) {
        mean.max(1) as u32
    } else {
        max_context
    }
}

/// Whether DeepGEMM's paged MQA logits takes `next_n` rows per request on
/// this GPU (vLLM `native_next_n_supported` / `_supports_native_decode`).
fn native_decode_next_n_supported(gpu_name: &str, next_n: u32) -> bool {
    match compute_capability(gpu_name) {
        Some((9, _)) => matches!(next_n, 1 | 2 | 4),
        Some((10, _)) => true,
        _ => matches!(next_n, 1 | 2),
    }
}

/// `sum over x in 1..=n of x / kpool`: the pools spanned by the rows at
/// positions `0..n` (as `glm53_dsa_attn_local`).
fn pool_prefix_sum(n: u32, kpool: u32) -> u64 {
    let (n, kpool) = (u64::from(n), u64::from(kpool));
    let (q, r) = (n / kpool, n % kpool);
    kpool * q * q.saturating_sub(1) / 2 + q * (r + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Glm53DsaSpecAttnLocalWorkletConfig {
        Glm53DsaSpecAttnLocalWorkletConfig {
            hidden: 4096.into(),
            num_heads: 16.into(),
            q_lora_rank: 1536.into(),
            kv_lora_rank: 512.into(),
            qk_nope_head_dim: 256.into(),
            v_head_dim: 256.into(),
            index_num_heads: 32.into(),
            index_head_dim: 128.into(),
            index_topk: 2048,
            index_kpool: 4,
            selected_k: 2176,
            max_model_len: 8192,
            cache_block_size: 64,
            rms_eps: 1e-5,
            verify_width: 6,
            kv_cache_dtype: DType::Bf16,
            gpu_name: "NVIDIA H200".into(),
            bf16_gemm_backends: vec!["torch_linear_vllm"],
            fp32_gemm_backends: vec!["torch_cublas"],
            qkv_norm_backends: vec!["vllm_triton"],
            mla_bmm_q_absorb_backends: vec!["torch_mla_q_absorb_no_rope"],
            mla_bmm_v_up_backends: vec!["torch_mla_v_up_unpadded"],
            mqa_logits_backends: vec!["deepgemm_fp8"],
            topk_backends: vec!["vllm_fork_cuda"],
            mqa_logits_prefill_backends: vec!["deepgemm_fp8"],
            topk_prefill_backends: vec!["vllm_cuda"],
            mla_cache_append_backends: vec!["vllm_cuda"],
            sparse_attention_backends: vec![crate::timing::kernels::FA3_SPARSE_BACKEND],
            elementwise_backends: vec!["triton"],
        }
    }

    #[test]
    fn verify_rows_reach_the_decode_indexer_k_plus_one_per_request() {
        let r = Glm53DsaSpecAttnLocalWorklet::resolve_config(&cfg());
        // SM90 DeepGEMM has no next_n = 6: one single-token row per query.
        assert!(r.flatten_decode_indexer);
        assert_eq!(r.mqa_logits_decode.next_n, 1);
        assert_eq!(r.topk_decode.next_n, 1);
        let b200 =
            Glm53DsaSpecAttnLocalWorklet::resolve_config(&Glm53DsaSpecAttnLocalWorkletConfig {
                gpu_name: "NVIDIA B200".into(),
                ..cfg()
            });
        assert!(!b200.flatten_decode_indexer);
        assert_eq!(b200.mqa_logits_decode.next_n, 6);
        let mtp3 =
            Glm53DsaSpecAttnLocalWorklet::resolve_config(&Glm53DsaSpecAttnLocalWorkletConfig {
                verify_width: 4,
                ..cfg()
            });
        assert!(!mtp3.flatten_decode_indexer);
        assert_eq!(mtp3.topk_decode.next_n, 4);
        assert_eq!(r.mla_cache_append.kv_dtype, DType::Bf16);
        assert_eq!((r.kv_b_proj.n.get(), r.kv_b_proj.k.get()), (8192, 512));
        let w = derive_work(
            &Glm53DsaSpecAttnLocalWorkletInput {
                prefill_chunk_pairs: Vec::new(),
                decode_kv_lens: vec![1006, 4006],
                decode_width: 6,
                skip_indexer: false,
            },
            &cfg(),
        )
        .unwrap();
        assert_eq!(
            (w.decode_requests, w.decode_rows, w.total_tokens),
            (2, 12, 12)
        );
        assert_eq!(w.decode_pools, 1002);
        assert_eq!((w.dense_prefill_rows, w.mqa_rows), (0, 12));
    }

    #[test]
    fn short_prefills_take_dense_mha_and_long_ones_sparse_mqa() {
        let short = derive_work(
            &Glm53DsaSpecAttnLocalWorkletInput {
                prefill_chunk_pairs: vec![(0, 2048), (0, 256)],
                decode_kv_lens: vec![300],
                decode_width: 6,
                skip_indexer: false,
            },
            &cfg(),
        )
        .unwrap();
        assert_eq!((short.dense_prefill_rows, short.mqa_rows), (2304, 6));
        assert_eq!(short.prefill_indexer_rows, 0);
        let long = derive_work(
            &Glm53DsaSpecAttnLocalWorkletInput {
                prefill_chunk_pairs: vec![(2304, 1792)],
                decode_kv_lens: Vec::new(),
                decode_width: 6,
                skip_indexer: false,
            },
            &cfg(),
        )
        .unwrap();
        assert_eq!((long.dense_prefill_rows, long.mqa_rows), (0, 1792));
        assert_eq!(long.prefill_indexer_rows, 1792);
    }

    #[test]
    fn index_sharing_draft_steps_may_decode_one_row() {
        assert!(derive_work(
            &Glm53DsaSpecAttnLocalWorkletInput {
                prefill_chunk_pairs: Vec::new(),
                decode_kv_lens: vec![500],
                decode_width: 1,
                skip_indexer: false,
            },
            &cfg(),
        )
        .is_err());
        let w = derive_work(
            &Glm53DsaSpecAttnLocalWorkletInput {
                prefill_chunk_pairs: Vec::new(),
                decode_kv_lens: vec![500, 700],
                decode_width: 1,
                skip_indexer: true,
            },
            &cfg(),
        )
        .unwrap();
        assert_eq!(w.mqa_rows, 2);
    }

    /// Catches the sparse attention leaf drifting from the profiled FA3
    /// identity (rope-free BF16 latent rows, 1/16 scale, pooled pattern) or
    /// the zero fill reading bytes or covering the 512-wide latent.
    #[test]
    fn sparse_attention_is_the_fa3_l1_with_its_q_zero_fill() {
        let r = Glm53DsaSpecAttnLocalWorklet::resolve_config(&cfg());
        let a = &r.sparse_attention;
        assert_eq!(a.backends, vec!["vllm_fa3_sparse_bf16"]);
        assert_eq!((a.num_heads.get(), a.num_kv_heads.get()), (16, 1));
        assert_eq!(
            (a.latent_dim.get(), a.rope_dim.get(), a.value_dim.get()),
            (512, 0, 512)
        );
        assert_eq!((a.selected_k, a.softmax_scale_denominator), (2176, 16));
        assert_eq!(
            (a.q_dtype, a.cache_dtype, a.output_dtype),
            (DType::Bf16, DType::Bf16, DType::Bf16)
        );
        assert_eq!(a.cache_layout, "token_major_mqa_bf16_latent");
        assert_eq!(
            a.valid_counts_pattern,
            ValidCountsPattern::PooledUniformFull {
                index_topk: 2048,
                index_kpool: 4
            }
        );
        assert_eq!(r.q_rope_zero_fill.input_bytes_per_token.get(), 0);
        assert_eq!(r.q_rope_zero_fill.output_bytes_per_token.get(), 16 * 64 * 2);
    }

    /// Catches a sparse call priced at the wrong total of valid slots: verify
    /// rows below `index_topk` read their own descending contexts, saturated
    /// rows a phase mix at the largest context.
    #[test]
    fn one_fa3_call_keeps_the_rows_valid_slot_total() {
        let work = |decode_kv_lens: Vec<u32>, prefill_chunk_pairs: Vec<(u32, u32)>| {
            derive_work(
                &Glm53DsaSpecAttnLocalWorkletInput {
                    prefill_chunk_pairs,
                    decode_kv_lens,
                    decode_width: 6,
                    skip_indexer: false,
                },
                &cfg(),
            )
            .unwrap()
        };
        // 1001..=1006 read themselves (6021); 4001..=4006 read 2048 + n mod 4
        // (12297): 18318 slots over 12 rows, mean 1526.5.
        let mixed = work(vec![1006, 4006], Vec::new());
        assert_eq!((mixed.mqa_rows, mixed.sparse_cache_tokens), (12, 1527));
        let saturated = work(vec![5000, 9000], Vec::new());
        assert_eq!(saturated.sparse_cache_tokens, 9000);
        // A long prefill joins the decode rows in the same call.
        let long = work(vec![300], vec![(2304, 1792)]);
        assert_eq!(long.mqa_rows, 6 + 1792);
        let slots = (295..=300).sum::<u64>() + (2305..=4096_u64).map(|n| 2048 + n % 4).sum::<u64>();
        assert_eq!(
            u64::from(long.sparse_cache_tokens),
            (2 * slots + 1798) / (2 * 1798)
        );
        // Short prefills take dense MHA and leave only the verify rows.
        let short = work(vec![300], vec![(0, 2048)]);
        assert_eq!((short.mqa_rows, short.sparse_cache_tokens), (6, 298));
    }

    /// The whole MQA batch is one FA3 call, so the measured Combine drop
    /// falls at the call's row count: 165 verifying requests (990 rows) still
    /// combine, 166 (996 rows) do not.
    #[test]
    fn the_combine_boundary_falls_at_the_calls_row_count() {
        use crate::timing::kernels::FA3_COMBINE_MAX_QUERIES;
        let rows = |requests: usize| {
            derive_work(
                &Glm53DsaSpecAttnLocalWorkletInput {
                    prefill_chunk_pairs: Vec::new(),
                    decode_kv_lens: vec![3000; requests],
                    decode_width: 6,
                    skip_indexer: false,
                },
                &cfg(),
            )
            .unwrap()
            .mqa_rows
        };
        assert!(rows(165) <= FA3_COMBINE_MAX_QUERIES);
        assert!(rows(166) > FA3_COMBINE_MAX_QUERIES);
    }
}
