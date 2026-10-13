"""Sparse MLA attention kernel kind over DSA-selected cache tokens."""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import CUPTI_METHOD, BackendDoc, KernelDoc, arg
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND: str = "dsa_sparse_mla_attention"


@dataclass(frozen=True)
class DsaSparseMlaAttentionArgs(KernelArgs):
    num_queries: int = arg(unit="tokens", doc="Query tokens in the attention batch.")
    num_cache_tokens: int = arg(
        unit="tokens", doc="Cache tokens available to the selected indices."
    )
    num_heads: int = arg(unit="heads", doc="Query heads on this GPU.")
    num_kv_heads: int = arg(unit="heads", doc="KV heads shared by the query heads.")
    selected_k: int = arg(unit="tokens", doc="Selected index slots per query.")
    latent_dim: int = arg(unit="elements", doc="Compressed KV latent width.")
    rope_dim: int = arg(
        unit="elements", doc="Rotary-position width appended to the latent; 0 when there is none."
    )
    value_dim: int = arg(unit="elements", doc="Value and output width per head.")
    softmax_scale: float = arg(unit="multiplier", doc="Multiplier applied to attention scores.")
    q_dtype: DType = arg(doc="Query element type.")
    cache_dtype: DType = arg(doc="Compressed KV cache element type.")
    index_dtype: str = arg(doc="Element type of selected indices.")
    output_dtype: DType = arg(doc="Attention output element type.")
    valid_counts: str = arg(doc="Encoded valid selected-index count per query.")
    index_distribution: str = arg(doc="Pattern used to construct selected indices.")
    cache_layout: str = arg(doc="Layout of compressed KV values in the cache.")


DOC = KernelDoc(
    title="Sparse MLA attention",
    summary="Attend to the DSA-selected cache tokens for each query head.",
    description=(
        "In sparse MLA attention, each query head attends only to the cache "
        "tokens the DSA indexer selected. Scores use the compressed "
        "latent and the rotary part, if any; values are the latent. valid_counts sets "
        "how many of the selected_k slots each query uses, and the rest are "
        "masked. Queries and cache values are synthetic, and the selected "
        "indices follow index_distribution."
    ),
    category="Attention",
    subcategory="DSA",
    formula=(
        "O = softmax(softmax_scale · q · K_selectedᵀ) · V_selected",
        "D = latent_dim + rope_dim; Q = num_queries; H = num_heads; K = selected_k",
        "TFLOPS = 2·Q·H·K·(D + value_dim) / time",
        "logical bytes = bq·Q·H·D + 4·Q·K + bc·Σvalid_counts·D + 2·Q·H·value_dim + 8·Q·H",
        "GB/s = logical bytes / time; bq = bc = 2 for BF16, 1 for FP8",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "torch counts every launch of its composite; flashinfer_trtllm_fp8 "
        "counts only fmhaSm100, and vllm_flashmla_bf16 only "
        "sparse_attn_fwd_kernel. vllm_fa3_sparse_bf16 counts every launch of "
        "one FA3 call: prepare_varlen_num_blocks, FlashAttnFwdSm90 and, when FA3 "
        "splits the keys, FlashAttnFwdCombine. vllm_flashinfer_sparse_sm90_bf16 counts "
        "only BatchMLAPageAttentionHopperKernel, the one launch of FlashInfer's planned "
        "MLA run; vllm_flashinfer_sparse_sm90_fp8 counts the same launch over an e4m3 "
        "cache. Setup and output checks run before timing."
    ),
    caveats=(
        "Selected positions and tensor values are constructed, not taken from "
        "an indexer or a serving KV cache.",
        "TFLOPS counts all selected_k slots even when fewer are valid.",
        "torch and vllm_flashmla_bf16 accept only rope_dim = 64; "
        "flashinfer_trtllm_fp8 also accepts rope_dim = 0, with its own cache "
        "layout; vllm_fa3_sparse_bf16 accepts only rope_dim = 0, with the "
        "token_major_mqa_bf16_latent layout. vllm_flashmla_bf16 needs "
        "selected_k to be a multiple of FlashMLA's top-k tile: 128 on SM90, "
        "and 64 at 64 heads or 128 otherwise on SM10x.",
        "GB/s includes 8 bytes per query head for the max-logit and log-sum-exp"
        " outputs, even for backends that return only the attention output.",
        "vllm_fa3_sparse_bf16 excludes vLLM's per-call zero fill of the 64-wide "
        "query that FA3's rope-free path takes in place of a RoPE query.",
        "vllm_flashinfer_sparse_sm90_bf16 accepts only rope_dim = 0 with the "
        "token_major_mqa_bf16_latent layout and GLM-5.3-Flash's pooled counts "
        "(at most index_topk + index_kpool - 1 = 2051 per row). It measures a fresh "
        "plan from the per-row host bounds plus vLLM's 32-token slack, trimmed on device "
        "to the exact counts; production reuses a plan while the bounds stay inside "
        "the slack. The plan, its kv_end clamp and the per-call index pack are "
        "separate launches outside the timed run.",
        "vllm_flashinfer_sparse_sm90_fp8 has the same limits with the "
        "token_major_mqa_fp8_latent layout: e4m3 latent rows at per-tensor scale "
        "1.0 (GLM-5.3-Flash ships no KV scales), the BF16 synthetic cache rounded "
        "to e4m3 and checked against the reference on its exact BF16 "
        "dequantization.",
    ),
    reference="profiling.runners.attention.dsa_sparse_mla_attention_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.BF16}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_sparse_mla_attention",
            function_name="profile_dsa_sparse_mla_attention_torch",
        ),
        table_name=KIND,
        args_schema=DsaSparseMlaAttentionArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary="A PyTorch gather, score, softmax and value-reduction composite on BF16 inputs."
        ),
    )
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer_trtllm_fp8",
        # TRTLLM-GEN sparse MLA (fmhaSm100) runs only on the SM10x family.
        supports=BackendSupport(
            compute=frozenset({DType.FP8_E4M3}),
            kv=frozenset({DType.FP8_E4M3}),
            sm_targets=frozenset({"sm_100f"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_sparse_mla_attention",
            function_name="profile_dsa_sparse_mla_attention_flashinfer_trtllm_fp8",
        ),
        table_name=KIND,
        args_schema=DsaSparseMlaAttentionArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "FlashInfer trtllm_batch_decode_with_kv_cache_mla in sparse mode, FP8 "
                "query and paged cache, on B200."
            ),
            url="https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/mla/_core.py",
        ),
    )
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_flashmla_bf16",
        # FlashMLA's Arch::is_sm90a() / is_sm100f(): exactly SM90, or any SM10x.
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.BF16}),
            sm_targets=frozenset({"sm_90a", "sm_100f"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_sparse_mla_attention",
            function_name="profile_dsa_sparse_mla_attention_vllm_flashmla_bf16",
        ),
        table_name=KIND,
        args_schema=DsaSparseMlaAttentionArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=("vLLM's flash_mla_sparse_fwd on BF16 selected KV tokens."),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/flashmla.py",
        ),
    )
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_fa3_sparse_bf16",
        # vLLM FlashAttnMLASparseBackend.supports_compute_capability: major == 9,
        # and flash_attn_supports_mla() needs FA3 on the SM90 family. Its
        # supports_combination admits only FP16/BF16 KV caches.
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.BF16}),
            sm_targets=frozenset({"sm_90a"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_sparse_mla_attention",
            function_name="profile_dsa_sparse_mla_attention_vllm_fa3_sparse_bf16",
        ),
        table_name=KIND,
        args_schema=DsaSparseMlaAttentionArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "vLLM's FLASH_ATTN_MLA_SPARSE FA3 varlen call on rope-free BF16 "
                "latent rows, with the selected rows as a page-size-1 block table, on H200."
            ),
            url=(
                "https://github.com/vllm-project/vllm/blob/main/"
                "vllm/v1/attention/backends/mla/flashattn_mla_sparse.py"
            ),
        ),
    )
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_flashinfer_sparse_sm90_bf16",
        # vLLM FlashInferMLASparseSM90Backend.supports_compute_capability: major
        # == 9; FlashInfer JIT-builds its BatchMLAPagedAttention FA3 (Hopper
        # wgmma/TMA) module for sm_90a. The FP8 cache is its own backend below.
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.BF16}),
            sm_targets=frozenset({"sm_90a"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_sparse_mla_flashinfer_sm90",
            function_name="profile_dsa_sparse_mla_attention_vllm_flashinfer_sparse_sm90_bf16",
        ),
        table_name=KIND,
        args_schema=DsaSparseMlaAttentionArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "vLLM's FLASHINFER_MLA_SPARSE_SM90: FlashInfer's planned "
                "BatchMLAPagedAttentionWrapper (FA3) run over rope-free BF16 latent rows, "
                "with the selected rows packed as a page-size-1 CSR page table, on H200."
            ),
            url=(
                "https://github.com/vllm-project/vllm/blob/main/"
                "vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py"
            ),
        ),
    )
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_flashinfer_sparse_sm90_fp8",
        # The same SM90-only backend and sm_90a FlashInfer module, instantiated
        # with an e4m3 DTypeKV: FlashInferMLASparseSM90Backend admits
        # kv_cache_dtype fp8 / fp8_e4m3 (`_FP8_KV_DTYPES`), the builder plans
        # float8_e4m3fn and FlashInfer's Hopper kernel repacks the FP8 tiles to
        # BF16 in shared memory (USE_KV_REPACK, mla_hopper.cuh). Queries and
        # output stay BF16 (`supports_quant_query_input = False`).
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.FP8_E4M3}),
            sm_targets=frozenset({"sm_90a"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_sparse_mla_flashinfer_sm90",
            function_name="profile_dsa_sparse_mla_attention_vllm_flashinfer_sparse_sm90_fp8",
        ),
        table_name=KIND,
        args_schema=DsaSparseMlaAttentionArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "vLLM's FLASHINFER_MLA_SPARSE_SM90 with --kv-cache-dtype fp8: the same "
                "planned FlashInfer MLA run over rope-free e4m3 latent rows (per-tensor "
                "scale 1.0), dequantized to BF16 in shared memory, on H200."
            ),
            url=(
                "https://github.com/vllm-project/vllm/blob/main/"
                "vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py"
            ),
        ),
    )
)
