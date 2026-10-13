"""Plain MLA paged-cache gather (and FP8 dequantize) into a contiguous workspace.

vLLM's ``gather_and_maybe_dequant_cache`` copies whole cached contexts out of
the paged MLA cache into a token-major workspace, converting an FP8 cache to
the workspace dtype as it copies. GLM-5.3-Flash's FP8-cache prefill on H200
(the opt-in ``VLLM_FP8_KV_PREFILL_BF16_WORKSPACE`` path of
``FLASHINFER_MLA_SPARSE_SM90``) runs it once per DSA layer over every
prefilling request's context.
"""

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

KIND: str = "mla_cache_gather"


@dataclass(frozen=True)
class MlaCacheGatherArgs(KernelArgs):
    num_requests: int = arg(unit="requests", doc="Requests whose cached context is gathered.")
    num_tokens: int = arg(
        unit="tokens",
        doc="Cache rows gathered over all requests, split as evenly as possible.",
    )
    kv_lora_rank: int = arg(unit="elements", doc="Width of each compressed KV latent.")
    rope_dim: int = arg(unit="elements", doc="Width of each rotary key part; 0 when there is none.")
    block_size: int = arg(unit="tokens", doc="Token positions in each cache page.")
    cache_dtype: DType = arg(doc="Element type of the cache entries.")
    output_dtype: DType = arg(doc="Element type of the gathered workspace rows.")
    cache_format: str = arg(doc="Layout of the MLA cache entries.")


DOC = KernelDoc(
    title="MLA cache gather",
    summary="Copy whole cached MLA contexts from the paged cache into a contiguous workspace.",
    description=(
        "Each request's context, in position order, is read page by page through "
        "its block table and written to consecutive workspace rows; an FP8 cache "
        "is converted to the workspace dtype at its per-tensor scale as it is "
        "copied. num_tokens rows are split as evenly as possible over "
        "num_requests requests, whose pages sit at shuffled physical blocks. "
        "Entries are kv_lora_rank + rope_dim wide in the plain format."
    ),
    category="Attention",
    subcategory="MLA",
    formula=(
        "workspace[cu_seq_lens[i] + p] = scale · cache[block_table[i, p / B], p mod B]",
        "B = block_size",
        "D = kv_lora_rank + rope_dim; G = num_tokens; R = num_requests; "
        "P = Σ ceil(len_i / block_size)",
        "GB/s = [G·D·(cache bytes + output bytes) + 4·P + 4·(R + 1)] / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "vllm_cuda counts only gather_and_maybe_dequant_cache_page, the one launch "
        "of the call, after checking every workspace row against the reference."
    ),
    caveats=(
        "TFLOPS is not reported: the copy has no matrix multiply.",
        "The scale is 1.0, as for a checkpoint without KV scales; E4M3 to BF16 is then exact.",
        "vllm_cuda accepts only entries 320, 512 or 576 wide (the kernel's "
        "specializations), and no seq_starts offsets: every context starts at "
        "position 0.",
    ),
    reference="profiling.runners.attention.mla_cache_gather_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_cuda",
        # A plain CUDA copy kernel built into vLLM's _C extension: no
        # architecture-specific instructions; FP8 e4m3 conversion needs SM89+.
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.BF16, DType.FP8_E4M3}),
            min_compute_capability=(8, 9),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.mla_cache_gather",
            function_name="profile_mla_cache_gather_vllm_cuda",
        ),
        table_name=KIND,
        args_schema=MlaCacheGatherArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "vLLM ops.gather_and_maybe_dequant_cache: one page-per-CTA CUDA "
                "launch that gathers and, for an FP8 cache, dequantizes."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/cache_kernels.cu",
        ),
    )
)

__all__ = ["KIND", "MlaCacheGatherArgs"]
