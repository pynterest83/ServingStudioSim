"""GDN causal-convolution decode kernel kind.

The ``vllm_triton`` backend times one call of vLLM's ``causal_conv1d_update``
(``_causal_conv1d_update_kernel``, grid ``(B, ceil(channels/256))``), made as
the GDN/KDA layer makes it on one of two branches, chosen by ``query_len``:

- ``query_len == 1``: the plain decode step. ``x`` is ``[B, channels]``, the
  state allocation is ``kernel_size - 1`` samples wide, and neither
  ``query_start_loc`` nor ``num_accepted_tokens`` is passed.
- ``query_len > 1``: the speculative verify step (``query_len = K + 1`` for K
  drafts), as ``Glm5NextLinearAttention._forward`` calls it:
  ``conv_state_indices = spec_state_indices[:, 0]``, ``num_accepted_tokens``
  ``[B]``, ``query_start_loc`` (uniform width) and ``max_query_len =
  query_len``. The kernel runs with ``IS_VARLEN`` and ``IS_SPEC_DECODING``: it
  reads the window at offset ``num_accepted - 1`` and stores a state that is
  ``kernel_size - 2 + query_len`` samples wide, so the state allocation is
  ``kernel_size - 1 + K`` wide (``kda_state_shape(num_spec=K)``).
  ``num_accepted_tokens`` only shifts the read offset, so the runner fixes it
  at 1; it is not an arg.

Both branches are the same kernel. The conv state uses vLLM's default "SD"
layout, ``[slots, state_len, channels]`` storage passed as its
``(-1, -2)`` transpose.

The ``torch`` backend measures the complete multi-launch semantic reference of
the plain decode step only (``query_len == 1``). It is a
correctness/performance baseline, not the production fused decode launch, and
must not be selected for production simulation.
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

KIND: str = "gdn_causal_conv_decode"


@dataclass(frozen=True)
class GdnCausalConvDecodeArgs(KernelArgs):
    batch_size: int = arg(unit="requests", doc="Requests decoded or verified together.")
    query_len: int = arg(
        unit="tokens",
        doc="Tokens per request: 1 for a plain decode step, K + 1 to verify K drafts.",
    )
    channels: int = arg(unit="channels", doc="Channels in each token and depthwise filter.")
    kernel_size: int = arg(unit="tokens", doc="Width of each causal filter.")
    dtype: DType = arg(doc="Element type of the input, filter and output.")
    state_dtype: DType = arg(doc="Element type of the convolution state.")


DOC = KernelDoc(
    title="Decode causal convolution",
    summary=(
        "Apply a depthwise causal convolution and SiLU to each request's decode "
        "or draft-verify tokens, updating the convolution state."
    ),
    description=(
        "In Gated DeltaNet and KDA layers, decode runs a short "
        "depthwise causal convolution over each channel: the request's last kernel_size −"
        " 1 samples plus the new token, weighted by that channel's filter, "
        "summed in FP32, then SiLU. A plain decode step has query_len 1 and "
        "shifts the new token into the convolution state. A speculative verify "
        "step convolves query_len = K + 1 tokens per request in order and keeps "
        "the last kernel_size − 2 history samples plus all query_len tokens in "
        "a state widened by K samples, so a rejected draft can be rolled back. "
        "Each request has its own state row in vLLM's default token-major "
        "layout; inputs are bounded BF16 values, and repeated calls keep "
        "advancing the states."
    ),
    category="Attention",
    subcategory="Gated DeltaNet",
    formula=(
        "y[b, t, c] = SiLU(Σᵢ window[b, t, c, i] · weight[c, i]), token by token",
        "T = batch_size · query_len",
        "FLOPs = T · channels · (2·kernel_size + 1)",
        (
            "bytes = (2·T·channels + channels·kernel_size)·bytes(dtype) + "
            "batch_size·channels·(2·kernel_size − 3 + query_len)·bytes(state_dtype) + "
            "4·batch_size (+ 8·batch_size + 4 when query_len > 1)"
        ),
        "TFLOPS = FLOPs / time",
        "GB/s = bytes / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "torch counts every launch of the reference, which updates the state in"
        " place; vllm_triton counts only _causal_conv1d_update_kernel. A verify "
        "call reads its window at offset 0 (one accepted token). Allocation and "
        "the vLLM output check run before timing."
    ),
    caveats=(
        "torch writes a new output tensor, while vllm_triton overwrites the "
        "input token; both update the state in place.",
        "The torch reference checks request indices on the GPU inside each timed call.",
        "torch measures only query_len 1.",
        "Every request in a verify call has the same query_len, and the number "
        "of accepted tokens, which only shifts the read offset, is fixed at 1.",
        "FLOPs and bytes count the operation itself, not the torch backend's "
        "extra tensors and launches.",
    ),
    reference="profiling.runners.attention.gdn_causal_conv_decode_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(compute=frozenset({DType.BF16})),
        runner_ref=RunnerRef(
            module_name=("profiling.runners.attention.gdn_causal_conv_decode_torch"),
            function_name="profile_gdn_causal_conv_decode",
        ),
        table_name=KIND,
        args_schema=GdnCausalConvDecodeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary=(
                "The PyTorch reference of the plain decode step, including its "
                "separate state update and SiLU launches."
            )
        ),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
        ),
        runner_ref=RunnerRef(
            module_name=("profiling.runners.attention.gdn_causal_conv_decode_vllm_triton"),
            function_name="profile_gdn_causal_conv_decode_vllm_triton",
        ),
        table_name=KIND,
        args_schema=GdnCausalConvDecodeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary=(
                "vLLM's causal_conv1d_update Triton call: convolution, SiLU and "
                "in-place state update in one kernel."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/causal_conv1d.py",
        ),
        subprocess_env="vllm_env",
    )
)
