"""KDA (Kimi Delta Attention) recurrent decode and spec-decode verify as one public call.

Wire string: ``"kda_recurrent_decode"``. The facade stem is
``get_kda_recurrent_decode_times`` / ``count_missing_kda_recurrent_decode``.

The measured boundary is one call of the vendored
``vllm.models.glm5next.nvidia.ops.third_party.kda.fused_recurrent_kda``, made
exactly as ``Glm5NextLinearAttention._forward`` makes it on one of two
branches, chosen by ``query_len``:

- ``query_len == 1``: the plain decode step (no prefill, no speculative
  decoding). ``ssm_state_indices`` is ``[B]``, ``cu_seqlens`` is
  ``arange(B + 1)`` and ``num_accepted_tokens`` is not passed.
- ``query_len > 1``: the MTP draft-verify step, ``query_len = K + 1`` tokens
  per request for K drafts. ``ssm_state_indices`` is the ``[B, query_len]``
  ``spec_state_indices_tensor``, ``cu_seqlens`` is
  ``spec_query_start_loc[:B + 1]`` (uniform width) and ``num_accepted_tokens``
  is ``[B]``. Each program reads the state at column ``num_accepted - 1``,
  runs the request's tokens serially and stores a state snapshot per token in
  column ``t``. ``num_accepted_tokens`` only selects which slot is read, so the
  runner fixes it at 1 (read column 0); it is not an arg.

Both branches launch the same ``fused_recurrent_gated_delta_rule_fwd_kernel``
(grid ``(1, ceil(V/8), N*H)``) in KDA mode. It computes the per-channel gate
(``COMPUTE_GATE``), the beta sigmoid and the q/k l2norm in the kernel and
reads q, k, v and beta in place: they are row-strided views into the merged
``in_proj_qkvbfg_a`` output (the short conv updates its q|k|v slice in place),
and v0.31's ``fused_recurrent_kda`` passes their token strides to the kernel
instead of copying them. ``g`` is already contiguous, so the call is one
launch. The 3f667d7 container image's older ``fused_recurrent_kda`` made four
``.contiguous()`` copies first (five launches); rows it measured carry that
image's ``backend_version``.

``_causal_conv1d_update`` runs before the call and is costed elsewhere
(``gdn_causal_conv_decode``, same ``query_len`` axis).

This is a separate kind, not a backend of ``gdn_recurrent_decode``. That kind
times vLLM's ``fused_recurrent_gated_delta_rule_packed_decode``, whose gate is
a scalar per value head (``-exp(A_log[h]) * softplus(a + dt_bias[h])``, from
``a[B,HV]``). Its operands are the packed ``mixed_qkv`` plus ``a``, ``b``,
``A_log[HV]`` and ``dt_bias[HV]``. KDA's gate is per key channel,
``lower_bound * sigmoid(exp(A_log[h]) * (raw_g + dt_bias[h,:]))``, and is
computed from a bf16 ``raw_g[T,H,K]`` and an fp32 ``dt_bias[H*K]``. KDA has no
separate q/k and value head counts, and a different entry point.

The state is fp32 ``[slots, H, V, K]``. ``kda_state_dtype`` hard-codes the
recurrent state to fp32 whatever ``mamba_cache_dtype`` is, so state dtype is
not an arg. ``lower_bound=-5.0``, ``scale=K**-0.5``, BV=8 and one warp are
fixed by the production path.
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

KIND: str = "kda_recurrent_decode"


@dataclass(frozen=True)
class KdaRecurrentDecodeArgs(KernelArgs):
    batch_size: int = arg(unit="requests", doc="Requests decoded or verified together.")
    query_len: int = arg(
        unit="tokens",
        doc="Tokens per request: 1 for a plain decode step, K + 1 to verify K drafts.",
    )
    num_heads: int = arg(unit="heads", doc="Heads, each with its own query, key, value and state.")
    head_dim: int = arg(unit="elements", doc="Features in each query, key and value head.")
    dtype: DType = arg(doc="Element type of the activations, gate input and output.")


DOC = KernelDoc(
    title="KDA recurrent decode",
    summary=(
        "Advance each request's KDA state, gated per key channel, through its "
        "decode or draft-verify tokens and write their outputs."
    ),
    description=(
        "The core of KDA decode. KDA is a gated delta rule like Gated DeltaNet, "
        "but its decay gate has one value per key channel instead of one per "
        "head. vLLM's fused_recurrent_kda launches one Triton kernel that reads "
        "q, k, v and beta in place from the merged input projection, computes "
        "the gate from the raw projection, applies the beta sigmoid and the q "
        "and k L2 norms, and runs each request's tokens in order on its FP32 "
        "state from a slot-indexed state pool. A plain decode step has "
        "query_len 1 and updates the state in place. A speculative verify step "
        "has query_len K + 1 for K drafts: it reads the state of the last "
        "accepted token and stores a state snapshot per token in its own slot, "
        "so a rejected draft can be rolled back. Activations are BF16."
    ),
    category="Attention",
    subcategory="Gated DeltaNet",
    formula=(
        "decay = exp(−5·sigmoid(exp(A_log)·(raw_g + dt_bias))), per head and key channel",
        "q, k = L2Norm(q, k); q = q·head_dim⁻¹ᐟ²; beta = sigmoid(b)",
        "per token: S′ = diag(decay)·S + k·(beta·(v − kᵀ(diag(decay)·S)))ᵀ; y = qᵀS′",
        "T = batch_size·query_len",
        "FLOPs = 7·T·num_heads·head_dim²",
        "bytes = 10·T·num_heads·head_dim + 2·T·num_heads + "
        "4·batch_size·(1 + query_len)·num_heads·head_dim²",
        "TFLOPS = FLOPs / time",
        "GB/s = bytes / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} Five warm-up calls run first, and every launch of the "
        "call is counted. Every call advances the states in place, as vLLM "
        "does; a verify call reads slot column 0 (one accepted token) and "
        "writes one snapshot per token. Before timing, one call is checked "
        "against a token-by-token PyTorch reference and the states are restored."
    ),
    caveats=(
        "The kernel takes any head_dim.",
        "Every request in a verify call has the same query_len, and the number "
        "of accepted tokens, which only picks the slot read, is fixed at 1.",
        "The short causal convolution before the call is a separate kind, gdn_causal_conv_decode.",
        "GB/s counts each input once and each state read once and written once per token.",
    ),
    reference="profiling.runners.attention.kda_recurrent_decode_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.kda_recurrent_decode_vllm_triton",
            function_name="profile_kda_recurrent_decode_vllm_triton",
        ),
        table_name=KIND,
        args_schema=KdaRecurrentDecodeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "vLLM's fused_recurrent_kda: one Triton kernel that reads the "
                "strided inputs in place, gates, normalizes and updates the "
                "indexed FP32 states."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/models/glm5next/nvidia/ops/third_party/kda/kernels.py",
        ),
        # vLLM defaults TRITON_CACHE_AUTOTUNING=1, which would let the first
        # shape tuned in a TRITON_CACHE_DIR fix the configs of every later row
        # and process. The runner instead tunes at a documented anchor per
        # worker and records it through row_provenance.
        worker_env=(("TRITON_CACHE_AUTOTUNING", "0"),),
        row_provenance_ref=RunnerRef(
            module_name="profiling.runners.attention.kda_recurrent_decode_vllm_triton",
            function_name="row_provenance",
        ),
    )
)
