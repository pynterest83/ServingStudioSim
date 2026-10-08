"""KDA (Kimi Delta Attention) chunked prefill as one public call.

Wire string: ``"kda_chunk_prefill"`` -- the facade stem is
``get_kda_chunk_prefill_times`` / ``count_missing_kda_chunk_prefill``.

The measured boundary is the prefill core call ``Glm5NextLinearAttention._forward``
makes on the prefill branch; ``_resolve_kda_prefill_backend`` picks which one:

- ``vllm_triton``: the vendored
  ``vllm.models.glm5next.nvidia.ops.third_party.kda.chunk_kda_with_fused_gate``.
  It launches a fixed chain of FLA Triton kernels at a 64-token chunk: the
  q/k/v ``.contiguous()`` copies, two l2norms, the fused safe-gate + chunk
  cumsum, scaled K.Kt (inter + intra), the 64x64 triangular solve, w/u
  recompute, the inter-chunk state scan, and the output kernel. It takes beta
  already passed through the fp32 sigmoid, a separate ``_cast_sigmoid`` launch
  outside the call.
- ``vllm_flashkda``: ``_flashkda_prefill``, i.e. the q/k/v/g ``.contiguous()``
  calls and one ``torch.ops._flashkda_C.fwd`` (vLLM's FlashKDA CUDA build,
  16-token chunks). It takes the raw bf16 beta logits and applies the sigmoid
  itself; ``fwd`` transposes beta to a dense ``[H, T]`` copy, then launches
  ``_flash_kda_fwd_prepare`` and ``_flash_kda_fwd_recurrence``. It is the
  "auto" choice on compute capability 9.x/10.x/12.x for bf16, head_dim 128 and
  a bounded gate.

Both compute the same output and final states from the same operands, except
that beta arrives sigmoided for one and as logits for the other. The
gather/scatter of recurrent states are separate launches outside the call and
are costed elsewhere.

This is a separate kind from ``gdn_chunk_delta_rule`` and the FLA
``gdn_chunk_*`` kinds, not a new backend of them. GDN has a scalar per-head
gate ``g[T,H]`` that the caller exponentiates; KDA has a per-channel gate
``gk[T,H,K]`` that the callable derives from the raw projection with
``lower_bound * sigmoid(exp(A_log) * (raw_g + dt_bias))``. It is also a
different callable (FlashInfer CUTLASS for gdn_chunk_delta_rule), and its
operand set includes raw_g, A_log and dt_bias.

The schema keeps ``num_decode_sequences`` separate. In a mixed iteration the
decode tokens are length-1 sequences in the same varlen call. Every one of them
costs a whole chunk (64 tokens for vllm_triton, 16 for vllm_flashkda) in the
chunk-parallel kernels and one more (sequence, head) program in the state
scan. For vllm_triton the capture's 2019 + 29 batch therefore has 61 chunks,
not the 33 that a 2048-token prefill-only batch with the same longest sequence
would have.

Canonical realization of ``(num_tokens, max_sequence_length,
num_decode_sequences)``: first the decode sequences, one token each (vLLM
places decodes first), then ``P = num_tokens - num_decode_sequences`` prefill
tokens as ``P // max_sequence_length`` full sequences plus one remainder, which
is the chunked-prefill shape. The backend's chunk size, fp32 state in
``[N,H,V,K]`` layout, ``safe_gate=True`` and ``lower_bound=-5.0`` are fixed by
the production path. They are not args.
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

KIND: str = "kda_chunk_prefill"


@dataclass(frozen=True)
class KdaChunkPrefillArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Tokens in the call, decode tokens included.")
    max_sequence_length: int = arg(unit="tokens", doc="Tokens in the longest prefill sequence.")
    num_decode_sequences: int = arg(
        unit="requests", doc="Decode requests in the same call, one token each."
    )
    num_heads: int = arg(unit="heads", doc="Heads, each with its own query, key, value and state.")
    head_dim: int = arg(unit="elements", doc="Features in each query, key and value head.")
    dtype: DType = arg(doc="Element type of queries, keys, values, gate input and output.")


DOC = KernelDoc(
    title="KDA chunked prefill",
    summary=(
        "Compute KDA prefill output and final recurrent states, with a decay gate "
        "per key channel, in one call."
    ),
    description=(
        "KDA is a gated delta rule like Gated DeltaNet, but its decay gate has "
        "one value per key channel instead of one per head. In prefill vLLM "
        "makes one chunked call that copies the strided q, k and v views, "
        "L2-normalizes q and k, computes the gate from the raw projection, and "
        "writes every token's output and each sequence's final state: FLA's "
        "chunked Triton kernels (vllm_triton) or the FlashKDA CUDA kernels "
        "(vllm_flashkda). Decode requests "
        "scheduled in the same step join the call as one-token sequences, "
        "placed first; the remaining tokens form full max_sequence_length "
        "sequences plus one shorter remainder. Decode sequences start from "
        "random FP32 states, prefill sequences from zero. vllm_triton receives "
        "beta already passed through a sigmoid; vllm_flashkda receives the raw "
        "logits and applies the sigmoid itself."
    ),
    category="Attention",
    subcategory="Gated DeltaNet",
    formula=(
        "decay = exp(−5·sigmoid(exp(A_log)·(raw_g + dt_bias))), per head and key channel",
        "q, k = L2Norm(q, k); q = q·head_dim⁻¹ᐟ²; beta = sigmoid(b) (inside the call for "
        "vllm_flashkda, before it for vllm_triton)",
        "S′ = diag(decay)·S + k·(beta·(v − kᵀ(diag(decay)·S)))ᵀ; y = qᵀS′, token by token",
        "N = num_decode_sequences + ⌈(num_tokens − num_decode_sequences) / max_sequence_length⌉",
        "FLOPs = 2·num_tokens·num_heads·(3·head_dim² + 4·64·head_dim)",
        "bytes = 10·num_tokens·num_heads·head_dim + 4·num_tokens·num_heads + "
        "8·N·num_heads·head_dim²",
        "TFLOPS = FLOPs / time",
        "GB/s = bytes / time",
    ),
    default_metric="tflops",
    method=(
        f"{CUPTI_METHOD} Five warm-up calls run first; every launch of the call "
        "is counted, the q, k and v copies included (and, for vllm_flashkda, "
        "its internal beta transpose). For vllm_triton, Triton autotuning runs "
        "once per worker, at 2048 tokens (one 2019-token prefill and 29 "
        "decodes); its keys omit the token count, so those configs serve every "
        "later shape. Before timing, the shape cut to at most 8 decodes and "
        "1024-token sequences is checked against a per-token PyTorch reference."
    ),
    caveats=(
        "FLA's chunk kernels (vllm_triton) take head_dim up to 256; FlashKDA "
        "(vllm_flashkda) takes head_dim 128 only.",
        "Each one-token decode sequence occupies a whole chunk in the "
        "chunk-parallel kernels: 64 tokens for vllm_triton, 16 for vllm_flashkda.",
        "For vllm_triton the timed calls reuse one cu_seqlens tensor, so FLA's "
        "cached chunk-index setup, which the first layer of a step pays, is not "
        "in the time.",
        "The gather and scatter of recurrent states run outside the call, and "
        "so does vllm_triton's beta sigmoid.",
        "FLOPs are logical counts at a 64-token chunk width for both backends; "
        "bytes count each input once and leave out the copies and intermediates.",
    ),
    reference="profiling.runners.attention.kda_chunk_prefill_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.kda_chunk_prefill_vllm_triton",
            function_name="profile_kda_chunk_prefill_vllm_triton",
        ),
        table_name=KIND,
        args_schema=KdaChunkPrefillArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "vLLM's chunk_kda_with_fused_gate: q, k and v copies, L2 norms, the "
                "fused gate and chunk cumsum, then FLA's chunked Triton kernels."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/models/glm5next/nvidia/ops/third_party/kda/kernels.py",
        ),
        # vLLM defaults TRITON_CACHE_AUTOTUNING=1, which would let the first
        # shape tuned in a TRITON_CACHE_DIR fix the configs of every later row
        # and process. The runner instead tunes at a documented anchor per
        # worker and records it through row_provenance.
        worker_env=(("TRITON_CACHE_AUTOTUNING", "0"),),
        row_provenance_ref=RunnerRef(
            module_name="profiling.runners.attention.kda_chunk_prefill_vllm_triton",
            function_name="row_provenance",
        ),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_flashkda",
        # vLLM builds _flashkda_C for 9.0a, 10.0f and 12.0f
        # (cmake/external_projects/flashkda.cmake; its SASS holds sm_90a,
        # sm_100 and sm_120). Only sm_90a (H200) has been run against the
        # reference, so only it is declared.
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            sm_targets=frozenset({"sm_90a"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.kda_chunk_prefill_vllm_flashkda",
            function_name="profile_kda_chunk_prefill_vllm_flashkda",
        ),
        table_name=KIND,
        args_schema=KdaChunkPrefillArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "vLLM's FlashKDA prefill: q, k and v copies, then the FlashKDA "
                "CUDA op, which takes raw beta logits and runs 16-token chunks."
            ),
            url="https://github.com/vllm-project/FlashKDA",
        ),
    )
)
