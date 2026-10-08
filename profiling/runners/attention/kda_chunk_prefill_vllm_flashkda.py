"""vLLM-fork FlashKDA runner for the GLM-5.3-Flash KDA chunked prefill.

This runner only allocates tensors, times one public call, and returns
metrics. DB writes, subprocess selection and registry routing live in L1b.

It times the body of ``Glm5NextLinearAttention._flashkda_prefill`` in
``vllm/models/glm5next/common/kda.py``: the ``.contiguous()`` calls on q, k, v
and g, then one ``torch.ops._flashkda_C.fwd`` (``vllm/_flashkda_C.abi3.so``,
vLLM's FlashKDA build). The method itself cannot be called without a layer and
vLLM's workspace manager, so the runner makes the same op call with the same
arguments. ``_resolve_kda_prefill_backend`` selects this path ("auto") on CUDA
compute-capability major 9, 10 or 12 with head_dim 128, bf16 and a bounded
gate. The operands are laid out the way ``_forward`` hands them over on a
prefill step without spec-decode tokens:

- q/k/v are row-strided ``[1,T,H,K]`` views into one contiguous ``[T, 3*H*K]``
  short-conv output (``causal_conv1d_fn`` writes a dense ``[T, 3*H*K]``
  buffer), so the three ``.contiguous()`` calls are three real copies.
- ``g`` is the contiguous bf16 ``f_b_proj`` output ``[1,T,H,K]``: raw gate
  logits; ``.contiguous()`` is a no-op.
- ``beta`` is the raw bf16 logit column of the merged input projection,
  ``[1,T,H]`` with the projection's row stride. FlashKDA applies the sigmoid
  itself and transposes beta to a dense ``[H,T]`` copy inside ``fwd``: the
  fourth copy launch.
- ``A_log`` is ``[1,1,H,1]`` fp32 passed as ``view(-1)``; ``dt_bias`` is
  ``[H*K]`` fp32 passed as ``view(-1, K)``.
- ``initial_state`` is fp32 ``[N,H,V,K]``, as ``gather_initial_states`` returns
  it: nonzero rows for decodes, zero rows for fresh prefills. ``final_state``
  is a separate fp32 ``[N,H,V,K]`` buffer and ``out`` is the contiguous
  ``[1,T,H,V]`` layer output.
- ``workspace`` is uint8, ``get_workspace_size(T, H, N)`` bytes (production
  slices a buffer sized for ``max_num_batched_tokens`` and ``max_num_seqs``).
- ``cu_seqlens`` is int32; ``scale = K**-0.5``; ``lower_bound=-5.0``; no
  checkpoint state.

Timing is the CUPTI sum over every launch of one call (``kernel_name=None``):
the three q/k/v copies, FlashKDA's internal beta transpose,
``_flash_kda_fwd_prepare`` and ``_flash_kda_fwd_recurrence``. Nothing the call
reads is written, so no reset is needed between calls. FlashKDA is CUDA with no
Triton autotuning, so there is no autotune anchor.

FLOPs and bytes reuse the kind's semantic logical counts (at a 64-token chunk
width for every backend); FlashKDA itself chunks by 16 tokens.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention.kda_chunk_prefill_vllm_triton import (
    LOWER_BOUND,
    KdaChunkPrefillShape,
    guard_shape,
    logical_bytes,
    semantic_flops,
    sequence_boundaries,
    sequence_lengths,
)
from profiling.runners.attention.kda_chunk_prefill_vllm_triton import (
    validate_args as _validate_common,
)
from profiling.runners.exceptions import (
    KernelLaunchFailed,
    OOMError,
    ProfilerNotImplemented,
)
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "kda_chunk_prefill:vllm_flashkda"
_EXTENSION = "vllm._flashkda_C"
# FlashKDA's build is specialized for one head width
# (`_resolve_kda_prefill_backend`: head_dim == 128).
HEAD_DIM = 128
# The fork's FlashKDA-vs-naive tolerance with an fp32 recurrent state
# (tests/models/kimi_k3/test_kda.py::test_kda_prefill_correctness, flashkda-fp32).
RMSE_RATIO_TOL = 1e-2


@dataclass(frozen=True)
class _Operands:
    qkv: Any
    q: Any
    k: Any
    v: Any
    g: Any
    projected_beta: Any
    beta: Any
    a_log: Any
    dt_bias: Any
    initial_state: Any
    final_state: Any
    workspace: Any
    out: Any
    cu_seqlens: Any
    boundaries: tuple[int, ...]


def validate_args(
    num_tokens: int,
    max_sequence_length: int,
    num_decode_sequences: int,
    num_heads: int,
    head_dim: int,
    dtype: DType | str,
) -> KdaChunkPrefillShape:
    """The Triton backend's topology rules plus FlashKDA's head width."""
    if DType.from_value(dtype) is not DType.BF16:
        raise ValueError(f"{_BACKEND} requires dtype=bf16, got {dtype}")
    if head_dim != HEAD_DIM or isinstance(head_dim, bool):
        raise ValueError(f"{_BACKEND} requires head_dim={HEAD_DIM}, got {head_dim!r}")
    return _validate_common(
        num_tokens, max_sequence_length, num_decode_sequences, num_heads, head_dim, dtype
    )


def build_operands(
    torch: Any,
    shape: KdaChunkPrefillShape,
    *,
    device: Any,
    workspace_size: Callable[[int, int, int], int],
) -> _Operands:
    """Operands as ``_forward`` lays them out; ``workspace_size(T, H, N)`` is
    ``torch.ops._flashkda_C.get_workspace_size``."""
    tokens, heads, dim = shape.num_tokens, shape.num_heads, shape.head_dim
    lengths = sequence_lengths(shape)
    boundaries = sequence_boundaries(shape)
    generator = torch.Generator(device=device).manual_seed(42)

    def normal(*size: int, dtype: Any) -> Any:
        return torch.randn(*size, generator=generator, dtype=torch.float32, device=device).to(dtype)

    projection = heads * dim
    qkv = normal(tokens, 3 * projection, dtype=torch.bfloat16)
    q, k, v = (part.reshape(1, tokens, heads, dim) for part in qkv.split(projection, dim=-1))
    g = normal(1, tokens, heads, dim, dtype=torch.bfloat16)
    # Raw beta logits: column `3*H*K .. 3*H*K + H` of the merged projection
    # (q | k | v | beta | f_a | g_a), so its row stride is the projection width.
    projected_beta = normal(tokens, 3 * projection + heads + 2 * dim, dtype=torch.bfloat16)
    beta = projected_beta[:, 3 * projection : 3 * projection + heads].unsqueeze(0)
    # Mamba/Kimi-style A init: exp(A_log) uniform in [1, 16].
    a_log = (
        torch.empty(1, 1, heads, 1, dtype=torch.float32, device=device)
        .uniform_(1.0, 16.0, generator=generator)
        .log()
    )
    dt_bias = normal(projection, dtype=torch.float32) * 0.1
    initial_state = normal(len(lengths), heads, dim, dim, dtype=torch.float32)
    # gather_initial_states zero-fills rows without prior state (fresh prefills).
    initial_state[shape.num_decode_sequences :] = 0
    final_state = torch.empty_like(initial_state)
    workspace = torch.empty(
        workspace_size(tokens, heads, len(lengths)), dtype=torch.uint8, device=device
    )
    out = torch.empty(1, tokens, heads, dim, dtype=torch.bfloat16, device=device)
    cu_seqlens = torch.tensor(boundaries, dtype=torch.int32, device=device)
    return _Operands(
        qkv=qkv,
        q=q,
        k=k,
        v=v,
        g=g,
        projected_beta=projected_beta,
        beta=beta,
        a_log=a_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        final_state=final_state,
        workspace=workspace,
        out=out,
        cu_seqlens=cu_seqlens,
        boundaries=boundaries,
    )


def invoke(fwd: Any, operands: _Operands) -> Any:
    """The op call ``Glm5NextLinearAttention._flashkda_prefill`` makes."""
    head_dim = operands.q.shape[-1]
    fwd(
        operands.q.contiguous(),
        operands.k.contiguous(),
        operands.v.contiguous(),
        operands.g.contiguous(),
        operands.beta,
        head_dim**-0.5,
        operands.out,
        operands.workspace,
        operands.a_log.view(-1),
        operands.dt_bias.view(-1, head_dim),
        LOWER_BOUND,
        operands.initial_state.contiguous(),
        operands.final_state,
        operands.cu_seqlens.contiguous(),
        None,
        None,
    )
    return operands.out, operands.final_state


def check_correctness(
    torch: Any, fwd: Any, operands: _Operands, shape: KdaChunkPrefillShape
) -> tuple[float, float]:
    """Compare one call against the naive oracle; return (o, state) RMSE ratios."""
    from profiling.runners.attention.kda_chunk_prefill_reference import (
        kda_chunk_prefill_reference,
        rmse_ratio,
    )

    snapshots = {
        name: getattr(operands, name).clone()
        for name in (
            "qkv",
            "g",
            "projected_beta",
            "a_log",
            "dt_bias",
            "initial_state",
            "cu_seqlens",
        )
    }
    output, final_state = invoke(fwd, operands)
    torch.cuda.synchronize()
    tokens, heads, dim = shape.num_tokens, shape.num_heads, shape.head_dim
    if tuple(output.shape) != (1, tokens, heads, dim) or output.dtype is not torch.bfloat16:
        raise AssertionError(f"unexpected output {tuple(output.shape)} {output.dtype}")
    if not (torch.isfinite(output).all() and torch.isfinite(final_state).all()):
        raise AssertionError("non-finite FlashKDA output or state")
    for name, snapshot in snapshots.items():
        if not torch.equal(getattr(operands, name), snapshot):
            raise AssertionError(f"FlashKDA fwd mutated {name}")

    expected_o, expected_state = kda_chunk_prefill_reference(
        operands.q.squeeze(0),
        operands.k.squeeze(0),
        operands.v.squeeze(0),
        operands.g.squeeze(0),
        operands.beta.squeeze(0),
        operands.a_log,
        operands.dt_bias,
        operands.initial_state,
        operands.boundaries,
        lower_bound=LOWER_BOUND,
        beta_is_logits=True,
    )
    o_err = rmse_ratio(expected_o, output.squeeze(0))
    state_err = rmse_ratio(expected_state, final_state)
    if not (o_err < RMSE_RATIO_TOL and state_err < RMSE_RATIO_TOL):
        raise AssertionError(
            f"FlashKDA mismatch vs naive oracle: o rmse ratio {o_err:.2e}, "
            f"state rmse ratio {state_err:.2e} (tol {RMSE_RATIO_TOL:.0e})"
        )
    return o_err, state_err


def load_ops() -> tuple[Any, Callable[[int, int, int], int]]:
    """``(fwd, get_workspace_size)`` from vLLM's FlashKDA extension."""
    try:
        importlib.import_module(_EXTENSION)
        import torch

        return torch.ops._flashkda_C.fwd, torch.ops._flashkda_C.get_workspace_size
    except Exception as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires vllm_env with {_EXTENSION}") from exc


def profile_kda_chunk_prefill_vllm_flashkda(
    num_tokens: int,
    max_sequence_length: int,
    num_decode_sequences: int,
    num_heads: int,
    head_dim: int,
    dtype: DType | str,
) -> ComputeMetrics:
    """Profile one GLM-5.3-Flash KDA FlashKDA chunked-prefill call."""
    shape = validate_args(
        num_tokens, max_sequence_length, num_decode_sequences, num_heads, head_dim, dtype
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {_BACKEND}") from exc
    fwd, workspace_size = load_ops()
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        guard = guard_shape(shape)
        check_correctness(
            torch,
            fwd,
            build_operands(torch, guard, device=device, workspace_size=workspace_size),
            guard,
        )
        operands = build_operands(torch, shape, device=device, workspace_size=workspace_size)

        def kernel() -> Any:
            return invoke(fwd, operands)

        time_ms = Timer.cupti(kernel, warmup=5, kernel_name=None)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed: {exc}") from exc

    elapsed = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=semantic_flops(shape) / elapsed / 1e12 if elapsed > 0 else 0.0,
        memory_bandwidth_gbps=logical_bytes(shape) / elapsed / 1e9 if elapsed > 0 else 0.0,
        energy_j=float(energy_j),
    )


__all__ = [
    "HEAD_DIM",
    "build_operands",
    "check_correctness",
    "invoke",
    "load_ops",
    "profile_kda_chunk_prefill_vllm_flashkda",
    "validate_args",
]
