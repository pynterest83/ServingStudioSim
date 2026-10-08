"""Profile vLLM v0.31's public decode top-k dispatcher, ``SparseIndexerTopk("auto")``.

The DSA and GLM-5.3 kpool indexers call ``get_indexer_topk(backend)(logits,
seq_lens, next_n, topk_indices, topk_tokens, max_seq_len)``
(``vllm/model_executor/layers/indexer_topk.py``). With the default
``sparse_indexer_topk_backend = "auto"`` it resolves per call:
``cooperative_topk`` for top_k in {512, 1024, 2048} and at most
``AUTO_COOPERATIVE_MAX_ROWS`` (64) rows with a row stride divisible by 4 on
SM90+ (except SM12x), else ``persistent_topk``. ``persistent_topk``
(``csrc/libtorch_stable/topk.cu``) then picks ``sampled_topk`` above 64 rows
when ``max_seq_len`` reaches its long-row threshold, ``FilteredTopK`` above 32
rows, and its persistent kernel (after a workspace memset) otherwise. The timed
closure is the public ``forward`` call, so every launch that dispatch makes is
counted; the workspace view comes from vLLM's workspace manager, initialized
once outside timing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention.dsa_persistent_topk_decode import require_vllm_top_k
from profiling.runners.attention.dsa_persistent_topk_decode_fork import _max_ragged_lengths
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "dsa_persistent_topk_decode:vllm_auto"
_CONTEXT_MODES = ("uniform", "max_ragged")
# Rows compared with the Torch reference: all of them up to this count, else an
# evenly spaced sample that keeps the first and last rows.
_MAX_CHECKED_ROWS = 64


@dataclass(frozen=True)
class _Operands:
    logits: Any
    lengths: Any
    indices: Any
    next_n: int
    top_k: int
    max_seq_len: int


def _row_lengths(
    batch_size: int, context_len: int, next_n: int, context_mode: str
) -> tuple[int, ...]:
    """Valid logits per row, request-major, as the kind's context_mode defines them."""
    if context_mode == "max_ragged":
        return _max_ragged_lengths(batch_size, context_len, next_n)
    first = context_len - next_n + 1
    return tuple(first + token for _ in range(batch_size) for token in range(next_n))


def _validate_args(
    batch_size: int,
    context_len: int,
    next_n: int,
    max_model_len: int,
    top_k: int,
    logits_row_stride: int,
    logits_dtype: object,
    index_dtype: str,
    context_mode: str,
) -> tuple[int, ...]:
    for name, value in (("batch_size", batch_size), ("next_n", next_n)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive int")
    if type(context_len) is not int or context_len < 0:
        raise ValueError("context_len must be a nonnegative int")
    require_vllm_top_k(_BACKEND, top_k)
    if (
        type(max_model_len) is not int
        or type(logits_row_stride) is not int
        or max_model_len < 1
        or context_len > max_model_len
        or logits_row_stride < max_model_len
    ):
        raise ValueError(
            f"{_BACKEND} requires context_len <= max_model_len, max_model_len >= 1, "
            "and logits_row_stride >= max_model_len"
        )
    if DType.from_value(logits_dtype) is not DType.FP32 or index_dtype != "int32":
        raise ProfilerNotImplemented(f"{_BACKEND} requires fp32 logits and int32 indices")
    if context_mode not in _CONTEXT_MODES:
        raise ValueError(f"context_mode must be one of {_CONTEXT_MODES}, got {context_mode!r}")
    if context_mode == "uniform" and context_len < next_n - 1:
        raise ValueError("uniform context_len must be >= next_n - 1")
    return _row_lengths(batch_size, context_len, next_n, context_mode)


def _build_operands(
    torch: Any,
    *,
    lengths: tuple[int, ...],
    batch_size: int,
    next_n: int,
    max_model_len: int,
    logits_row_stride: int,
    top_k: int,
    context_mode: str,
    device: Any,
) -> _Operands:
    """Padded FP32 logits and request-shaped lengths, as the indexer passes them.

    The row template follows the kind: uniform rows share one increasing
    sequence, max_ragged rows one seeded random row. Logits are a
    ``[:, :max_model_len]`` view of rows spaced ``logits_row_stride`` apart.
    """
    num_rows = len(lengths)
    backing = torch.empty((num_rows, logits_row_stride), dtype=torch.float32, device=device)
    if context_mode == "uniform":
        template = torch.linspace(-1.0, 1.0, logits_row_stride, dtype=torch.float32, device=device)
    else:
        generator = torch.Generator(device=device).manual_seed(0)
        template = torch.randn(
            logits_row_stride, dtype=torch.float32, device=device, generator=generator
        )
    backing.copy_(template)
    return _Operands(
        logits=backing[:, :max_model_len],
        lengths=torch.tensor(lengths, dtype=torch.int32, device=device).view(batch_size, next_n),
        indices=torch.empty((num_rows, top_k), dtype=torch.int32, device=device),
        next_n=next_n,
        top_k=top_k,
        max_seq_len=max(lengths),
    )


def _checked_rows(num_rows: int) -> list[int]:
    if num_rows <= _MAX_CHECKED_ROWS:
        return list(range(num_rows))
    step = (num_rows - 1) / (_MAX_CHECKED_ROWS - 1)
    return sorted({round(i * step) for i in range(_MAX_CHECKED_ROWS)})


def _check_output(torch: Any, operands: _Operands) -> None:
    """Compare sampled rows with the kind's Torch reference.

    Rows no longer than top_k must equal the reference exactly (natural
    positions, then -1). Longer rows must hold top_k distinct valid positions
    whose logit values equal the reference selection's; the kernels do not
    define output order or equal-value tie choice.
    """
    from profiling.runners.attention.dsa_persistent_topk_decode_reference import (
        dsa_persistent_topk_decode_reference,
    )

    rows = torch.tensor(_checked_rows(operands.indices.shape[0]), device=operands.logits.device)
    logits = operands.logits.index_select(0, rows)
    lengths = operands.lengths.view(-1).index_select(0, rows).contiguous()
    actual = operands.indices.index_select(0, rows)
    expected = torch.empty_like(actual)
    dsa_persistent_topk_decode_reference(
        logits, lengths, expected, top_k=operands.top_k, max_seq_len=operands.max_seq_len
    )
    for position in range(rows.numel()):
        length = int(lengths[position].item())
        if length <= operands.top_k:
            if not torch.equal(actual[position], expected[position]):
                raise KernelLaunchFailed(
                    f"{_BACKEND} row {int(rows[position])} (length {length}) differs from "
                    "the natural-position reference"
                )
            continue
        selected = actual[position].long()
        if (
            selected.min() < 0
            or selected.max() >= length
            or selected.unique().numel() != operands.top_k
        ):
            raise KernelLaunchFailed(f"{_BACKEND} returned invalid or duplicate indices")
        torch.testing.assert_close(
            logits[position].index_select(0, selected).sort().values,
            logits[position].index_select(0, expected[position].long()).sort().values,
            atol=0,
            rtol=0,
        )


def profile_dsa_persistent_topk_decode_vllm_auto(
    batch_size: int,
    context_len: int,
    next_n: int,
    max_model_len: int,
    top_k: int,
    logits_row_stride: int,
    logits_dtype: object,
    index_dtype: str,
    context_mode: str,
) -> ComputeMetrics:
    lengths = _validate_args(
        batch_size,
        context_len,
        next_n,
        max_model_len,
        top_k,
        logits_row_stride,
        logits_dtype,
        index_dtype,
        context_mode,
    )
    try:
        import torch
        from vllm.model_executor.layers.indexer_topk import SparseIndexerTopk
        from vllm.v1.worker.workspace import (
            init_workspace_manager,
            is_workspace_manager_initialized,
        )
    except (ImportError, OSError) as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires vllm_env") from exc

    try:
        device = torch.device("cuda", torch.cuda.current_device())
        if not is_workspace_manager_initialized():
            init_workspace_manager(device)
        topk = SparseIndexerTopk("auto")
        operands = _build_operands(
            torch,
            lengths=lengths,
            batch_size=batch_size,
            next_n=next_n,
            max_model_len=max_model_len,
            logits_row_stride=logits_row_stride,
            top_k=top_k,
            context_mode=context_mode,
            device=device,
        )

        def launch() -> None:
            topk(
                operands.logits,
                operands.lengths,
                operands.next_n,
                operands.indices,
                operands.top_k,
                operands.max_seq_len,
            )

        launch()
        torch.cuda.synchronize(device)
        _check_output(torch, operands)
        time_ms = Timer.cupti(launch, kernel_name=None)
        energy_j = Energy.perf(launch, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of memory") from exc
    except (KernelLaunchFailed, ProfilerNotImplemented, TypeError, ValueError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed: {exc}") from exc

    num_rows = len(lengths)
    logical_bytes = 4 * sum(lengths) + 4 * num_rows + 4 * num_rows * top_k
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(logical_bytes / (time_ms / 1000.0) / 1e9),
        energy_j=float(energy_j),
    )


__all__ = ["profile_dsa_persistent_topk_decode_vllm_auto"]
