"""Profile vLLM's ``gather_and_maybe_dequant_cache`` over a plain MLA cache.

Production call (vLLM v0.31 fork, branch exp/fp8-prefill-ws,
``vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py``
``forward_mqa``): with ``VLLM_FP8_KV_PREFILL_BF16_WORKSPACE=1`` and an FP8
cache, each DSA layer gathers the whole context of the step's prefilling
requests into a BF16 workspace::

    ops.gather_and_maybe_dequant_cache(
        src_cache=kv_c_and_k_pe_cache,      # [num_blocks, block_size, 512] uint8
        dst=workspace,                      # [num_tokens, 512] bf16
        block_table=prefill_ws.block_table, # the prefill requests' rows
        cu_seq_lens=prefill_ws.cu_seq_lens,
        token_to_seq=prefill_ws.request_ids,  # unused by the kernel
        num_tokens=prefill_ws.num_tokens,
        kv_cache_dtype=self.kv_cache_dtype,  # "fp8"
        scale=layer._k_scale,
    )

The call is one ``gather_and_maybe_dequant_cache_page`` launch of
``ceil(num_tokens / block_size) + 2 * num_reqs`` CTAs, each copying one page.
The runner builds the cache with shuffled physical blocks so page lookup is
exercised, checks the workspace against the Torch reference outside timing,
and times only that launch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "mla_cache_gather:vllm_cuda"
_CACHE_FORMAT = "plain"
_KERNEL_NAME = "gather_and_maybe_dequant_cache_page"
# `TORCH_CHECK(head_dim == 320 || head_dim == 512 || head_dim == 576)`.
_ENTRY_DIMS = frozenset({320, 512, 576})
_SEED = 20261013


@dataclass(frozen=True)
class _ValidatedArgs:
    lengths: tuple[int, ...]
    entry_dim: int
    block_size: int
    cache_dtype: DType
    output_dtype: DType


@dataclass(frozen=True)
class _Operands:
    cache: Any  # [num_blocks, block_size, D]: bf16, or float8_e4m3fn for FP8
    # What the call receives: vLLM allocates an FP8 cache as uint8 storage,
    # which the kernel reinterprets; a BF16 cache is passed as is.
    src_cache: Any
    workspace: Any  # [num_tokens, D] output dtype
    block_table: Any  # [R, max_blocks] int32
    cu_seq_lens: Any  # [R + 1] int32
    token_to_seq: Any  # [num_tokens] int32, passed through as production does
    scale: Any  # [1] float32, the layer's k_scale


def request_lengths(num_requests: int, num_tokens: int) -> tuple[int, ...]:
    """Split ``num_tokens`` over ``num_requests``; earlier requests get the remainder."""
    base, extra = divmod(num_tokens, num_requests)
    return tuple(base + (1 if index < extra else 0) for index in range(num_requests))


def _validate_args(
    *,
    num_requests: int,
    num_tokens: int,
    kv_lora_rank: int,
    rope_dim: int,
    block_size: int,
    cache_dtype: DType | str,
    output_dtype: DType | str,
    cache_format: str,
) -> _ValidatedArgs:
    for name, value in {
        "num_requests": num_requests,
        "num_tokens": num_tokens,
        "kv_lora_rank": kv_lora_rank,
        "rope_dim": rope_dim,
        "block_size": block_size,
    }.items():
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer")
    if num_requests < 1:
        raise ProfilerNotImplemented(f"num_requests must be >= 1, got {num_requests}")
    if num_tokens < num_requests:
        raise ProfilerNotImplemented(
            f"every request gathers at least one row: num_tokens {num_tokens} < "
            f"num_requests {num_requests}"
        )
    if block_size < 1:
        raise ProfilerNotImplemented(f"block_size must be >= 1, got {block_size}")
    if rope_dim < 0 or kv_lora_rank < 1:
        raise ProfilerNotImplemented("kv_lora_rank must be >= 1 and rope_dim >= 0")
    entry_dim = kv_lora_rank + rope_dim
    if entry_dim not in _ENTRY_DIMS:
        raise ProfilerNotImplemented(
            f"{_BACKEND} entries must be {sorted(_ENTRY_DIMS)} wide, got {entry_dim}"
        )
    cache = DType.from_value(cache_dtype)
    output = DType.from_value(output_dtype)
    if cache not in {DType.BF16, DType.FP8_E4M3}:
        raise ProfilerNotImplemented(f"cache_dtype must be bf16 or fp8_e4m3, got {cache.value}")
    if output is not DType.BF16:
        raise ProfilerNotImplemented(f"output_dtype must be bf16, got {output.value}")
    if cache_format != _CACHE_FORMAT:
        raise ProfilerNotImplemented(f"cache_format must be {_CACHE_FORMAT!r}")
    return _ValidatedArgs(
        lengths=request_lengths(num_requests, num_tokens),
        entry_dim=entry_dim,
        block_size=block_size,
        cache_dtype=cache,
        output_dtype=output,
    )


def _num_pages(lengths: tuple[int, ...], block_size: int) -> int:
    return sum(math.ceil(length / block_size) for length in lengths)


def logical_bytes(validated: _ValidatedArgs) -> int:
    """Rows read and written, one int32 block id per page, the cu_seq_lens array."""
    rows = sum(validated.lengths)
    cache_bytes = 1 if validated.cache_dtype is DType.FP8_E4M3 else 2
    row_bytes = validated.entry_dim * (cache_bytes + 2)
    pages = _num_pages(validated.lengths, validated.block_size)
    return rows * row_bytes + 4 * pages + 4 * (len(validated.lengths) + 1)


def _build_operands(torch: Any, validated: _ValidatedArgs, *, device: Any) -> _Operands:
    block_size = validated.block_size
    pages = [math.ceil(length / block_size) for length in validated.lengths]
    num_blocks = sum(pages) + 1
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_SEED)
    physical = torch.randperm(num_blocks, generator=generator, dtype=torch.int64)
    width = max(pages)
    # Unused table entries point at a real block, as vLLM's padded rows do.
    block_table = torch.zeros((len(pages), width), dtype=torch.int32)
    start = 0
    for request, count in enumerate(pages):
        block_table[request, :count] = physical[start : start + count].to(torch.int32)
        start += count
    cu = [0]
    for length in validated.lengths:
        cu.append(cu[-1] + length)

    entry = validated.entry_dim
    feature = torch.linspace(-0.875, 0.875, entry, device=device)
    rows = (torch.arange(block_size, dtype=torch.float32, device=device) - block_size / 2) / 256.0
    template = (rows[:, None] + feature[None, :]).to(torch.bfloat16)
    cache = torch.empty((num_blocks, block_size, entry), dtype=torch.bfloat16, device=device)
    cache.copy_(template.expand_as(cache))
    # Each block carries its own offset so a wrong page lookup cannot pass.
    cache += ((torch.arange(num_blocks, dtype=torch.float32, device=device) % 29 - 14) / 64.0).to(
        torch.bfloat16
    )[:, None, None]
    if validated.cache_dtype is DType.FP8_E4M3:
        cache = cache.to(torch.float8_e4m3fn)
    num_tokens = cu[-1]
    src_cache = cache.view(torch.uint8) if validated.cache_dtype is DType.FP8_E4M3 else cache
    return _Operands(
        cache=cache,
        src_cache=src_cache,
        workspace=torch.empty((num_tokens, entry), dtype=torch.bfloat16, device=device),
        block_table=block_table.to(device),
        cu_seq_lens=torch.tensor(cu, dtype=torch.int32, device=device),
        token_to_seq=torch.zeros(num_tokens, dtype=torch.int32, device=device),
        scale=torch.ones(1, dtype=torch.float32, device=device),
    )


def _kv_cache_dtype(cache_dtype: DType) -> str:
    """vLLM's ``kv_cache_dtype`` string: ``auto`` for a BF16 cache, ``fp8`` else."""
    return "fp8" if cache_dtype is DType.FP8_E4M3 else "auto"


def _launch(ops: Any, operands: _Operands, *, cache_dtype: DType) -> None:
    num_tokens = int(operands.workspace.shape[0])
    ops.gather_and_maybe_dequant_cache(
        src_cache=operands.src_cache,
        dst=operands.workspace,
        block_table=operands.block_table,
        cu_seq_lens=operands.cu_seq_lens,
        token_to_seq=operands.token_to_seq,
        num_tokens=num_tokens,
        kv_cache_dtype=_kv_cache_dtype(cache_dtype),
        scale=operands.scale,
    )


def _check_correctness(torch: Any, ops: Any, operands: _Operands, *, cache_dtype: DType) -> None:
    from profiling.runners.attention.mla_cache_gather_reference import (
        mla_cache_gather_reference,
    )

    operands.workspace.fill_(float("nan"))
    _launch(ops, operands, cache_dtype=cache_dtype)
    torch.cuda.synchronize()
    expected = mla_cache_gather_reference(
        operands.cache,
        operands.block_table,
        operands.cu_seq_lens,
        output_dtype=torch.bfloat16,
        scale=float(operands.scale.item()),
    )
    # E4M3 -> FP32 -> BF16 at scale 1.0 is exact, as is the BF16 copy.
    if not torch.equal(operands.workspace, expected):
        raise AssertionError(f"{_BACKEND} workspace differs from the reference gather")


def profile_mla_cache_gather_vllm_cuda(
    *,
    num_requests: int,
    num_tokens: int,
    kv_lora_rank: int,
    rope_dim: int,
    block_size: int,
    cache_dtype: DType | str,
    output_dtype: DType | str,
    cache_format: str,
) -> ComputeMetrics:
    """Profile the one-launch ``gather_and_maybe_dequant_cache`` call."""
    validated = _validate_args(
        num_requests=num_requests,
        num_tokens=num_tokens,
        kv_lora_rank=kv_lora_rank,
        rope_dim=rope_dim,
        block_size=block_size,
        cache_dtype=cache_dtype,
        output_dtype=output_dtype,
        cache_format=cache_format,
    )
    try:
        import torch
        from vllm import _custom_ops as ops
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires the instrumented vLLM env") from exc

    try:
        device = torch.device("cuda", torch.cuda.current_device())
        operands = _build_operands(torch, validated, device=device)
        _check_correctness(torch, ops, operands, cache_dtype=validated.cache_dtype)

        def kernel() -> None:
            _launch(ops, operands, cache_dtype=validated.cache_dtype)

        time_ms = Timer.cupti(kernel, kernel_name=_KERNEL_NAME)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except (KernelLaunchFailed, ProfilerNotImplemented):
        raise
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of GPU memory") from exc
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} native callable failed") from exc

    seconds = time_ms / 1000.0
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise KernelLaunchFailed(f"{_BACKEND} measured a non-positive time {time_ms} ms")
    return ComputeMetrics(
        time_ms=float(time_ms),
        energy_j=float(energy_j),
        tflops=0.0,
        memory_bandwidth_gbps=logical_bytes(validated) / seconds / 1e9,
    )


__all__ = ["profile_mla_cache_gather_vllm_cuda", "request_lengths", "logical_bytes"]
