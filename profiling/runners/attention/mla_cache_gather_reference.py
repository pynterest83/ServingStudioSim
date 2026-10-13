"""Torch semantic reference for the plain MLA paged-cache gather."""

from __future__ import annotations

import torch


def mla_cache_gather_reference(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    *,
    output_dtype: torch.dtype,
    scale: float = 1.0,
) -> torch.Tensor:
    """Gather every request's context, in position order, into one workspace.

    ``cache`` is ``[num_blocks, block_size, D]`` (BF16, or ``float8_e4m3fn`` for
    an FP8 cache); request ``i`` owns workspace rows
    ``[cu_seq_lens[i], cu_seq_lens[i + 1])``, and its position ``p`` lives at
    ``cache[block_table[i, p // block_size], p % block_size]``. FP8 entries are
    widened to FP32, multiplied by ``scale`` and rounded to ``output_dtype``,
    as vLLM's ``fp8::scaled_convert`` does.

    Source boundary: vLLM v0.31 ``gather_and_maybe_dequant_cache`` /
    ``gather_and_maybe_dequant_cache_page`` (csrc/libtorch_stable/cache_kernels.cu)
    without ``seq_starts``.
    """
    if cache.dim() != 3:
        raise ValueError(f"cache must be [num_blocks, block_size, D], got {tuple(cache.shape)}")
    block_size = cache.shape[1]
    bounds = cu_seq_lens.tolist()
    rows: list[torch.Tensor] = []
    for request, (start, stop) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        positions = torch.arange(stop - start, device=cache.device)
        blocks = block_table[request, positions // block_size].long()
        entries = cache[blocks, positions % block_size]
        rows.append((entries.float() * scale).to(output_dtype))
    if not rows:
        return torch.empty((0, cache.shape[2]), dtype=output_dtype, device=cache.device)
    return torch.cat(rows)


__all__ = ["mla_cache_gather_reference"]
