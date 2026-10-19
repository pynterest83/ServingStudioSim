"""Profile vLLM's FLASHINFER_MLA_SPARSE_SM90 sparse-MLA call on H200.

Production source (vLLM v0.31 fork, alignment/profiler/vllm):
``vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py``. When nvcc is
on PATH, vLLM's ``has_flashinfer()`` is true and ``vllm/platforms/cuda.py``
puts this backend first for head_size 512 (GLM-5.3-Flash's rope-free latent).

Parity contract. The runner drives the production objects, not a rewrite:

- ``_SM90State`` is the builder-owned state vLLM constructs: FlashInfer's
  ``BatchMLAPagedAttentionWrapper(use_cuda_graph=True, backend="fa3")`` over a
  128 MiB workspace and reserved ``max_num_batched_tokens``-sized CSR buffers.
- ``plan()`` takes the per-row host bounds ``FlashInferMLASparseSM90Builder.
  _kv_lens_host`` computes under the always-present ``seq_lens_cpu_upper_bound``
  (``clamp(context, max_valid)``), pads them by ``_PLAN_SLACK``, and runs the
  production ``_clamp_work_kv_end`` kernel that trims every work item to the
  exact valid count from the device batch layout. A fresh plan is measured;
  production reuses a plan while the bounds stay inside its slack.
- ``pack_indices`` (the production ``_pack_topk_indices`` kernel) writes each
  row's valid prefix of the converted ``[Q, selected_k]`` slot buffer, -1 past
  it, at the planned offsets.
- The timed closure is ``state.wrapper.run(q_nope, q_pe, ckv, kpe)``, exactly
  ``forward_mqa``'s call for a BF16 cache: one ``BatchMLAPageAttentionHopperKernel``
  launch (merge of split-KV partials happens inside it). Pack, plan, the clamp
  and the shared index remap are separate launches outside this kind, as the
  FA3 backend's query zero fill is.

FP8 cache (``--kv-cache-dtype fp8``, backend
``vllm_flashinfer_sparse_sm90_fp8``). The builder plans ``float8_e4m3fn``
(``_planned_kv_data_type``), ``forward_mqa`` views the uint8 cache as
``float8_e4m3fn`` and passes ``ckv_scale=layer._k_scale_float or 1.0`` and
``kpe_scale=1.0``; GLM-5.3-Flash's checkpoint carries no KV scales, so both are
1.0. FlashInfer instantiates the same Hopper kernel with an e4m3 ``DTypeKV``
(``USE_KV_REPACK``, ``mla_hopper.cuh``): KV tiles land in shared memory as FP8
and are dequantized to BF16 there, so the math is the BF16 kernel's on the
dequantized cache, which the reference reads.

GLM-5.3-Flash's pooled indexer fixes the row law the plan reproduces: a row at
context ``n`` keeps ``min(n, index_topk + n mod index_kpool)`` slots. The kind's
``valid_counts`` give the counts; each row becomes a one-query request at a
context with that count.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention.dsa_sparse_mla_attention import (
    _FA3_ATOL,
    _FA3_CACHE_LAYOUT,
    _FA3_COSINE_LIMIT,
    _FA3_RTOL,
    _LATENT_DIM,
    _VALUE_DIM,
    _build_operands,
    _logical_bytes,
    _logical_flops,
    _validate_args,
    _ValidatedArgs,
)
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "dsa_sparse_mla_attention:vllm_flashinfer_sparse_sm90_bf16"
_FP8_BACKEND = "dsa_sparse_mla_attention:vllm_flashinfer_sparse_sm90_fp8"
_FP8_B1_BACKEND = "dsa_sparse_mla_attention:vllm_flashinfer_sparse_sm90_fp8_b1"
# The fork module that applies VLLM_FLASHINFER_INCLUDE_OVERLAY (vLLM fork branch
# exp/fp8-repack-b1) and the one header the B1 overlay changes.
_OVERLAY_MODULE = "vllm.utils.flashinfer_overlay"
_B1_HEADER = "flashinfer/attention/mla_hopper.cuh"
# Messages shared by both cache dtypes name the vLLM backend.
_LABEL = "FLASHINFER_MLA_SPARSE_SM90"
# The same rope-free 512-wide latent rows, one byte per element: vLLM's FP8
# cache is a contiguous [num_blocks, block_size, 512] uint8 tensor viewed as
# float8_e4m3fn, with one per-tensor scale.
_FP8_CACHE_LAYOUT = "token_major_mqa_fp8_latent"
# forward_mqa's FP8 scales: `layer._k_scale_float or 1.0` and a fixed kpe
# scale. GLM-5.3-Flash ships no KV scales, so k_scale stays at its 1.0 default.
_FP8_SCALE_KWARGS = {"ckv_scale": 1.0, "kpe_scale": 1.0}
_PRODUCTION_MODULE = "vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90"
# The one launch of `BatchMLAPagedAttentionWrapper.run` on the SM90 FA3 path
# (flashinfer/data/csrc/batch_mla_sm90_run.cu); the H200 case-10 capture shows
# one per sparse-MLA layer call.
_KERNEL_NAME = "BatchMLAPageAttentionHopperKernel"
# GLM-5.3-Flash config.json: index_topk 2048, index_kpool 4. The builder passes
# them to `_SM90State` as the clamp kernel's TOPK / KPOOL.
_INDEX_TOPK = 2048
_INDEX_KPOOL = 4
# scheduler_config.max_num_batched_tokens of the H200 deployment (chunk 8192):
# the size of the reserved plan buffers, padded with empty rows past Q.
_MAX_NUM_BATCHED_TOKENS = 8192
# MLAPlan (flashinfer/data/include/flashinfer/attention/scheduler.cuh) sizes its
# per-work arrays for `max_total_num_works = 16384` and does not check it. A row
# is at least one work item, so a larger call writes past them (an illegal
# address on H200 at 16500 rows); production plans at most 8192 rows.
_MAX_PLAN_WORKS = 16384


@dataclass(frozen=True)
class _CacheVariant:
    """What the cache dtype changes about the planned call."""

    backend: str  # error-message label
    cache_dtype: DType
    cache_layout: str
    cache_bytes: int  # bytes per cached element in the logical traffic
    fp8: bool  # e4m3 cache: planned as float8_e4m3fn, run with ckv/kpe scales
    b1_overlay: bool = False  # FlashInfer JIT-built from the B1 header overlay


_BF16_VARIANT = _CacheVariant(
    backend=_BACKEND,
    cache_dtype=DType.BF16,
    cache_layout=_FA3_CACHE_LAYOUT,
    cache_bytes=2,
    fp8=False,
)
_FP8_VARIANT = _CacheVariant(
    backend=_FP8_BACKEND,
    cache_dtype=DType.FP8_E4M3,
    cache_layout=_FP8_CACHE_LAYOUT,
    cache_bytes=1,
    fp8=True,
)
_FP8_B1_VARIANT = _CacheVariant(
    backend=_FP8_B1_BACKEND,
    cache_dtype=DType.FP8_E4M3,
    cache_layout=_FP8_CACHE_LAYOUT,
    cache_bytes=1,
    fp8=True,
    b1_overlay=True,
)


@dataclass(frozen=True)
class _Sm90Operands:
    q: Any  # [Q, H, 512] q_nope (contiguous token-major, as mla_attention writes it)
    q_pe: Any  # [Q, H, 0] the rope-free model's zero-width rope query
    cache: Any  # [N, 1, 512] latent rows (the flattened paged cache), BF16 or e4m3
    selected_indices: Any  # [Q, 1, K] global rows, -1 after each valid prefix
    contexts: tuple[int, ...]  # per-row context with the row's valid count
    # The BF16 values the kernel computes on: the cache itself for BF16, its
    # exact e4m3 -> BF16 dequantization (scale 1.0) for FP8.
    reference_cache: Any


def _max_valid(selected_k: int) -> int:
    """`_SM90State.max_valid`: the largest count the convert kernel can produce."""
    return min(_INDEX_TOPK + _INDEX_KPOOL - 1, selected_k)


def _row_contexts(valid_counts: Sequence[int], *, selected_k: int) -> tuple[int, ...]:
    """A context per row whose pooled valid count is the row's count.

    Below or at ``index_topk`` the count is the context itself. Above it the
    count is ``index_topk + n mod index_kpool``, so ``count + index_kpool`` is a
    context past ``index_topk`` with that remainder (index_topk is a multiple
    of index_kpool).
    """
    assert _INDEX_TOPK % _INDEX_KPOOL == 0
    limit = _max_valid(selected_k)
    contexts: list[int] = []
    for count in valid_counts:
        if count > limit:
            raise ProfilerNotImplemented(
                f"{_LABEL} rows keep at most min(index_topk + index_kpool - 1, "
                f"selected_k) = {limit} slots, got {count}"
            )
        contexts.append(count if count <= _INDEX_TOPK else count + _INDEX_KPOOL)
    return tuple(contexts)


def _valid_count(context: int) -> int:
    """The convert kernel's count for a row at ``context`` (the clamp's law)."""
    if context <= _INDEX_TOPK:
        return context
    return _INDEX_TOPK + context % _INDEX_KPOOL


def _host_kv_lens(torch: Any, contexts: Sequence[int], *, selected_k: int) -> Any:
    """`_kv_lens_host` on the upper-bound path: ``clamp(context, max_valid)``, CPU int32."""
    lens = torch.tensor(contexts, dtype=torch.int64)
    return torch.clamp(lens, max=_max_valid(selected_k)).to(torch.int32)


def _b1_overlay_digest(active: Any) -> str:
    """Check the worker's applied overlay is exactly the B1 patch; its digest."""
    import hashlib

    from profiling.kernels.dsa_sparse_mla_attention import B1_INCLUDE_ENV, B1_MLA_HOPPER_SHA256

    if active is None:
        raise ProfilerNotImplemented(
            f"{_FP8_B1_BACKEND} needs the B1 FlashInfer overlay: set {B1_INCLUDE_ENV} "
            "to its include root (host worker)"
        )
    if tuple(active.changed) != (_B1_HEADER,):
        raise ProfilerNotImplemented(
            f"{_FP8_B1_BACKEND}: the overlay must change only {_B1_HEADER}, "
            f"it changes {list(active.changed)}"
        )
    digest = hashlib.sha256((active.include_dir / _B1_HEADER).read_bytes()).hexdigest()
    if digest != B1_MLA_HOPPER_SHA256:
        raise ProfilerNotImplemented(
            f"{_FP8_B1_BACKEND}: overlay {_B1_HEADER} sha256 {digest} is not the "
            f"pinned B1 patch {B1_MLA_HOPPER_SHA256}"
        )
    return digest


def _require_b1_overlay() -> str:
    """Import the fork's overlay hook (``import vllm`` applies it) and verify it."""
    try:
        overlay = importlib.import_module(_OVERLAY_MODULE)
    except (ImportError, ModuleNotFoundError) as exc:
        raise ProfilerNotImplemented(
            f"{_FP8_B1_BACKEND} requires a vLLM fork with {_OVERLAY_MODULE}"
        ) from exc
    return _b1_overlay_digest(overlay.active_overlay())


def b1_row_provenance(**_kwargs: Any) -> str | None:
    """``backend_version`` note of a B1 row: the overlay header digest and JIT dir."""
    overlay = importlib.import_module(_OVERLAY_MODULE)
    active = overlay.active_overlay()
    return f"fi_overlay={_b1_overlay_digest(active)[:12]} jit={active.jit_dir}"


def _load_sm90_state_cls() -> Any:
    """Load the production `_SM90State` only inside a vllm_env worker."""
    try:
        module = importlib.import_module(_PRODUCTION_MODULE)
    except (ImportError, ModuleNotFoundError) as exc:
        raise ProfilerNotImplemented(
            f"{_LABEL} requires vllm_env with {_PRODUCTION_MODULE}"
        ) from exc
    state_cls = getattr(module, "_SM90State", None)
    if state_cls is None:
        raise ProfilerNotImplemented(f"{_LABEL} requires {_PRODUCTION_MODULE}._SM90State")
    return state_cls


def _build_sm90_operands(
    torch: Any,
    validated: _ValidatedArgs,
    *,
    num_heads: int,
    selected_k: int,
    device: Any,
    fp8: bool = False,
) -> _Sm90Operands:
    contexts = _row_contexts(validated.valid_counts, selected_k=selected_k)
    base = _build_operands(
        torch,
        validated,
        num_heads=num_heads,
        selected_k=selected_k,
        device=device,
        score_dim=_LATENT_DIM,
    )
    q_pe = torch.empty((validated.num_queries, num_heads, 0), dtype=torch.bfloat16, device=device)
    cache = reference_cache = base.cache
    if fp8:
        # concat_and_cache_mla's fp8 path stores `scaled_convert(x, 1.0)`: the
        # round-to-nearest e4m3 of each BF16 latent element. The BF16 staging
        # copy is dropped once both views exist.
        cache = base.cache.to(torch.float8_e4m3fn)
        reference_cache = cache.to(torch.bfloat16)
    return _Sm90Operands(
        q=base.q,
        q_pe=q_pe,
        cache=cache,
        selected_indices=base.selected_indices,
        contexts=contexts,
        reference_cache=reference_cache,
    )


def _make_state(
    torch: Any,
    state_cls: Any,
    *,
    num_queries: int,
    num_heads: int,
    selected_k: int,
    softmax_scale: float,
    device: Any,
    fp8: bool = False,
) -> Any:
    """`FlashInferMLASparseSM90Builder.__init__`'s `_SM90State` for this model.

    The KV dtype is `_planned_kv_data_type`'s: ``float8_e4m3fn`` for an FP8
    cache (the spec's uint8 storage dtype would fail FlashInfer's plan), else
    the BF16 spec dtype.
    """
    return state_cls(
        device,
        num_heads,
        torch.float8_e4m3fn if fp8 else torch.bfloat16,
        # Production never plans past max_num_batched_tokens; a larger synthetic
        # batch sizes the buffers to itself.
        max(_MAX_NUM_BATCHED_TOKENS, num_queries),
        selected_k,
        kv_lora_rank=_LATENT_DIM,
        qk_rope_head_dim=0,
        sm_scale=softmax_scale,
        index_topk=_INDEX_TOPK,
        index_kpool=_INDEX_KPOOL,
    )


def _plan_and_pack(torch: Any, state: Any, operands: _Sm90Operands, *, selected_k: int) -> None:
    """The builder's `state.plan(...)` then `forward_mqa`'s `pack_indices`.

    Every row is its own one-query request, so the clamp kernel's
    ``seq_len - query_end + row + 1`` is the row's context.
    """
    device = operands.q.device
    num_rows = len(operands.contexts)
    cam = SimpleNamespace(
        seq_lens=torch.tensor(operands.contexts, dtype=torch.int32, device=device),
        query_start_loc=torch.arange(num_rows + 1, dtype=torch.int32, device=device),
    )
    req_id_per_token = torch.arange(num_rows, dtype=torch.int32, device=device)
    kv_lens = _host_kv_lens(torch, operands.contexts, selected_k=selected_k)
    state.plan(num_rows, kv_lens, cam, req_id_per_token)
    # `triton_convert_req_index_to_global_index(..., return_valid_counts=True)`
    # output: [Q, selected_k] int32 global slots, -1 past each valid prefix.
    state.pack_indices(operands.selected_indices[:, 0, :].contiguous())


def _launch(state: Any, operands: _Sm90Operands, *, fp8: bool = False) -> Any:
    """`forward_mqa`'s run call: no scales for a BF16 cache, ckv/kpe scales of
    1.0 for an e4m3 one."""
    flat = operands.cache.reshape(-1, 1, _LATENT_DIM)
    ckv = flat[..., :_LATENT_DIM]
    kpe = flat[..., _LATENT_DIM:]
    scale_kwargs = _FP8_SCALE_KWARGS if fp8 else {}
    return state.wrapper.run(operands.q, operands.q_pe, ckv, kpe, **scale_kwargs)


def _check_correctness(
    torch: Any,
    state: Any,
    operands: _Sm90Operands,
    *,
    softmax_scale: float,
    fp8: bool = False,
) -> None:
    """Validate the planned production call against the Torch reference.

    The reference reads ``reference_cache``: for an e4m3 cache its exact BF16
    dequantization, which is what the kernel's shared-memory repack produces at
    scale 1.0, so the BF16 tolerances apply unchanged.
    """
    from profiling.runners.attention.dsa_sparse_mla_attention_reference import (
        dsa_sparse_mla_attention_reference,
    )

    inputs = (
        ("q", operands.q),
        ("cache", operands.cache),
        ("selected_indices", operands.selected_indices),
    )
    snapshots = tuple(tensor.clone() for _, tensor in inputs)

    def invoke() -> Any:
        returned = _launch(state, operands, fp8=fp8)
        torch.cuda.synchronize()
        return returned

    output = invoke()
    expected = dsa_sparse_mla_attention_reference(
        operands.q,
        operands.reference_cache,
        operands.selected_indices,
        softmax_scale=softmax_scale,
    )
    expected_shape = (operands.q.shape[0], operands.q.shape[1], _VALUE_DIM)
    if tuple(output.shape) != expected_shape or output.dtype is not torch.bfloat16:
        raise AssertionError(
            f"FlashInfer SM90 sparse output must be BF16 {expected_shape}, "
            f"got {output.dtype} {tuple(output.shape)}"
        )
    nonempty = torch.tensor(
        [_valid_count(context) > 0 for context in operands.contexts], device=output.device
    )
    checked, reference = output[nonempty], expected[nonempty]
    if not torch.isfinite(checked).all():
        raise AssertionError("FlashInfer SM90 sparse output must be finite for nonempty rows")
    # FlashInfer's FA3 MLA path rounds P to BF16 before PV and keeps split-KV
    # partials in BF16 before its in-kernel merge, as FA3 does.
    torch.testing.assert_close(checked.float(), reference.float(), atol=_FA3_ATOL, rtol=_FA3_RTOL)
    numerator = 2.0 * torch.sum(checked.float() * reference.float()).item()
    denominator = (
        torch.sum(checked.float().square()).item() + torch.sum(reference.float().square()).item()
    )
    cosine_difference = 0.0 if denominator == 0.0 else 1.0 - numerator / denominator
    if abs(cosine_difference) > _FA3_COSINE_LIMIT:
        raise AssertionError(
            f"FlashInfer SM90 sparse cosine difference {cosine_difference} exceeds "
            f"{_FA3_COSINE_LIMIT}"
        )
    if not torch.equal(invoke(), output):
        raise AssertionError("FlashInfer SM90 sparse repeated calls must be deterministic")
    for (name, tensor), snapshot in zip(inputs, snapshots, strict=True):
        if not torch.equal(tensor, snapshot):
            raise AssertionError(f"FlashInfer SM90 sparse mutated {name}")
    input_pointers = {tensor.untyped_storage().data_ptr() for _, tensor in inputs}
    if output.untyped_storage().data_ptr() in input_pointers:
        raise AssertionError("FlashInfer SM90 sparse output aliases an input")


def _profile(
    variant: _CacheVariant,
    *,
    num_queries: int,
    num_cache_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    selected_k: int,
    latent_dim: int,
    rope_dim: int,
    value_dim: int,
    softmax_scale: float,
    q_dtype: DType | str,
    cache_dtype: DType | str,
    index_dtype: str,
    output_dtype: DType | str,
    valid_counts: str,
    index_distribution: str,
    cache_layout: str,
) -> ComputeMetrics:
    """Profile `BatchMLAPagedAttentionWrapper.run` as FLASHINFER_MLA_SPARSE_SM90 plans it."""
    backend = variant.backend
    validated = _validate_args(
        num_queries=num_queries,
        num_cache_tokens=num_cache_tokens,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        selected_k=selected_k,
        latent_dim=latent_dim,
        rope_dim=rope_dim,
        value_dim=value_dim,
        softmax_scale=softmax_scale,
        q_dtype=q_dtype,
        cache_dtype=cache_dtype,
        index_dtype=index_dtype,
        output_dtype=output_dtype,
        valid_counts=valid_counts,
        index_distribution=index_distribution,
        cache_layout=cache_layout,
        # The same rope-free latent rows FA3 reads (kv_lora_rank 512,
        # qk_rope_head_dim 0), BF16 or e4m3. num_heads is the per-rank q-head
        # count; the planner packs any head count into its 64-row query tiles.
        expected_cache_dtype=variant.cache_dtype,
        expected_cache_layout=variant.cache_layout,
        expected_rope_dim=0,
    )
    # Reject counts the pooled indexer cannot produce, and calls the planner's
    # fixed work arrays cannot hold, before allocating.
    _row_contexts(validated.valid_counts, selected_k=selected_k)
    if num_queries > _MAX_PLAN_WORKS:
        raise ProfilerNotImplemented(
            f"{backend} plans at most {_MAX_PLAN_WORKS} work items (one per row at "
            f"least), got {num_queries} rows"
        )

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ProfilerNotImplemented(f"{backend} requires PyTorch in vllm_env") from exc

    fp8 = variant.fp8
    if variant.b1_overlay:
        # Before anything imports flashinfer (the overlay sets its JIT workspace).
        _require_b1_overlay()
    try:
        state_cls = _load_sm90_state_cls()
        device = torch.device("cuda", torch.cuda.current_device())
        operands = _build_sm90_operands(
            torch, validated, num_heads=num_heads, selected_k=selected_k, device=device, fp8=fp8
        )
        state = _make_state(
            torch,
            state_cls,
            num_queries=num_queries,
            num_heads=num_heads,
            selected_k=selected_k,
            softmax_scale=float(softmax_scale),
            device=device,
            fp8=fp8,
        )
        _plan_and_pack(torch, state, operands, selected_k=selected_k)
        _check_correctness(torch, state, operands, softmax_scale=float(softmax_scale), fp8=fp8)

        def kernel() -> Any:
            return _launch(state, operands, fp8=fp8)

        time_ms = Timer.cupti(kernel, kernel_name=_KERNEL_NAME)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except (KernelLaunchFailed, ProfilerNotImplemented):
        raise
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{backend} ran out of GPU memory") from exc
    except Exception as exc:
        raise KernelLaunchFailed(f"{backend} native callable failed") from exc

    flops = _logical_flops(
        num_queries=num_queries,
        num_heads=num_heads,
        selected_k=selected_k,
        score_dim=_LATENT_DIM,
    )
    logical_bytes = _logical_bytes(
        num_queries=num_queries,
        num_heads=num_heads,
        selected_k=selected_k,
        valid_counts=validated.valid_counts,
        cache_bytes=variant.cache_bytes,
        score_dim=_LATENT_DIM,
    )
    seconds = time_ms / 1000.0
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise KernelLaunchFailed(f"{backend} measured a non-positive time {time_ms} ms")
    return ComputeMetrics(
        time_ms=float(time_ms),
        energy_j=float(energy_j),
        tflops=flops / seconds / 1e12,
        memory_bandwidth_gbps=logical_bytes / seconds / 1e9,
    )


def profile_dsa_sparse_mla_attention_vllm_flashinfer_sparse_sm90_bf16(
    **kwargs: Any,
) -> ComputeMetrics:
    """The planned FlashInfer SM90 call over a BF16 latent cache."""
    return _profile(_BF16_VARIANT, **kwargs)


def profile_dsa_sparse_mla_attention_vllm_flashinfer_sparse_sm90_fp8(
    **kwargs: Any,
) -> ComputeMetrics:
    """The same call over an e4m3 latent cache (`--kv-cache-dtype fp8`)."""
    return _profile(_FP8_VARIANT, **kwargs)


def profile_dsa_sparse_mla_attention_vllm_flashinfer_sparse_sm90_fp8_b1(
    **kwargs: Any,
) -> ComputeMetrics:
    """The FP8-cache call with FlashInfer built from the B1 header overlay."""
    return _profile(_FP8_B1_VARIANT, **kwargs)


__all__ = [
    "b1_row_provenance",
    "profile_dsa_sparse_mla_attention_vllm_flashinfer_sparse_sm90_bf16",
    "profile_dsa_sparse_mla_attention_vllm_flashinfer_sparse_sm90_fp8",
    "profile_dsa_sparse_mla_attention_vllm_flashinfer_sparse_sm90_fp8_b1",
]
