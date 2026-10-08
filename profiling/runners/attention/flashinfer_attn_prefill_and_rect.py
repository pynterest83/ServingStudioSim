"""FlashInfer ragged attention runner — prefill (causal) + rect (non-causal).

Both kinds drive the same ragged wrapper via one core caller (``_run_ragged``,
keyed on ``q_len``/``kv_len`` + ``causal``); they differ only in how their wire
schema maps to those kernel params, so they live together here rather than
duplicating the wrapper/plan/run logic:

- **prefill** (causal) merges the ref's pure-prefill + chunked ops. Its wire
  schema is ``(prefix_len, append_len)``; ``_run_prefill`` derives ``q_len =
  append_len``, ``kv_len = prefix_len + append_len`` (the ONE place that mapping
  lives). Pure prefill = ``prefix_len == 0``.
- **rect** (non-causal) is parametrized by ``(q_len, kv_len)`` *directly* — for
  non-causal there is no causal prefix/append split, only the two lengths, and
  this also lets rect express ``q_len > kv_len`` (which the prefill encoding
  cannot). rect entries pass ``q_len``/``kv_len`` straight to ``_run_ragged``.

Each ``(kind, backend)`` registers its own thin entry function because the worker
strips ``backend`` before calling the runner — it routes via the registry, so the
runner is selected by backend, not told it (see ``profiling/exec/local_worker.py``).

The **fa2** path sweeps ``fixed_split_size`` internally (``_measure_best_split``)
and records the fastest, because fa2's default KV-split is load-imbalanced for
long-kv ragged calls; fa3 self-tunes and keeps the single default plan. The split
set is internal to profiling, not a wire/cache axis.

The **vllm_fa3** backend (both kinds) is not FlashInfer: it times vLLM v0.31's
own FA3 varlen call from MLA dense-MHA prefill (``_run_vllm_fa3``) and runs in
``vllm_env``.

Shared op-agnostic mechanics (tensor/indptr/fp8 construction, dtype/backend
helpers, do_bench + metrics) live in ``_common``. The heavy libs (``flashinfer`` /
``torch``) are imported lazily inside the functions; this module is only loaded
inside the profiling worker subprocess via ``RunnerRef``.
"""

from __future__ import annotations

from profiling.db.args import DType
from profiling.profilers.timer import Timer
from profiling.runners.attention import _common
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

# fa2's KV-split heuristic is load-imbalanced for long-kv ragged calls (the
# default can inflate kernel time ~1.6x); the simulator prices each shape at its
# best achievable split, so the fa2 profiler sweeps fixed_split_size internally
# and keeps the fastest. Mirrors the validated best-split sweep (see
# agent-trace/attention_cache_fidelity.md). fa3 self-tunes — it keeps the single
# default plan. The split set is NOT a wire/cache axis; it stays internal here.
_FA2_SPLIT_SIZES: tuple[int, ...] = (1024, 2048, 4096, 8192, 16384)

_VLLM_FA3_LABEL = "vllm_fa3"
# BF16 output against an FP32 softmax reference: about 3 BF16 ulps relative,
# plus an absolute floor for outputs near zero.
_VLLM_FA3_ATOL = 4e-3
_VLLM_FA3_RTOL = 3.01 / 128
_VLLM_FA3_COSINE_LIMIT = 1e-5
# Query rows checked against the reference: the first and last rows plus an
# evenly spaced middle sample, so the causal diagonal and the full-width rows
# are both covered without an O(q_len * kv_len) FP32 score matrix.
_VLLM_FA3_CHECK_ROWS = 64
# Bound one FP32 reference score block to about 1 GiB.
_VLLM_FA3_REFERENCE_SCORE_ELEMENTS = 1 << 28


def _run_cudnn_attention(
    *,
    q_len: int,
    kv_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_dtype: DType | str,
    kv_dtype: DType | str,
    o_dtype: DType | str,
    causal: bool,
) -> ComputeMetrics:
    """cuDNN SDPA path (torch), used for the ``cudnn`` backend."""
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if any(_common.is_fp8(d) for d in (q_dtype, kv_dtype, o_dtype)):
        raise ProfilerNotImplemented("cudnn backend does not support fp8")
    if causal and q_len != kv_len:
        raise ProfilerNotImplemented(
            "cudnn causal attention requires q_len == kv_len (prefix_len == 0)"
        )

    torch_dtype = _common.to_torch_dtype(q_dtype)
    q = torch.randn(1, num_qo_heads, q_len, head_dim, dtype=torch_dtype, device="cuda")
    k = torch.randn(1, num_kv_heads, kv_len, head_dim, dtype=torch_dtype, device="cuda")
    v = torch.randn(1, num_kv_heads, kv_len, head_dim, dtype=torch_dtype, device="cuda")
    enable_gqa = num_qo_heads != num_kv_heads

    def benchmark_fn():
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=enable_gqa)

    o_elem = torch.tensor([], dtype=_common.to_torch_dtype(o_dtype)).element_size()
    bytes_accessed = int(
        q.numel() * q.element_size()
        + k.numel() * k.element_size()
        + v.numel() * v.element_size()
        + num_qo_heads * q_len * head_dim * o_elem
    )
    flops = _common.attention_flops(
        q_len=q_len,
        kv_len=kv_len,
        num_qo_heads=num_qo_heads,
        head_dim=head_dim,
        causal=causal,
    )
    return _common.measure(benchmark_fn, flops=flops, bytes_accessed=bytes_accessed)


def _vllm_fa3_check_rows(q_len: int) -> list[int]:
    """Query rows the vllm_fa3 correctness check compares (sorted, unique)."""
    edge = _VLLM_FA3_CHECK_ROWS // 4
    rows = set(range(min(edge, q_len))) | set(range(max(0, q_len - edge), q_len))
    step = max(1, q_len // _VLLM_FA3_CHECK_ROWS)
    rows |= set(range(0, q_len, step))
    return sorted(rows)


def _vllm_fa3_reference_rows(torch, q, k, v, rows: list[int], *, causal: bool, scale: float):
    """FP32 softmax attention for the selected query rows of one request.

    The causal mask is FlashAttention's bottom-right alignment: query row ``i``
    sees keys ``j <= i + kv_len - q_len``. K/V heads are repeated for GQA.
    """
    q_len, num_qo_heads, _ = q.shape
    kv_len, num_kv_heads, _ = k.shape
    group = num_qo_heads // num_kv_heads
    k32 = k.float().repeat_interleave(group, dim=1)
    v32 = v.float().repeat_interleave(group, dim=1)
    row_index = torch.tensor(rows, dtype=torch.long, device=q.device)
    chunk = max(1, _VLLM_FA3_REFERENCE_SCORE_ELEMENTS // (num_qo_heads * kv_len))
    outputs = []
    for start in range(0, len(rows), chunk):
        sub = row_index[start : start + chunk]
        scores = torch.einsum("rhd,khd->hrk", q.index_select(0, sub).float(), k32) * scale
        if causal:
            limit = sub + (kv_len - q_len)
            keys = torch.arange(kv_len, device=q.device)
            scores.masked_fill_(keys.view(1, 1, -1) > limit.view(1, -1, 1), float("-inf"))
        outputs.append(torch.einsum("hrk,khd->rhd", torch.softmax(scores, dim=-1), v32))
    return torch.cat(outputs, dim=0)


def _check_vllm_fa3(torch, launch, q, k, v, *, causal: bool, scale: float) -> None:
    """Compare the production FA3 call with the FP32 reference before timing."""
    snapshots = (q.clone(), k.clone(), v.clone())
    output = launch()
    torch.cuda.synchronize()
    expected_shape = (q.shape[0], q.shape[1], v.shape[2])
    if tuple(output.shape) != expected_shape or output.dtype is not torch.bfloat16:
        raise KernelLaunchFailed(
            f"{_VLLM_FA3_LABEL} output must be BF16 {expected_shape}, "
            f"got {output.dtype} {tuple(output.shape)}"
        )
    rows = _vllm_fa3_check_rows(q.shape[0])
    actual = output.index_select(0, torch.tensor(rows, device=q.device)).float()
    expected = _vllm_fa3_reference_rows(torch, q, k, v, rows, causal=causal, scale=scale)
    if not torch.isfinite(actual).all():
        raise KernelLaunchFailed(f"{_VLLM_FA3_LABEL} output is not finite")
    torch.testing.assert_close(actual, expected, atol=_VLLM_FA3_ATOL, rtol=_VLLM_FA3_RTOL)
    numerator = 2.0 * torch.sum(actual * expected).item()
    denominator = torch.sum(actual.square()).item() + torch.sum(expected.square()).item()
    cosine_difference = 0.0 if denominator == 0.0 else 1.0 - numerator / denominator
    if abs(cosine_difference) > _VLLM_FA3_COSINE_LIMIT:
        raise KernelLaunchFailed(
            f"{_VLLM_FA3_LABEL} cosine difference {cosine_difference} exceeds "
            f"{_VLLM_FA3_COSINE_LIMIT}"
        )
    for name, tensor, snapshot in zip(("q", "k", "v"), (q, k, v), snapshots, strict=True):
        if not torch.equal(tensor, snapshot):
            raise KernelLaunchFailed(f"{_VLLM_FA3_LABEL} mutated {name}")


def _run_vllm_fa3(
    *,
    causal: bool,
    q_len: int,
    kv_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_dtype: DType | str,
    kv_dtype: DType | str,
    o_dtype: DType | str,
) -> ComputeMetrics:
    """vLLM v0.31's FA3 varlen call from MLA dense-MHA prefill, for one request.

    Causal is ``FlashAttnPrefillBackend.run_prefill_new_tokens``
    (``vllm/v1/attention/backends/mla/prefill/flash_attn.py``): the new tokens
    attend to themselves, ``cu_seqlens_k = cu_seqlens_q``, so only
    ``q_len == kv_len`` (``prefix_len == 0``) is that call. Non-causal is
    ``run_prefill_context_chunk``: the new tokens over one cached context chunk,
    returning the LSE that ``merge_attn_states`` (not timed) consumes.

    ``MLACommonBaseImpl.forward_mha`` passes K and V as the two halves of the
    ``kv_b_proj`` output viewed ``[tokens, heads, 2 * head_dim]`` (NoPE MLA
    returns ``k_nope`` without a copy), so K and V are strided views here too.
    One call launches FA3's ``prepare_varlen_num_blocks``, ``FlashAttnFwdSm90``
    and, when FA3 splits KV, ``FlashAttnFwdCombine``; all are timed.
    """
    q_len = int(q_len)
    kv_len = int(kv_len)
    for name, dtype in (("q_dtype", q_dtype), ("kv_dtype", kv_dtype), ("o_dtype", o_dtype)):
        if DType.from_value(dtype) is not DType.BF16:
            raise ProfilerNotImplemented(
                f"{_VLLM_FA3_LABEL} measures vLLM's BF16 MLA prefill only, got {name}={dtype}"
            )
    if min(q_len, kv_len, num_qo_heads, num_kv_heads, head_dim) <= 0:
        raise ValueError(f"{_VLLM_FA3_LABEL} requires positive lengths, heads and head_dim")
    if num_qo_heads % num_kv_heads:
        raise ValueError(
            f"{_VLLM_FA3_LABEL} requires num_qo_heads to be a multiple of num_kv_heads"
        )
    if causal and q_len != kv_len:
        raise ProfilerNotImplemented(
            f"{_VLLM_FA3_LABEL} causal is vLLM's new-token call (cu_seqlens_k = cu_seqlens_q); "
            "it requires prefix_len == 0, and cached context is the non-causal rect call"
        )

    try:
        import torch
        from vllm.v1.attention.backends.fa_utils import (
            flash_attn_varlen_func,
            get_flash_attn_version,
        )
    except (ImportError, OSError) as exc:
        raise ProfilerNotImplemented(f"{_VLLM_FA3_LABEL} requires vllm_env") from exc

    # The version FlashAttnPrefillBackend selects for these head dims on this
    # device: FA3 on SM90 up to head_dim 256, FA4 above it.
    fa_version = get_flash_attn_version(head_size=head_dim, head_size_v=head_dim)
    if fa_version != 3:
        raise ProfilerNotImplemented(
            f"{_VLLM_FA3_LABEL}: vLLM selects FA{fa_version} for head_dim {head_dim} here"
        )

    scale = head_dim**-0.5
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        generator = torch.Generator(device=device).manual_seed(0)
        q = torch.randn(
            (q_len, num_qo_heads, head_dim),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        kv = torch.randn(
            (kv_len, num_kv_heads, 2 * head_dim),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        k, v = kv.split([head_dim, head_dim], dim=-1)
        cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)
        cu_seqlens_k = torch.tensor([0, kv_len], dtype=torch.int32, device=device)

        def launch():
            result = flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=q_len,
                max_seqlen_k=kv_len,
                softmax_scale=scale,
                causal=causal,
                return_softmax_lse=not causal,
                fa_version=3,
            )
            return result if causal else result[0]

        _check_vllm_fa3(torch, launch, q, k, v, causal=causal, scale=scale)
        flops = _common.attention_flops(
            q_len=q_len,
            kv_len=kv_len,
            num_qo_heads=num_qo_heads,
            head_dim=head_dim,
            causal=causal,
        )
        # Logical bytes: Q and O, the K and V halves read, and the FP32 LSE.
        bytes_accessed = 2 * (2 * q_len * num_qo_heads * head_dim) + 2 * (
            2 * kv_len * num_kv_heads * head_dim
        )
        bytes_accessed += 4 * q_len * num_qo_heads
        return _common.measure(launch, flops=flops, bytes_accessed=bytes_accessed)
    except (KernelLaunchFailed, ProfilerNotImplemented):
        raise
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_VLLM_FA3_LABEL} ran out of GPU memory") from exc
    except (AssertionError, RuntimeError) as exc:
        raise KernelLaunchFailed(f"{_VLLM_FA3_LABEL} failed: {exc}") from exc


def _run_ragged(
    *,
    backend: str,
    causal: bool,
    q_len: int,
    kv_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_dtype: DType | str,
    kv_dtype: DType | str,
    o_dtype: DType | str,
) -> ComputeMetrics:
    """Core ragged caller keyed on (q_len, kv_len) + causal, shared by both kinds.

    Picks the wrapper, builds ``plan()`` kwargs + the ``run()`` closure, and
    measures. The wire-schema -> (q_len, kv_len) mapping is the caller's job
    (``_run_prefill`` for prefill; rect passes them directly).
    """
    if backend == "trt":
        # trtllm-gen has no variable-length/ragged prefill kernel, and on sm90
        # (H200) even the paged path errors with "Unsupported architecture" —
        # trtllm-gen FMHA is B200/sm100-only in this build, and ref only ever
        # profiled trt as paged on B200. Until the paged path lands (B200), trt
        # is cleanly unsupported here rather than a launch failure.
        raise ProfilerNotImplemented(
            "trt (trtllm-gen) prefill/rect is unavailable on the ragged path: it "
            "has no variable-length kernel and is B200/sm100-only; the paged path "
            "on B200 is not yet implemented"
        )

    q_len = int(q_len)
    kv_len = int(kv_len)

    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented("torch is required for attention runners") from exc

    if backend == "cudnn":
        return _run_cudnn_attention(
            q_len=q_len,
            kv_len=kv_len,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            o_dtype=o_dtype,
            causal=causal,
        )

    try:
        import flashinfer
    except ImportError as exc:
        raise ProfilerNotImplemented("flashinfer is required for attention runners") from exc

    try:
        inp = _common.build_ragged_inputs(
            q_len=q_len,
            kv_len=kv_len,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            o_dtype=o_dtype,
        )
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            _common.make_workspace(),
            kv_layout="NHD",
            backend=_common.flashinfer_backend_name(backend),
        )
        plan_kwargs = dict(
            qo_indptr=inp.qo_indptr,
            kv_indptr=inp.kv_indptr,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
            causal=causal,
            q_data_type=_common.to_torch_dtype(q_dtype),
        )
        if inp.scales is not None:  # fp8: q == kv == fp8, o non-fp8
            plan_kwargs["q_data_type"] = torch.float8_e4m3fn
            plan_kwargs["kv_data_type"] = torch.float8_e4m3fn
            plan_kwargs["o_data_type"] = _common.to_torch_dtype(o_dtype)
        elif _common.to_torch_dtype(q_dtype) != _common.to_torch_dtype(o_dtype):
            plan_kwargs["o_data_type"] = _common.to_torch_dtype(o_dtype)
        # plan() is deferred: fa2 sweeps fixed_split_size (re-plans per candidate),
        # so we do not plan once up front. The run() closure is split-agnostic.

        if inp.scales is not None:
            s_q, s_k, s_v = inp.scales

            def benchmark_fn():
                return wrapper.run(inp.q, inp.k, inp.v, s_q, s_k, s_v)
        else:

            def benchmark_fn():
                return wrapper.run(inp.q, inp.k, inp.v)

        flops = _common.attention_flops(
            q_len=q_len,
            kv_len=kv_len,
            num_qo_heads=num_qo_heads,
            head_dim=head_dim,
            causal=causal,
        )
        if backend == "fa2":
            return _measure_best_split(
                wrapper=wrapper,
                plan_kwargs=plan_kwargs,
                benchmark_fn=benchmark_fn,
                kv_len=kv_len,
                flops=flops,
                bytes_accessed=inp.bytes_accessed,
            )
        wrapper.plan(**plan_kwargs)
        # FlashInfer's first run may emit one-time initialization kernels. The
        # CUPTI launch-pattern probe warms the callable itself, for every
        # backend, so no warm-up is needed here.
        return _common.measure(benchmark_fn, flops=flops, bytes_accessed=inp.bytes_accessed)
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc


def _measure_best_split(
    *,
    wrapper,
    plan_kwargs: dict,
    benchmark_fn,
    kv_len: int,
    flops: int,
    bytes_accessed: int,
) -> ComputeMetrics:
    """fa2 only: time the run under each candidate ``fixed_split_size`` (plus the
    flashinfer default) and re-measure at the fastest.

    The sweep uses cold-L2 kernel-only ``Timer.cupti`` (cheap, no energy); the
    winning split is then re-planned and run through the full ``_common.measure``
    (cupti + energy) once. Candidates ``>= kv_len`` are dropped (a split that
    large means no split = the default). A fixed split that fails to plan/run is
    skipped; the default (no ``fixed_split_size``) is always tried and its failure
    surfaces (caught as ``KernelLaunchFailed`` by the caller).
    """
    candidates: list[int | None] = [None]
    candidates += [s for s in _FA2_SPLIT_SIZES if s < kv_len]

    best_time = float("inf")
    best_split: int | None = None
    for split in candidates:
        kwargs = dict(plan_kwargs)
        if split is not None:
            kwargs["fixed_split_size"] = split
        try:
            wrapper.plan(**kwargs)
            time_ms = Timer.cupti(benchmark_fn)
        except Exception:  # noqa: BLE001 — a bad fixed split is skipped, not fatal
            if split is None:
                raise
            continue
        if time_ms < best_time:
            best_time = time_ms
            best_split = split

    kwargs = dict(plan_kwargs)
    if best_split is not None:
        kwargs["fixed_split_size"] = best_split
    wrapper.plan(**kwargs)
    return _common.measure(benchmark_fn, flops=flops, bytes_accessed=bytes_accessed)


def _run_prefill(
    *,
    backend: str,
    prefix_len: int,
    append_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_dtype: DType | str,
    kv_dtype: DType | str,
    o_dtype: DType | str,
) -> ComputeMetrics:
    """Prefill caller: maps the cache dims to kernel params, then runs causal.

    Cache dims -> kernel params (the ONE place this mapping lives):
    ``q_len = append_len``, ``kv_len = prefix_len + append_len``.
    """
    return _run_ragged(
        backend=backend,
        causal=True,
        q_len=int(append_len),
        kv_len=int(prefix_len) + int(append_len),
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        o_dtype=o_dtype,
    )


# --- flashinfer_attn_prefill (causal) entry points, one per backend ----------


def profile_flashinfer_attn_prefill_fa2(**kwargs) -> ComputeMetrics:
    return _run_prefill(backend="fa2", **kwargs)


def profile_flashinfer_attn_prefill_fa3(**kwargs) -> ComputeMetrics:
    return _run_prefill(backend="fa3", **kwargs)


def profile_flashinfer_attn_prefill_trt(**kwargs) -> ComputeMetrics:
    return _run_prefill(backend="trt", **kwargs)


def profile_flashinfer_attn_prefill_cudnn(**kwargs) -> ComputeMetrics:
    return _run_prefill(backend="cudnn", **kwargs)


def profile_flashinfer_attn_prefill_vllm_fa3(
    *, prefix_len: int, append_len: int, **kwargs
) -> ComputeMetrics:
    return _run_vllm_fa3(
        causal=True,
        q_len=int(append_len),
        kv_len=int(prefix_len) + int(append_len),
        **kwargs,
    )


# --- flashinfer_attn_rect (non-causal) entry points, one per backend ---------
# rect is parametrized by (q_len, kv_len) directly, so kwargs already carry the
# kernel params; pass them straight through with causal=False.


def profile_flashinfer_attn_rect_fa2(**kwargs) -> ComputeMetrics:
    return _run_ragged(backend="fa2", causal=False, **kwargs)


def profile_flashinfer_attn_rect_fa3(**kwargs) -> ComputeMetrics:
    return _run_ragged(backend="fa3", causal=False, **kwargs)


def profile_flashinfer_attn_rect_trt(**kwargs) -> ComputeMetrics:
    return _run_ragged(backend="trt", causal=False, **kwargs)


def profile_flashinfer_attn_rect_cudnn(**kwargs) -> ComputeMetrics:
    return _run_ragged(backend="cudnn", causal=False, **kwargs)


def profile_flashinfer_attn_rect_vllm_fa3(**kwargs) -> ComputeMetrics:
    return _run_vllm_fa3(causal=False, **kwargs)
