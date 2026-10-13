"""CPU tests for the FLASHINFER_MLA_SPARSE_SM90 sparse-MLA profiler backend."""

from __future__ import annotations

import pytest
import torch

from profiling.db.args import DType
from profiling.db.registry import find_kernel_profiler_spec
from profiling.kernels.dsa_sparse_mla_attention import KIND, DsaSparseMlaAttentionArgs
from profiling.runners.attention import dsa_sparse_mla_attention as shared
from profiling.runners.attention import dsa_sparse_mla_flashinfer_sm90 as runner
from profiling.runners.exceptions import ProfilerNotImplemented

_BACKEND = "vllm_flashinfer_sparse_sm90_bf16"
_SPEC = {
    "num_queries": 2,
    "num_cache_tokens": 8192,
    "num_heads": 16,
    "num_kv_heads": 1,
    "selected_k": 2176,
    "latent_dim": 512,
    "rope_dim": 0,
    "value_dim": 512,
    "softmax_scale": 0.0625,
    "q_dtype": "bf16",
    "cache_dtype": "bf16",
    "index_dtype": "int32",
    "output_dtype": "bf16",
    "valid_counts": "r:2048..2049",
    "index_distribution": "unique_scattered_pages",
    "cache_layout": "token_major_mqa_bf16_latent",
}


def test_registration_is_sm90_bf16_in_vllm_env() -> None:
    """Catches a registration that would let the launcher pick this backend on a
    non-Hopper GPU or for an FP8 cache it was never validated on."""
    spec = find_kernel_profiler_spec(KIND, _BACKEND)
    assert spec.args_schema is DsaSparseMlaAttentionArgs
    assert spec.table_name == KIND
    assert spec.supports.sm_targets == frozenset({"sm_90a"})
    assert spec.supports.allows(DType.BF16, DType.BF16, gpu="NVIDIA H200")
    assert not spec.supports.allows(DType.BF16, DType.BF16, gpu="NVIDIA B200")
    assert not spec.supports.allows(DType.BF16, DType.FP8_E4M3, gpu="NVIDIA H200")
    assert spec.subprocess_env == "vllm_env"
    assert spec.runner_ref.module_name == (
        "profiling.runners.attention.dsa_sparse_mla_flashinfer_sm90"
    )
    assert spec.runner_ref.function_name == (
        "profile_dsa_sparse_mla_attention_vllm_flashinfer_sparse_sm90_bf16"
    )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({}, "reached the build"),
        ({"num_heads": 64}, "reached the build"),
        ({"num_queries": 1, "valid_counts": "u:2051x1"}, "reached the build"),
        ({"valid_counts": "u:2052x2"}, "at most"),
        ({"num_queries": 16384, "valid_counts": "u:2051x16384"}, "reached the build"),
        ({"num_queries": 16385, "valid_counts": "u:2051x16385"}, "plans at most 16384"),
        ({"selected_k": 2048, "valid_counts": "u:2048x2"}, "reached the build"),
        ({"selected_k": 2048, "valid_counts": "r:2047..2048"}, "reached the build"),
        ({"rope_dim": 64}, "rope_dim must be 0"),
        ({"cache_layout": "token_major_mqa_bf16_latent_rope"}, "cache_layout must be"),
        ({"cache_dtype": "fp8_e4m3"}, "cache_dtype must be bf16"),
        ({"q_dtype": "fp8_e4m3"}, "q_dtype must be bf16"),
    ],
)
def test_accepts_only_the_rope_free_bf16_layout_and_pooled_counts(
    monkeypatch, overrides, match
) -> None:
    """Catches a backend that would measure a shape production cannot plan: a
    rope or FP8 layout, or a row count past the pooled indexer's 2051 cap."""

    def reached(*_args, **_kwargs):
        raise ProfilerNotImplemented("reached the build")

    monkeypatch.setattr(runner, "_load_sm90_state_cls", lambda: object())
    monkeypatch.setattr(runner, "_build_sm90_operands", reached)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    with pytest.raises(ProfilerNotImplemented, match=match):
        runner.profile_dsa_sparse_mla_attention_vllm_flashinfer_sparse_sm90_bf16(
            **(_SPEC | overrides)
        )


def test_row_contexts_reproduce_each_pooled_count() -> None:
    """Catches a context choice whose pooled count (the device clamp's law)
    differs from the row's valid count, which would plan and trim the wrong
    number of slots, and host bounds that are not clamp(context, 2051)."""
    counts = (1, 100, 2047, 2048, 2049, 2050, 2051)
    contexts = runner._row_contexts(counts, selected_k=2176)
    assert [runner._valid_count(context) for context in contexts] == list(counts)
    assert all(context > 2048 for context in contexts[-3:])
    assert runner._host_kv_lens(torch, contexts, selected_k=2176).tolist() == [
        1,
        100,
        2047,
        2048,
        2051,
        2051,
        2051,
    ]
    assert runner._host_kv_lens(torch, contexts, selected_k=2176).dtype is torch.int32
    # A narrower page table caps max_valid at selected_k, as _SM90State does.
    assert runner._max_valid(2048) == 2048
    with pytest.raises(ProfilerNotImplemented, match="at most"):
        runner._row_contexts((2049,), selected_k=2048)


class _FakeWrapper:
    def __init__(self, log):
        self.log = log

    def run(self, *args, **kwargs):
        self.log.append(("run", args, kwargs))
        return "output"


class _FakeState:
    def __init__(self, *args, **kwargs):
        self.log = [("init", args, kwargs)]
        self.wrapper = _FakeWrapper(self.log)

    def plan(self, *args):
        self.log.append(("plan", args))

    def pack_indices(self, slots):
        self.log.append(("pack", slots))


def test_state_plan_pack_and_run_match_the_production_builder() -> None:
    """Catches a drift from vLLM's builder and forward_mqa: a state built with
    other plan buffers or indexer constants, a plan without the device batch
    layout (no kv_end clamp), slots that are not the converted [Q, K] buffer, or
    a run call whose q / zero-width q_pe / ckv / kpe are not the cache views."""
    validated = shared._ValidatedArgs(
        num_queries=3,
        num_cache_tokens=40,
        valid_counts=(5, 8, 2),
        index_distribution="unique_scattered_pages",
    )
    cpu = torch.device("cpu")
    operands = runner._build_sm90_operands(torch, validated, num_heads=4, selected_k=8, device=cpu)
    assert operands.q.shape == (3, 4, 512) and operands.q.is_contiguous()
    assert operands.q_pe.shape == (3, 4, 0) and operands.q_pe.dtype is torch.bfloat16
    assert operands.cache.shape == (40, 1, 512)
    assert operands.contexts == (5, 8, 2)

    state = runner._make_state(
        torch,
        _FakeState,
        num_queries=3,
        num_heads=4,
        selected_k=8,
        softmax_scale=0.0625,
        device=cpu,
    )
    (_, args, kwargs) = state.log[0]
    assert args == (cpu, 4, torch.bfloat16, 8192, 8)
    assert kwargs == {
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 0,
        "sm_scale": 0.0625,
        "index_topk": 2048,
        "index_kpool": 4,
    }
    big = runner._make_state(
        torch,
        _FakeState,
        num_queries=9000,
        num_heads=4,
        selected_k=8,
        softmax_scale=0.0625,
        device=cpu,
    )
    assert big.log[0][1][3] == 9000

    runner._plan_and_pack(torch, state, operands, selected_k=8)
    (_, plan_args), (_, slots) = state.log[1], state.log[2]
    num_rows, kv_lens, cam, req_ids = plan_args
    assert num_rows == 3
    assert kv_lens.tolist() == [5, 8, 2] and kv_lens.device.type == "cpu"
    assert cam.seq_lens.tolist() == [5, 8, 2]
    assert cam.query_start_loc.tolist() == [0, 1, 2, 3]
    assert req_ids.tolist() == [0, 1, 2]
    assert slots.shape == (3, 8) and slots.dtype is torch.int32
    assert torch.equal(slots, operands.selected_indices[:, 0, :])
    assert slots[2, 2:].tolist() == [-1] * 6

    assert runner._launch(state, operands) == "output"
    (_, (q, q_pe, ckv, kpe), run_kwargs) = state.log[3]
    assert run_kwargs == {}
    assert q is operands.q and q_pe is operands.q_pe
    assert ckv.shape == (40, 1, 512) and kpe.shape == (40, 1, 0)
    assert ckv.data_ptr() == operands.cache.data_ptr()


def test_profile_times_only_the_hopper_kernel_after_plan_and_check(monkeypatch) -> None:
    """Catches timing that would include the plan, clamp or pack launches, a
    check run after timing, or score accounting that counts a rope dimension."""
    calls: list[object] = []
    operands = object()

    monkeypatch.setattr(runner, "_load_sm90_state_cls", lambda: "state_cls")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        runner,
        "_build_sm90_operands",
        lambda *_a, **kwargs: calls.append(("build", kwargs["selected_k"])) or operands,
    )
    monkeypatch.setattr(
        runner,
        "_make_state",
        lambda _torch, cls, **kwargs: (
            calls.append(("state", cls, kwargs["num_queries"])) or "state"
        ),
    )
    monkeypatch.setattr(
        runner,
        "_plan_and_pack",
        lambda _torch, state, ops, *, selected_k: calls.append(("plan", state, ops, selected_k)),
    )
    monkeypatch.setattr(runner, "_check_correctness", lambda *_a, **_k: calls.append("correctness"))
    monkeypatch.setattr(
        runner,
        "_launch",
        lambda state, ops, **kwargs: calls.append(("launch", state, ops, kwargs)) or "out",
    )

    def fake_timer(callable_, *, kernel_name):
        calls.append(("timer", kernel_name))
        assert callable_() == "out"
        return 2.0

    monkeypatch.setattr(runner.Timer, "cupti", fake_timer)
    monkeypatch.setattr(runner.Energy, "perf", lambda *_a, **_k: 0.5)
    metrics = runner.profile_dsa_sparse_mla_attention_vllm_flashinfer_sparse_sm90_bf16(**_SPEC)

    assert calls == [
        ("build", 2176),
        ("state", "state_cls", 2),
        ("plan", "state", operands, 2176),
        "correctness",
        ("timer", "BatchMLAPageAttentionHopperKernel"),
        ("launch", "state", operands, {"fp8": False}),
    ]
    flops = 2 * 2 * 16 * 2176 * (512 + 512)
    logical_bytes = 2 * 2 * 16 * 512 + 4 * 2 * 2176 + 2 * (2048 + 2049) * 512
    logical_bytes += 2 * 2 * 16 * 512 + 8 * 2 * 16
    assert metrics.time_ms == 2.0
    assert metrics.tflops == pytest.approx(flops / 0.002 / 1e12)
    assert metrics.memory_bandwidth_gbps == pytest.approx(logical_bytes / 0.002 / 1e9)


_FP8_BACKEND = "vllm_flashinfer_sparse_sm90_fp8"
_FP8_SPEC = _SPEC | {"cache_dtype": "fp8_e4m3", "cache_layout": "token_major_mqa_fp8_latent"}


def test_fp8_registration_is_the_sm90_e4m3_cache_in_vllm_env() -> None:
    """Catches the FP8 cache leaking onto the BF16 rows (or the reverse): the
    launcher picks a backend by the config's KV dtype, so each dtype must map
    to exactly one SM90 FlashInfer backend."""
    spec = find_kernel_profiler_spec(KIND, _FP8_BACKEND)
    assert spec.args_schema is DsaSparseMlaAttentionArgs
    assert spec.table_name == KIND
    assert spec.supports.allows(DType.BF16, DType.FP8_E4M3, gpu="NVIDIA H200")
    assert not spec.supports.allows(DType.BF16, DType.BF16, gpu="NVIDIA H200")
    assert not spec.supports.allows(DType.FP8_E4M3, DType.FP8_E4M3, gpu="NVIDIA H200")
    assert not spec.supports.allows(DType.BF16, DType.FP8_E4M3, gpu="NVIDIA B200")
    assert spec.subprocess_env == "vllm_env"
    assert spec.runner_ref.function_name == (
        "profile_dsa_sparse_mla_attention_vllm_flashinfer_sparse_sm90_fp8"
    )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({}, "reached the build"),
        ({"cache_dtype": "bf16"}, "cache_dtype must be fp8_e4m3"),
        ({"cache_layout": "token_major_mqa_bf16_latent"}, "cache_layout must be"),
        ({"q_dtype": "fp8_e4m3"}, "q_dtype must be bf16"),
        ({"valid_counts": "u:2052x2"}, "at most"),
    ],
)
def test_fp8_runner_accepts_only_the_e4m3_latent_layout(monkeypatch, overrides, match) -> None:
    """Catches the FP8 backend measuring a BF16 cache (or an FP8 query) under
    the FP8 key, which would price --kv-cache-dtype fp8 with BF16 times."""

    def reached(*_args, **kwargs):
        assert kwargs["fp8"] is True
        raise ProfilerNotImplemented("reached the build")

    monkeypatch.setattr(runner, "_load_sm90_state_cls", lambda: object())
    monkeypatch.setattr(runner, "_build_sm90_operands", reached)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    with pytest.raises(ProfilerNotImplemented, match=match):
        runner.profile_dsa_sparse_mla_attention_vllm_flashinfer_sparse_sm90_fp8(
            **(_FP8_SPEC | overrides)
        )


def test_fp8_state_cache_and_run_match_forward_mqa() -> None:
    """Catches a drift from vLLM's FP8 path: a plan declared with the uint8
    spec dtype instead of float8_e4m3fn, a cache that is not e4m3, a run without
    forward_mqa's ckv/kpe scales, or a reference that reads values other than
    the cache's exact dequantization."""
    validated = shared._ValidatedArgs(
        num_queries=2,
        num_cache_tokens=70,
        valid_counts=(5, 3),
        index_distribution="unique_scattered_pages",
    )
    cpu = torch.device("cpu")
    operands = runner._build_sm90_operands(
        torch, validated, num_heads=4, selected_k=8, device=cpu, fp8=True
    )
    assert operands.cache.dtype is torch.float8_e4m3fn
    assert operands.cache.shape == (70, 1, 512)
    assert operands.reference_cache.dtype is torch.bfloat16
    assert torch.equal(operands.reference_cache, operands.cache.to(torch.bfloat16))
    bf16 = runner._build_sm90_operands(torch, validated, num_heads=4, selected_k=8, device=cpu)
    assert bf16.reference_cache is bf16.cache
    # Same synthetic values, rounded to e4m3: the indices and queries match.
    assert torch.equal(operands.selected_indices, bf16.selected_indices)
    assert torch.equal(operands.q, bf16.q)
    assert torch.equal(
        bf16.cache.to(torch.float8_e4m3fn).view(torch.uint8), operands.cache.view(torch.uint8)
    )

    state = runner._make_state(
        torch,
        _FakeState,
        num_queries=2,
        num_heads=4,
        selected_k=8,
        softmax_scale=0.0625,
        device=cpu,
        fp8=True,
    )
    assert state.log[0][1][2] is torch.float8_e4m3fn
    assert runner._launch(state, operands, fp8=True) == "output"
    (_, (_q, _q_pe, ckv, kpe), run_kwargs) = state.log[1]
    assert run_kwargs == {"ckv_scale": 1.0, "kpe_scale": 1.0}
    assert ckv.dtype is torch.float8_e4m3fn and kpe.shape == (70, 1, 0)


def test_fp8_profile_counts_one_byte_per_cached_element(monkeypatch) -> None:
    """Catches FP8 rows logging BF16 cache traffic, which would halve the
    reported bandwidth of the e4m3 reads."""
    monkeypatch.setattr(runner, "_load_sm90_state_cls", lambda: "state_cls")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    flags: list[object] = []
    monkeypatch.setattr(
        runner, "_build_sm90_operands", lambda *_a, **kwargs: flags.append(kwargs["fp8"])
    )
    monkeypatch.setattr(
        runner, "_make_state", lambda _torch, _cls, **kwargs: flags.append(kwargs["fp8"])
    )
    monkeypatch.setattr(runner, "_plan_and_pack", lambda *_a, **_k: None)
    monkeypatch.setattr(
        runner, "_check_correctness", lambda *_a, **kwargs: flags.append(kwargs["fp8"])
    )
    monkeypatch.setattr(runner, "_launch", lambda *_a, **kwargs: flags.append(kwargs["fp8"]))

    def fake_timer(callable_, *, kernel_name):
        callable_()
        return 2.0

    monkeypatch.setattr(runner.Timer, "cupti", fake_timer)
    monkeypatch.setattr(runner.Energy, "perf", lambda *_a, **_k: 0.5)
    metrics = runner.profile_dsa_sparse_mla_attention_vllm_flashinfer_sparse_sm90_fp8(**_FP8_SPEC)
    assert flags == [True, True, True, True]
    logical_bytes = 2 * 2 * 16 * 512 + 4 * 2 * 2176 + 1 * (2048 + 2049) * 512
    logical_bytes += 2 * 2 * 16 * 512 + 8 * 2 * 16
    assert metrics.memory_bandwidth_gbps == pytest.approx(logical_bytes / 0.002 / 1e9)
