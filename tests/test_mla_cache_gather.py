"""CPU tests for the mla_cache_gather kind (vLLM gather_and_maybe_dequant_cache)."""

from __future__ import annotations

import pytest
import torch

from profiling.db.args import DType
from profiling.db.registry import find_kernel_profiler_spec
from profiling.kernels.mla_cache_gather import KIND, MlaCacheGatherArgs
from profiling.runners.attention import mla_cache_gather as runner
from profiling.runners.attention.mla_cache_gather_reference import mla_cache_gather_reference
from profiling.runners.exceptions import ProfilerNotImplemented

_SPEC = {
    "num_requests": 3,
    "num_tokens": 200,
    "kv_lora_rank": 512,
    "rope_dim": 0,
    "block_size": 64,
    "cache_dtype": "fp8_e4m3",
    "output_dtype": "bf16",
    "cache_format": "plain",
}


def test_registration_admits_bf16_and_fp8_caches_in_vllm_env() -> None:
    """Catches the launcher refusing the FP8 cache the prefill workspace reads,
    or the row being run outside the vLLM env that owns the op."""
    spec = find_kernel_profiler_spec(KIND, "vllm_cuda")
    assert spec.args_schema is MlaCacheGatherArgs
    assert spec.subprocess_env == "vllm_env"
    assert spec.supports.allows(DType.BF16, DType.FP8_E4M3, gpu="NVIDIA H200")
    assert spec.supports.allows(DType.BF16, DType.BF16, gpu="NVIDIA H200")


def test_requests_split_evenly_with_the_remainder_first() -> None:
    """Catches a split that drops or duplicates rows: the profiled call must
    gather exactly num_tokens rows."""
    assert runner.request_lengths(3, 200) == (67, 67, 66)
    assert runner.request_lengths(1, 5) == (5,)
    assert sum(runner.request_lengths(4, 262_145)) == 262_145


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"num_tokens": 2}, "at least one row"),
        ({"rope_dim": 64, "kv_lora_rank": 448}, None),
        ({"rope_dim": 1}, "entries must be"),
        ({"output_dtype": "fp8_e4m3"}, "output_dtype must be bf16"),
        ({"cache_format": "fp8_ds_mla"}, "cache_format must be"),
    ],
)
def test_validation_rejects_shapes_the_kernel_cannot_take(overrides, match) -> None:
    """Catches rows the CUDA op would reject (entry widths outside its 320 / 512
    / 576 specializations) reaching a GPU worker."""
    if match is None:
        runner._validate_args(**(_SPEC | overrides))
        return
    with pytest.raises(ProfilerNotImplemented, match=match):
        runner._validate_args(**(_SPEC | overrides))


def test_operands_page_through_shuffled_blocks_and_match_the_reference() -> None:
    """Catches a workspace layout that is not each request's context in
    position order, or a cache whose pages are contiguous (page lookup
    untested): the reference walk must read every block exactly once."""
    validated = runner._validate_args(**_SPEC)
    operands = runner._build_operands(torch, validated, device=torch.device("cpu"))
    assert operands.cache.dtype is torch.float8_e4m3fn
    assert operands.src_cache.dtype is torch.uint8
    assert operands.src_cache.data_ptr() == operands.cache.data_ptr()
    assert operands.cu_seq_lens.tolist() == [0, 67, 134, 200]
    assert operands.workspace.shape == (200, 512)
    assert operands.workspace.dtype is torch.bfloat16
    used = operands.block_table[0, :2].tolist() + operands.block_table[1, :2].tolist()
    used += operands.block_table[2, :2].tolist()
    assert len(set(used)) == 6 and used != sorted(used)

    expected = mla_cache_gather_reference(
        operands.cache, operands.block_table, operands.cu_seq_lens, output_dtype=torch.bfloat16
    )
    # Request 1, position 66 lives in its second block at offset 2.
    block = operands.block_table[1, 1]
    assert torch.equal(expected[67 + 66], operands.cache[block, 2].to(torch.bfloat16))
    assert expected.shape == (200, 512)


def test_launch_forwards_the_production_arguments() -> None:
    """Catches the timed call drifting from forward_mqa's: the uint8 cache view,
    the 'fp8' dtype string, the k_scale tensor and the gathered row count."""
    validated = runner._validate_args(**_SPEC)
    operands = runner._build_operands(torch, validated, device=torch.device("cpu"))
    calls = []

    class _Ops:
        @staticmethod
        def gather_and_maybe_dequant_cache(**kwargs):
            calls.append(kwargs)

    runner._launch(_Ops, operands, cache_dtype=DType.FP8_E4M3)
    (call,) = calls
    assert call["src_cache"] is operands.src_cache
    assert call["kv_cache_dtype"] == "fp8"
    assert call["num_tokens"] == 200
    assert call["scale"].tolist() == [1.0]
    runner._launch(_Ops, operands, cache_dtype=DType.BF16)
    assert calls[-1]["kv_cache_dtype"] == "auto"


def test_logical_bytes_count_one_byte_per_fp8_element() -> None:
    """Catches FP8 rows logged with BF16 cache traffic (bandwidth off by 4/3)."""
    fp8 = runner._validate_args(**_SPEC)
    bf16 = runner._validate_args(**(_SPEC | {"cache_dtype": "bf16"}))
    pages = 2 + 2 + 2
    assert runner.logical_bytes(fp8) == 200 * 512 * 3 + 4 * pages + 4 * 4
    assert runner.logical_bytes(bf16) == 200 * 512 * 4 + 4 * pages + 4 * 4
