"""Focused CPU tests for the ``kda_chunk_prefill`` ``vllm_flashkda`` backend.

No GPU, no DB. Each test names the defect it guards against. The H200
comparison of FlashKDA against the oracle runs inside the runner, and its
evidence comes from the profiling smoke.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from profiling.db.args import DType
from profiling.db.registry import find_kernel_profiler_spec
from profiling.kernels.kda_chunk_prefill import KIND, KdaChunkPrefillArgs
from profiling.runners.attention import kda_chunk_prefill_vllm_flashkda as runner
from profiling.runners.attention.kda_chunk_prefill_reference import kda_chunk_prefill_reference

# The GLM-5.3-Flash TP4 H200 prefill layer: 16 heads of 128.
_SHAPE = dict(
    num_tokens=70,
    max_sequence_length=66,
    num_decode_sequences=4,
    num_heads=2,
    head_dim=128,
    dtype="bf16",
)


def test_runner_takes_the_kind_args() -> None:
    # Defect: the new backend's runner kwargs drifting from the shared schema.
    names = [field for field in KdaChunkPrefillArgs.__dataclass_fields__]
    assert list(inspect.signature(runner.profile_kda_chunk_prefill_vllm_flashkda).parameters) == (
        names
    )


def test_registration_declares_only_the_verified_hopper_build() -> None:
    # Defect: offering FlashKDA on a GPU its build was never run against, or for
    # a dtype it does not take.
    spec = find_kernel_profiler_spec(KIND, "vllm_flashkda")
    assert spec.args_schema is KdaChunkPrefillArgs and spec.subprocess_env == "vllm_env"
    assert spec.supports.compute == frozenset({DType.BF16})
    assert spec.supports.sm_targets == frozenset({"sm_90a"})
    assert spec.supports.allows(DType.BF16, gpu="NVIDIA H200")
    assert not spec.supports.allows(DType.BF16, gpu="NVIDIA A100")
    assert not spec.supports.allows(DType.FP16, gpu="NVIDIA H200")


@pytest.mark.parametrize(
    "override",
    [{"head_dim": 64}, {"head_dim": 256}, {"dtype": "fp16"}, {"max_sequence_length": 1}],
)
def test_rejects_shapes_flashkda_never_runs(override: dict) -> None:
    # Defect: timing a head width FlashKDA is not built for (vLLM falls back to
    # Triton there) or an unreachable query-length-1 prefill.
    with pytest.raises(ValueError):
        runner.validate_args(**{**_SHAPE, **override})


def test_operands_and_call_reproduce_the_production_flashkda_prefill() -> None:
    # Defect: timing contiguous q/k/v (skips the three copy launches), a
    # pre-sigmoided or dense beta (skips FlashKDA's own sigmoid and transpose),
    # a wrong A_log/dt_bias view, or a workspace sized for other dimensions.
    shape = runner.validate_args(**_SHAPE)
    sizes: list[tuple[int, int, int]] = []

    def workspace_size(tokens: int, heads: int, sequences: int) -> int:
        sizes.append((tokens, heads, sequences))
        return 1024

    operands = runner.build_operands(
        torch, shape, device=torch.device("cpu"), workspace_size=workspace_size
    )
    assert sizes == [(70, 2, 5)] and operands.workspace.numel() == 1024
    seen: list = []
    runner.invoke(lambda *args: seen.extend(args), operands)

    q, k, v, g, beta, scale, out, workspace, a_log, dt_bias, lower_bound = seen[:11]
    initial_state, final_state, cu_seqlens, checkpoint, offsets = seen[11:]
    for tensor in (q, k, v, g):
        assert tensor.shape == (1, 70, 2, 128) and tensor.is_contiguous()
    assert not operands.q.is_contiguous() and operands.q.stride(1) == 3 * 256
    assert q.data_ptr() != operands.q.data_ptr()  # a real copy
    assert g.data_ptr() == operands.g.data_ptr()  # already dense: no copy
    assert beta is operands.beta and beta.dtype is torch.bfloat16
    assert beta.shape == (1, 70, 2) and beta.stride(1) == 3 * 256 + 2 + 2 * 128
    assert scale == 128**-0.5 and lower_bound == -5.0
    assert out is operands.out and out.shape == (1, 70, 2, 128) and out.is_contiguous()
    assert workspace is operands.workspace
    assert a_log.shape == (2,) and dt_bias.shape == (2, 128)
    assert a_log.dtype is dt_bias.dtype is torch.float32
    assert initial_state.shape == final_state.shape == (5, 2, 128, 128)
    assert final_state is operands.final_state and final_state.dtype is torch.float32
    assert bool(initial_state[4:].eq(0).all()) and bool(initial_state[:4].ne(0).any())
    assert cu_seqlens.tolist() == [0, 1, 2, 3, 4, 70] and cu_seqlens.dtype is torch.int32
    assert checkpoint is None and offsets is None


def test_oracle_sigmoids_raw_beta_logits_like_flashkda() -> None:
    # Defect: comparing FlashKDA's raw-logit beta against an oracle that treats
    # it as an already-sigmoided write strength.
    generator = torch.Generator().manual_seed(3)
    q, k, v, raw_g = (torch.randn(5, 2, 4, generator=generator) for _ in range(4))
    logits = torch.randn(5, 2, generator=generator)
    a_log, dt_bias = torch.zeros(2), torch.zeros(8)
    initial = torch.randn(2, 2, 4, 4, generator=generator)
    expected = kda_chunk_prefill_reference(
        q, k, v, raw_g, logits.sigmoid(), a_log, dt_bias, initial, [0, 2, 5], lower_bound=-5.0
    )
    actual = kda_chunk_prefill_reference(
        q,
        k,
        v,
        raw_g,
        logits,
        a_log,
        dt_bias,
        initial,
        [0, 2, 5],
        lower_bound=-5.0,
        beta_is_logits=True,
    )
    for left, right in zip(expected, actual, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
