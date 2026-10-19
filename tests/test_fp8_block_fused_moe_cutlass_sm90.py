"""Behavioral tests for the SM90 CUTLASS FP8 block-scale backend of `nvfp4_fused_moe` (no GPU)."""

from __future__ import annotations

import pytest
import torch

from profiling.runners.moe.fp8_block_fused_moe_cutlass_sm90 import (
    _logical_bytes,
    _validate_args,
    precomputed_moe_reference,
    prepare_weights_for_cutlass,
)


def _spec(**overrides: object) -> dict:
    # GLM-5.3-Flash on one TP4 rank (all 288 experts, I/rank 512), 16 tokens.
    batches = [0] * 288
    for expert in range(0, 64, 2):
        batches[expert] = 4
    spec = {
        "num_tokens": 16,
        "hidden_size": 4096,
        "intermediate_size": 512,
        "num_experts": 288,
        "num_local_experts": 288,
        "top_k": 8,
        "input_dtype": "bf16",
        "weight_format": "fp8_e4m3",
        "group_size": 128,
        "routing_method": "precomputed",
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_numerator": 1,
        "routed_scaling_denominator": 1,
        "per_expert_batches": tuple(batches),
    }
    spec.update(overrides)
    return spec


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"routing_method": "deepseek_v3"}, "precomputed"),
        ({"routed_scaling_numerator": 5, "routed_scaling_denominator": 2}, "1/1"),
        ({"num_local_experts": 72}, "expert parallelism"),
        ({"input_dtype": "fp16"}, "BF16"),
        ({"weight_format": "nvfp4_e2m1", "group_size": 16}, "fp8_e4m3"),
    ],
)
def test_validation_rejects_keys_the_measured_call_would_not_honor(
    override: dict, message: str
) -> None:
    """The call routes nothing and holds every expert: a DeepSeekV3, scaled or
    EP-sliced key would file this TP-only timing under work it never did."""
    with pytest.raises(ValueError, match=message):
        _validate_args(**_spec(**override))


def test_logical_bytes_charge_fp8_weights_only_for_active_experts() -> None:
    """Moving one row onto an idle expert adds exactly one expert's FP8 weights
    plus its FP32 128x128 block scales; activations stay per token."""
    spec = _validate_args(**_spec())
    moved = list(spec["per_expert_batches"])
    moved[0] -= 1
    moved[1] += 1
    moved_spec = dict(spec, per_expert_batches=tuple(moved))
    hidden, intermediate = 4096, 512
    per_expert = 3 * intermediate * hidden + 4 * 3 * (intermediate // 128) * (hidden // 128)
    assert _logical_bytes(moved_spec) - _logical_bytes(spec) == per_expert


def _swap_w13_to_w31(x: torch.Tensor) -> torch.Tensor:
    # vLLM flashinfer_utils.swap_w13_to_w31, verbatim.
    return x.reshape(-1, 2, x.shape[-2] // 2, x.shape[-1]).flip(dims=[1]).reshape(x.shape)


def test_cutlass_weights_swap_gate_and_up_with_their_scales_and_clamp_dead_scales() -> None:
    """FlashInfer's SwiGLU reads [up; gate]. A swap of the weights without their
    block scales (or no swap) would multiply the wrong half by the wrong scale;
    an unclamped ~0 scale makes the SM90 kernel emit NaN."""
    w13 = torch.arange(256 * 128, dtype=torch.float32).reshape(1, 256, 128)
    w13_scale = torch.tensor([[[1e-23], [2.0]]])  # gate block, up block
    w2 = torch.zeros((1, 128, 128))
    w2_scale = torch.tensor([[[0.0]]])
    prepared = prepare_weights_for_cutlass(_swap_w13_to_w31, w13, w13_scale, w2, w2_scale)
    torch.testing.assert_close(prepared["fc1_expert_weights"][0, :128], w13[0, 128:])
    torch.testing.assert_close(prepared["fc1_expert_weights"][0, 128:], w13[0, :128])
    assert prepared["fc1_scales"].flatten().tolist() == [2.0, pytest.approx(1e-10)]
    assert prepared["fc2_scales"].item() == pytest.approx(1e-10)


def test_reference_clamps_gate_above_and_up_on_both_sides() -> None:
    """GLM's swiglu_limit as vLLM applies it (and FlashInfer's SwigluBias with
    alpha 1, beta 0): silu(min(gate, L)) * clamp(up, -L, L), weighted by the
    precomputed top-k weight."""
    fp8 = torch.float8_e4m3fn
    hidden_size = intermediate = 128
    hidden = torch.ones((1, hidden_size))
    w13 = torch.cat(
        [torch.ones((1, intermediate, hidden_size)), -torch.ones((1, intermediate, hidden_size))],
        dim=1,
    ).to(fp8)
    w13_scale = torch.full((1, 2, 1), 0.25)  # gate = 32 -> 10, up = -32 -> -10
    w2 = torch.ones((1, hidden_size, intermediate)).to(fp8)
    w2_scale = torch.full((1, 1, 1), 1.0 / 128)
    out = precomputed_moe_reference(
        torch,
        hidden=hidden,
        w13=w13,
        w13_scale=w13_scale,
        w2=w2,
        w2_scale=w2_scale,
        topk_ids=torch.zeros((1, 1), dtype=torch.int32),
        topk_weights=torch.full((1, 1), 0.5),
        clamp_limit=10.0,
        quantize_activations=False,
    )
    act = torch.nn.functional.silu(torch.tensor(10.0)) * -10.0
    torch.testing.assert_close(out, torch.full((1, hidden_size), float(act) * 0.5))
