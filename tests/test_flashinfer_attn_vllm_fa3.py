"""CPU tests for the vllm_fa3 backend of flashinfer_attn_prefill / flashinfer_attn_rect."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from profiling.runners.attention import flashinfer_attn_prefill_and_rect as runner
from profiling.runners.exceptions import ProfilerNotImplemented

_DIMS = dict(num_qo_heads=4, num_kv_heads=2, head_dim=16)
_BF16 = dict(q_dtype="bf16", kv_dtype="bf16", o_dtype="bf16")


def test_prefill_entry_maps_cache_dims_and_rejects_cached_prefix(monkeypatch):
    # vLLM's causal FA3 call is new tokens over themselves; a prefix would be
    # a different (bottom-right causal) call that production never makes.
    calls = []
    monkeypatch.setattr(runner, "_run_vllm_fa3", lambda **kwargs: calls.append(kwargs))
    runner.profile_flashinfer_attn_prefill_vllm_fa3(prefix_len=0, append_len=96, **_DIMS, **_BF16)
    runner.profile_flashinfer_attn_rect_vllm_fa3(q_len=64, kv_len=200, **_DIMS, **_BF16)
    assert [(c["causal"], c["q_len"], c["kv_len"]) for c in calls] == [
        (True, 96, 96),
        (False, 64, 200),
    ]
    monkeypatch.undo()
    with pytest.raises(ProfilerNotImplemented, match="prefix_len == 0"):
        runner.profile_flashinfer_attn_prefill_vllm_fa3(
            prefix_len=32, append_len=96, **_DIMS, **_BF16
        )


def test_non_bf16_is_rejected_before_any_import():
    with pytest.raises(ProfilerNotImplemented, match="BF16"):
        runner.profile_flashinfer_attn_rect_vllm_fa3(
            q_len=8, kv_len=8, **_DIMS, q_dtype="fp8_e4m3", kv_dtype="fp8_e4m3", o_dtype="bf16"
        )


@pytest.mark.parametrize(
    ("causal", "q_len", "kv_len"), [(True, 40, 40), (True, 24, 40), (False, 40, 72)]
)
def test_reference_rows_match_sdpa_with_bottom_right_causal_and_gqa(causal, q_len, kv_len):
    # The correctness oracle must use FlashAttention's bottom-right causal
    # alignment and repeat KV heads for GQA, or a correct kernel would fail.
    generator = torch.Generator().manual_seed(0)
    q = torch.randn(q_len, 4, 16, generator=generator)
    k = torch.randn(kv_len, 2, 16, generator=generator)
    v = torch.randn(kv_len, 2, 16, generator=generator)
    rows = runner._vllm_fa3_check_rows(q_len)
    actual = runner._vllm_fa3_reference_rows(torch, q, k, v, rows, causal=causal, scale=0.25)
    mask = None
    if causal:
        mask = torch.ones(q_len, kv_len, dtype=torch.bool).tril(diagonal=kv_len - q_len)
    expected = F.scaled_dot_product_attention(
        q.transpose(0, 1),
        k.repeat_interleave(2, dim=1).transpose(0, 1),
        v.repeat_interleave(2, dim=1).transpose(0, 1),
        attn_mask=mask,
        scale=0.25,
    ).transpose(0, 1)
    torch.testing.assert_close(actual, expected[rows], atol=1e-5, rtol=1e-5)


def test_checked_rows_cover_both_ends_of_the_causal_diagonal():
    rows = runner._vllm_fa3_check_rows(2048)
    assert rows[0] == 0 and rows[-1] == 2047
    assert len(rows) == len(set(rows)) and rows == sorted(rows)
    assert runner._vllm_fa3_check_rows(3) == [0, 1, 2]
