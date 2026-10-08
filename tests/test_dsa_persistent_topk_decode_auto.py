import pytest
import torch

from profiling.runners.attention.dsa_persistent_topk_decode_auto import (
    _build_operands,
    _check_output,
    _validate_args,
)
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented

_BASE = (4, 2048, 1, 2048, 512, 2048, "fp32", "int32", "max_ragged")


def test_context_modes_keep_the_kind_row_lengths():
    # max_ragged: request b holds context_len - b logits.
    assert _validate_args(*_BASE) == (2048, 2047, 2046, 2045)
    # uniform next_n=6 is GLM-5.3's flattened MTP5 verify: each request's rows
    # end at L-5..L, the same lengths vLLM's flattened decode metadata builds.
    uniform = _validate_args(2, 600, 6, *_BASE[3:8], "uniform")
    assert uniform == (595, 596, 597, 598, 599, 600) * 2


def test_unsupported_identities_fail_closed():
    with pytest.raises(ProfilerNotImplemented, match="top_k in"):
        _validate_args(*_BASE[:4], 256, *_BASE[5:])
    with pytest.raises(ProfilerNotImplemented):
        _validate_args(*_BASE[:6], "bf16", *_BASE[7:])
    with pytest.raises(ValueError, match="context_mode"):
        _validate_args(*_BASE[:8], "ragged")
    with pytest.raises(ValueError, match="logits_row_stride"):
        _validate_args(*_BASE[:5], 1024, *_BASE[6:])
    with pytest.raises(ValueError, match="next_n - 1"):
        _validate_args(1, 4, 6, *_BASE[3:8], "uniform")


def _operands(lengths, *, batch_size, next_n, stride=1024, width=1024, mode="max_ragged"):
    return _build_operands(
        torch,
        lengths=lengths,
        batch_size=batch_size,
        next_n=next_n,
        max_model_len=width,
        logits_row_stride=stride,
        top_k=512,
        context_mode=mode,
        device="cpu",
    )


def test_operands_keep_the_padded_row_stride_and_request_shape():
    # The resolved backend depends on logits.stride(0) and the row count, so the
    # visible width must be a view of rows spaced logits_row_stride apart.
    operands = _operands((600,) * 12, batch_size=2, next_n=6, stride=1028, width=1024)
    assert tuple(operands.logits.shape) == (12, 1024)
    assert operands.logits.stride(0) == 1028
    assert tuple(operands.lengths.shape) == (2, 6)
    assert operands.max_seq_len == 600


def _fill_like_kernel(operands):
    for row, length in enumerate(operands.lengths.view(-1).tolist()):
        operands.indices[row].fill_(-1)
        count = min(length, operands.top_k)
        if length <= operands.top_k:
            operands.indices[row, :count] = torch.arange(count, dtype=torch.int32)
        else:
            picked = torch.topk(operands.logits[row, :length], count).indices
            operands.indices[row] = picked.flip(0).to(torch.int32)  # order is free


def test_check_accepts_kernel_output_and_rejects_a_wrong_selection():
    operands = _operands((700, 513, 300, 0), batch_size=4, next_n=1)
    _fill_like_kernel(operands)
    _check_output(torch, operands)
    operands.indices[0, 0] = int(torch.argmin(operands.logits[0, :700]))
    with pytest.raises((AssertionError, KernelLaunchFailed)):
        _check_output(torch, operands)


def test_check_requires_the_minus_one_tail_on_short_rows():
    operands = _operands((700, 300), batch_size=2, next_n=1)
    _fill_like_kernel(operands)
    operands.indices[1, 300] = 0
    with pytest.raises(KernelLaunchFailed, match="natural-position"):
        _check_output(torch, operands)
