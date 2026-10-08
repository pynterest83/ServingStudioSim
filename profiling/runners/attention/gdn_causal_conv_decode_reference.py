"""Torch semantics for GDN/KDA causal-convolution decode and speculative verify."""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["gdn_causal_conv_decode_reference"]

_SUPPORTED_KERNEL_SIZES = frozenset(range(2, 7))


def gdn_causal_conv_decode_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    slot_indices: torch.Tensor,
    *,
    num_accepted_tokens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one GDN depthwise-convolution decode or speculative verify step.

    ``x`` is ``[B, C]``, depthwise ``weight`` is ``[C, W]``, dim-first
    slot-indexed ``state`` is ``[S, C, W - 1]``, and ``slot_indices`` is
    ``[B]``. Qwen3.6 uses ``C=8192`` and ``W=4``. Activations, weights, state,
    and output are BF16, while the channelwise dot product and SiLU are FP32.

    Slot zero is reserved by vLLM's cache layout, so every active row must name
    a distinct slot in ``[1, S)``. Selected state slots are shifted left and
    receive ``x`` as their newest sample. State is mutated in place and returned
    by identity; output is newly allocated, and ``x`` and ``weight`` are not
    mutated. All validation completes before state mutation.

    With ``num_accepted_tokens`` (int32 ``[B]``) this is the speculative verify
    step that ``causal_conv1d_update`` runs with ``num_accepted_tokens``,
    ``query_start_loc`` and ``max_query_len``: ``x`` is ``[B * L, C]``, the
    ``L`` tokens of each request adjacent, and the state is at least ``W - 2 +
    L`` samples wide (vLLM allocates ``W - 1 + num_spec``). Request ``b`` reads
    its window at offset ``o = num_accepted_tokens[b] - 1``: the ``W - 1``
    samples ``state[slot, :, o : o + W - 1]`` precede its first token, and each
    token convolves the ``W - 1`` samples before it. The state then holds
    ``state[slot, :, o + 1 : o + W - 1]`` followed by all ``L`` tokens, written
    from sample 0; samples past ``W - 2 + L`` are left alone.

    The reference excludes non-uniform query lengths, bias, activations other
    than SiLU, non-BF16 data, duplicate/invalid slots, non-dim-first state, and
    kernel widths outside the production Triton implementation's supported
    range of 2 through 6.
    """
    _validate(x, weight, state, slot_indices, num_accepted_tokens)

    slots = slot_indices.to(torch.int64)
    batch_size, channels = slot_indices.shape[0], x.shape[1]
    query_len = x.shape[0] // batch_size
    history = weight.shape[1] - 1
    selected_state = state.index_select(0, slots)
    if num_accepted_tokens is None:
        prior = selected_state
    else:
        offsets = num_accepted_tokens.to(torch.int64) - 1
        positions = offsets.view(-1, 1, 1) + torch.arange(history, device=x.device)
        prior = selected_state.gather(-1, positions.expand(batch_size, channels, history))
    tokens = x.reshape(batch_size, query_len, channels).transpose(1, 2)  # [B, C, L]
    window = torch.cat((prior, tokens), dim=-1)  # [B, C, W - 1 + L]
    taps = window.unfold(-1, weight.shape[1], 1)  # [B, C, L, W]
    accumulator = torch.sum(taps.float() * weight.float().view(1, channels, 1, -1), dim=-1)
    output = F.silu(accumulator).to(torch.bfloat16).transpose(1, 2).reshape(x.shape)
    updated_state = window[..., 1:]

    if num_accepted_tokens is None:
        state.index_copy_(0, slots, updated_state)
    else:
        written = selected_state.clone()
        written[..., : updated_state.shape[-1]] = updated_state
        state.index_copy_(0, slots, written)
    return output, state


def _validate(
    x: object,
    weight: object,
    state: object,
    slot_indices: object,
    num_accepted_tokens: object,
) -> None:
    tensors = {
        "x": x,
        "weight": weight,
        "state": state,
        "slot_indices": slot_indices,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    assert all(isinstance(tensor, torch.Tensor) for tensor in tensors.values())
    x = tensors["x"]
    weight = tensors["weight"]
    state = tensors["state"]
    slot_indices = tensors["slot_indices"]

    expected_ranks = {"x": 2, "weight": 2, "state": 3, "slot_indices": 1}
    for name, tensor in tensors.items():
        expected_rank = expected_ranks[name]
        if tensor.ndim != expected_rank:
            raise ValueError(f"{name} must be rank {expected_rank}, got rank {tensor.ndim}")
        if any(dimension <= 0 for dimension in tensor.shape):
            raise ValueError(f"{name} dimensions must be positive")

    for name in ("x", "weight", "state"):
        tensor = tensors[name]
        if tensor.dtype is not torch.bfloat16:
            raise TypeError(f"{name} dtype must be torch.bfloat16, got {tensor.dtype}")
    if slot_indices.dtype is not torch.int32:
        raise TypeError(f"slot_indices dtype must be torch.int32, got {slot_indices.dtype}")

    for name, tensor in tensors.items():
        if tensor.layout is not torch.strided:
            raise ValueError(f"{name} must have torch.strided layout")
        if tensor.device != x.device:
            raise ValueError(
                f"{name} must be on the same device as x, got {tensor.device} and {x.device}"
            )
    if x.device.type == "meta":
        raise ValueError("meta tensors are not supported")

    num_tokens, channels = x.shape
    batch_size = slot_indices.shape[0]
    if num_accepted_tokens is None:
        query_len = 1
        if slot_indices.shape != (num_tokens,):
            raise ValueError(
                f"slot_indices must have shape ({num_tokens},), got {tuple(slot_indices.shape)}"
            )
    else:
        if num_tokens % batch_size:
            raise ValueError(
                f"x holds {num_tokens} tokens, not a whole number per {batch_size} requests"
            )
        query_len = num_tokens // batch_size
        if not isinstance(num_accepted_tokens, torch.Tensor):
            raise TypeError("num_accepted_tokens must be a torch.Tensor")
        if num_accepted_tokens.dtype is not torch.int32:
            raise TypeError(
                f"num_accepted_tokens dtype must be torch.int32, got {num_accepted_tokens.dtype}"
            )
        if num_accepted_tokens.shape != (batch_size,):
            raise ValueError(
                f"num_accepted_tokens must have shape ({batch_size},), "
                f"got {tuple(num_accepted_tokens.shape)}"
            )
        if num_accepted_tokens.device != x.device:
            raise ValueError("num_accepted_tokens must be on the same device as x")
        if torch.any((num_accepted_tokens < 1) | (num_accepted_tokens > query_len)).item():
            raise ValueError(f"num_accepted_tokens must lie in [1, {query_len}]")
    if weight.shape[0] != channels:
        raise ValueError(f"weight must have {channels} channels, got {weight.shape[0]}")

    kernel_size = weight.shape[1]
    if kernel_size not in _SUPPORTED_KERNEL_SIZES:
        raise ValueError(
            "kernel_size must be supported by the production Triton kernel "
            f"({min(_SUPPORTED_KERNEL_SIZES)}..{max(_SUPPORTED_KERNEL_SIZES)}), "
            f"got {kernel_size}"
        )
    if num_accepted_tokens is None:
        expected_state_shape = (state.shape[0], channels, kernel_size - 1)
        if state.shape != expected_state_shape:
            raise ValueError(
                f"state must use dim-first shape {expected_state_shape}, got {tuple(state.shape)}"
            )
    else:
        min_width = kernel_size - 2 + query_len
        if state.shape[1] != channels or state.shape[2] < min_width:
            raise ValueError(
                f"state must use dim-first shape ({state.shape[0]}, {channels}, >= {min_width}), "
                f"got {tuple(state.shape)}"
            )

    if torch.any(slot_indices <= 0).item():
        raise ValueError("slot_indices must not select reserved slot zero or negative slots")
    if torch.any(slot_indices >= state.shape[0]).item():
        raise ValueError(f"slot_indices must be less than slot count {state.shape[0]}")
    if torch.unique(slot_indices).numel() != batch_size:
        raise ValueError("slot_indices must be unique for independent state mutation")
