"""Naive token-by-token Torch oracle for ``kda_recurrent_decode``.

Recomputes what ``fused_recurrent_kda`` computes on the GLM-5.3-Flash plain
decode and speculative-verify paths. It uses no FLA kernel:

1. l2-normalize q and k per head in fp32 (``eps=1e-6``). The recurrent kernel
   normalizes in registers, so unlike the chunk oracle nothing is rounded back
   to bf16.
2. Compute the per-channel safe gate
   ``g = lower_bound * sigmoid(exp(A_log[h]) * (raw_g + dt_bias[h, :]))`` and
   ``beta = sigmoid(raw_beta)`` in fp32.
3. Apply one gated-delta-rule step per token to each request's fp32 state
   ``S[h, k, v]``, its tokens in order: ``S *= exp(g)``,
   ``S += beta * k (v - k^T S)``, ``o = (scale * q)^T S``.

States live in the ``[slots, H, V, K]`` pool, the production layout, addressed
the way the kernel addresses them:

- plain decode: ``state_indices`` is ``[B]`` and each request has one token;
  its state is read from and written back to ``state_indices[n]``.
- speculative verify: ``state_indices`` is ``[B, L]`` for ``L`` tokens per
  request and ``num_accepted_tokens`` is ``[B]``. The state is read from
  ``state_indices[n, num_accepted_tokens[n] - 1]`` and the state after token
  ``t`` is stored to ``state_indices[n, t]``.

Only torch is imported, so the unit tests run this module on CPU.
"""

from __future__ import annotations

import torch

from profiling.runners.attention.kda_chunk_prefill_reference import kda_safe_gate

L2NORM_EPS = 1e-6


def kda_recurrent_decode_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_pool: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    lower_bound: float,
    num_accepted_tokens: torch.Tensor | None = None,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(o [T,H,V] in v.dtype, updated copy of state_pool)``.

    ``q``/``k``/``raw_g`` are ``[T,H,K]`` and ``v`` is ``[T,H,V]`` with the
    ``L`` tokens of each request adjacent (``T = B * L``); ``raw_beta`` is the
    un-sigmoided ``[T,H]``. ``state_indices`` is ``[B]`` (plain decode, ``L =
    1``) or ``[B, L]`` with ``num_accepted_tokens`` ``[B]`` (verify). The input
    pool is not mutated.
    """
    if state_indices.ndim == 1:
        if num_accepted_tokens is not None:
            raise ValueError("num_accepted_tokens needs [B, L] state_indices")
        read_slots = state_indices.long()
        write_slots = read_slots.unsqueeze(-1)
    elif state_indices.ndim == 2:
        if num_accepted_tokens is None:
            raise ValueError("[B, L] state_indices need num_accepted_tokens")
        write_slots = state_indices.long()
        column = num_accepted_tokens.long().unsqueeze(-1) - 1
        read_slots = write_slots.gather(1, column).squeeze(-1)
    else:
        raise ValueError(f"state_indices must be [B] or [B, L], got {tuple(state_indices.shape)}")
    batch, query_len = write_slots.shape
    tokens, heads, head_dim = q.shape
    if tokens != batch * query_len:
        raise ValueError(f"{tokens} tokens do not split into {batch} requests of {query_len}")
    if scale is None:
        scale = head_dim**-0.5

    def per_request(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(batch, query_len, *x.shape[1:])

    qf, kf = q.float(), k.float()
    qn = per_request(qf * torch.rsqrt(qf.square().sum(-1, keepdim=True) + L2NORM_EPS) * scale)
    kn = per_request(kf * torch.rsqrt(kf.square().sum(-1, keepdim=True) + L2NORM_EPS))
    decay = per_request(kda_safe_gate(raw_g, a_log, dt_bias, lower_bound).exp())  # [B,L,H,K]
    beta = per_request(raw_beta.float().sigmoid())
    vf = per_request(v.float())

    updated = state_pool.clone()
    state = state_pool[read_slots].float().transpose(-1, -2)  # [B, H, K, V]
    outputs = []
    for t in range(query_len):
        state = state * decay[:, t].unsqueeze(-1)
        predicted = torch.einsum("bhk,bhkv->bhv", kn[:, t], state)
        correction = beta[:, t].unsqueeze(-1) * (vf[:, t] - predicted)
        state = state + kn[:, t].unsqueeze(-1) * correction.unsqueeze(-2)
        outputs.append(torch.einsum("bhk,bhkv->bhv", qn[:, t], state))
        updated[write_slots[:, t]] = state.transpose(-1, -2).to(state_pool.dtype)
    output = torch.stack(outputs, dim=1).reshape(tokens, heads, -1)
    return output.to(v.dtype), updated
