"""One-launch vLLM Triton runner for GDN/KDA causal-convolution decode and verify.

The timed callable is exactly vLLM's fused causal-convolution state update,
``causal_conv1d_update``, on one of the two branches the GDN/KDA layers take:

- ``query_len == 1`` (plain decode): ``x`` is ``[B, channels]`` and the call
  passes only ``conv_state_indices`` (``[B]``, slots ``1..B``).
- ``query_len > 1`` (speculative verify, as ``Glm5NextLinearAttention._forward``
  calls it): ``x`` is ``[B * query_len, channels]``, ``conv_state_indices`` is
  column 0 of the ``[B, query_len]`` ``spec_state_indices`` (a strided view;
  slots ``1..B*query_len``), ``num_accepted_tokens`` is ``[B]`` of ones,
  ``query_start_loc`` is ``arange(0, T + 1, query_len)`` and
  ``max_query_len = query_len``.

The conv state uses vLLM's default "SD" layout (``VLLM_SSM_CONV_STATE_LAYOUT``
unset): ``[slots, width, channels]`` storage, passed as its ``(-1, -2)``
transpose as ``_forward`` does. ``width`` is ``kernel_size - 1 + (query_len -
1)``, the ``kda_state_shape(num_spec=K)`` allocation. ``x`` is a contiguous
token-major buffer; production passes a column slice of the input projection,
which changes only the token stride the kernel is given.

Allocation, the semantic correctness guard, and operand restoration are setup
work outside CUPTI and energy measurement. Reported FLOPs and bytes are semantic
logical counts, not physical Triton instructions or device traffic.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention._gdn_common import load_required_callable
from profiling.runners.attention.gdn_causal_conv_decode_torch import (
    _logical_bytes,
    _semantic_flops,
)
from profiling.runners.exceptions import (
    KernelLaunchFailed,
    OOMError,
    ProfilerNotImplemented,
)
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "gdn_causal_conv_decode:vllm_triton"
_CALLABLE_MODULE = "vllm.model_executor.layers.mamba.ops.causal_conv1d"
_CALLABLE_NAME = "causal_conv1d_update"
_KERNEL_NAME = "_causal_conv1d_update_kernel"
# Kernel limit, not a measured set: `_causal_conv1d_update_kernel` has
# KERNEL_WIDTH branches for 2 through 6 only.
_SUPPORTED_KERNEL_SIZES = frozenset(range(2, 7))
_OUTPUT_ATOL = 1e-2
_OUTPUT_RTOL = 1e-2
# The verify step's accepted-token count. It only shifts the window the kernel
# reads (offset ``accepted - 1``), so it is fixed, not an arg.
NUM_ACCEPTED_TOKENS = 1


@dataclass(frozen=True)
class _ValidatedArgs:
    batch_size: int
    query_len: int
    channels: int
    kernel_size: int
    dtype: DType
    state_dtype: DType

    @property
    def num_tokens(self) -> int:
        return self.batch_size * self.query_len

    @property
    def is_verify(self) -> bool:
        return self.query_len > 1


@dataclass(frozen=True)
class _OperandShapes:
    x: tuple[int, int]
    weight: tuple[int, int]
    # SD storage: [slots, width, channels]; the call gets its (-1, -2) transpose.
    state_storage: tuple[int, int, int]
    slot_indices: tuple[int]
    output: tuple[int, int]


@dataclass(frozen=True)
class _Operands:
    x: Any
    weight: Any
    # The dim-first [slots, channels, width] view the call receives.
    state: Any
    slot_indices: Any
    # Verify only: the [B, query_len] spec_state_indices whose column 0 is
    # slot_indices, the accepted-token counts and the query start offsets.
    spec_state_indices: Any = None
    num_accepted_tokens: Any = None
    query_start_loc: Any = None


def _validate_args(
    batch_size: int,
    query_len: int,
    channels: int,
    kernel_size: int,
    dtype: DType | str,
    state_dtype: DType | str,
) -> _ValidatedArgs:
    validated = _ValidatedArgs(
        batch_size=int(batch_size),
        query_len=int(query_len),
        channels=int(channels),
        kernel_size=int(kernel_size),
        dtype=DType.from_value(dtype),
        state_dtype=DType.from_value(state_dtype),
    )
    if validated.batch_size <= 0 or validated.channels <= 0:
        raise ValueError(
            "batch_size and channels must be > 0, got "
            f"batch_size={validated.batch_size}, channels={validated.channels}"
        )
    if validated.query_len < 1:
        raise ValueError(f"query_len must be >= 1, got {validated.query_len}")
    if validated.kernel_size not in _SUPPORTED_KERNEL_SIZES:
        raise ValueError(
            "kernel_size must be supported by the production Triton kernel "
            f"({min(_SUPPORTED_KERNEL_SIZES)}..{max(_SUPPORTED_KERNEL_SIZES)}), "
            f"got {validated.kernel_size}"
        )
    if validated.dtype is not DType.BF16 or validated.state_dtype is not DType.BF16:
        raise ValueError(
            "vllm_triton gdn_causal_conv_decode requires dtype=bf16 and "
            "state_dtype=bf16, got "
            f"{validated.dtype.value} and {validated.state_dtype.value}"
        )
    return validated


def _operand_shapes(args: _ValidatedArgs) -> _OperandShapes:
    return _OperandShapes(
        x=(args.num_tokens, args.channels),
        weight=(args.channels, args.kernel_size),
        state_storage=(
            args.num_tokens + 1,
            args.kernel_size - 1 + (args.query_len - 1),
            args.channels,
        ),
        slot_indices=(args.batch_size,),
        output=(args.num_tokens, args.channels),
    )


def _valid_slot_indices(batch_size: int, query_len: int = 1) -> tuple[int, ...]:
    """Each request's conv slot: column 0 of ``arange(1, B*L + 1).view(B, L)``."""
    return tuple(range(1, batch_size * query_len + 1, query_len))


def _load_fused_callable() -> Any:
    return load_required_callable(
        importlib.import_module,
        backend=_BACKEND,
        module_name=_CALLABLE_MODULE,
        callable_name=_CALLABLE_NAME,
    )


def _build_operands(torch: Any, args: _ValidatedArgs, *, device: Any) -> _Operands:
    shapes = _operand_shapes(args)
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    activation_dtype = args.dtype.torch()
    x = torch.empty(shapes.x, dtype=activation_dtype, device=device).uniform_(
        -0.25, 0.25, generator=generator
    )
    weight = torch.empty(shapes.weight, dtype=activation_dtype, device=device).uniform_(
        -0.125, 0.125, generator=generator
    )
    state = (
        torch.empty(shapes.state_storage, dtype=args.state_dtype.torch(), device=device)
        .uniform_(-0.25, 0.25, generator=generator)
        .transpose(-1, -2)
    )
    if not args.is_verify:
        slot_indices = torch.tensor(
            _valid_slot_indices(args.batch_size), dtype=torch.int32, device=device
        )
        return _Operands(
            x=x.contiguous(), weight=weight.contiguous(), state=state, slot_indices=slot_indices
        )
    spec_state_indices = torch.arange(
        1, args.num_tokens + 1, dtype=torch.int32, device=device
    ).view(args.batch_size, args.query_len)
    return _Operands(
        x=x.contiguous(),
        weight=weight.contiguous(),
        state=state,
        slot_indices=spec_state_indices[:, 0],
        spec_state_indices=spec_state_indices,
        num_accepted_tokens=torch.full(
            (args.batch_size,), NUM_ACCEPTED_TOKENS, dtype=torch.int32, device=device
        ),
        query_start_loc=torch.arange(
            0, args.num_tokens + 1, args.query_len, dtype=torch.int32, device=device
        ),
    )


def _invoke_fused(fused_callable: Any, operands: _Operands) -> Any:
    if operands.num_accepted_tokens is None:
        return fused_callable(
            operands.x,
            operands.state,
            operands.weight,
            bias=None,
            activation="silu",
            conv_state_indices=operands.slot_indices,
            validate_data=False,
        )
    return fused_callable(
        operands.x,
        operands.state,
        operands.weight,
        bias=None,
        activation="silu",
        conv_state_indices=operands.slot_indices,
        num_accepted_tokens=operands.num_accepted_tokens,
        query_start_loc=operands.query_start_loc,
        max_query_len=operands.spec_state_indices.shape[1],
        validate_data=False,
    )


def _shares_storage(torch: Any, left: Any, right: Any) -> bool:
    return (
        left.data_ptr() == right.data_ptr()
        and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
        and left.device == right.device
    )


def _check_correctness(
    torch: Any,
    fused_callable: Any,
    operands: _Operands,
    args: _ValidatedArgs,
    *,
    synchronize: Any | None = None,
) -> None:
    """Check aliasing and semantics from identical operands, then restore them."""
    from profiling.runners.attention.gdn_causal_conv_decode_reference import (
        gdn_causal_conv_decode_reference,
    )

    synchronize = torch.cuda.synchronize if synchronize is None else synchronize
    x_snapshot = operands.x.clone()
    state_snapshot = operands.state.clone()
    weight_snapshot = operands.weight.clone()
    indices_snapshot = operands.slot_indices.clone()
    reference_state = state_snapshot.clone()
    expected_output, expected_state = gdn_causal_conv_decode_reference(
        x_snapshot.clone(),
        weight_snapshot,
        reference_state,
        indices_snapshot,
        num_accepted_tokens=operands.num_accepted_tokens,
    )
    try:
        returned = _invoke_fused(fused_callable, operands)
        synchronize()
        if not args.is_verify and returned is operands.x:
            # The plain branch unsqueezes x to [B, C, 1] and returns the squeeze.
            raise AssertionError("fused output must be a distinct view of x")
        if not _shares_storage(torch, returned, operands.x):
            raise AssertionError("fused output must alias x storage")
        if tuple(returned.shape) != _operand_shapes(args).output:
            raise AssertionError(f"unexpected fused output shape {tuple(returned.shape)}")
        if returned.dtype is not torch.bfloat16:
            raise AssertionError(f"unexpected fused output dtype {returned.dtype}")
        torch.testing.assert_close(
            returned.float(),
            expected_output.float(),
            atol=_OUTPUT_ATOL,
            rtol=_OUTPUT_RTOL,
        )
        if not torch.equal(operands.state, expected_state):
            raise AssertionError("fused updated state must exactly match the reference")
        if torch.equal(operands.state[1:], state_snapshot[1:]):
            raise AssertionError("fused call did not mutate selected state slots")
        if not torch.equal(operands.state[0], state_snapshot[0]):
            raise AssertionError("reserved state slot zero was mutated")
        if not torch.equal(operands.weight, weight_snapshot):
            raise AssertionError("fused call mutated weight")
        if not torch.equal(operands.slot_indices, indices_snapshot):
            raise AssertionError("fused call mutated slot indices")
        if not torch.isfinite(returned).all() or not torch.isfinite(operands.state).all():
            raise AssertionError("fused output and updated state must remain finite")
    finally:
        # The fused callable overwrites x and state. Restore both outside the
        # timed region so subsequent calls begin from bounded valid operands.
        operands.x.copy_(x_snapshot)
        operands.state.copy_(state_snapshot)


def profile_gdn_causal_conv_decode_vllm_triton(
    batch_size: int,
    query_len: int,
    channels: int,
    kernel_size: int,
    dtype: DType | str,
    state_dtype: DType | str,
) -> ComputeMetrics:
    """Profile vLLM's fused causal-convolution decode or verify."""
    args = _validate_args(batch_size, query_len, channels, kernel_size, dtype, state_dtype)
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {_BACKEND}") from exc

    try:
        fused_callable = _load_fused_callable()
        device = torch.device("cuda", torch.cuda.current_device())
        operands = _build_operands(torch, args, device=device)
        _check_correctness(torch, fused_callable, operands, args)

        def kernel() -> Any:
            return _invoke_fused(fused_callable, operands)

        # No reset is included in either measurement region. Every callable is
        # one fused decode: state advances in place and the prior in-place output
        # becomes the next bounded input token. Allocation and the one-time guard
        # remain outside timing and energy measurement.
        time_ms = Timer.cupti(kernel, kernel_name=_KERNEL_NAME)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed") from exc

    flops = _semantic_flops(
        batch_size=args.batch_size,
        query_len=args.query_len,
        channels=args.channels,
        kernel_size=args.kernel_size,
    )
    logical_bytes = _logical_bytes(
        batch_size=args.batch_size,
        query_len=args.query_len,
        channels=args.channels,
        kernel_size=args.kernel_size,
        dtype=args.dtype,
        state_dtype=args.state_dtype,
    )
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        energy_j=float(energy_j),
        tflops=flops / elapsed_s / 1e12 if time_ms > 0.0 else 0.0,
        memory_bandwidth_gbps=(logical_bytes / elapsed_s / 1e9 if time_ms > 0.0 else 0.0),
    )
