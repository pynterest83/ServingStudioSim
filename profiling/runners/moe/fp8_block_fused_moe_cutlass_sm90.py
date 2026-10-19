"""Whole FlashInfer CUTLASS DeepSeek-FP8 block-scale fused MoE on SM90.

The timed callable is ``flashinfer.fused_moe.cutlass_fused_moe`` (vLLM's
``flashinfer_cutlass_fused_moe``) invoked exactly as vLLM's
``FlashInferExperts.apply`` does for ``--moe-backend flashinfer_cutlass`` on
an FP8 128x128 block checkpoint (``vllm/model_executor/layers/fused_moe/
experts/flashinfer_cutlass_moe.py``, ``use_deepseek_fp8_block_scale``):

* BF16 hidden states (``expects_unquantized_inputs``: the prepare step does
  not quantize, the call quantizes per token group of 128 itself);
* finished top-k ids (int32) and FP32 top-k weights: vLLM's router runs
  before the call, so the routing is precomputed;
* W31 FP8 weights and FP32 128x128 block scales, i.e. vLLM's
  ``prepare_fp8_moe_layer_for_fi`` without the TRT-LLM shuffle: gate/up
  swapped to up/gate (weights and scales) and the scales clamped to 1e-10;
* ``quant_scales=[w13_scale, w2_scale]``, ``input_sf=None``, SwiGLU with the
  per-local-expert ``swiglu_limit`` tensor (GLM-5.3-Flash ``swiglu_limit``),
  which FlashInfer runs as its SwigluBias adaptor (alpha 1, beta 0):
  ``silu(min(gate, L)) * clamp(up, -L, L)``, vLLM's clamp;
* FlashInfer's own defaults for PDL, ``tune_max_num_tokens`` (8192) and the
  fused finalize, as vLLM passes none of them.

One call launches the expert-map/sort kernels, the input expansion, the BF16
-> FP8 1x128 quantization, the FC1 DeepGEMM-style grouped GEMM (TensorRT-LLM's
``fp8_blockscale_gemm`` ``GroupedWithOffset``, NVRTC-compiled on first use and
cached under ``~/.tensorrt_llm``), the clamped SwiGLU, the second quantization,
the FC2 grouped GEMM and the finalize (top-k weighted reduce). All launches are
timed together with CUPTI, PDL overlap counted once.

``tp_size`` is passed as 1: TP only gates the FC bias (applied on rank 0),
and this layer has none, so every TP rank runs this same call on its
``intermediate_size`` shard.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.autotune_cache import autotune_cached
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics
from profiling.runners.moe.exact_topk import exact_topk_ids

WEIGHT_FORMAT = DType.FP8_E4M3
GROUP_SIZE = 128
#: Finished top-k ids and weights enter the call; nothing is routed inside it.
ROUTING_METHOD = "precomputed"
#: GLM-5.3-Flash ``swiglu_limit`` (config.json), forwarded by vLLM as the
#: per-local-expert ``gemm1_clamp_limit``.
SWIGLU_LIMIT = 10.0
#: ``prepare_fp8_moe_layer_for_fi``: near-zero block scales make the SM90
#: CUTLASS kernel emit NaN, so vLLM clamps them.
MIN_BLOCK_SCALE = 1e-10
_W13_SCALE = (0.05, 0.15)
_W2_SCALE = (0.005, 0.015)
_WEIGHT_CHUNK = 8
_PREPARED_WEIGHT_CACHE: dict[tuple[int, int, int, int, int], dict[str, Any]] = {}


def _load_runtime() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        import vllm.model_executor.layers.fused_moe  # noqa: F401  (breaks an import cycle)
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
            activation_to_flashinfer_type,
            swap_w13_to_w31,
        )
        from vllm.utils.flashinfer import flashinfer_cutlass_fused_moe
    except ImportError as exc:
        raise ProfilerNotImplemented("FlashInfer and the vLLM fork are required") from exc
    activation = activation_to_flashinfer_type(MoEActivation.SILU)
    return torch, swap_w13_to_w31, flashinfer_cutlass_fused_moe, activation


def _validate_args(**kwargs: Any) -> dict[str, Any]:
    args = dict(kwargs)
    for name in (
        "num_tokens",
        "hidden_size",
        "intermediate_size",
        "num_experts",
        "num_local_experts",
        "top_k",
        "group_size",
        "n_group",
        "topk_group",
        "routed_scaling_numerator",
        "routed_scaling_denominator",
    ):
        args[name] = int(args[name])
        if args[name] <= 0:
            raise ValueError(f"{name} must be positive")
    args["per_expert_batches"] = tuple(int(value) for value in args["per_expert_batches"])
    if DType.from_value(args["input_dtype"]) is not DType.BF16:
        raise ValueError("the CUTLASS FP8 block-scale MoE takes BF16 hidden states")
    if (
        DType.from_value(args["weight_format"]) is not WEIGHT_FORMAT
        or args["group_size"] != GROUP_SIZE
    ):
        raise ValueError(
            f"the CUTLASS FP8 block-scale MoE requires {WEIGHT_FORMAT.value} weights with "
            f"group_size={GROUP_SIZE}"
        )
    # Routing happens in vLLM's router before the call; a key naming a router
    # rule, a group restriction or a routed scale would describe work this call
    # never does.
    if args["routing_method"] != ROUTING_METHOD:
        raise ValueError(f"routing method must be {ROUTING_METHOD!r} (top-k ids come in)")
    if args["n_group"] != 1 or args["topk_group"] != 1:
        raise ValueError("precomputed routing has no expert groups (n_group = topk_group = 1)")
    if args["routed_scaling_numerator"] != args["routed_scaling_denominator"]:
        raise ValueError("precomputed routing carries its scale in the weights (scaling 1/1)")
    if args["hidden_size"] % GROUP_SIZE or args["intermediate_size"] % GROUP_SIZE:
        raise ValueError("hidden_size and intermediate_size must be multiples of 128")
    # TP without EP: every rank holds every expert (the H200 deployment).
    if args["num_local_experts"] != args["num_experts"]:
        raise ValueError("only TP without expert parallelism is measured (local == global)")
    if len(args["per_expert_batches"]) != args["num_experts"]:
        raise ValueError("per_expert_batches must contain one count per global expert")
    exact_topk_ids(
        num_tokens=args["num_tokens"],
        top_k=args["top_k"],
        per_expert_batches=args["per_expert_batches"],
    )
    return args


def _random_fp8(torch: Any, shape: tuple[int, ...], generator: Any) -> Any:
    values = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    return values.to(torch.float8_e4m3fn)


def _random_scales(
    torch: Any, shape: tuple[int, ...], bounds: tuple[float, float], generator: Any
) -> Any:
    low, high = bounds
    values = torch.rand(shape, generator=generator, device="cuda", dtype=torch.float32)
    return low + (high - low) * values


def prepare_weights_for_cutlass(
    swap_w13_to_w31: Any, w13: Any, w13_scale: Any, w2: Any, w2_scale: Any
) -> dict[str, Any]:
    """vLLM's ``prepare_fp8_moe_layer_for_fi`` for the CUTLASS (not TRT-LLM) path.

    Checkpoint ``w13 = [gate; up]`` becomes ``[up; gate]`` for weights and block
    scales alike; no BlockMajorK shuffle; scales clamped to 1e-10.
    """

    return {
        "fc1_expert_weights": swap_w13_to_w31(w13).contiguous(),
        "fc1_scales": swap_w13_to_w31(w13_scale).clamp(min=MIN_BLOCK_SCALE).contiguous(),
        "fc2_expert_weights": w2.contiguous(),
        "fc2_scales": w2_scale.clamp(min=MIN_BLOCK_SCALE).contiguous(),
    }


def _prepared_weights(
    torch: Any,
    swap_w13_to_w31: Any,
    *,
    experts: int,
    hidden_size: int,
    intermediate_size: int,
    seed: int,
    keep_checkpoint: bool,
) -> dict[str, Any]:
    key = (torch.cuda.current_device(), experts, hidden_size, intermediate_size, seed)
    cached = _PREPARED_WEIGHT_CACHE.get(key)
    if cached is not None and (not keep_checkpoint or "w13" in cached):
        return cached

    generator = torch.Generator(device="cuda").manual_seed(seed)
    w13 = torch.empty(
        (experts, 2 * intermediate_size, hidden_size), dtype=torch.float8_e4m3fn, device="cuda"
    )
    w2 = torch.empty(
        (experts, hidden_size, intermediate_size), dtype=torch.float8_e4m3fn, device="cuda"
    )
    for start in range(0, experts, _WEIGHT_CHUNK):
        stop = min(start + _WEIGHT_CHUNK, experts)
        w13[start:stop] = _random_fp8(torch, tuple(w13[start:stop].shape), generator)
        w2[start:stop] = _random_fp8(torch, tuple(w2[start:stop].shape), generator)
    w13_scale = _random_scales(
        torch,
        (experts, 2 * intermediate_size // GROUP_SIZE, hidden_size // GROUP_SIZE),
        _W13_SCALE,
        generator,
    )
    w2_scale = _random_scales(
        torch,
        (experts, hidden_size // GROUP_SIZE, intermediate_size // GROUP_SIZE),
        _W2_SCALE,
        generator,
    )
    prepared = prepare_weights_for_cutlass(swap_w13_to_w31, w13, w13_scale, w2, w2_scale)
    if keep_checkpoint:
        prepared.update(w13=w13, w13_scale=w13_scale, w2=w2, w2_scale=w2_scale)
    else:
        del w13
    _PREPARED_WEIGHT_CACHE.clear()
    _PREPARED_WEIGHT_CACHE[key] = prepared
    return prepared


def cutlass_call_kwargs(
    torch: Any,
    activation_type: Any,
    weights: dict[str, Any],
    *,
    hidden: Any,
    topk_ids: Any,
    topk_weights: Any,
    num_local_experts: int,
) -> dict[str, Any]:
    """Keyword arguments of ``FlashInferExperts.apply``'s block-scale call."""

    return {
        "input": hidden,
        "token_selected_experts": topk_ids.to(torch.int),
        "token_final_scales": topk_weights,
        "fc1_expert_weights": weights["fc1_expert_weights"],
        "fc2_expert_weights": weights["fc2_expert_weights"],
        "fc1_expert_biases": None,
        "fc2_expert_biases": None,
        "swiglu_alpha": None,
        "swiglu_beta": None,
        "swiglu_limit": torch.full(
            (num_local_experts,), SWIGLU_LIMIT, dtype=torch.float32, device=hidden.device
        ),
        "output": torch.empty_like(hidden),
        "output_dtype": torch.bfloat16,
        "quant_scales": [weights["fc1_scales"], weights["fc2_scales"]],
        "input_sf": None,
        "tp_size": 1,
        "tp_rank": 0,
        "ep_size": 1,
        "ep_rank": 0,
        "activation_type": activation_type,
        "use_deepseek_fp8_block_scale": True,
        "use_mxfp8_act_scaling": False,
        "use_w4_group_scaling": False,
    }


def _build_case(
    args: dict[str, Any], *, seed: int = 0, keep_checkpoint: bool = False
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Return ``(torch, callable, call_kwargs, weights)`` for one validated spec."""

    torch, swap_w13_to_w31, fused_moe, activation_type = _load_runtime()
    weights = _prepared_weights(
        torch,
        swap_w13_to_w31,
        experts=args["num_local_experts"],
        hidden_size=args["hidden_size"],
        intermediate_size=args["intermediate_size"],
        seed=seed,
        keep_checkpoint=keep_checkpoint,
    )
    generator = torch.Generator(device="cuda").manual_seed(seed + 1)
    hidden = torch.randn(
        (args["num_tokens"], args["hidden_size"]),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    ids = torch.tensor(
        exact_topk_ids(
            num_tokens=args["num_tokens"],
            top_k=args["top_k"],
            per_expert_batches=args["per_expert_batches"],
        ),
        dtype=torch.int32,
        device="cuda",
    )
    topk_weights = torch.full(
        tuple(ids.shape), 1.0 / args["top_k"], dtype=torch.float32, device="cuda"
    )
    call_kwargs = cutlass_call_kwargs(
        torch,
        activation_type,
        weights,
        hidden=hidden,
        topk_ids=ids,
        topk_weights=topk_weights,
        num_local_experts=args["num_local_experts"],
    )
    return torch, fused_moe, call_kwargs, weights


def _logical_bytes(args: dict[str, Any]) -> int:
    """Useful algorithmic traffic of one rank's call.

    BF16 hidden states read once per token, FP8 weights and their FP32 128x128
    block scales only for experts with a row, int32 ids + FP32 weights of the
    precomputed routing, and one BF16 output row per token (the call finalizes).
    """

    batches = args["per_expert_batches"][: args["num_local_experts"]]
    active_experts = sum(batch > 0 for batch in batches)
    tokens, hidden = args["num_tokens"], args["hidden_size"]
    intermediate = args["intermediate_size"]
    blocks = GROUP_SIZE * GROUP_SIZE
    routing = 8 * tokens * args["top_k"]
    activations = 2 * tokens * hidden
    weights = active_experts * (3 * intermediate * hidden + 4 * 3 * intermediate * hidden // blocks)
    output = 2 * tokens * hidden
    return routing + activations + weights + output


def profile_fp8_block_fused_moe_cutlass_sm90(
    *,
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    num_local_experts: int,
    top_k: int,
    input_dtype: DType | str,
    weight_format: DType | str,
    group_size: int,
    routing_method: str,
    n_group: int,
    topk_group: int,
    routed_scaling_numerator: int,
    routed_scaling_denominator: int,
    per_expert_batches: tuple[int, ...],
) -> ComputeMetrics:
    args = _validate_args(**dict(locals()))
    try:
        from flashinfer.autotuner import autotune

        torch, fused_moe, call_kwargs, _weights = _build_case(args)

        def run_once() -> None:
            fused_moe(**call_kwargs)

        # vLLM's kernel warmup autotunes FlashInfer ops before serving; the
        # first call also NVRTC-compiles (or loads) the grouped-GEMM recipe.
        with autotune_cached(autotune, "nvfp4_fused_moe.fp8_block_cutlass_sm90"):
            run_once()
        run_once()
        torch.cuda.synchronize()
        time_ms = Timer.cupti(run_once, warmup=3, interval_union=True)
        energy_j = Energy.perf(run_once, per_iter_time_ms=time_ms)
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError("SM90 CUTLASS FP8 block-scale fused MoE ran out of memory") from exc
        raise KernelLaunchFailed(f"SM90 CUTLASS FP8 block-scale fused MoE failed: {exc}") from exc

    local_rows = sum(args["per_expert_batches"][: args["num_local_experts"]])
    h, i = args["hidden_size"], args["intermediate_size"]
    flops = 2 * local_rows * (h * 2 * i + i * h)
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=time_ms,
        tflops=flops / elapsed_s / 1e12,
        memory_bandwidth_gbps=_logical_bytes(args) / elapsed_s / 1e9,
        energy_j=energy_j,
    )


def precomputed_moe_reference(
    torch: Any,
    *,
    hidden: Any,
    w13: Any,
    w13_scale: Any,
    w2: Any,
    w2_scale: Any,
    topk_ids: Any,
    topk_weights: Any,
    clamp_limit: float | None,
    quantize_activations: bool = True,
) -> Any:
    """FP32 ``[T, H]`` routed-expert output for precomputed top-k (checkpoint order).

    Block-dequantized weights; with ``quantize_activations`` the hidden states and
    the intermediate are round-tripped through per-token-group-128 FP8, as both
    vLLM recipes (Triton and CUTLASS) do.
    """

    from profiling.runners.moe.fp8_block_fused_moe_reference import (
        dequantize_blocks,
        quantize_token_groups,
    )

    x = hidden.float()
    if quantize_activations:
        x = quantize_token_groups(torch, x)
    intermediate = w13.shape[1] // 2
    output = torch.zeros_like(x)
    for expert in torch.unique(topk_ids).tolist():
        rows, slots = torch.nonzero(topk_ids == expert, as_tuple=True)
        gate_up = x[rows] @ dequantize_blocks(torch, w13[expert], w13_scale[expert]).t()
        gate, up = gate_up[:, :intermediate], gate_up[:, intermediate:]
        if clamp_limit is not None:
            gate = torch.clamp(gate, max=clamp_limit)
            up = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
        act = torch.nn.functional.silu(gate) * up
        if quantize_activations:
            act = quantize_token_groups(torch, act)
        down = act @ dequantize_blocks(torch, w2[expert], w2_scale[expert]).t()
        output.index_add_(0, rows, down * topk_weights[rows, slots, None].float())
    return output


def compare_with_reference(*, seed: int = 0, **spec: Any) -> dict[str, float]:
    """Run the production call once and compare it with the Torch reference."""

    args = _validate_args(**spec)
    torch, fused_moe, call_kwargs, weights = _build_case(args, seed=seed, keep_checkpoint=True)
    fused_moe(**call_kwargs)
    actual = call_kwargs["output"].float()
    torch.cuda.synchronize()
    expected = precomputed_moe_reference(
        torch,
        hidden=call_kwargs["input"],
        w13=weights["w13"],
        w13_scale=weights["w13_scale"],
        w2=weights["w2"],
        w2_scale=weights["w2_scale"],
        topk_ids=call_kwargs["token_selected_experts"],
        topk_weights=call_kwargs["token_final_scales"],
        clamp_limit=SWIGLU_LIMIT,
    )
    rel = float((actual - expected).norm() / expected.norm().clamp(min=1e-30))
    return {
        "rel_l2": rel,
        "max_abs": float((actual - expected).abs().max()),
        "ref_abs_max": float(expected.abs().max()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
        ),
        "finite": bool(torch.isfinite(actual).all()),
    }


__all__ = [
    "compare_with_reference",
    "cutlass_call_kwargs",
    "precomputed_moe_reference",
    "prepare_weights_for_cutlass",
    "profile_fp8_block_fused_moe_cutlass_sm90",
]
