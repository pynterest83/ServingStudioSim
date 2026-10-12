"""Behavioral tests for the ``vllm_fused_moe:vllm_triton`` runner (no GPU)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from profiling.kernels.vllm_fused_moe import LAUNCH_ROLE_DOWN, LAUNCH_ROLE_GATE_UP
from profiling.runners.moe import vllm_fused_moe_triton as runner

# GLM-5.3-Flash at TP4: 288 experts, top-8, H = 4096, per-shard I = 512.
_EXPERTS, _TOP_K, _HIDDEN, _INTERMEDIATE = 288, 8, 4096, 512


def _args(launch_role: str, num_tokens: int = 96) -> runner._ValidatedArgs:
    batches = [0] * _EXPERTS
    for slot in range(num_tokens * _TOP_K):
        batches[slot % _EXPERTS] += 1
    n, k = (
        (2 * _INTERMEDIATE, _HIDDEN)
        if launch_role == LAUNCH_ROLE_GATE_UP
        else (_HIDDEN, _INTERMEDIATE)
    )
    return runner._validate_args(
        n=n,
        k=k,
        dtype="fp8_e4m3",
        num_local_experts=_EXPERTS,
        num_tokens=num_tokens,
        experts_per_token=_TOP_K,
        launch_role=launch_role,
        block_size=128,
        per_group_batches=batches,
    )


def test_both_launches_look_up_the_layer_config_keyed_by_intermediate_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches keying vLLM's tuned-config lookup on one launch's own (n, k).

    Production ``fused_experts`` resolves one config per layer from the real
    ``w1 = (E, 2I, H)`` / ``w2 = (E, H, I)`` and vLLM names the tuned file by
    ``N = w2.shape[2] = I``. A per-launch shape gives N = 2I (w13) or N = H
    (w2), which names a missing file and silently measures the default config.
    """
    calls: list[tuple[Any, ...]] = []
    tuned = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256}

    def record_config(w1_shape, w2_shape, top_k, dtype, M, block_shape=None):
        calls.append((tuple(w1_shape), tuple(w2_shape), top_k, dtype, M, block_shape))
        return tuned

    def prepare_assignment(topk_ids, config, *rest, **kwargs):
        return (torch.empty(0), torch.empty(0), torch.empty(1))

    triton_language = SimpleNamespace(bfloat16="bf16")
    monkeypatch.setattr(
        runner,
        "_load_vllm",
        lambda: (lambda *a, **k: None, record_config, prepare_assignment, triton_language),
    )

    launches = [
        runner._build_launch(torch, _args(role), device="cpu")
        for role in (LAUNCH_ROLE_GATE_UP, LAUNCH_ROLE_DOWN)
    ]

    layer_w1 = (_EXPERTS, 2 * _INTERMEDIATE, _HIDDEN)
    layer_w2 = (_EXPERTS, _HIDDEN, _INTERMEDIATE)
    expected = (layer_w1, layer_w2, _TOP_K, "fp8_w8a8", 96, [128, 128])
    assert calls == [expected, expected]
    assert all(call[1][2] == _INTERMEDIATE for call in calls)
    # The resolved config is the one the kernel is launched with.
    assert all(launch.positional[-1] is tuned for launch in launches)


# ── the tuned-config backend ─────────────────────────────────────────────────


def test_the_tuned_folder_holds_the_file_vllm_looks_up_with_the_recorded_bytes() -> None:
    """Catches a tuned folder vLLM would silently ignore or read other bytes from.

    ``get_moe_configs`` opens exactly ``E=<E>,N=<I>,device_name=<gpu>,dtype=fp8_w8a8,
    block_shape=[128,128].json``; any other name falls back to the default
    config and the backend's rows would measure the default kernel under a
    tuned name. The sha256 ties the rows to the config the Tock measured.
    """
    import hashlib

    from profiling.kernels import vllm_fused_moe as kernel

    folder = kernel.TUNED_212D200C_CONFIG_DIR
    expected_name = (
        f"E={_EXPERTS},N={_INTERMEDIATE},device_name=NVIDIA_H200,dtype=fp8_w8a8,"
        "block_shape=[128,128].json"
    )
    assert sorted(path.name for path in folder.glob("*.json")) == [expected_name]
    digest = hashlib.sha256((folder / expected_name).read_bytes()).hexdigest()
    assert digest == kernel.TUNED_212D200C_SHA256
    assert (folder / "sha256.txt").read_text().split() == [digest, expected_name]


def test_only_the_tuned_backend_points_vllm_at_the_tuned_folder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches the tuned config leaking into the default rows, or the tuned rows
    being measured without it (the runner itself never sets the variable)."""
    from profiling.db.registry import find_kernel_profiler_spec
    from profiling.kernels import vllm_fused_moe as kernel

    default = find_kernel_profiler_spec(kernel.KIND, "vllm_triton")
    tuned = find_kernel_profiler_spec(kernel.KIND, kernel.TUNED_212D200C_BACKEND)
    assert "VLLM_TUNED_CONFIG_FOLDER" not in dict(default.worker_env)
    assert dict(tuned.worker_env)["VLLM_TUNED_CONFIG_FOLDER"] == kernel._worker_config_dir(
        kernel.TUNED_212D200C_CONFIG_DIR
    )
    # Same launch, same table and cache identity: only the environment differs.
    assert (tuned.runner_ref, tuned.table_name, tuned.args_schema) == (
        default.runner_ref,
        default.table_name,
        default.args_schema,
    )

    # The folder is spelled as the worker sees it: the host path for a host
    # worker, the image's /opt/vibesim mount otherwise.
    monkeypatch.setenv("VIBESIM_VLLM_PROFILE_ENV", "host")
    assert kernel._worker_config_dir(kernel.TUNED_212D200C_CONFIG_DIR) == str(
        kernel.TUNED_212D200C_CONFIG_DIR
    )
    monkeypatch.setenv("VIBESIM_VLLM_PROFILE_ENV", "container")
    assert kernel._worker_config_dir(kernel.TUNED_212D200C_CONFIG_DIR) == (
        "/opt/vibesim/profiling/configs/vllm_fused_moe/tuned_212d200c"
    )
