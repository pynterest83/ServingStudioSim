"""Turn one (case, variant, host) into a runnable alignment case directory.

The output is exactly what the five existing phase commands already read — this
module writes configs, it does not introduce a new runtime. Rendering replaces
the two 400-500 line f-string generators the two GLM batches each grew their own
copy of; configs are built as dicts and dumped, so a malformed nesting is a
`yaml.safe_dump` of a wrong tree rather than broken indentation.

## Where each rendered value comes from

| rendered field                       | source                                  |
|--------------------------------------|-----------------------------------------|
| `server.model_path` / `tokenizer`    | host `checkpoints[variant.checkpoint]`  |
| `workload.text_file`                 | host `text_corpus`, or the host's       |
|                                      | `text_corpora[case.text_corpus.host_key]`|
| `nsys.executable`                    | host `nsys_executable`                  |
| `fork_python`                        | host `fork_python` (omitted when empty) |
| `cuda_visible_devices`               | host `device_roles[case.device_role]`   |
| `server.port` / `startup_timeout`    | host                                    |
| `server.gpu_memory_utilization`      | case (it decides KV capacity)           |
| `server.extra_args`                  | variant + `--max-model-len` from case   |
| `--speculative-config` `model`       | host `checkpoints[variant.draft_checkpoint]` |
| `arch.max_model_len`                 | case                                    |
| `worker.attn_gpu_memory_gb`          | case (calibrated)                       |
| `worker.gpu_time_multiplier`         | always 1.0; kernel-align only reports a |
|                                      | recommendation, nothing injects it      |
| `worker.prefill_gpu_time_multiplier` | variant, verbatim (a pack-declared      |
|                                      | calibration, never the recommendation)  |
| `workload.arrival_mode` / `rate`     | case                                    |
| `workload.frontend`                  | `independent` over the shape trace, or  |
|                                      | `session` over `case.session_trace`     |

## Session-trace cases

A case with `session_trace` replays a committed `session-execution-v2` CSV
instead of a shape recipe. `trace.csv` is that file byte for byte, driven by
req-frontend's session frontend on the measured side and read with
`input_file_format: text-generation-session-execution-v2` by the simulator.
A speculative case adds `trace_speculative.csv`: the same rows plus the
`accept_rate` column of the `speculative` tag, which req-frontend's session
loader accepts and the simulator turns into each round's decoding strategy,
one chain per round from `acceptance_chain` (its `output_len`).

`rate` keeps req-frontend's meaning, a target top-level arrival rate (sessions
per second here). req-frontend rescales the trace by `trace_rate / rate`, the
trace rate being (sessions - 1) / span of the session start times; the
simulator divides arrival times by `request_rate`. The simulation therefore gets
`request_rate = rate / trace_rate`, which equals `rate` for the 1 req/s shape
traces.

## Two normalizations that are not cosmetic

`analyze_e2e.yaml` is rendered **without** `profile_log_dir`. Every accepted
GLM case carries it, and `load_analyze_config` now rejects it on a workload/e2e
phase ("profile_log_dir does not belong to workload/e2e alignment"). Those
configs predate that rule, which is also why cases 01-10 have
`throughput.measured_server_gpu_*` in their e2e report and later ones do not. A
re-run from this pack produces the current, narrower shape.

Host paths are rendered **absolute**. The accepted configs reach the engine venv
with `../../../../alignment/...`, which only works at one directory depth; a
campaign that may render anywhere cannot rely on that. `fork_python` is anchored
without resolving its final symlink, matching what `load_profile_config` does,
so the venv's `site-packages` discovery is unchanged.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from alignment.load_generator.config import (
    IndependentFrontendConfig,
    SessionFrontendConfig,
    routes_backend,
)

from ..corpus import ROUTING_FILES
from .pack import (
    Case,
    HostProfile,
    Pack,
    PackError,
    TraceSpec,
    Variant,
    load_per_request_chains,
    read_session_trace,
)

#: Trace file names inside a rendered case directory. Fixed rather than derived
#: from the pack's file names so a run directory reads the same for every pack.
WORKLOAD_TRACE_NAME = "trace.csv"
SPECULATIVE_TRACE_NAME = "trace_speculative.csv"
KERNEL_TRACE_NAME = "trace_nsys.csv"
#: The session trace's `tracegen` manifest, copied beside `trace.csv` so a run
#: directory carries the provenance of the bytes it replayed.
SESSION_MANIFEST_NAME = "trace.manifest.json"

#: The four phases that follow the variant's profile passes. `--phase` values are
#: artifact directory names throughout; only these two analyze rows have a config
#: stem that differs from their directory, which is pre-existing naming in
#: `alignment/README.md` and so is spelled out rather than left implicit.
TIMING_PREDICT_PHASE = "timing_predict"
ANALYSIS_KERNEL_PHASE = "analysis_kernel"
SIMULATION_PHASE = "simulation"
ANALYSIS_E2E_PHASE = "analysis_e2e"

PHASE_CONFIG_STEMS = {
    TIMING_PREDICT_PHASE: "timing_predict",
    ANALYSIS_KERNEL_PHASE: "analyze_kernel",
    SIMULATION_PHASE: "simulation",
    ANALYSIS_E2E_PHASE: "analyze_e2e",
}

#: Engine CLI spelling for the per-case context limit.
#:
#: Most server fields use the launcher's vLLM-shaped ``ServerConfig`` and are
#: translated by each driver. This value is derived by the campaign and appended
#: directly to ``extra_args``, so it must already use the target engine's CLI.
CONTEXT_LIMIT_FLAG = {
    "vllm": "--max-model-len",
    "sglang": "--context-length",
}

#: Fixed pipeline order after the profile passes. Used for `--phase` validation
#: and for the readiness graph; it is never used to chain phases automatically.
PIPELINE_PHASES = (
    TIMING_PREDICT_PHASE,
    ANALYSIS_KERNEL_PHASE,
    SIMULATION_PHASE,
    ANALYSIS_E2E_PHASE,
)


@dataclass(frozen=True)
class RenderedCase:
    """Where one case's run directory is and what phases it holds."""

    case_slug: str
    directory: Path
    phases: tuple[str, ...]
    files: tuple[Path, ...]


def phase_names(variant: Variant) -> tuple[str, ...]:
    """Every `--phase` value for this variant, in pipeline order."""
    return tuple(item.name for item in variant.profile_passes) + PIPELINE_PHASES


def config_stem(variant: Variant, phase: str) -> str:
    """Config file stem for a phase. A profile pass names its own config; the
    fixed tail uses the table above (`analysis_kernel` reads `analyze_kernel`)."""
    if variant.pass_named(phase) is not None:
        return phase
    try:
        return PHASE_CONFIG_STEMS[phase]
    except KeyError:
        raise PackError(
            f"unknown phase {phase!r}; this variant offers {list(phase_names(variant))}"
        ) from None


# ── traces ───────────────────────────────────────────────────────────────────

def acceptance_chain(case: Case, output_len: int) -> list[float]:
    """The conditional acceptance chain one request of `output_len` tokens gets.

    Without output-position buckets every request gets the case chain. With
    them, bucket b covers decode positions [from_b, from_b+1) of the request's
    [1, output_len); the request spends n_b = tokens_b / E_b verify rounds
    there, E_b = sum_a (a + 1) p_b(a). Its per-round distribution is
    q = sum_b n_b p_b / sum_b n_b, and position k's conditional acceptance is
    P(a >= k + 1) / P(a >= k) under q.
    """
    assert case.speculative_acceptance is not None
    if case.speculative_acceptance_by_output_position is None:
        return list(case.speculative_acceptance.value)
    buckets = case.speculative_acceptance_by_output_position.value
    depth = len(buckets[0]["p"]) - 1
    mixed = [0.0] * (depth + 1)
    total_rounds = 0.0
    for index, bucket in enumerate(buckets):
        end = buckets[index + 1]["from"] if index + 1 < len(buckets) else output_len
        tokens = max(0, min(output_len, end) - max(1, bucket["from"]))
        if tokens == 0:
            continue
        dist = bucket["p"]
        rounds = tokens / sum((accepted + 1) * dist[accepted] for accepted in range(depth + 1))
        total_rounds += rounds
        for accepted in range(depth + 1):
            mixed[accepted] += rounds * dist[accepted]
    if total_rounds == 0:
        # One output token: no decode round, so the chain is never consulted.
        mixed, total_rounds = list(buckets[0]["p"]), 1.0
    tail = [sum(mixed[k:]) / total_rounds for k in range(depth + 2)]
    return [tail[k + 1] / tail[k] if tail[k] > 0 else 0.0 for k in range(depth)]


def _trace_hash(request_id: str, seed: int = 0) -> int:
    key = request_id if seed == 0 else f"{seed}:{request_id}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def _chain_pool(
    entries: tuple[tuple[int, int, tuple[float, ...]], ...], input_len: int, output_len: int
) -> list[tuple[float, ...]]:
    """Measured chains a request of this shape draws from: the same (input,
    output) shape, else the same output length, else every chain."""
    pool = [chain for i, o, chain in entries if (i, o) == (input_len, output_len)]
    pool = pool or [chain for _, o, chain in entries if o == output_len]
    return pool or [chain for _, _, chain in entries]


def per_request_chains(
    entries: tuple[tuple[int, int, tuple[float, ...]], ...],
    requests: list[tuple[str, int, int]],
    seed: int = 0,
) -> dict[str, list[float]]:
    """Trace request id -> its measured chain. Within one shape, trace requests
    ordered by a fixed hash (sha256) of their trace id take that shape's chains
    in turn, so each measured chain is used once per pass over the pool (the
    measured per-request distribution, not a resample of it) -- and the pairing
    never follows the measured request id. Acceptance depends on the request
    (its length among others), hence the per-shape pools. Deterministic across
    runs and machines; `seed` picks another hash order (0 = the rendered trace)
    so stochastic scoring can sample the pairing."""
    by_shape: dict[tuple[int, int], list[str]] = {}
    for request_id, input_len, output_len in requests:
        by_shape.setdefault((input_len, output_len), []).append(request_id)
    assigned: dict[str, list[float]] = {}
    for (input_len, output_len), ids in by_shape.items():
        pool = _chain_pool(entries, input_len, output_len)
        start = _trace_hash(f"{input_len}x{output_len}", seed) % len(pool)
        for rank, request_id in enumerate(sorted(ids, key=lambda i: _trace_hash(i, seed))):
            assigned[request_id] = list(pool[(start + rank) % len(pool)])
    return assigned


def trace_text(
    case: Case,
    spec: TraceSpec,
    *,
    speculative_acceptance: Callable[[str, int, int], list[float]] | None = None,
) -> str:
    """Regenerate one trace from its shapes.

    Request ids carry the case slug so two cases replayed into one server are
    never confused, and arrival times are a deterministic 1 req/s baseline that
    `saturated` discards and `trace-timed` rescales by `rate`.
    `speculative_acceptance` maps a request's trace id, input and output length to its chain.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    columns = ["id", "input_len", "output_len", "arrival_time"]
    if speculative_acceptance is not None:
        columns.append("accept_rate")
    writer.writerow(columns)
    rows = [shape for _ in range(spec.repeats) for shape in spec.shapes]
    prefix = case.slug if spec.id_suffix is None else f"{case.slug}-{spec.id_suffix}"
    for index, (input_len, output_len) in enumerate(rows):
        if input_len + output_len >= case.max_model_len:
            raise PackError(
                f"{case.slug} trace row {index} reaches max_model_len: "
                f"{input_len}+{output_len} >= {case.max_model_len}"
            )
        request_id = f"{prefix}-{index:04d}"
        row = [request_id, input_len, output_len, float(index * 1000)]
        if speculative_acceptance is not None:
            row.append(
                json.dumps(
                    speculative_acceptance(request_id, input_len, output_len),
                    separators=(",", ":"),
                )
            )
        writer.writerow(row)
    return buffer.getvalue()


def session_rows(text: str) -> list[list[str]]:
    """A session trace's rows as the strings in the file (header excluded)."""
    return list(csv.reader(io.StringIO(text)))[1:]


def session_arrival_rate(text: str) -> float:
    """Sessions per second of a session trace, as req-frontend computes it:
    (sessions - 1) over the span of each session's first-round arrival."""
    first_arrival: dict[str, float] = {}
    for row in session_rows(text):
        first_arrival.setdefault(row[1], float(row[3]))
    arrivals = list(first_arrival.values())
    span_seconds = (max(arrivals, default=0.0) - min(arrivals, default=0.0)) / 1000.0
    if len(arrivals) < 2 or span_seconds <= 0:
        raise PackError(
            "a trace-timed session trace needs at least two sessions with distinct start times"
        )
    return (len(arrivals) - 1) / span_seconds


def session_trace_with_acceptance(case: Case, text: str) -> str:
    """The session trace plus the `speculative` tag's `accept_rate` column,
    one chain per round from its `output_len`. Every other field keeps the
    file's own spelling."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    writer.writerow([*header, "accept_rate"])
    output_column = header.index("output_len")
    for row in reader:
        chain = acceptance_chain(case, int(row[output_column]))
        writer.writerow([*row, json.dumps(chain, separators=(",", ":"))])
    return buffer.getvalue()


def _check_session_lengths(case: Case, text: str) -> None:
    """The shape traces' context rule, applied to a session round's whole
    prompt (declared prefix plus fresh input) and its output."""
    for index, row in enumerate(session_rows(text)):
        total = int(row[4]) + int(row[5]) + int(row[6])
        if total >= case.max_model_len:
            raise PackError(
                f"{case.slug} session trace row {index} ({row[0]}) reaches max_model_len: "
                f"{row[4]}+{row[5]}+{row[6]} >= {case.max_model_len}"
            )


# ── config bodies ────────────────────────────────────────────────────────────

SPECULATIVE_CONFIG_FLAG = "--speculative-config"


def _with_draft_checkpoint(variant: Variant, host: HostProfile, args: list[str]) -> list[str]:
    """Point the variant's `--speculative-config` at its proposer checkpoint.

    The pack names the proposer by host checkpoint key, never by path, and the
    config it writes must leave `model` to this function -- a path there would
    be one machine's HF cache committed into a pack.
    """
    if variant.draft_checkpoint is None:
        return args
    where = f"variants.{variant.name}"
    positions = [
        index
        for index, argument in enumerate(args)
        if argument.partition("=")[0].replace("_", "-") == SPECULATIVE_CONFIG_FLAG
    ]
    if len(positions) != 1:
        raise PackError(
            f"{where}.draft_checkpoint needs exactly one {SPECULATIVE_CONFIG_FLAG} "
            f"in server.extra_args, found {len(positions)}"
        )
    index = positions[0]
    inline = "=" in args[index]
    if not inline and index + 1 >= len(args):
        raise PackError(f"{where}.server.extra_args: {SPECULATIVE_CONFIG_FLAG} has no value")
    raw = args[index].partition("=")[2] if inline else args[index + 1]
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as error:
        raise PackError(f"{where}: {SPECULATIVE_CONFIG_FLAG} is not JSON: {error}") from None
    if not isinstance(config, dict) or "model" in config:
        raise PackError(
            f"{where}: {SPECULATIVE_CONFIG_FLAG} must be a JSON object without `model` "
            f"when draft_checkpoint is set; rendering supplies the path"
        )
    config["model"] = host.checkpoint_path(variant.draft_checkpoint)
    rendered = json.dumps(config, separators=(",", ":"))
    result = list(args)
    if inline:
        result[index] = f"{args[index].partition('=')[0]}={rendered}"
    else:
        result[index + 1] = rendered
    return result


def _extra_args(variant: Variant, case: Case) -> list[str]:
    """Variant flags plus the context limit in the target engine's spelling.

    Both known spellings are reserved, including ``--flag=value`` and vLLM's
    underscore aliases, so a variant cannot introduce a second static limit
    beside the per-case value.
    """
    try:
        flag = CONTEXT_LIMIT_FLAG[variant.engine]
    except KeyError:
        raise PackError(
            f"variants.{variant.name}.engine {variant.engine!r} has no known context-limit "
            f"flag; known engines: {sorted(CONTEXT_LIMIT_FLAG)}"
        ) from None
    args = [str(item) for item in variant.server.get("extra_args", [])]
    for argument in args:
        argument_name = argument.partition("=")[0].replace("_", "-")
        if argument_name in CONTEXT_LIMIT_FLAG.values():
            raise PackError(
                f"variants.{variant.name}.server.extra_args must not set {argument_name}; "
                f"the context limit is derived from each case's max_model_len and "
                f"rendered as {flag} for engine {variant.engine!r}"
            )
    return args + [flag, str(case.max_model_len)]


def _server_block(pack: Pack, variant: Variant, case: Case, host: HostProfile) -> dict[str, Any]:
    body = {
        "model_path": host.checkpoint_path(variant.checkpoint),
        "port": host.port,
    }
    for key, value in variant.server.items():
        if key == "extra_args":
            continue
        body[key] = value
    body["gpu_memory_utilization"] = case.gpu_memory_utilization
    if case.chunk_size is not None:
        body["chunk_size"] = case.chunk_size
    body["startup_timeout"] = host.startup_timeout
    body["extra_args"] = _with_draft_checkpoint(variant, host, _extra_args(variant, case))
    return body


def _workload_block(
    variant: Variant, case: Case, host: HostProfile, trace_name: str, profile_kind: str
) -> dict[str, Any]:
    backend = variant.backend
    if profile_kind == "token_corpus":
        backend = routes_backend(backend)
    frontend = (
        IndependentFrontendConfig if case.session_trace is None else SessionFrontendConfig
    )
    corpus_key = None if case.text_corpus is None else case.text_corpus.host_key
    body: dict[str, Any] = {
        "frontend": {"type": frontend.type, "path": f"./{trace_name}"},
        "backend": {"type": backend},
        "text_file": host.text_corpus_path(corpus_key),
        "tokenizer": host.checkpoint_path(variant.tokenizer),
        # A session trace draws its fresh prompt tokens from the pool across many long sessions (the agentic sample
        # needs ~12M); a 1M pool would wrap and hand two sessions the same prompt prefix, a fabricated cross-session
        # cache hit. The shape traces keep their original 1M pool.
        "token_pool_limit": 1_000_000 if case.session_trace is None else 100_000_000,
        "arrival_mode": case.profile_arrival_mode,
    }
    if case.max_concurrency is not None:
        body["max_concurrency"] = case.max_concurrency
    if case.rate is not None:
        body["rate"] = case.rate.value
    body["max_model_len"] = case.max_model_len
    return body


def profile_document(
    pack: Pack,
    case: Case,
    variant: Variant,
    host: HostProfile,
    pass_name: str,
    repo_root: Path,
) -> dict[str, Any]:
    """One `alignment profile` config — the `<pass.name>.yaml` of a case."""
    profile_pass = variant.pass_named(pass_name)
    if profile_pass is None:
        raise PackError(f"variants.{variant.name} declares no profile pass {pass_name!r}")
    short = pass_name.removeprefix("profile_")
    prefix = variant.run_name_prefix or pack.name
    trace_name = (
        KERNEL_TRACE_NAME
        if profile_pass.trace == "kernel" and case.kernel_trace is not None
        else WORKLOAD_TRACE_NAME
    )
    document: dict[str, Any] = {
        "schema_version": 1,
        "name": f"{prefix}_{case.slug}_{short}",
        "log_dir": f"./{pass_name}",
        "gpu": variant.gpu,
        "cuda_visible_devices": host.devices(case.device_role),
        "profile_kind": profile_pass.kind,
        "engine": variant.engine,
    }
    if host.fork_python:
        document["fork_python"] = _absolute(host.fork_python, repo_root)
    if variant.python_runtime is not None:
        document["python_runtime"] = dict(variant.python_runtime)
    document["server"] = _server_block(pack, variant, case, host)
    if profile_pass.kind == "nsys":
        nsys: dict[str, Any] = {
            "executable": host.nsys_executable,
            "capture_mode": "cuda_profiler_api",
            "capture_duration_seconds": case.capture_seconds,
            "cuda_graph_trace": "node",
            "cuda_event_trace": False,
        }
        if case.analyze_iterations is not None:
            # Omitted unless the case says otherwise, so the profiler's own
            # default stays the default rather than being restated per case.
            start, end = case.analyze_iterations
            nsys["analyze_iteration_start"] = start
            nsys["analyze_iteration_end"] = end
        document["nsys"] = nsys
    document["workload"] = _workload_block(variant, case, host, trace_name, profile_pass.kind)
    if profile_pass.warmup:
        document["workload"]["warmup"] = True
    return document


def timing_predict_document(variant: Variant) -> dict[str, Any]:
    kernel_pass = _kernel_pass(variant)
    return {
        "schema_version": 1,
        "simulation_preset": f"./{PHASE_CONFIG_STEMS[SIMULATION_PHASE]}.yaml",
        "profile_log_dir": f"./{kernel_pass.name}",
        "log_dir": f"./{TIMING_PREDICT_PHASE}",
        "input_builder": dict(variant.input_builder),
    }


def analyze_kernel_document(variant: Variant) -> dict[str, Any]:
    kernel_pass = _kernel_pass(variant)
    return {
        "schema_version": 1,
        "profile_log_dir": f"./{kernel_pass.name}",
        "timing_predict_log_dir": f"./{TIMING_PREDICT_PHASE}",
        "log_dir": f"./{ANALYSIS_KERNEL_PHASE}",
        "iteration": {
            "enabled": True,
            "labeled_kernel_sequences_file": "./kernel_sequences_labeled.json",
        },
    }


def analyze_e2e_document(variant: Variant) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "simulation_log_dir": f"./{SIMULATION_PHASE}",
        "workload_profile_log_dir": f"./{_workload_pass(variant).name}",
        "log_dir": f"./{ANALYSIS_E2E_PHASE}",
        "workload": {"enabled": True},
        "e2e": {"enabled": True, "throughput_bins": 20},
    }


def simulation_document(
    pack: Pack, case: Case, variant: Variant, case_dir: Path, repo_root: Path
) -> dict[str, Any]:
    """The ordinary ServingStudioSim preset.

    Unlike the four launcher-owned phase configs, a preset's paths are
    repo-root-relative, not config-relative — so this is the one document that
    cannot be written once and moved. Paths are emitted repo-relative when the
    run directory is inside the repository and absolute otherwise.
    """
    # The speculative worker is the contract that consumes per-position
    # acceptance, whichever proposer the arch models, so it is what decides.
    speculative = variant.worker.get("type") == "speculative"
    if speculative != (case.speculative_acceptance is not None):
        raise PackError(f"{case.slug}: speculative worker and acceptance must be configured together")
    if speculative:
        # Stated on both sides rather than defaulted: the two speculative archs
        # default to different depths, and a silent default is how they drift.
        depth = variant.arch.get("draft_tokens")
        if (
            depth is None
            or variant.worker.get("draft_tokens") != depth
            or len(case.speculative_acceptance.value) != depth
            or (
                case.speculative_acceptance_by_output_position is not None
                and any(
                    len(bucket["p"]) != depth + 1
                    for bucket in case.speculative_acceptance_by_output_position.value
                )
            )
        ):
            raise PackError(
                f"{case.slug}: arch and worker must both set draft_tokens, and the "
                f"acceptance vector must have that many positions"
            )
    trace_name = SPECULATIVE_TRACE_NAME if speculative else WORKLOAD_TRACE_NAME
    session_text = None
    if case.session_trace is not None:
        session_text, _ = read_session_trace(pack.root, case)
    frontend = IndependentFrontendConfig if session_text is None else SessionFrontendConfig
    workload: dict[str, Any] = {
        "trace_files": [_preset_path(case_dir / trace_name, repo_root)],
        "input_file_format": frontend.input_file_format,
        "arrival_mode": case.simulation_arrival_mode,
    }
    if speculative:
        workload["input_file_tags"] = ["speculative"]
    if case.max_concurrency is not None:
        workload["max_concurrency"] = case.max_concurrency
    if case.rate is not None:
        # The client's `rate` is absolute; the simulator's rescales a rate-1
        # trace (see "Session-trace cases" above).
        workload["request_rate"] = (
            case.rate.value
            if session_text is None
            else case.rate.value / session_arrival_rate(session_text)
        )
    workload["run_to_end"] = True

    arch = dict(variant.arch)
    if "max_model_len" in arch:
        raise PackError(
            f"variants.{variant.name}.arch must not set max_model_len; it is per-case"
        )
    arch["max_model_len"] = case.max_model_len
    # Both measured-routing artifacts are named pack-relative in a variant and
    # must reach the simulator as paths it can resolve from the repository root.
    # A corpus manifest is resolved relative to the process, so leaving it
    # pack-relative fails the build after the capture has already run.
    for key in ROUTING_FILES.values():
        reference = arch.get(key)
        if isinstance(reference, str) and reference and not reference.startswith("hf://"):
            arch[key] = _preset_path(pack.root / reference, repo_root)

    worker = dict(variant.worker)
    for derived in ("attn_gpu_memory_gb", "gpu_time_multiplier"):
        if derived in worker:
            raise PackError(
                f"variants.{variant.name}.worker must not set {derived}; "
                "attn_gpu_memory_gb is per-case calibrated and gpu_time_multiplier "
                "is fixed at 1.0 (kernel-align reports the measured duty cycle)"
            )
    worker["attn_gpu_memory_gb"] = case.attn_gpu_memory_gb.value
    if case.chunk_size is not None:
        worker["max_batch_tokens"] = case.chunk_size
    # Rendered as the neutral 1.0 on purpose: the E2E checks judge the model
    # without a duty-cycle correction, and kernel-align reports the measured
    # multiplier as its own metric. Nothing reads that report back into the sim.
    worker["gpu_time_multiplier"] = 1.0

    document = {
        "deployment": variant.deployment,
        "workload": workload,
        "io": {"log_dir": _preset_path(case_dir / SIMULATION_PHASE, repo_root)},
        "pools": {
            "main": {
                "groups": [
                    {
                        "gpu": variant.gpu,
                        "replicas": variant.replicas,
                        "arch": arch,
                        "worker": worker,
                    }
                ]
            }
        },
    }
    for key, value in variant.raw_overrides.items():
        document[key] = value
    return document


def _kernel_pass(variant: Variant):
    for item in variant.profile_passes:
        if item.kind == "nsys":
            return item
    raise PackError(
        f"variants.{variant.name} has no nsys profile pass; kernel alignment needs one"
    )


def _workload_pass(variant: Variant):
    for item in variant.profile_passes:
        if item.kind == "workload_metrics":
            return item
    raise PackError(
        f"variants.{variant.name} has no workload_metrics profile pass; "
        "workload/e2e alignment needs one"
    )


def _absolute(value: str, base: Path) -> str:
    """Anchor a host path against the repository, without resolving its final
    symlink — a venv's `bin/python` must keep its lexical path or
    `site-packages` discovery moves. A relative host path means "inside this
    repository", which is how `fork_python` reaches the engine venv."""
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base / candidate
    return os.path.abspath(os.path.normpath(candidate))


def _preset_path(path: Path, repo_root: Path) -> str:
    path = Path(os.path.abspath(path))
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


# ── writing ──────────────────────────────────────────────────────────────────

def case_documents(
    pack: Pack, case: Case, host: HostProfile, case_dir: Path, repo_root: Path
) -> dict[str, dict[str, Any]]:
    """Every phase config for one case, keyed by file name. Pure — the CPU tier
    renders into a temp dir and the migration diff renders into memory."""
    variant = pack.variant_of(case)
    documents: dict[str, dict[str, Any]] = {}
    for profile_pass in variant.profile_passes:
        documents[f"{profile_pass.name}.yaml"] = profile_document(
            pack, case, variant, host, profile_pass.name, repo_root
        )
    documents[f"{PHASE_CONFIG_STEMS[TIMING_PREDICT_PHASE]}.yaml"] = timing_predict_document(variant)
    documents[f"{PHASE_CONFIG_STEMS[ANALYSIS_KERNEL_PHASE]}.yaml"] = analyze_kernel_document(
        variant
    )
    documents[f"{PHASE_CONFIG_STEMS[ANALYSIS_E2E_PHASE]}.yaml"] = analyze_e2e_document(variant)
    documents[f"{PHASE_CONFIG_STEMS[SIMULATION_PHASE]}.yaml"] = simulation_document(
        pack, case, variant, case_dir, repo_root
    )
    return documents


def case_traces(pack: Pack, case: Case) -> dict[str, str]:
    """Trace file name → contents for one rendered case directory."""
    if case.session_trace is not None:
        text, _ = read_session_trace(pack.root, case)
        _check_session_lengths(case, text)
        traces = {WORKLOAD_TRACE_NAME: text}
        if case.speculative_acceptance is not None:
            traces[SPECULATIVE_TRACE_NAME] = session_trace_with_acceptance(case, text)
        return traces
    traces = {WORKLOAD_TRACE_NAME: trace_text(case, case.workload_trace)}
    if case.speculative_acceptance is not None:
        if case.speculative_acceptance_per_request is not None:
            entries = load_per_request_chains(pack.root, case)
            spec = case.workload_trace
            prefix = case.slug if spec.id_suffix is None else f"{case.slug}-{spec.id_suffix}"
            shapes = [shape for _ in range(spec.repeats) for shape in spec.shapes]
            assigned = per_request_chains(
                entries,
                [(f"{prefix}-{index:04d}", i, o) for index, (i, o) in enumerate(shapes)],
            )
            chain_of = lambda request_id, _input_len, _output_len: assigned[request_id]
        else:
            chain_of = lambda _id, _input_len, output_len: acceptance_chain(case, output_len)
        traces[SPECULATIVE_TRACE_NAME] = trace_text(
            case, case.workload_trace, speculative_acceptance=chain_of
        )
    if case.kernel_trace is not None:
        traces[KERNEL_TRACE_NAME] = trace_text(case, case.kernel_trace)
    return traces


def _verify_text_corpus(case: Case, host: HostProfile) -> None:
    """Refuse to render a case whose declared replay corpus has other bytes.

    A corpus missing on the rendering machine is left to the profile phase,
    whose runner refuses it before the engine starts: a simulation-only host
    never needs the corpus at all.
    """
    if case.text_corpus is None:
        return
    path = Path(host.text_corpus_path(case.text_corpus.host_key))
    if not path.is_file():
        return
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    if digest.hexdigest() != case.text_corpus.sha256:
        raise PackError(
            f"cases[{case.slug}].text_corpus: {path} hashes to {digest.hexdigest()}, "
            f"the pack records {case.text_corpus.sha256}"
        )


def render_case(
    pack: Pack, case: Case, host: HostProfile, out_root: Path, repo_root: Path
) -> RenderedCase:
    """Write one case's run directory. Overwrites configs and traces; never
    touches artifact directories, so re-rendering after a config fix does not
    discard an expensive capture."""
    case_dir = Path(out_root).resolve() / case.slug
    _verify_text_corpus(case, host)
    case_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, text in case_traces(pack, case).items():
        target = case_dir / name
        target.write_text(text, encoding="utf-8")
        written.append(target)
    if case.session_trace is not None:
        target = case_dir / SESSION_MANIFEST_NAME
        target.write_bytes((pack.root / case.session_trace.manifest).read_bytes())
        written.append(target)
    for name, document in case_documents(pack, case, host, case_dir, repo_root).items():
        target = case_dir / name
        target.write_text(
            yaml.safe_dump(document, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        written.append(target)
    return RenderedCase(
        case_slug=case.slug,
        directory=case_dir,
        phases=phase_names(pack.variant_of(case)),
        files=tuple(written),
    )
