"""Fit the speculative worker's host-time model to one kernel-alignment case.

The worker's host-time model (``worker`` knobs ``host_fixed_ms`` and
``host_floor_{full_graph,piecewise,eager_prefill,eager_decode}_ms``) bills one
engine step::

    cycle = max(kernel * multiplier + fixed_ms, floor_ms[graph mode])

where eager splits by whether the step schedules prefill. This tool fits
``fixed_ms`` and the four floors to the measured GPU cycles of a kernel-align
run and compares the fit with the multiplier-only clock.

Inputs
  --report   ``analysis_kernel/reports/alignment_iteration_report.json``: per
             iteration ``measured_gpu_cycle_ms`` (the target), ``simulated_ms``
             (the kernel prediction), ``stage`` and ``case_index``.
  --predict  the timing-predict ``log_dir`` whose ``raw/cost_log`` carries the
             ``graph_mode`` column. Defaults to the report's
             ``meta.predict_log_dir``; a run from a simulator that predates the
             column fails here. Its ``total_time_ms`` must match the report's
             ``simulated_ms`` per case, so the modes describe the same batches.

Method (the reference is ``analysis_host/fit_host_models.py`` of the
GLM-5.3-Flash H200 campaign): minimize mean |relative error| (L1, robust to the
bimodal host state) over the iterations kernel-align keeps for its multiplier.
For a fixed ``fixed_ms`` each floor is independent and piecewise linear, so its
optimum is one of the breakpoints {cycle_i, kernel_i + fixed_ms}; ``fixed_ms``
is a grid search. A mode with no iterations has no identified floor (``null``).

Baselines: the report's pooled ``recommended_gpu_time_multiplier`` and the best
stage-split multiplier pair (pooled ratio per stage, as
``gpu_time_multiplier`` / ``prefill_gpu_time_multiplier``).

    uv run python tools/host-time-fit/fit_host_time.py \\
        --report <case>/analysis_kernel/reports/alignment_iteration_report.json \\
        --predict <log_dir>/timing_predict [--out fit.json]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

MODES = ("full_graph", "piecewise", "eager_prefill", "eager_decode")
FIXED_GRID_MS = np.round(np.arange(0.0, 5.0001, 0.01), 4)


def floor_key(graph_mode: str, schedules_prefill: bool) -> str:
    """The worker's floor lookup (`HostTimeModel::floor_ms`)."""
    if graph_mode == "eager":
        return "eager_prefill" if schedules_prefill else "eager_decode"
    return graph_mode


def load_rows(report_path: Path, predict_dir: Path | None) -> tuple[list[dict], dict]:
    report = json.loads(report_path.read_text())
    meta = report["meta"]
    predict_dir = predict_dir or Path(meta["predict_log_dir"])
    cost_logs = sorted((predict_dir / "raw" / "cost_log").glob("*.parquet"))
    if len(cost_logs) != 1:
        raise SystemExit(f"{predict_dir}: expected one cost_log parquet, got {cost_logs}")
    table = pq.read_table(cost_logs[0])
    if "graph_mode" not in table.column_names:
        raise SystemExit(
            f"{cost_logs[0]} has no graph_mode column: rerun timing-predict with a "
            "simulator that logs it"
        )
    by_case = {}
    for row in table.to_pylist():
        prefill = any(group["prefill_tokens"] > 0 for group in row["groups"])
        by_case[row["iter_id"]] = (row["graph_mode"], prefill, row["total_time_ms"])

    excluded = {item["iteration"] for item in meta.get("multiplier_excluded_iterations", [])}
    rows = []
    for item in report["iterations"]:
        cycle = item.get("measured_gpu_cycle_ms")
        if cycle is None or item["iteration_id"] in excluded:
            continue
        graph_mode, prefill, predicted = by_case[item["case_index"]]
        if graph_mode is None:
            raise SystemExit(f"case {item['case_index']}: the arch reported no graph mode")
        if not math.isclose(predicted, item["simulated_ms"], rel_tol=1e-6, abs_tol=1e-6):
            raise SystemExit(
                f"case {item['case_index']}: cost_log total_time_ms {predicted} != report "
                f"simulated_ms {item['simulated_ms']}; the predict dir is not this report's"
            )
        rows.append(
            {
                "iteration": item["iteration_id"],
                "stage": item["stage"],
                "mode": floor_key(graph_mode, prefill),
                "kernel": float(item["simulated_ms"]),
                "cycle": float(cycle),
            }
        )
    info = {
        "report": str(report_path),
        "predict": str(predict_dir),
        "excluded_iterations": sorted(excluded),
        "recommended_gpu_time_multiplier": meta.get("recommended_gpu_time_multiplier"),
    }
    return rows, info


def best_floor(kernel_plus_fixed: np.ndarray, cycle: np.ndarray) -> tuple[float, float]:
    """L1-optimal floor for one mode; returns (floor, summed |rel err|)."""
    candidates = np.unique(np.concatenate([[0.0], cycle, kernel_plus_fixed]))
    pred = np.maximum(kernel_plus_fixed[None, :], candidates[:, None])
    loss = np.abs(pred - cycle[None, :]) / cycle[None, :]
    total = loss.sum(axis=1)
    best = int(np.argmin(total))
    return float(candidates[best]), float(total[best])


def fit_host(rows: list[dict], multiplier: float) -> dict:
    kernel = np.array([r["kernel"] for r in rows]) * multiplier
    cycle = np.array([r["cycle"] for r in rows])
    modes = np.array([r["mode"] for r in rows])
    best = None
    for fixed in FIXED_GRID_MS:
        floors, loss = {}, 0.0
        for mode in MODES:
            mask = modes == mode
            if not mask.any():
                floors[mode] = None
                continue
            floors[mode], mode_loss = best_floor(kernel[mask] + fixed, cycle[mask])
            loss += mode_loss
        if best is None or loss < best[0] - 1e-12:
            best = (loss, float(fixed), floors)
    _, fixed, floors = best
    # A floor is identified only by the iterations it binds on (floor > kernel +
    # fixed); one that binds on none says only "at most min(kernel + fixed)".
    binding = {
        mode: None
        if floors[mode] is None
        else int((floors[mode] > kernel[modes == mode] + fixed).sum())
        for mode in MODES
    }
    return {"multiplier": multiplier, "fixed_ms": fixed, "floor_ms": floors, "binding": binding}


def host_prediction(row: dict, params: dict) -> float:
    floor = params["floor_ms"][row["mode"]] or 0.0
    return max(row["kernel"] * params["multiplier"] + params["fixed_ms"], floor)


def error_table(rows: list[dict], predict) -> dict:
    """p50/p90 |relative error| and signed total error, per stage and overall."""
    out = {}
    for stage in ("all", *sorted({r["stage"] for r in rows})):
        sel = [r for r in rows if stage == "all" or r["stage"] == stage]
        pred = np.array([predict(r) for r in sel])
        cycle = np.array([r["cycle"] for r in sel])
        rel = np.abs(pred - cycle) / cycle
        out[stage] = {
            "n": len(sel),
            "p50_pct": float(np.percentile(rel, 50) * 100),
            "p90_pct": float(np.percentile(rel, 90) * 100),
            "total_signed_pct": float((pred.sum() / cycle.sum() - 1) * 100),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--predict", type=Path, default=None)
    parser.add_argument(
        "--multiplier",
        type=float,
        default=1.0,
        help="gpu_time_multiplier the host model runs with (default 1.0: kernels are aligned)",
    )
    parser.add_argument(
        "--evaluate",
        type=float,
        nargs=5,
        metavar=("FIXED", "FULL", "PIECEWISE", "EAGER_PREFILL", "EAGER_DECODE"),
        help="also score a given host model, e.g. the host analysis's fast-state priors",
    )
    parser.add_argument("--out", type=Path, default=None, help="write the fit and tables as JSON")
    args = parser.parse_args()

    rows, info = load_rows(args.report, args.predict)
    counts = {mode: sum(r["mode"] == mode for r in rows) for mode in MODES}
    host = fit_host(rows, args.multiplier)
    pooled = info["recommended_gpu_time_multiplier"]
    stage_k = {
        stage: sum(r["cycle"] for r in rows if r["stage"] == stage)
        / sum(r["kernel"] for r in rows if r["stage"] == stage)
        for stage in {r["stage"] for r in rows}
    }
    models = {
        "kernel only": lambda r: r["kernel"],
        f"multiplier k={pooled:.3f} (report)": lambda r: r["kernel"] * pooled,
        "multiplier per stage ("
        + ", ".join(f"{s} {k:.3f}" for s, k in sorted(stage_k.items()))
        + ")": lambda r: r["kernel"] * stage_k[r["stage"]],
        "host model (fit)": lambda r: host_prediction(r, host),
    }
    if args.evaluate:
        given = {
            "multiplier": args.multiplier,
            "fixed_ms": args.evaluate[0],
            "floor_ms": dict(zip(MODES, args.evaluate[1:])),
        }
        models["host model (--evaluate)"] = lambda r: host_prediction(r, given)
    tables = {name: error_table(rows, fn) for name, fn in models.items()}

    print(f"report  {info['report']}\npredict {info['predict']}")
    print(f"iterations {len(rows)} (excluded {info['excluded_iterations']}); per floor {counts}")
    print(f"\nhost model fit (multiplier {host['multiplier']}):")
    print(f"  host_fixed_ms: {host['fixed_ms']:.2f}")
    for mode in MODES:
        value = host["floor_ms"][mode]
        if value is None:
            shown = "null  # no iterations: not identified"
        else:
            shown = f"{value:.2f}  # binds on {host['binding'][mode]} of {counts[mode]}"
        print(f"  host_floor_{mode}_ms: {shown}")
    stages = [s for s in next(iter(tables.values())) if s != "all"] + ["all"]
    header = "model".ljust(48) + "".join(f"{s:>26}" for s in stages)
    print("\n|rel err| p50 / p90 % (signed total %)\n" + header)
    for name, table in tables.items():
        cells = "".join(
            f"{t['p50_pct']:>9.1f} /{t['p90_pct']:>6.1f} ({t['total_signed_pct']:+6.1f})"
            for t in (table[s] for s in stages)
        )
        print(name.ljust(48) + cells)
    by_mode = {
        mode: error_table(
            [r for r in rows if r["mode"] == mode], lambda r: host_prediction(r, host)
        )["all"]
        for mode in MODES
        if counts[mode]
    }
    print(
        "\nhost model by floor: "
        + json.dumps({m: {k: round(v, 1) for k, v in t.items()} for m, t in by_mode.items()})
    )
    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "inputs": info,
                    "counts": counts,
                    "host_fit": host,
                    "stage_multipliers": stage_k,
                    "errors": tables,
                    "host_errors_by_floor": by_mode,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
