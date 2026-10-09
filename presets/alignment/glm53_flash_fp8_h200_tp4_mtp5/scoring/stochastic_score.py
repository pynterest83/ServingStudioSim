"""Stochastic scoring of a speculative (MTP) pack case: the `stochastic_scoring` policy of acceptance.yaml.

With per-request acceptance chains, the iteration count of one simulation is a random variable: it depends on
which trace request draws which measured chain (who finishes last sets the drain tail). The measured run is one
sample of that variable. This tool therefore scores in two parts.

(i) Deterministic, 1% tolerance:
  - rounds_total_pct: simulated decode (verify) rounds of the whole case, mean over seeds, vs measured.
    Per (input, output) shape the same deviation is a diagnostic only, flagged when it exceeds 3 standard errors:
    bootstrapped from the shape's measured per-request rounds, or (one request in the shape) the renewal
    approximation Var(R) ~= N v / m^3 of per-round draw noise (m, v = mean/variance of tokens per verify round
    under the request's chain, N = its output tokens).
  - replica_iterations_pct: replica.py with vLLM v0.31's scheduling rules on the MEASURED per-request rounds vs the
    measured iteration count (closed-loop saturated cases only; rate-driven cases are n/a).
(ii) Stochastic: the case is simulated under >= N hash orders of the chain assignment (render seed 1..N):
  - the measured iteration count must lie in the simulated p5-p95 band; its percentile is reported. A band of
    zero width (no re-pairing freedom, e.g. one request per shape) falls back to the deterministic 1% bound;
  - e2e / throughput / TPOT / TTFT and the workload-structure metrics are reported as simulated mean +- sd (and
    p5/p95) of the pack's metric formula, next to the measured noise band when given.
A diagnostic run (`--measured-id`) assigns every trace request the chain of the SAME measured request id; it is
not used for scoring, it shows what remains when the pairing matches the measured sample (scheduling rules).

usage (from the repo root, CPU only):
  uv run python presets/alignment/<pack>/scoring/stochastic_score.py --pack presets/alignment/<pack> \\
      --run-root <campaign out root> --cases 03,04 --seeds 50 --jobs 16 [--measured-id] [--out score.json]
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import glob
import io
import json
import statistics as st
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]  # presets/alignment/<pack>/scoring -> repository root
sys.path[:0] = [str(REPO_ROOT), str(HERE)]

from launcher.alignment_campaign.metrics import _load_report, metric_specs  # noqa: E402
from launcher.alignment_campaign.pack import load_pack, load_per_request_chains  # noqa: E402
from launcher.alignment_campaign.render import per_request_chains  # noqa: E402

from replica import Config, run as replica_run  # noqa: E402

STOCHASTIC_METRICS = (
    "iterations_pct", "decode_batch_mean_pct", "prefill_tokens_mean_pct", "scheduled_kv_mean_pct",
    "iteration_cycle_mean_pct", "e2e_mean_pct", "output_tps_pct", "server_tpot_mean_pct", "server_ttft_mean_pct",
)


def measured_chains_by_id(case_dir: Path, depth: int) -> dict[str, list[float]]:
    """Per measured request (trace id), its conditional acceptance chain (as acceptance/extract_per_request.py)."""
    counts: dict[str, list[int]] = defaultdict(lambda: [0] * (depth + 1))
    metrics = glob.glob(f"{case_dir}/profile_workload/vllm/*_metrics.jsonl")[0]
    for line in open(metrics):
        for progress in json.loads(line).get("decode_request_progress") or []:
            if progress.get("request_finished_before") or progress.get("drafted_tokens") != depth:
                continue
            row = counts[progress["request_id"].removeprefix("independent_")]
            for k in range(progress["accepted_draft_tokens"] + 1):
                row[k] += 1
    pooled = [sum(r[k] for r in counts.values()) for k in range(depth + 1)]
    pooled_chain = [pooled[k] / pooled[k - 1] for k in range(1, depth + 1)]
    return {
        rid: [round(r[k] / r[k - 1], 4) if r[k - 1] else round(pooled_chain[k - 1], 4) for k in range(1, depth + 1)]
        for rid, r in counts.items()
    }


def measured_rounds(case_dir: Path) -> tuple[Counter, dict[str, int], dict[str, bool]]:
    """Measured verify rounds per trace id (and whether the request ran a wasted async round)."""
    per_id: Counter = Counter()
    wasted: dict[str, bool] = defaultdict(bool)
    metrics = glob.glob(f"{case_dir}/profile_workload/vllm/*_metrics.jsonl")[0]
    for line in open(metrics):
        for progress in json.loads(line).get("decode_request_progress") or []:
            rid = progress["request_id"].removeprefix("independent_")
            if progress.get("request_finished_before"):
                wasted[rid] = True
            else:
                per_id[rid] += 1
    return per_id, dict(per_id), wasted


def write_variant(case_dir: Path, name: str, chains: dict[str, list[float]], profile_workload: Path) -> Path:
    """One simulation variant: the case's speculative trace with replaced chains, its preset and analysis config."""
    out = case_dir / "stochastic" / name
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(case_dir / "trace_speculative.csv")))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        row["accept_rate"] = json.dumps(chains[row["id"]], separators=(",", ":"))
        writer.writerow(row)
    (out / "trace_speculative.csv").write_text(buffer.getvalue())
    preset = yaml.safe_load(open(case_dir / "simulation.yaml"))
    preset["workload"]["trace_files"] = [str(out / "trace_speculative.csv")]
    preset["io"]["log_dir"] = str(out / "simulation")
    (out / "simulation.yaml").write_text(yaml.safe_dump(preset, sort_keys=False))
    (out / "analyze_e2e.yaml").write_text(yaml.safe_dump({
        "schema_version": 1, "simulation_log_dir": "./simulation",
        "workload_profile_log_dir": str(profile_workload), "log_dir": "./analysis_e2e",
        "workload": {"enabled": True}, "e2e": {"enabled": True, "throughput_bins": 20},
    }, sort_keys=False))
    return out


def simulate(variant: Path) -> bool:
    if (variant / "analysis_e2e" / "reports" / "alignment_e2e_report.json").is_file():
        return True
    for cmd in (["alignment", "sim", str(variant / "simulation.yaml")],
                ["alignment", "analyze", str(variant / "analyze_e2e.yaml")]):
        result = subprocess.run([sys.executable, "-m", "launcher", *cmd], capture_output=True, text=True)
        if result.returncode != 0:
            (variant / "failed.log").write_text(result.stdout + result.stderr)
            return False
    return True


def variant_metrics(variant: Path, specs) -> dict[str, float]:
    out: dict[str, float] = {}
    docs = {kind: _load_report(variant, kind)[0] for kind in ("e2e", "workload")}
    for spec in specs:
        if spec.name in STOCHASTIC_METRICS and docs.get(spec.report) is not None:
            value = spec.read(docs[spec.report])
            if value is not None:
                out[spec.name] = value
    meta = (docs["workload"] or {}).get("meta", {})
    out["simulated_iterations"] = meta.get("simulated_iterations")
    out["measured_iterations"] = meta.get("measured_iterations")
    return out


def simulated_rounds_by_shape(variant: Path, shape_of: dict[str, tuple[int, int]], ids: list[str]) -> Counter:
    table = pq.read_table(variant / "simulation" / "raw" / "request_slo.parquet",
                          columns=["request_id", "speculative_progress"]).to_pylist()
    rounds: Counter = Counter()
    for row in table:
        progress = row["speculative_progress"]
        progress = json.loads(progress) if isinstance(progress, str) else progress
        rounds[shape_of[ids[row["request_id"]]]] += progress["decode_rounds"]
    return rounds


def tokens_per_round_moments(chain: list[float]) -> tuple[float, float]:
    """Mean and variance of tokens emitted per verify round (1 + accepted drafts) under a conditional chain."""
    reach, probs = 1.0, []
    for p_k in chain:
        probs.append(reach * (1 - p_k))
        reach *= p_k
    probs.append(reach)  # all drafts accepted
    mean = sum((a + 1) * p for a, p in enumerate(probs))
    var = sum(((a + 1) - mean) ** 2 * p for a, p in enumerate(probs))
    return mean, var


def shape_rounds_se(rounds: list[int], chains: list[list[float]], outputs: list[int], resamples: int) -> tuple[float, str]:
    """Standard error of a shape's total decode rounds: bootstrap over its measured requests, or the renewal
    approximation of per-round draw noise when the shape has a single request."""
    if len(rounds) >= 2:
        rng = random.Random(0)
        sums = [sum(rng.choice(rounds) for _ in rounds) for _ in range(resamples)]
        return st.pstdev(sums), "bootstrap"
    var = 0.0
    for chain, n in zip(chains, outputs):
        mean, v = tokens_per_round_moments(chain)
        var += max(n - 1, 0) * v / mean ** 3
    return math.sqrt(var), "renewal"


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))]


def score_case(pack, case, run_root: Path, seeds: int, jobs: int, measured_id: bool) -> dict:
    case_dir = run_root / case.slug
    depth = len(case.speculative_acceptance.value)
    entries = load_per_request_chains(pack.root, case)
    trace_rows = list(csv.DictReader(open(case_dir / "trace_speculative.csv")))
    ids = [r["id"] for r in trace_rows]
    shape_of = {r["id"]: (int(r["input_len"]), int(r["output_len"])) for r in trace_rows}
    requests = [(rid, *shape_of[rid]) for rid in ids]
    variants = {f"s{seed:02d}": per_request_chains(entries, requests, seed) for seed in range(1, seeds + 1)}
    by_id = measured_chains_by_id(case_dir, depth)
    if measured_id:
        variants["measured_id"] = {rid: by_id.get(rid, list(case.speculative_acceptance.value)) for rid in ids}
    paths = {name: write_variant(case_dir, name, chains, case_dir / "profile_workload")
             for name, chains in variants.items()}
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        ok = dict(zip(paths, pool.map(simulate, paths.values())))
    specs = metric_specs()
    results = {name: variant_metrics(path, specs) for name, path in paths.items() if ok[name]}
    seeded = [name for name in results if name.startswith("s")]

    # (ii) stochastic
    stoch: dict[str, dict] = {}
    for metric in STOCHASTIC_METRICS:
        values = [results[n][metric] for n in seeded if metric in results[n]]
        if values:
            stoch[metric] = {"mean": st.mean(values), "sd": st.pstdev(values),
                             "p5": pct(values, 0.05), "p95": pct(values, 0.95), "n": len(values)}
    sim_iters = [results[n]["simulated_iterations"] for n in seeded if results[n].get("simulated_iterations")]
    measured_iters = next((results[n]["measured_iterations"] for n in seeded if results[n].get("measured_iterations")), None)
    band = {"p5": pct(sim_iters, 0.05), "p95": pct(sim_iters, 0.95), "mean": st.mean(sim_iters),
            "measured": measured_iters,
            "measured_percentile": 100.0 * sum(x < measured_iters for x in sim_iters) / len(sim_iters),
            "measured_in_band": pct(sim_iters, 0.05) <= measured_iters <= pct(sim_iters, 0.95)}

    # (i) deterministic: rounds by shape (mean over seeds) and the replica on measured rounds
    rounds_mean: dict[tuple[int, int], float] = defaultdict(float)
    for n in seeded:
        for shape, value in simulated_rounds_by_shape(paths[n], shape_of, ids).items():
            rounds_mean[shape] += value / len(seeded)
    per_id_rounds, _, wasted = measured_rounds(case_dir)
    measured_by_shape: Counter = Counter()
    for rid, value in per_id_rounds.items():
        measured_by_shape[shape_of[rid]] += value
    by_shape = {f"{s[0]}x{s[1]}": 100.0 * (rounds_mean[s] / measured_by_shape[s] - 1) for s in measured_by_shape}
    total = 100.0 * (sum(rounds_mean.values()) / sum(measured_by_shape.values()) - 1)
    policy = (pack.acceptance.get("stochastic_scoring") or {})
    flag_sigma = float((policy.get("rounds_by_shape_diagnostic") or {}).get("flag_sigma", 3.0))
    resamples = int((policy.get("rounds_by_shape_diagnostic") or {}).get("bootstrap_resamples", 2000))
    shape_diag = {}
    for shape in measured_by_shape:
        members = [rid for rid in ids if shape_of[rid] == shape and rid in per_id_rounds]
        se, method = shape_rounds_se([per_id_rounds[r] for r in members], [by_id[r] for r in members if r in by_id],
                                     [shape[1] for r in members], resamples)
        diff = rounds_mean[shape] - measured_by_shape[shape]
        shape_diag[f"{shape[0]}x{shape[1]}"] = {
            "pct": by_shape[f"{shape[0]}x{shape[1]}"], "diff_rounds": diff, "se_rounds": se, "se_method": method,
            "z": diff / se if se > 0 else None, "flagged": se > 0 and abs(diff) > flag_sigma * se}
    replica = None
    if case.profile_arrival_mode == "saturated":
        trace = [shape_of[rid] for rid in ids]
        rounds = [per_id_rounds.get(rid, 0) for rid in ids]
        steps, _ = replica_run(trace, case.max_concurrency, rounds, [wasted.get(rid, False) for rid in ids],
                               Config(split="vllm", lag="vllm", ramp="vllm", order="vllm"))
        replica = {"steps": len(steps), "measured": measured_iters,
                   "pct": 100.0 * (len(steps) / measured_iters - 1) if measured_iters else None}
    return {
        "case": case.slug, "seeds": len(seeded), "failed_variants": [n for n, good in ok.items() if not good],
        "deterministic": {"rounds_total_pct": total, "rounds_by_shape": shape_diag,
                          "replica_iterations": replica},
        "stochastic": {"iterations_band": band, "metrics": stoch},
        "diagnostic_measured_id": results.get("measured_id"),
        "verdict": verdict(pack, policy, total, replica, band, stoch),
    }


def verdict(pack, policy, rounds_total, replica, band, stoch) -> dict:
    """Pass/fail per scored item of acceptance.yaml's stochastic_scoring."""
    tol = (policy.get("deterministic_tolerances") or {})
    default = (pack.acceptance.get("tolerances") or {}).get("default", {})
    out = {"rounds_total": abs(rounds_total) <= float(tol.get("rounds_total_pct", 1.0))}
    if replica and replica.get("pct") is not None:
        out["replica_iterations"] = abs(replica["pct"]) <= float(tol.get("replica_iterations_pct", 1.0))
    if band["p5"] == band["p95"]:
        out["iterations"] = abs(stoch["iterations_pct"]["mean"]) <= float(policy.get("zero_width_band_iterations_pct", 1.0))
        out["iterations_rule"] = "zero-width band: deterministic bound"
    else:
        out["iterations"] = band["measured_in_band"]
        out["iterations_rule"] = "measured within simulated p5-p95"
    for metric in ("e2e_mean_pct", "output_tps_pct", "server_tpot_mean_pct", "server_ttft_mean_pct"):
        if metric in stoch and metric in default:
            out[metric] = abs(stoch[metric]["mean"]) <= float(default[metric])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cases", required=True, help="comma-separated case indexes or slugs")
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--measured-id", action="store_true", help="also run the measured-id diagnostic")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    pack = load_pack(args.pack)
    report = []
    for key in args.cases.split(","):
        case = pack.case_named(key)
        if case is None or case.speculative_acceptance_per_request is None:
            raise SystemExit(f"case {key}: no per-request acceptance in this pack")
        result = score_case(pack, case, args.run_root.resolve(), args.seeds, args.jobs, args.measured_id)
        report.append(result)
        band = result["stochastic"]["iterations_band"]
        det = result["deterministic"]
        flagged = [k for k, v in det["rounds_by_shape"].items() if v["flagged"]]
        print(f"{case.slug}: iterations measured {band['measured']} sim p5-p95 [{band['p5']}, {band['p95']}] "
              f"(percentile {band['measured_percentile']:.0f}); rounds total {det['rounds_total_pct']:+.2f}%; "
              f"shapes flagged >3SE {flagged}; replica {det['replica_iterations']}; verdict {result['verdict']}",
              flush=True)
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
