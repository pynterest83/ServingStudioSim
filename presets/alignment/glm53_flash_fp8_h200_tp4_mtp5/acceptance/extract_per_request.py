"""Per-request MTP acceptance chains from one workload pass (the pack's `speculative_acceptance_per_request`).

For every request, over its verify rounds (`decode_request_progress` entries of the vLLM iteration metrics,
excluding the wasted round of an already-finished request), count n_k = rounds that accepted at least k drafts.
Its chain is the conditional acceptance per draft position, p_k = n_k / n_{k-1} for k = 1..K, kept per position
(never collapsed to a mean) so a shorter speculation depth is the chain's prefix. A position the request never
reached (n_{k-1} = 0) takes the case's pooled conditional value. One chain per measured request; rendering assigns
chains to trace requests by a fixed hash of the trace request id, not by measured request id.

Measured on the alignment matrix's synthetic prompts (enwik9 text, tokens-only replay); production traffic may
have a different per-request distribution.

usage: extract_per_request.py <case_dir> <out.json> [K=5]
"""
import glob, json, sys
from collections import defaultdict

case_dir, out = sys.argv[1], sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 5
metrics = glob.glob(f"{case_dir}/profile_workload/vllm/*_metrics.jsonl")[0]
counts = defaultdict(lambda: [0] * (K + 1))
for line in open(metrics):
    record = json.loads(line)
    for progress in record.get("decode_request_progress") or []:
        if progress.get("request_finished_before") or progress.get("drafted_tokens") != K:
            continue
        accepted = progress["accepted_draft_tokens"]
        row = counts[progress["request_id"]]
        for k in range(accepted + 1):
            row[k] += 1
pooled_n = [sum(row[k] for row in counts.values()) for k in range(K + 1)]
pooled = [pooled_n[k] / pooled_n[k - 1] if pooled_n[k - 1] else 0.0 for k in range(1, K + 1)]
chains, filled = [], 0
for request_id in sorted(counts):
    row = counts[request_id]
    chain = []
    for k in range(1, K + 1):
        if row[k - 1]:
            chain.append(round(row[k] / row[k - 1], 4))
        else:
            chain.append(round(pooled[k - 1], 4)); filled += 1
    chains.append(chain)
json.dump({
    "schema_version": 1,
    "draft_tokens": K,
    "meaning": "per request, conditional acceptance P(accept draft k | accepted k-1), k = 1..K",
    "evidence": metrics.split("/logs/", 1)[-1] if "/logs/" in metrics else metrics,
    "requests": len(chains),
    "rounds": pooled_n[0],
    "pooled_chain": [round(x, 4) for x in pooled],
    "positions_filled_from_pool": filled,
    "note": "synthetic enwik9 prompts (tokens-only replay); production traffic may differ",
    "chains": chains,
}, open(out, "w"), indent=None, separators=(",", ":"))
print(f"{case_dir.rstrip('/').split('/')[-1]}: {len(chains)} requests, {pooled_n[0]} rounds, pooled {[round(x,4) for x in pooled]}, filled {filled}")
