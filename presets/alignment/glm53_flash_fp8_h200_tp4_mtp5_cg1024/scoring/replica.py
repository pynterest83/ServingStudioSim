"""Step-level replica of the two schedulers' RULES, used to attribute the
iteration-count / batch / TTFT gap to individual rule differences.

This is a diagnostic only. It never reads a measured batch composition,
admission order or chunk boundary: inputs are the trace (prompt, output),
the client concurrency, and per-request decode-round counts (taken either
from the measured run or from the simulator, to separate acceptance from
scheduling). Each toggle switches one rule between the simulator's behavior
and vLLM v0.31's behavior.

Toggles (value 'sim' or 'vllm'):
  split  : chunk-end stop at the prompt's last cacheable block boundary.
           sim  = P//B*B (chunked_prefill_admission.rs:849)
           vllm = P//B*B - B when use_eagle_block_drop (scheduler.py:441-444)
  lag    : when a closed-loop replacement becomes schedulable after its
           predecessor's final decode step N.
           sim  = N+2 (completion stamped after the next iteration began)
           vllm = N+3, and the finished request runs one wasted decode in
                  N+1 (async scheduling one step behind; core.py:760-876,
                  scheduler.py:640-655); N+2 if the engine would otherwise idle.
  ramp   : sim  = all initial requests visible at step 1
           vllm = request 0 alone at step 1, the rest from step 3
  order  : sim  = all live decodes reserve budget first, then started prefills
           vllm = running list in admission order (scheduler.py:630)
  kv     : none | sim (full footprint P+O+state tokens reserved at admission)
           | vllm (block-granular, prompt must fit at admission
             (scheduler_reserve_full_isl), incremental growth, LIFO preemption
             with recompute)
  state_blocks: fixed recurrent-state blocks per request for kv=vllm/sim:
           19 = 3 KDA groups x (1 + K) + 1, what a decoding request holds
           in mamba align mode (the startup sizer's 22 adds a prefill
           checkpoint per group, freed by remove_skipped_blocks).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

B = 1152
QW = 6
BUDGET = 8192
LOOKAHEAD = 5


@dataclass
class Req:
    idx: int
    P: int
    O: int
    rounds: int
    wasted: bool
    tok_per_round: float
    visible_from: int = 1
    arrival_step_end: int = 0  # step index whose end is the arrival time (0 = t0)
    computed: int = 0  # prefill tokens computed
    target: int = 0  # prefill target (P, or P+generated after preemption)
    rounds_done: int = 0
    generated: float = 0.0
    admitted_step: int = -1
    first_token_step: int = -1
    finish_step: int = -1
    state: str = "new"  # new|prefill|decode|wasted|done
    preemptions: int = 0
    blocks: int = 0


def split_chunk(start, end_target, chunk, rule):
    """vLLM _mamba_block_aligned_split for this deployment (no internal
    checkpoint, no partial-tail stop: all prompts are multiples of 16)."""
    if chunk <= 0 or start >= end_target:
        return chunk
    end = start + chunk
    if end < end_target:
        aligned = end // B * B
        if aligned > start or B <= BUDGET:
            end = aligned
    last = end_target - end_target % B
    if rule == "vllm":
        last = max(last - B, 0)
    stops = []
    if start % B != 0:
        stops.append((start // B + 1) * B)
    stops.append(last)
    cands = [s for s in stops if start < s < end]
    if cands:
        end = min(cands)
    return max(end - start, 0)


@dataclass
class Config:
    split: str = "sim"
    lag: str = "sim"
    ramp: str = "sim"
    order: str = "sim"
    kv: str = "none"
    state_blocks: int = 19
    total_blocks: int = 3144
    sim_kv_tokens: int = 3622674


def run(trace, conc, rounds, wasted, cfg: Config, max_steps=200000):
    reqs = []
    for i, (P, O) in enumerate(trace):
        r = max(int(rounds[i]), 0)
        reqs.append(
            Req(i, P, O, r, bool(wasted[i]), (O - 1) / r if r else 0.0, target=P)
        )
    n = len(reqs)
    waiting = deque()
    nxt = 0
    for i in range(min(conc, n)):
        if cfg.ramp == "vllm":
            reqs[i].visible_from = 1 if i == 0 else 3
        else:
            reqs[i].visible_from = 1
        reqs[i].arrival_step_end = 0
        waiting.append(reqs[i])
    nxt = min(conc, n)
    running: list[Req] = []
    free_blocks = cfg.total_blocks
    sim_tokens_free = cfg.sim_kv_tokens
    steps = []
    k = 0
    done = 0
    pending_wasted: list[Req] = []

    def blocks_for(r, tokens):
        return math.ceil(min(tokens + LOOKAHEAD, 8192 + LOOKAHEAD) / B) + cfg.state_blocks

    while done < n and k < max_steps:
        k += 1
        budget = BUDGET
        sched = []  # (req, kind, tokens)
        # wasted async decodes of requests that finished in step k-1
        wasted_now = [r for r in pending_wasted]
        pending_wasted = []
        # ---------- running ----------
        if cfg.order == "sim":
            order = [r for r in running if r.state == "decode"] + [
                r for r in running if r.state == "prefill"
            ]
        else:
            order = list(running)
        # wasted decodes sit in running order position; put them first (oldest)
        for r in wasted_now:
            take = min(QW, budget)
            budget -= take
            sched.append((r, "wasted", take))
        preempted_any = False
        for r in order:
            if budget <= 0:
                break
            if r.state == "decode":
                take = min(QW, budget)
                if cfg.kv == "vllm":
                    need = blocks_for(r, r.target + r.generated + take) - r.blocks
                    while need > free_blocks:
                        victim = running[-1]
                        running.remove(victim)
                        free_blocks += victim.blocks
                        victim.blocks = 0
                        victim.state = "new"
                        victim.computed = 0
                        victim.target = victim.P + int(round(victim.generated))
                        victim.preemptions += 1
                        waiting.appendleft(victim)
                        preempted_any = True
                        if victim is r:
                            break
                    if r.state != "decode":
                        continue
                    if need > 0:
                        free_blocks -= need
                        r.blocks += need
                budget -= take
                sched.append((r, "decode", take))
            elif r.state == "prefill":
                rem = r.target - r.computed
                c = split_chunk(r.computed, r.target, min(rem, budget), cfg.split)
                if c <= 0:
                    continue
                if cfg.kv == "vllm":
                    need = blocks_for(r, r.computed + c) - r.blocks
                    if need > free_blocks:
                        continue
                    free_blocks -= need
                    r.blocks += need
                budget -= c
                sched.append((r, "prefill", c))
        # ---------- waiting ----------
        if not preempted_any:
            while budget > 0 and waiting:
                r = waiting[0]
                if r.visible_from > k:
                    break
                c = split_chunk(0, r.target, min(r.target, budget), cfg.split)
                if c <= 0:
                    break
                if cfg.kv == "vllm":
                    full = blocks_for(r, r.target) - LOOKAHEAD // B
                    if full > free_blocks:
                        break
                    need = blocks_for(r, c)
                    free_blocks -= need
                    r.blocks = need
                elif cfg.kv == "sim":
                    fp = r.P + r.O + cfg.state_blocks * B
                    if fp > sim_tokens_free:
                        break
                    sim_tokens_free -= fp
                    r.blocks = fp
                waiting.popleft()
                r.state = "prefill"
                if r.admitted_step < 0:
                    r.admitted_step = k
                running.append(r)
                budget -= c
                sched.append((r, "prefill", c))
        if not sched:
            # engine idle: wait for the in-flight step's output, then reschedule
            # (the lag shrinks by one step: nothing was launched at k)
            for r in waiting:
                if r.visible_from > k:
                    r.visible_from -= 1
            k -= 1
            if not waiting and not running and not pending_wasted:
                break
            # guard against livelock
            if all(r.visible_from > k + 1 for r in waiting) and not running:
                for r in waiting:
                    r.visible_from = min(r.visible_from, k + 1)
            continue
        # ---------- execute step k ----------
        pf_tok = sum(t for _, kind, t in sched if kind == "prefill")
        n_dec = sum(1 for _, kind, _ in sched if kind in ("decode", "wasted"))
        n_chunks = sum(1 for _, kind, _ in sched if kind == "prefill")
        steps.append(
            dict(step=k, pf=pf_tok, n_dec=n_dec, n_chunks=n_chunks,
                 n_wasted=sum(1 for _, kind, _ in sched if kind == "wasted"),
                 dec_tokens=sum(t for _, kind, t in sched if kind in ("decode", "wasted")))
        )
        for r, kind, t in sched:
            if kind == "prefill":
                r.computed += t
                if r.computed >= r.target:
                    r.state = "decode"
                    if r.first_token_step < 0:
                        r.first_token_step = k
                    if r.rounds_done >= r.rounds:
                        _finish(r, k)
            elif kind == "decode":
                r.rounds_done += 1
                r.generated += r.tok_per_round
                if r.rounds_done >= r.rounds:
                    _finish(r, k)
        for r, kind, t in sched:
            if r.state == "done" and r.finish_step == k:
                if r in running:
                    running.remove(r)
                if cfg.kv == "vllm":
                    free_blocks += r.blocks
                elif cfg.kv == "sim":
                    sim_tokens_free += r.blocks
                r.blocks = 0
                done += 1
                if cfg.lag == "vllm" and r.wasted:
                    pending_wasted.append(r)
                if nxt < n:
                    q = reqs[nxt]
                    nxt += 1
                    q.visible_from = k + (3 if cfg.lag == "vllm" else 2)
                    q.arrival_step_end = k
                    waiting.append(q)
    return steps, reqs


def _finish(r, k):
    r.state = "done"
    r.finish_step = k
