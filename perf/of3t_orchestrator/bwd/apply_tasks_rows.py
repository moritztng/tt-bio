#!/usr/bin/env python3
"""Idempotently add the backward sprint's four JOBS rows to ~/.coworker/TASKS.md.

Why this is a script and not a one-off edit: `reconcile_tasks.sh` does an unlocked
read-modify-write of TASKS.md (`mapfile -t all_lines < $T` ... `mv $tmp $T`) and `fleet.sh`
runs it from cron every 2 minutes, so any hand edit made in that window is silently reverted
with no error and no log line. It ate this dispatch twice on 2026-09-25: once on queue.tsv
(my error -- that file is DOCUMENTED as generated from TASKS.md ws-tags, so editing it by hand
is always wrong) and once on TASKS.md itself (not my error -- that is a real race).

Re-run until `verify()` reports 4/4 present. Idempotent: a row already present is left alone.
"""
import sys, pathlib

T = pathlib.Path.home() / ".coworker" / "TASKS.md"
ANCHOR = "<!--ws:of3t-orchestrator-->"
PRE = ("- [ ] (2026-09-25 16:2x CEST, dispatched by `of3t-orchestrator` pass 445 from "
       "**`of3t-bwsurvey`'s JOBS list**, which concluded GO the same hour) ")

ROWS = [
 ("of3t-bwattrib", PRE +
  "**OF3T J0: attribute the 2.107 ms per backward verb. Not a kernel, and worth more than every "
  "kernel job combined times four.** `of3t-bwsurvey` measured it: at the forward's own warm rate "
  "the backward's 168,922 verb calls would cost **8.03 s**; they cost **356.00 s**, so **~348 s "
  "-- 97.7 % of the backward -- is not the model's arithmetic**. A backward verb costs **44.3x a "
  "forward verb on operands of the same shape**, warm, JIT burned off, allocator probe off, and "
  "nothing in the record explains why. At a pessimistic 0.24 ms/verb the step lands near 51 s, "
  "which is **9.1x** on its own. Suspects named and separable: `Tensor.evict` over PCIe, "
  "`add_grad`'s typecast to fp32, `to_layout` at fan-in, allocation churn. Settles J3's 14x "
  "bracket as a side effect. Brief `workstreams/of3t-bwattrib.txt`. DONE_CHECK declared. "
  "<!--ws:of3t-bwattrib-->"),
 ("of3t-lnbw", PRE +
  "**OF3T J1: the fused LayerNorm backward, adapted from `ttml::metal::layernorm_bw`. The biggest "
  "kernel job, worth at least 46.4 s and plausibly 150 s+.** LayerNorm backward is **1,296 of "
  "2,473 tape nodes, 52.4 %** -- an order of magnitude larger than triangle attention. The cost is "
  "`_sum_leading`: ~96 verbs per call on fp32 via `_tree_sum`, ~24 on bf16. Three things the "
  "survey already settled and this row must not rediscover: `mean`/`rstd` are not a blocker (our "
  "backward recomputes both, pass them in); precision is a gate not a blocker (`ttnn.sum` at "
  "8.3e-4 against the tree's 7.7e-8, 60x under the 5.0e-02 bar); and **do not grade it on the four "
  "LayerNorm affine leaves**, whose 92.68 % error mass is injected upstream by AttentionPairBias. "
  "**Unified, not per-model** -- serves Boltz-2, OF3T, BC2 and RFD3. Brief "
  "`workstreams/of3t-lnbw.txt`. DONE_CHECK declared. <!--ws:of3t-lnbw-->"),
 ("of3t-softbw", PRE +
  "**OF3T J2: the fused softmax backward, ~32.8 s, and it may ship without anyone touching C++.** "
  "3,888 softmax backwards on the tape (3,840 TRI_ATT + 48 APB) at ~5 verbs each = 19,440 verbs = "
  "41.0 s, falling to ~8.2 s. **Try `ttnn.moreh_softmax_backward` from the wheel FIRST** -- same "
  "expression, zero build cost. Two caveats: last-dim only (all callers qualify, confirm rather "
  "than assume), and **decide whether `SOFTMAX_BW_RENORM` is load-bearing BEFORE dropping it** -- "
  "it went default-on at pass 274 for a reason and costs 1.32 s. On the SHARED triangle path, so "
  "Moritz's inference constraint binds: A/B against an A/A floor on every model that executes it. "
  "Brief `workstreams/of3t-softbw.txt`. DONE_CHECK declared. <!--ws:of3t-softbw-->"),
 ("of3t-wheelbw", PRE +
  "**OF3T J4: nine one-line `*_bw` substitutions from the wheel, 5-15 s, one afternoon.** `mul "
  "scale add relu sigmoid silu reshape narrow concat` in `tt_bio/autograd.py` are hand-composed "
  "from 1-6 verbs each while `ttnn.mul_bw`, `ttnn.silu_bw`, `ttnn.sigmoid_bw`, `ttnn.relu_bw` and "
  "`ttnn.concat_bw` ship in 0.68.0. Small individually, on **every node in the tape** "
  "collectively. Nine ops, nine float64 gradient checks, no sampling -- the low-risk claim is what "
  "this job has to earn. `autograd.py` is CONTESTED (BCX is rewriting the head-verb backwards in "
  "it), so list the exact line ranges touched and re-run BCX's bar. Brief "
  "`workstreams/of3t-wheelbw.txt`. DONE_CHECK declared. <!--ws:of3t-wheelbw-->"),
]

# R203 retired the framing this row was written with, and prose is what a reader reads (R189).
STALE = ("Recorded: **one taped call switched fused triangle attention off for the whole process** "
         "— a state-dependent decline cached as a permanent refusal. If it still holds, part "
         "of the 6-7x is kernels we already have being bypassed under taping, a wiring fix not a "
         "kernel project. Count at runtime which of the ~11 fused forwards fire and which "
         "decompose; price the bypass in seconds of the 466.70 s step.")
FRESH = ("**Root-caused by `of3t-orchestrator` pass 445 (R203), so this row now PRICES rather than "
         "hunts: `ttnn.generic_op` has no backward, so all eleven fused kernels decline under "
         "taping BY DESIGN** — thirteen guard sites in eight modules, each with a correct "
         "comment saying so, plus a DRAM-for-L1 residency downgrade at `tenstorrent.py:9016`. A "
         "taped training step runs a decomposed, DRAM-resident model. What is owed is the "
         "measurement: which guards a real crop-384 step actually reaches, and the seconds each "
         "costs inside the 466.70 s.")


def verify():
    t = T.read_text()
    return [s for s, _ in ROWS if f"<!--ws:{s}-->" in t]


def apply():
    t = T.read_text()
    lines = t.split("\n")
    idx = [i for i, l in enumerate(lines) if ANCHOR in l]
    if len(idx) != 1:
        print(f"anchor {ANCHOR} found {len(idx)} times, refusing to guess"); return 1
    missing = [(s, r) for s, r in ROWS if f"<!--ws:{s}-->" not in t]
    if missing:
        lines[idx[0] + 1:idx[0] + 1] = [r for _, r in missing]
        t = "\n".join(lines)
    if STALE in t:
        t = t.replace(STALE, FRESH, 1)
    T.write_text(t)
    have = verify()
    print(f"applied {len(missing)} row(s); present now: {len(have)}/4 -> {have}")
    return 0 if len(have) == 4 else 1


if __name__ == "__main__":
    sys.exit(verify() and 0 if "--verify-only" in sys.argv and len(verify()) == 4 else apply())
