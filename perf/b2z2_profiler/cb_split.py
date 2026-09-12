#!/usr/bin/env python3
"""The math thread's wait/compute split for one profiled block, per core.

`--enable-sum-profiling` adds two compute-thread stall accumulators to the ops report,
`DEVICE COMPUTE CB WAIT FRONT [ns]` (blocked on input tiles) and
`DEVICE COMPUTE CB RESERVE BACK [ns]` (blocked on room to write output). Both are SUMMED
OVER THE CORES THAT RAN THE OP, while `DEVICE TRISC1 KERNEL DURATION [ns]` is a duration.
Adding them straight out of the CSV is how you get a stall that is 4400 % of the thread it
stalls. Divide each op by its own `CORE COUNT` first -- not the block's, because op core
counts differ (72 and 64 in the same Pairformer block).

The divisor is exact only when every core of the op ran the compute kernel for the whole op.
`ratio_gt_1` below is the honest error bar on that: ops where the per-core stall comes out
longer than the per-core TRISC1 residency it is a part of.

Reads the same fence convention as `perf/b2z_kernel_census/census_report.py`: the profiled
region is what lies between the last two runs of >=3 32x32 unary ops.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

FENCE_DIM, FENCE_N = 32, 3
WAIT, RES = "DEVICE COMPUTE CB WAIT FRONT [ns]", "DEVICE COMPUTE CB RESERVE BACK [ns]"
TRISC1, CORES = "DEVICE TRISC1 KERNEL DURATION [ns]", "CORE COUNT"


def f(row, key, default=0.0):
    v = row.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def is_fence(row):
    if not row.get("OP CODE", "").startswith("Unary"):
        return False
    sh = tuple(int(f(row, f"INPUT_0_{d}", 0)) for d in ("Y", "X"))
    return sh in ((FENCE_DIM, FENCE_DIM), (0, 0))


def find_region(rows):
    runs, i = [], 0
    while i < len(rows):
        if is_fence(rows[i]):
            j = i
            while j < len(rows) and is_fence(rows[j]):
                j += 1
            if j - i >= FENCE_N:
                runs.append((i, j))
            i = j
        else:
            i += 1
    if len(runs) < 2:
        raise SystemExit(f"expected >=2 fence runs, found {len(runs)}")
    return rows[runs[-2][1]:runs[-1][0]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", required=True)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv)))
    meta = json.loads(a.meta.read_text())
    reps = int(meta["env"]["reps"])
    region = find_region(rows)
    if len(region) % reps:
        raise SystemExit(f"{len(region)} ops in the region is not divisible by {reps} reps")
    n_ops = len(region) // reps
    chunks = [region[i * n_ops:(i + 1) * n_ops] for i in range(reps)]

    ns_per_cycle = st.median([f(r, "DEVICE FW DURATION [ns]") /
                              (f(r, "DEVICE FW END CYCLE") - f(r, "DEVICE FW START CYCLE"))
                              for r in region
                              if f(r, "DEVICE FW END CYCLE") > f(r, "DEVICE FW START CYCLE")])

    per_rep, by_code, bad = [], defaultdict(lambda: [0.0, 0.0, 0.0, 0]), 0
    for ch in chunks:
        span = (f(ch[-1], "DEVICE FW END CYCLE") - f(ch[0], "DEVICE FW START CYCLE")) * ns_per_cycle
        kern = sum(f(r, "DEVICE KERNEL DURATION [ns]") for r in ch)
        t1 = w = rb = 0.0
        for r in ch:
            n = max(f(r, CORES), 1.0)
            t, wi, wo = f(r, TRISC1), f(r, WAIT) / n, f(r, RES) / n
            t1, w, rb = t1 + t, w + wi, rb + wo
            if wi + wo > t and t > 0:
                bad += 1
            c = by_code[r.get("OP CODE", "?")]
            c[0] += t; c[1] += wi; c[2] += wo; c[3] += 1
        per_rep.append({"span_ms": span / 1e6, "kernel_ms": kern / 1e6, "trisc1_ms": t1 / 1e6,
                        "wait_in_ms": w / 1e6, "wait_out_ms": rb / 1e6,
                        "compute_ms": (t1 - w - rb) / 1e6})

    med = {k: st.median([r[k] for r in per_rep]) for k in per_rep[0]}
    span = med["span_ms"]
    out = {"label": a.label, "csv": str(a.csv), "ops_per_block": n_ops, "reps": reps,
           "ns_per_cycle": ns_per_cycle, "per_rep_ms": per_rep, "median_ms": med,
           "synced_wall_ms_per_call": meta.get("synced_wall_ms_per_call"),
           "trisc1_pct": {k: 100.0 * med[k] / med["trisc1_ms"]
                          for k in ("wait_in_ms", "wait_out_ms", "compute_ms")},
           "non_resident_ms": span - med["trisc1_ms"],
           "gap_ms": span - med["kernel_ms"],
           "ops_where_stall_exceeds_residency": bad // reps,
           "by_op_code": {k: {"n": v[3] // reps, "trisc1_ms": v[0] / reps / 1e6,
                              "wait_in_ms": v[1] / reps / 1e6, "wait_out_ms": v[2] / reps / 1e6}
                          for k, v in sorted(by_code.items(), key=lambda kv: -kv[1][0])}}
    a.out.write_text(json.dumps(out, indent=1))

    print(f"{a.label}: {n_ops} ops/block, span {span:.4f} ms")
    print(f"  TRISC1 resident        {med['trisc1_ms']:9.4f} ms  "
          f"{100 * med['trisc1_ms'] / span:5.1f} % of span")
    for k, lbl in (("wait_in_ms", "CB wait-front (input) "),
                   ("wait_out_ms", "CB reserve-back (out) "),
                   ("compute_ms", "not stalled on a CB   ")):
        print(f"    {lbl} {med[k]:9.4f} ms  {out['trisc1_pct'][k]:5.1f} % of TRISC1")
    print(f"  non-resident           {out['non_resident_ms']:9.4f} ms")
    print(f"  gap (span - kernels)   {out['gap_ms']:9.4f} ms")
    print(f"  divisor error bar: {out['ops_where_stall_exceeds_residency']}/{n_ops} ops "
          f"have per-core stall > per-core TRISC1")
    print("  by op code (ms/block, per core):")
    for code, v in list(out["by_op_code"].items())[:8]:
        print(f"    {code[:32]:32s} n={v['n']:3d}  TRISC1 {v['trisc1_ms']:8.3f}  "
              f"in {v['wait_in_ms']:8.3f} ({100 * v['wait_in_ms'] / max(v['trisc1_ms'], 1e-9):4.1f} %)"
              f"  out {v['wait_out_ms']:7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
