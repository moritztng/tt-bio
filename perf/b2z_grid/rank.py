#!/usr/bin/env python3
"""Rank one unit's ops by idle core-seconds, from a Tracy ops_perf_results CSV.

The census window is the tail of the CSV: `--repeats` identical repeats of the same unit. The
period is recovered by finding the smallest L for which the last `repeats` groups of L rows have
identical OP CODE sequences, so no marker bookkeeping is needed and a mis-split is impossible to
miss (it would not repeat).

Idle core-seconds per op = (available - used) / available * device_kernel_duration.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics as st
from pathlib import Path

CORES = "CORE COUNT"
AVAIL = "AVAILABLE WORKER CORE COUNT"
DUR = "DEVICE KERNEL DURATION [ns]"


def find_period(codes: list[str], repeats: int, lo: int = 20) -> int:
    n = len(codes)
    for L in range(lo, n // repeats + 1):
        tail = codes[n - repeats * L:]
        if all(tail[i * L:(i + 1) * L] == tail[:L] for i in range(1, repeats)):
            return L
    raise SystemExit(f"no repeating period found in {n} rows for {repeats} repeats")


def shape(r: str, i: int) -> str:
    return "x".join(r[f"INPUT_{i}_{d}_PAD[LOGICAL]"] for d in "WZYX")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--calls-per-fold", type=int, default=280)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(a.csv)) if r.get(DUR)]
    codes = [r["OP CODE"] for r in rows]
    L = find_period(codes, a.repeats)
    n = len(rows)
    reps = [rows[n - a.repeats * L + i * L: n - a.repeats * L + (i + 1) * L]
            for i in range(a.repeats)]
    print(f"period {L} ops/unit, {a.repeats} repeats, {n} rows total")

    # median over repeats, per op position
    ops = []
    for j in range(L):
        rs = [rep[j] for rep in reps]
        r0 = rs[0]
        dur = st.median(float(x[DUR]) for x in rs)
        used, avail = int(r0[CORES]), int(r0[AVAIL])
        ops.append({
            "i": j, "code": r0["OP CODE"], "used": used, "avail": avail,
            "util": used / avail, "ns": dur,
            "idle_ns": (avail - used) / avail * dur,
            "in0": shape(r0, 0), "in0_mem": r0.get("INPUT_0_MEMORY", ""),
            "in0_dt": r0.get("INPUT_0_DATATYPE", ""),
            "out_mem": r0.get("OUTPUT_0_MEMORY", ""),
            "par": r0.get("PARALLELIZATION STRATEGY", ""),
            "fid": r0.get("MATH FIDELITY", ""),
        })

    tot_ns = sum(o["ns"] for o in ops)
    idle_ns = sum(o["idle_ns"] for o in ops)
    wmean = 1 - idle_ns / tot_ns
    under = [o for o in ops if o["util"] < 0.5]
    per_fold = a.calls_per_fold

    print(f"\nunit device time {tot_ns/1e6:.4f} ms over {L} ops  "
          f"({per_fold} calls/fold = {tot_ns*per_fold/1e9:.3f} s)")
    print(f"duration-weighted mean core utilization {100*wmean:.2f} %")
    print(f"UNDER-GRID-OPS (util < 0.5): {len(under)} of {L}  "
          f"({100*sum(o['ns'] for o in under)/tot_ns:.1f} % of unit device time)")
    print(f"idle core-fraction-seconds per fold: {idle_ns*per_fold/1e9:.4f} s")

    byclass: dict = collections.defaultdict(lambda: {"n": 0, "ns": 0.0, "idle": 0.0})
    for o in ops:
        b = byclass[o["code"]]
        b["n"] += 1
        b["ns"] += o["ns"]
        b["idle"] += o["idle_ns"]
    print("\nper op class (sorted by idle core-seconds/fold):")
    print(f"{'op class':36s} {'n':>4s} {'ms/unit':>9s} {'s/fold':>8s} {'util%':>7s} {'idle s/fold':>12s}")
    for k, b in sorted(byclass.items(), key=lambda kv: -kv[1]["idle"]):
        print(f"{k:36s} {b['n']:4d} {b['ns']/1e6:9.4f} {b['ns']*per_fold/1e9:8.3f} "
              f"{100*(1-b['idle']/b['ns']):7.2f} {b['idle']*per_fold/1e9:12.4f}")

    print("\ntop 20 individual ops by idle core-seconds/fold:")
    for o in sorted(ops, key=lambda x: -x["idle_ns"])[:20]:
        print(f"  {o['code']:30s} {o['used']:3d}/{o['avail']:3d} "
              f"{o['ns']/1e3:9.2f} us  idle {o['idle_ns']*per_fold/1e9:7.4f} s/fold  "
              f"in0 {o['in0']:18s} {o['in0_dt']:10s} {o['in0_mem'][:28]}")

    res = {"period": L, "repeats": a.repeats, "calls_per_fold": per_fold,
           "unit_ms": tot_ns / 1e6, "unit_s_per_fold": tot_ns * per_fold / 1e9,
           "weighted_mean_util": wmean, "under_grid_ops": len(under),
           "idle_core_seconds_per_fold": idle_ns * per_fold / 1e9, "ops": ops}
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
