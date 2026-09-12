#!/usr/bin/env python3
"""Attribute a Pairformer block's input-tile wait to op codes.

The zone split in ``arrival_split.py`` only covers tt-bio's own ``generic_op`` matmul
kernels. This says how much of the block's wait those programs carry in the first place,
so the split is quoted against the right denominator.

Both stall columns are summed over the cores that ran the op, while TRISC1 residency is a
duration, so each op is divided by its OWN ``CORE COUNT`` before anything is added up
(``state/b2z2/PROFILER-WHGLX.md``).
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

COLS = (("dur", "DEVICE KERNEL DURATION [ns]", False),
        ("trisc1", "DEVICE TRISC1 KERNEL DURATION [ns]", False),
        ("wait", "DEVICE COMPUTE CB WAIT FRONT [ns]", True),
        ("res", "DEVICE COMPUTE CB RESERVE BACK [ns]", True))


def fl(r, k):
    try:
        return float(r.get(k, "").strip())
    except ValueError:
        return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ops_csv", type=Path)
    ap.add_argument("--first", type=int, required=True, help="first row index of the window")
    ap.add_argument("--last", type=int, required=True, help="last row index, inclusive")
    ap.add_argument("--reps", type=int, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    rows = list(csv.DictReader(a.ops_csv.open()))[a.first:a.last + 1]
    agg = collections.defaultdict(lambda: dict.fromkeys([c[0] for c in COLS] + ["n"], 0.0))
    tot = dict.fromkeys([c[0] for c in COLS] + ["n"], 0.0)
    for r in rows:
        cc = max(fl(r, "CORE COUNT"), 1.0)
        d = agg[r["OP CODE"]]
        for key, col, percore in COLS:
            v = fl(r, col) / (cc if percore else 1.0)
            d[key] += v
            tot[key] += v
        d["n"] += 1
        tot["n"] += 1

    R = float(a.reps)
    hdr = f"{'op code':40s} {'n':>6s} {'kernel_us':>10s} {'TRISC1_us':>10s} {'wait_us':>9s} {'res_us':>8s}"
    print(f"per block, WH, us (stalls per core):\n{hdr}")
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1]["wait"]):
        print(f"{k:40s} {v['n']/R:6.1f} {v['dur']/R/1e3:10.1f} {v['trisc1']/R/1e3:10.1f} "
              f"{v['wait']/R/1e3:9.1f} {v['res']/R/1e3:8.1f}")
    print(f"{'TOTAL':40s} {tot['n']/R:6.1f} {tot['dur']/R/1e3:10.1f} {tot['trisc1']/R/1e3:10.1f} "
          f"{tot['wait']/R/1e3:9.1f} {tot['res']/R/1e3:8.1f}")
    out = {"per_block_us": {k: {kk: round(vv / R / 1e3, 4) if kk != "n" else vv / R
                                for kk, vv in v.items()} for k, v in agg.items()},
           "total_us": {k: round(v / R / 1e3, 4) if k != "n" else v / R for k, v in tot.items()},
           "reps": a.reps, "window": [a.first, a.last]}
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
