#!/usr/bin/env python3
"""Read a BindCraft 2 trajectory the way BC2's own run log reads it.

A BC2 campaign exposes the same stage three different ways and they do not agree:

  summary.csv          per-stage MEAN over the stage's rounds
  <traj>_losses.csv    every round, so mean / last / best are all recoverable
  run.log              "passed <stage> design stage i_pTM=X" -- the stage BEST

Comparing a device arm's summary.csv against a reference arm's run.log therefore
compares a mean against a best and invents a gap. On the shipped pdl1 pool the
reference's screen stage reads 0.598 as a mean, 0.220 as its last round and 0.870
as its best; only the last of those is what the log prints.

Usage:  stage_instrument.py <losses.csv> [<losses.csv> ...]
"""
import csv
import statistics as st
import sys
from collections import OrderedDict

# examples/pdl1.json, unedited
PLAN = {"screen": 50, "refine": 25, "anneal": 45, "harden": 5, "mutate": 15}


def _floats(rows, key):
    out = []
    for r in rows:
        v = (r.get(key) or "").strip()
        if v:
            try:
                out.append(float(v))
            except ValueError:
                pass
    return out


def read(path):
    rows = list(csv.DictReader(open(path)))
    stages = OrderedDict()
    for r in rows:
        stages.setdefault(r["phase"], []).append(r)
    return rows, stages


def report(path):
    rows, stages = read(path)
    print("=" * 84)
    print(f"{path}   rounds={len(rows)}")
    print(f"{'stage':8} {'n/plan':>9} {'iptm mean':>9} {'iptm last':>9} "
          f"{'iptm BEST':>9} {'ptm BEST':>9}")
    for s, rs in stages.items():
        ip = _floats(rs, "hPDL1.iptm")
        pt = _floats(rs, "hPDL1.ptm")
        plan = PLAN.get(s)
        # A non-finite loss BREAKS the stage instead of raising (trajectory.py:136),
        # so a short stage is a truncated experiment, not a clean verdict.
        short = "  <-- SHORT vs budget" if plan and len(rs) != plan else ""
        n = f"{len(rs)}/{plan if plan else '-'}"
        if ip:
            print(f"{s:8} {n:>9} {st.mean(ip):9.3f} {ip[-1]:9.3f} "
                  f"{max(ip):9.3f} {max(pt) if pt else float('nan'):9.3f}{short}")
        else:
            print(f"{s:8} {n:>9} {'--':>9} {'--':>9} {'--':>9} {'--':>9}{short}")
    sat = [s for s, rs in stages.items()
           if _floats(rs, "hPDL1.iptm")
           and all(v == 1.0 for v in _floats(rs, "hPDL1.iptm"))]
    if sat:
        print(f"  DEGENERATE: {', '.join(sat)} reports i_pTM exactly 1.0 on every round.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        report(p)
