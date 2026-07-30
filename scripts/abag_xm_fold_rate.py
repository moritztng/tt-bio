#!/usr/bin/env python3
"""Did restarting the labels loop slow folding down?

The labels campaign is running ~8 hot processes against a declared budget of 2 workers x 2 threads.
That overshoot only matters if it costs fold throughput, so compare per-fold wall_s for folds that
completed BEFORE the labels loop restarted against those completed after, per generator (generators
have very different costs, so pooling them would hide the effect).

Uses progress.jsonl wall_s, which is what the driver measured, not a derived estimate.
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# Labels loop restarted 09:54 on qb1 and 09:56 on qb2, 2026-07-28. progress.jsonl has no timestamp
# field, so split on the record ORDER relative to the last record present before the restart --
# passed in as a count so the split point is explicit rather than guessed.
CUT = int(sys.argv[1]) if len(sys.argv) > 1 else None
P = Path.home() / "abag_xm" / "tier_a" / "progress.jsonl"
recs = [json.loads(l) for l in open(P) if l.strip()]
ok = [r for r in recs if r.get("status") == "ok" and r.get("wall_s")]
print(f"{len(recs)} records, {len(ok)} ok with a wall_s")

if CUT is None:
    print("\npass the number of records that existed before the labels restart as argv[1]")
    print("per-generator wall_s over ALL ok records:")
    by = defaultdict(list)
    for r in ok:
        by[r["model"]].append(r["wall_s"])
    for g, v in sorted(by.items()):
        print(f"  {g:14} n={len(v):3} median={statistics.median(v):7.1f}s "
              f"mean={statistics.mean(v):7.1f}s")
    sys.exit(0)

before = [r for r in recs[:CUT] if r.get("status") == "ok" and r.get("wall_s")]
after = [r for r in recs[CUT:] if r.get("status") == "ok" and r.get("wall_s")]
print(f"split at record {CUT}: {len(before)} ok before, {len(after)} ok after\n")
gb, ga = defaultdict(list), defaultdict(list)
for r in before:
    gb[r["model"]].append(r["wall_s"])
for r in after:
    ga[r["model"]].append(r["wall_s"])

print(f"{'generator':14}{'n_before':>9}{'med_before':>12}{'n_after':>9}{'med_after':>11}{'change':>9}")
for g in sorted(set(gb) | set(ga)):
    b, a = gb.get(g, []), ga.get(g, [])
    mb = statistics.median(b) if b else float("nan")
    ma = statistics.median(a) if a else float("nan")
    ch = f"{100 * (ma - mb) / mb:+.0f}%" if b and a and mb else "--"
    print(f"{g:14}{len(b):>9}{mb:>12.1f}{len(a):>9}{ma:>11.1f}{ch:>9}")
print("\nSmall n after the cut -- treat as indicative, not a measurement, and re-run once more "
      "folds land.")
