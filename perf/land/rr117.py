#!/usr/bin/env python3
"""Score the round-robin 117 aa sweep: per-round pairs first, medians second.

The point of the round-robin is that a monotone host-load drift now lands on every arm, so the
honest statistic is the per-round ratio (arms minutes apart under the same load), not the ratio of
pooled medians (arms tens of minutes apart).
"""
import json, statistics, sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
ROUNDS = [1, 2, 3]
SETS = [("protenix-v2", ["L0", "L2", "L4"]), ("opendde", ["L0o", "L4o"])]


def load(tag, model):
    p = OUT / f"{tag}_{model}_117.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


for model, arms in SETS:
    print(f"\n=== {model}, 117 aa, qb1 card 3, ttnn 0.67.4 ===")
    base = arms[0]
    per_round = {a: [] for a in arms}
    ratios = {a: [] for a in arms}
    for r in ROUNDS:
        recs = {a: load(f"r{r}{a}", model) for a in arms}
        if any(v is None for v in recs.values()):
            print(f"  round {r}: incomplete, skipped")
            continue
        b = recs[base]["warm_median_s"]
        cells = []
        for a in arms:
            m = recs[a]["warm_median_s"]
            per_round[a].append(m)
            ratios[a].append(b / m)
            cells.append(f"{a}={m:.3f}s ({b/m:.4f}x)")
        print(f"  round {r}: " + "  ".join(cells))
    print("  ---")
    for a in arms:
        if not per_round[a]:
            continue
        med = statistics.median(per_round[a])
        lo, hi = min(per_round[a]), max(per_round[a])
        rmed = statistics.median(ratios[a])
        print(f"  {a:4s} median {med:.3f}s  [min {lo:.3f}, max {hi:.3f}, spread "
              f"{100*(hi-lo)/med:.1f}%]   paired ratio vs {base}: median {rmed:.4f}x  "
              f"all {[f'{x:.4f}' for x in ratios[a]]}")
    # plddt is a determinism check, not a perf one, but it costs nothing to print
    for a in arms:
        vals = {load(f"r{r}{a}", model)["confidence"]["plddt"]
                for r in ROUNDS if load(f"r{r}{a}", model)}
        if vals:
            print(f"  {a:4s} pLDDT across rounds: {sorted(vals)}")
