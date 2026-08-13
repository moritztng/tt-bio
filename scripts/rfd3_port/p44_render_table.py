"""Render `docs/rfd3-design.md`'s throughput table from the sweep's JSONL.

The doc cell is the **faster** of the point's two interleaved rounds, not their mean: a
co-tenant can only make a run slower, so on a box that is not exclusively ours the max is the
closer estimate of the quiet-card rate, and it is the same rule in both columns so the
`batch 8 vs 1` ratio stays a ratio of like things. The off arm never reaches the doc -- it is
here so the state doc can say what the flip bought on this chip, separately from what a month of
main bought.

`--jsonl` may be given more than once. A later file's (point, batch, arm) rounds *replace* an
earlier file's, so a re-measurement of one column on a quiet card supersedes the contaminated
runs from the full sweep without being averaged into them.

    python3 scripts/rfd3_port/p44_render_table.py --jsonl perf/p44/throughput.jsonl \
        --jsonl perf/p44/throughput_quiet_b1.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, action="append", default=None)
    a = ap.parse_args()
    files = a.jsonl or [Path("perf/p44/throughput.jsonl")]

    rate: dict[tuple, list[float]] = {}   # (point, batch, arm) -> [designs/sec per round]
    load: dict[tuple, list[float]] = {}
    meta: dict[int, tuple[str, int]] = {}
    parity, runs = [], []
    for path in files:
        seen: set[tuple] = set()
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["arm"] == "parity":
                parity.append(r)
                continue
            runs.append(r)
            key = (r["point"], r["batch"], r["arm"])
            if key not in seen:      # first round of this key in this file: supersede earlier files
                seen.add(key)
                rate[key], load[key] = [], []
            rate[key].append(r["designs_per_sec"])
            load[key].append(max(r.get("load_before", 0.0), r.get("load_after", 0.0)))
            meta[r["point"]] = (r["name"], r["atoms"])

    print("| design | atoms | batch 1 | batch 8 | batch 8 vs 1 |")
    print("|---|---:|---:|---:|---:|")
    for p in sorted(meta):
        name, atoms = meta[p]
        cells = [max(rate.get((p, b, "on"), []), default=None) for b in (1, 8)]
        if None in cells:
            print(f"| {name} | {atoms} | INCOMPLETE | | |")
            continue
        one, eight = cells
        first = f"{one:.4f} designs/sec" if p == min(meta) else f"{one:.4f}"
        print(f"| {name} | {atoms} | {first} | {eight:.4f} | {eight / one:.2f}x |")

    print("\noff vs on, same chip, faster of two rounds (state doc only):")
    print("| design | atoms | batch | off | on | delta | A/A floor | peak load |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for p in sorted(meta):
        name, atoms = meta[p]
        for b in (1, 8):
            off, on = rate.get((p, b, "off"), []), rate.get((p, b, "on"), [])
            if not off or not on:
                continue
            mo, mn = max(off), max(on)
            # The A/A floor is the spread between one arm's own two rounds, taken as the wider
            # of the two arms -- a delta inside it is not a measurement.
            spreads = [(max(v) - min(v)) / min(v) * 100 for v in (off, on) if len(v) > 1]
            floor = f"{max(spreads):.2f} %" if spreads else "1 round"
            peak = max(load.get((p, b, "off"), [0]) + load.get((p, b, "on"), [0]))
            print(f"| {name} | {atoms} | {b} | {mo:.4f} | {mn:.4f} | "
                  f"{(mn / mo - 1) * 100:+.2f} % | {floor} | {peak:.1f} |")

    bad = [r for r in parity if not r["equal"]]
    print(f"\nparity checks: {len(parity)}, all equal: {not bad}"
          + (f"  MISMATCHES: {bad}" if bad else ""))
    nonfinite = [r for r in runs if not r["finite"]]
    print(f"finite outputs: {len(runs) - len(nonfinite)}/{len(runs)}")
    served = {(r["point"], r["batch"]): (r["sparse_served"], r["fused_served"])
              for r in runs if r["arm"] == "on"}
    print(f"on-arm served (sparse/fused) per point,batch: {served}")


if __name__ == "__main__":
    main()
