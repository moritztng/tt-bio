"""Render `docs/rfd3-design.md`'s throughput table from the sweep's JSONL.

The doc cell is the mean of the point's two interleaved rounds. The off arm never reaches the
doc -- it is here so the state doc can say what the flip bought on this chip, separately from
what the host bought.

    python3 scripts/rfd3_port/p44_render_table.py --jsonl perf/p44/throughput.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, default=Path("perf/p44/throughput.jsonl"))
    a = ap.parse_args()

    rows = [json.loads(l) for l in a.jsonl.read_text().splitlines() if l.strip()]
    runs = [r for r in rows if r["arm"] in ("off", "on")]
    parity = [r for r in rows if r["arm"] == "parity"]

    rate = defaultdict(list)          # (point, batch, arm) -> [designs/sec per round]
    meta = {}
    for r in runs:
        rate[(r["point"], r["batch"], r["arm"])].append(r["designs_per_sec"])
        meta[r["point"]] = (r["name"], r["atoms"])

    print("| design | atoms | batch 1 | batch 8 | batch 8 vs 1 |")
    print("|---|---:|---:|---:|---:|")
    for p in sorted(meta):
        name, atoms = meta[p]
        cells = []
        for b in (1, 8):
            v = rate.get((p, b, "on"), [])
            cells.append(sum(v) / len(v) if v else None)
        if None in cells:
            print(f"| {name} | {atoms} | INCOMPLETE | | |")
            continue
        one, eight = cells
        first = f"{one:.4f} designs/sec" if p == min(meta) else f"{one:.4f}"
        print(f"| {name} | {atoms} | {first} | {eight:.4f} | {eight / one:.2f}x |")

    print("\noff vs on, same chip, mean of two rounds (state doc only):")
    print("| design | atoms | batch | off | on | delta | A/A floor |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for p in sorted(meta):
        name, atoms = meta[p]
        for b in (1, 8):
            off, on = rate.get((p, b, "off"), []), rate.get((p, b, "on"), [])
            if not off or not on:
                continue
            mo, mn = sum(off) / len(off), sum(on) / len(on)
            # The A/A floor is the spread between one arm's own two rounds, taken as the wider
            # of the two arms -- a delta inside it is not a measurement.
            spreads = [(max(v) - min(v)) / min(v) * 100 for v in (off, on) if len(v) > 1]
            floor = f"{max(spreads):.2f} %" if spreads else "1 round"
            print(f"| {name} | {atoms} | {b} | {mo:.4f} | {mn:.4f} | "
                  f"{(mn / mo - 1) * 100:+.2f} % | {floor} |")

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
