"""Compose p27_sweep.log's 20-step rows into real 200-timestep design numbers.

A 200-timestep design pays the one-time gather-table build once and then 199 steady
steps, so `build + 199 * steady_per_step` is the run, and quoting the steady per-step
alone is what would overstate the change (see p27_real_design_timing.py's docstring).
The build is a property of the (1,L,L,16) table, not of the batch, so both batches'
measurements of it are pooled into one estimate per fixture and tree.

  python3 scripts/rfd3_port/p27_reduce_sweep.py [--log <path>]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORDER = [("iai40", "40 residues"), ("iai80", "80 residues"), ("iai150", "150 residues"),
         ("mpro", "Mpro + nirmatrelvir"), ("iai250", "250 residues")]

ap = argparse.ArgumentParser()
ap.add_argument("--log", type=Path, default=ROOT / "scripts/rfd3_port/p27_sweep.log")
ap.add_argument("--steps", type=int, default=200, help="timesteps the run is composed to")
args = ap.parse_args()

rows = [json.loads(line.split("RESULT ", 1)[1])
        for line in args.log.read_text().splitlines() if "RESULT " in line]
steps_200 = args.steps - 1

cells: dict[tuple[str, str, int], list[dict]] = {}
builds: dict[tuple[str, str], list[float]] = {}
for row in rows:
    if row["steps"] != 20:          # anchors and smokes are reported separately
        continue
    fixture, variant, _ = row["tag"].split("/")
    cells.setdefault((fixture, variant, row["D"]), []).append(row)
    builds.setdefault((fixture, variant), []).append(row["build_s"])


def composed(fixture: str, variant: str, batch: int) -> tuple[float, float, float, int]:
    reps = cells[(fixture, variant, batch)]
    steady = statistics.mean(r["steady_ms_per_step"] for r in reps) / 1000
    # a negative build measurement is the second run reading warmer, not a negative
    # setup cost; the base tree has no table to build at all
    build = max(statistics.mean(builds[(fixture, variant)]), 0.0)
    total = build + steps_200 * steady
    return total, steady, build, len(reps)


print(f"composed to {args.steps} timesteps ({steps_200} steps) from 20-step measurements\n")
head = (f"{'fixture':<20} {'L':>5} {'D':>2} {'base_s':>8} {'fix_s':>8} {'speedup':>8} "
        f"{'base_ms/st':>10} {'fix_ms/st':>10} {'build_s':>8} {'reps':>5}")
print(head)
for fixture, _ in ORDER:
    for batch in (1, 8):
        if (fixture, "fix", batch) not in cells or (fixture, "base", batch) not in cells:
            continue
        base_t, base_st, _, n_b = composed(fixture, "base", batch)
        fix_t, fix_st, build, n_f = composed(fixture, "fix", batch)
        length = cells[(fixture, "fix", batch)][0]["L"]
        print(f"{fixture:<20} {length:>5} {batch:>2} {base_t:>8.2f} {fix_t:>8.2f} "
              f"{base_t / fix_t:>7.3f}x {base_st * 1000:>10.1f} {fix_st * 1000:>10.1f} "
              f"{build:>8.3f} {min(n_b, n_f):>5}")

print("\ndesigns/sec at the composed step count (what docs/rfd3-design.md quotes)")
print(f"{'fixture':<20} {'L':>5} {'base_D1':>9} {'base_D8':>9} {'fix_D1':>9} {'fix_D8':>9} "
      f"{'fix_8v1':>8} {'break_even_steps_D1':>20} {'D8':>6}")
for fixture, _ in ORDER:
    if (fixture, "fix", 1) not in cells:
        continue
    out = {}
    for variant in ("base", "fix"):
        for batch in (1, 8):
            total, steady, build, _ = composed(fixture, variant, batch)
            out[(variant, batch)] = (batch / total, steady, build)
    length = cells[(fixture, "fix", 1)][0]["L"]
    be = []
    for batch in (1, 8):
        saving = out[("base", batch)][1] - out[("fix", batch)][1]
        build = out[("fix", batch)][2]
        be.append(build / saving if saving > 0 else float("inf"))
    print(f"{fixture:<20} {length:>5} {out[('base', 1)][0]:>9.4f} {out[('base', 8)][0]:>9.4f} "
          f"{out[('fix', 1)][0]:>9.4f} {out[('fix', 8)][0]:>9.4f} "
          f"{out[('fix', 8)][0] / out[('fix', 1)][0]:>7.2f}x {be[0]:>20.1f} {be[1]:>6.1f}")

anchors = [r for r in rows if r["steps"] == 199]
if anchors:
    print("\nend-to-end 200-timestep anchors (no composition)")
    for row in anchors:
        print(f"  {row['tag']:<24} L={row['L']} D={row['D']} run_s={row['run_s']:.2f} "
              f"ms/step={row['ms_per_step']:.1f} finite={row['finite']}")
