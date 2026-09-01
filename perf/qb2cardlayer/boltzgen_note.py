#!/usr/bin/env python3
"""Compose the boltzgen cell note from the draw index, so no number in it is typed by hand.

The retired-box control is the v0.7.2 release gate of 2026-08-28, which ran this same protocol on
the previous physical qb2 and read 0.021410 designs/s (state/tt-bio-release-v0-7-2.md).
"""
import datetime, pathlib, statistics

CONTROL = 0.021410
rows = pathlib.Path(
    "/home/ttuser/.coworker/wt/qb2-card-layer-baseline-reseed/perf/qb2cardlayer/draws.tsv"
).read_text().splitlines()
vals = []
for r in rows:
    f = dict(p.split("=", 1) for p in r.split("\t") if "=" in p)
    if f.get("model") == "boltzgen" and f.get("value") not in (None, "NA"):
        vals.append(float(f["value"]))
if len(vals) < 4:
    raise SystemExit(f"only {len(vals)} usable boltzgen draws, refusing to seed a cell on that")
med, band = statistics.median(vals), (max(vals) / min(vals) - 1) * 100
print(
    f"Reseeded {datetime.datetime.now(datetime.UTC):%Y-%m-%d} on the replacement qb2, p300c card 0, "
    f"benchlocked, one fresh process per draw: {len(vals)} draws, median {med:.6f} designs/s, "
    f"band {band:.1f}%. This cell is the reseed run own draw, not the median. The old cell 0.017057 "
    f"dated 2026-07-18 was 25% low, the direction that hides a regression, and the gap predates the "
    f"2026-09-01 qb2 hardware swap: the retired box read {CONTROL:.6f} designs/s at v0.7.2 on "
    f"2026-08-28, {(med/CONTROL-1)*100:+.1f}% from the draw median here. boltzgen is a single-shot "
    f"cold-inflated proxy, each rep re-pays first-kernel compile, so it carries a wider spread than "
    f"the warm fold legs. Draws perf/qb2cardlayer/draws.tsv, see "
    f"state/qb2-card-layer-baseline-reseed.md and state/qb2-new-hardware-baseline-crosscheck.md."
)
