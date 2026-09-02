#!/usr/bin/env python3
"""Build the baseline note for one leg out of the draw index, so no number in a committed note is
typed by hand. A median-of-N written next to the logs instead of computed from them is a documented
past failure of this exact protocol (qb2-new-hardware-baseline-crosscheck, "Defect found and fixed").
"""
import pathlib, statistics, sys

WHY = {
  "rf3": "RF3 shipped as a predict --model choice in v0.6.6 and has p150a cells on qb1 and pc, but "
         "never had a p300c cell at either layer, so a full-SPECS gate on qb2 failed on it as NO "
         "BASELINE. Same light TRPCAGE single-seq fold protocol as every other fold leg here "
         "(1 recycle / 10 steps / 1 sample). NOT the site perf-page 512 aa cdk2x2 cell, which is a "
         "different fixture at the shipped 10 recycles / 50 steps.",
  "esmc-300m-single": "The batch-1 ESMC-300M embed leg, the path the ttnn trace capture replays. It "
         "had p150a cells on qb1 and pc but no p300c cell at either layer, so a full-SPECS gate on "
         "qb2 failed on it as NO BASELINE. The measurement child opens the device with a 256MB "
         "trace region, per its SPECS entry, or the forward stays eager and the leg measures the "
         "wrong path.",
  "nesso1": "tt-bio affinity / nesso1 had a p150a cell on qb1 but no p300c cell at either layer, so "
         "a full-SPECS gate on qb2 failed on it as NO BASELINE. Same FKBP12+SB3 fixture as the "
         "boltz2-affinity leg at shipped CLI defaults. Do not quote the ratio between the two as a "
         "headline: that number is a 512 aa measurement and this fixture is 107 aa.",
}

model = sys.argv[1]
idx = pathlib.Path("/home/ttuser/.coworker/wt/p300c-baseline-coverage-gap/perf/qb2p300cgap/draws.tsv")
v = []
for r in idx.read_text().splitlines():
    f = dict(p.split("=", 1) for p in r.split("\t") if "=" in p)
    if f.get("model") == model and f.get("value") not in (None, "NA"):
        v.append(float(f["value"]))
if not v:
    sys.exit(f"{model}: no drawn values in {idx}")
draws = " / ".join(f"{x:.6f}" for x in v)
print(f"seed p300c: {WHY[model]} Measured on qb2 physical card 2 (TT_VISIBLE_DEVICES=2), "
      f"benchlocked, one fresh perf_regression.py process per draw. "
      f"{len(v)} draws before this one: {draws} (median {statistics.median(v):.6f}, band "
      f"{(max(v)/min(v)-1)*100:.1f} %). The cell is this reseed run's own measurement, not the draw "
      f"median, so it is a number some run actually produced.")
