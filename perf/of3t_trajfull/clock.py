#!/usr/bin/env python3
"""AICLK sampled DURING the arm, from `run_trajfull.sh`'s 60 s sampler.

`tt-smi -s` reports AICLK as a hex string ("0x546"), and the sampler runs with
TT_VISIBLE_DEVICES set so the line is a reading of the chip that did the work rather than of
its board partner. A number without a clock is not a measurement on this hardware.
"""
import json
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/ttuser/of3t_runs/trajwide/aiclk_trajfull.log"
mhz, stamps = [], []
for line in open(path):
    m = re.match(r"(\S+)\s", line)
    v = re.search(r'"AICLK":"(0x[0-9a-fA-F]+|\d+)"', line)
    if not (m and v):
        continue
    stamps.append(m.group(1))
    mhz.append(int(v.group(1), 16) if v.group(1).startswith("0x") else int(v.group(1)))
mhz.sort()
n = len(mhz)
print(json.dumps({
    "samples": n, "window_start": stamps[0] if stamps else None,
    "window_end": stamps[-1] if stamps else None,
    "median_mhz": (mhz[n // 2] if n else None),
    "min_mhz": (mhz[0] if n else None), "max_mhz": (mhz[-1] if n else None),
    "samples_below_1200_mhz": sum(1 for x in mhz if x < 1200),
    "log": path}, indent=1))
