#!/usr/bin/env python3
"""Does a big resident set alone reproduce the harness's 3x on the loss heads? Card-free.

`of3t-xroof` measured that the step's own resident set binds the host crossing (a 4 GiB
balloon took a crossing 0.348 -> 0.765 s at flat loadavg). `step_exact_off_384.json` records
`avail_at_start_gib: 7.741`, so its loss heads ran in a process that was already squeezing
the page cache. This holds `--gib` of touched pages and re-times `lossprofile`'s region.
"""
import argparse, subprocess, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--gib", type=float, default=0.0)
ap.add_argument("--tokens", type=int, default=384)
ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()

hold = None
if a.gib > 0:
    n = int(a.gib * (1024 ** 3) / 8)
    hold = np.empty(n, np.float64)
    hold[:] = 1.0            # touch every page; np.empty alone is not resident
    print(f"holding {a.gib} GiB resident, sum {float(hold[::4096].sum()):.0f}", flush=True)

sys.argv = ["lossprofile", "--tokens", str(a.tokens), "--reps", str(a.reps),
            "--out", str(REPO / f"perf/of3t_p10host/out/lossprofile_b{a.gib:g}.json")]
sys.path.insert(0, str(REPO / "perf/of3t_p10host"))
import lossprofile
lossprofile.main()
print(f"(balloon still held: {hold is not None})")
