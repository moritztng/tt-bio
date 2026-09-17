#!/usr/bin/env python3
"""Where the 64-of-110 core pin the matmul-key row handed over actually lives.

`c12-matmul-key-attribution` handed off `_triangle_mul_program_config(16)` engaging 64 of 110
cores as a 1.72x occupancy headroom on an arithmetic-bound op. This reads the core count of every
program in the same executed graph the per-site rates come from, grouped by op code, so the pin is
located by measurement rather than by reading the config constructor.
"""
import csv, gzip, sys
from collections import Counter, defaultdict
from pathlib import Path

base = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/gr")
for name in ("pfl_prof2", "msal_prof"):
    rows = list(csv.DictReader(gzip.open(base / name / "ops.csv.gz", "rt")))
    print("=====", name, len(rows), "programs")
    by = defaultdict(Counter)
    dur = defaultdict(float)
    for r in rows:
        c = int(r["CORE COUNT"])
        by[r["OP CODE"]][c] += 1
        dur[(r["OP CODE"], c)] += float(r["DEVICE KERNEL DURATION [ns]"]) / 1e6
    for op in sorted(by, key=lambda o: -sum(by[o].values())):
        parts = ", ".join("%d cores x%d (%.3f ms)" % (c, n, dur[(op, c)])
                          for c, n in sorted(by[op].items()))
        print("  %-32s %s" % (op, parts))
