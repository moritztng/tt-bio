#!/usr/bin/env python3
"""AICLK over a run's window, all four nodes, from the 1 Hz sysfs sampler.

Prints one line per node. The fold's OWN window is reconstructed by clkwin.py afterwards; this
is the process-wide read, which is diluted by import and featurization and is here only so a
run that never boosted is visible immediately.
"""
import sys
from pathlib import Path

out, s, e = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
rows = [ln.split("\t") for ln in (out / "aiclk.tsv").read_text().splitlines() if "\t" in ln]
for i in range(4):
    v = [int(r[i + 1]) for r in rows
         if len(r) > i + 1 and r[i + 1].strip() and s <= int(r[0]) <= e]
    if v:
        print(f"AICLK whole process card{i}: n={len(v)} mean={sum(v)/len(v):.0f} MHz "
              f"min={min(v)} max={max(v)}")
