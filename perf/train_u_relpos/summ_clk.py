#!/usr/bin/env python3
"""Summarise a streamed AICLK log over a time window: summ_clk.py <log> <startZ> <endZ>"""
import statistics as st, sys
from collections import defaultdict
log, lo, hi = sys.argv[1], sys.argv[2], sys.argv[3]
acc = defaultdict(list)
for line in open(log):
    p = line.split()
    if len(p) == 3 and lo <= p[0] <= hi:
        acc[p[1]].append(int(p[2]))
for bdf, v in sorted(acc.items()):
    print(f"{bdf}  n={len(v)}  median {st.median(v):.0f} MHz  min {min(v)}  max {max(v)}")
