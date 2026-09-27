#!/usr/bin/env python3
"""Summarise a footprint.json into duotraj's leg-1 table: floor, banked, in-seam peak per round.

    fpsum.py out/fp/footprint.json
"""
import json
import sys

s = [x for x in json.load(open(sys.argv[1]))["samples"] if x[2] >= 0]
total = s[0][3]
cuts = [i for i, x in enumerate(s) if x[1] == "round_boundary"]
G = 2 ** 30  # GiB, the unit duotraj's leg-1 table used
print(f"card total {total / G:.3f} GiB, {len(s)} samples, {len(cuts)} round boundaries")
for k, (a, b) in enumerate(zip(cuts, cuts[1:] + [len(s)])):
    seg = s[a:b]
    if len(seg) < 3:
        continue
    floor = seg[0][2]
    banked = max(x[2] for x in seg if x[1].endswith(":exit") and "_backward" not in x[1])
    peak = max(seg, key=lambda x: x[2])
    lcf = min(x[4] for x in seg)
    print(f"round seg {k}: floor {floor / G:.3f}  banked {banked / G:.3f}  "
          f"peak {peak[2] / G:.3f} at {peak[1]}  min contiguous free {lcf / G:.2f}  "
          f"pair worst case {(peak[2] + banked - floor) / G:.3f} GiB")
