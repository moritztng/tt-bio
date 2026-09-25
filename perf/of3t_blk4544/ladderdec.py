#!/usr/bin/env python3
"""Decompose the width step into a NORM change and a DIRECTION change, off files already written.

`rel_l2` alone cannot tell "our cotangent points somewhere else" from "our cotangent is the right
direction with the wrong length". The identity is exact:

    rel_l2^2 = 1 + norm_ratio^2 - 2 * norm_ratio * cos

so norm_ratio and cos together reconstruct rel_l2, and the pair says which of the two moved
between the widths. D195: a pooled figure is decomposed before it is attributed to anything.
"""
from __future__ import annotations

import json
import math
import sys

A = json.load(open(sys.argv[1]))          # COTSCAN at padded 384
B = json.load(open(sys.argv[2]))          # COTSCAN at padded 64

SKIP = ("floor_masked", "floor_unmasked", "ref_norm_masked")
nmA = [k for k in A["rungs"]["44"]["ds"] if k not in SKIP][0]
nmB = [k for k in B["rungs"]["44"]["ds"] if k not in SKIP][0]
print(f"arm keys: {nmA} (384), {nmB} (64)")

for track in ("ds", "dz"):
    print(f"\n=== {track}, masked to the 56 real tokens ===")
    print(f"{'rung':>4} {'rel@384':>10} {'rel@64':>10} {'R':>6} "
          f"{'nr@384':>8} {'nr@64':>8} {'cos@384':>10} {'cos@64':>10} "
          f"{'1-cos@384':>10} {'1-cos@64':>10} {'Rcos':>7} {'Rnorm':>7}")
    for k in sorted((int(x) for x in A["rungs"]), reverse=True):
        ra = A["rungs"][str(k)].get(track)
        rb = B["rungs"][str(k)].get(track)
        if not ra or not rb or nmA not in ra or nmB not in rb:
            continue
        a, b = ra[nmA]["masked"], rb[nmB]["masked"]
        # the two parts of the identity, each scored on its own
        da = math.sqrt(max(0.0, 2 * a["norm_ratio"] * (1 - a["cos"])))
        db = math.sqrt(max(0.0, 2 * b["norm_ratio"] * (1 - b["cos"])))
        na = abs(a["norm_ratio"] - 1.0)
        nb = abs(b["norm_ratio"] - 1.0)
        print(f"{k:>4} {a['rel_l2']:10.4e} {b['rel_l2']:10.4e} "
              f"{a['rel_l2']/b['rel_l2']:6.3f} "
              f"{a['norm_ratio']:8.4f} {b['norm_ratio']:8.4f} "
              f"{a['cos']:10.6f} {b['cos']:10.6f} "
              f"{1-a['cos']:10.3e} {1-b['cos']:10.3e} "
              f"{(da/db if db else float('nan')):7.3f} {(na/nb if nb else float('nan')):7.3f}")
