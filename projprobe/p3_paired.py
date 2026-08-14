#!/usr/bin/env python3
"""Paired per-seed differences between two arms of the box-256 campaign.

Section 8.2's secondary acceptance is "bf16_dev no worse than tex8, in the same harness on the
same seeds", which is a per-seed statement and so wants a per-seed difference.

It does NOT buy a variance reduction, and the script should not be read as claiming one. MEASURED
at box 256, paired sd over the sd of an independent difference: 1.005 for bf16_dev - tex8, 0.920 for
bf16_dev_pess - bf16_dev, 1.000 for twopass - tri. Each arm's delta_A is already differenced against
the same per-seed fp64 control, so the shared noise realisation cancels inside each delta and no
common mode survives to be removed across arms. What the paired form does buy is that it cannot
silently compare arms measured on different seed sets, and that the near-absence of common mode is
visible rather than assumed.

    python3 p3_paired.py <variantA> <precA> <variantB> <precB> [box]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
     9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 15: 2.131, 20: 2.086}


def arm(variant, prec, box):
    # `_s` also matches `snr0.05`, so anchor the seed field on the digits that follow it.
    pat = re.compile(rf"^p3fsc_box{box}_snr0\.05_s(\d+)_{variant}_{prec}\.json$")
    out = {}
    for p in HERE.glob(f"p3fsc_box{box}_snr0.05_s*_{variant}_{prec}.json"):
        m = pat.match(p.name)
        if m:
            out[int(m.group(1))] = json.load(open(p))["delta_A"]
    return out


def interval(v):
    n = len(v)
    mean = float(v.mean())
    if n < 2:
        return mean, float("nan"), (float("nan"), float("nan"))
    sd = float(v.std(ddof=1))
    h = T.get(n - 1, 1.96) * sd / np.sqrt(n)
    return mean, sd, (mean - h, mean + h)


def main():
    va, pa, vb, pb = sys.argv[1:5]
    box = int(sys.argv[5]) if len(sys.argv) > 5 else 256
    A, B = arm(va, pa, box), arm(vb, pb, box)
    seeds = sorted(set(A) & set(B))
    if not seeds:
        sys.exit(f"no shared seeds between {va}/{pa} and {vb}/{pb} at box {box}")
    d = np.array([A[s] - B[s] for s in seeds], float)

    print(f"box {box}, {len(seeds)} shared seeds: {','.join(str(s) for s in seeds)}")
    for label, v in ((f"{va}/{pa}", np.array([A[s] for s in seeds])),
                     (f"{vb}/{pb}", np.array([B[s] for s in seeds]))):
        m, sd, (lo, hi) = interval(v)
        print(f"  {label:<24} mean {m:+.6f} A   sd {sd:.6f}   95% CI [{lo:+.6f}, {hi:+.6f}]")
    m, sd, (lo, hi) = interval(d)
    print(f"  PAIRED difference        mean {m:+.6f} A   sd {sd:.6f}   95% CI [{lo:+.6f}, {hi:+.6f}]")
    print("  per-seed: " + "  ".join(f"s{s}:{A[s]-B[s]:+.6f}" for s in seeds))
    verdict = "straddles zero -> indistinguishable" if lo < 0 < hi else (
        "entirely above zero -> A is worse" if lo > 0 else "entirely below zero -> A is better")
    print(f"  -> {verdict}")


if __name__ == "__main__":
    main()
