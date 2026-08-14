#!/usr/bin/env python3
"""Where in frequency does an arm's FSC deficit live?

Section 5 of state/relion-precision-fsc.md set a stop condition before the campaign ran: an arm
whose FSC deficit is concentrated in the top two shells rather than spread across them must be
re-examined, because the 0.143 crossing is decided by the high-frequency shells and a concentrated
deficit there would move the crossing while an aggregate stayed quiet. The delta_A in each result
JSON is the crossing only, so this reads the stored curves and checks the condition directly.

Deficit is `ref_fsc - arm_fsc` per shell, averaged over seeds (paired: same seed, same noise), then
binned into octiles of the resolution range. Reported as a fraction of the total absolute deficit,
so "concentrated in the top two shells" would show as the last two octiles carrying most of it.

    python3 p3_shells.py <variant> <precision> [box]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    variant, prec = sys.argv[1], sys.argv[2]
    box = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    pat = re.compile(rf"^p3fsc_box{box}_snr0\.05_s(\d+)_{variant}_{prec}\.json$")

    defs, seeds = [], []
    for p in sorted(HERE.glob(f"p3fsc_box{box}_snr0.05_s*_{variant}_{prec}.json")):
        m = pat.match(p.name)
        if not m:
            continue
        d = json.load(open(p))
        ref, arm = np.array(d["arms"]["ref"]["fsc"], float), np.array(d["arms"]["arm"]["fsc"], float)
        defs.append(ref - arm)
        seeds.append(int(m.group(1)))
    if not defs:
        sys.exit(f"no results for {variant}/{prec} at box {box}")

    D = np.array(defs).mean(axis=0)          # mean per-shell deficit over seeds
    n = len(D)
    edges = np.linspace(0, n, 9).astype(int)
    tot = np.abs(D).sum()
    print(f"box {box} {variant}/{prec}, n={len(seeds)} seeds ({','.join(map(str, seeds))}), "
          f"{n} shells")
    print(f"{'octile':>7} {'shells':>10} {'mean deficit':>13} {'|deficit| share':>16}")
    for i in range(8):
        lo, hi = edges[i], edges[i + 1]
        seg = D[lo:hi]
        share = np.abs(seg).sum() / tot if tot else 0.0
        print(f"{i:>7} {f'{lo}-{hi-1}':>10} {seg.mean():>+13.2e} {share:>15.1%}")
    top2 = np.abs(D[edges[6]:]).sum() / tot if tot else 0.0
    print(f"  top two octiles carry {top2:.1%} of the absolute deficit "
          f"(uniform would be 25.0%)")


if __name__ == "__main__":
    main()
