#!/usr/bin/env python3
"""Predict the walk ranking from DRAM bank arithmetic alone, and check it against the measurements.

The device A/Bs say `block` is the worst walk at every one of the 12 points measured across the two
architectures, but they disagree about which of `stride` and `rotate` is better, and the depth rule
fitted on Blackhole mispicks twice on Wormhole. So this drops the threshold and computes the thing
the threshold was standing in for: which DRAM bank every page of every concurrent group lands on.

Interleaved DRAM puts page p in bank p % banks, and the banks serve in parallel, so a wave costs
what its BUSIEST bank has to move. For each wave index j, the active cores are the ones with more
than j groups, each sits at the group its walk puts it on, and each issues a known page set. The
score is sum over waves of max over banks of the page count, which is the wave-serialised lower
bound in page-times. Host only: this opens no device, it is pure arithmetic over the kernels' own
index expressions, which are quoted beside each leg.
"""
from __future__ import annotations

import json, sys
from math import gcd
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tt_bio import core_split                                            # noqa: E402
from tt_bio.reblock_permute import _NO_WRAP, _walk                       # noqa: E402

GATED_SLICE_C = 128


def pages(leg, g, Nt, Ct, Ctw):
    """Every DRAM page the kernels touch for group ``g``, reader then writer.

    fwd    reader (it*32 + il)*Nt*Ct + jt*Ct + ct      writer (ct*32 + c)*Nt*Nt + it*Nt + jt
    back   reader (ct*32 + cl)*Nt*Nt + it*Nt + jt      writer (it*32 + il)*Nt*Ct + jt*Ct + ct
    gated  reader (it*32 + il)*Nt*Ctw + jt*Ctw + off + ct, for both the p and the g slice;
           writer as fwd
    """
    out = []
    if leg == "fwd":
        it, jt = divmod(g, Nt)
        for ct in range(Ct):
            out += [(it * 32 + il) * Nt * Ct + jt * Ct + ct for il in range(32)]
            out += [(ct * 32 + c) * Nt * Nt + it * Nt + jt for c in range(32)]
        return out
    it, rem = divmod(g, Nt * Ct)
    jt, ct = divmod(rem, Ct)
    if leg == "back":
        out += [(ct * 32 + cl) * Nt * Nt + it * Nt + jt for cl in range(32)]
        out += [(it * 32 + il) * Nt * Ct + jt * Ct + ct for il in range(32)]
        return out
    assert leg == "gated"
    for off in (0, Ct):                      # the p slice and the g slice
        out += [(it * 32 + il) * Nt * Ctw + jt * Ctw + off + ct for il in range(32)]
    out += [(ct * 32 + c) * Nt * Nt + it * Nt + jt for c in range(32)]
    return out


def score(leg, N, C, cores, banks):
    """``{walk: wave-serialised page cost}`` for one shape on one machine."""
    Nt = N // 32
    Ctw = C // 32
    Ct = GATED_SLICE_C // 32 if leg == "gated" else Ctw
    units = Nt * Nt * Ct if leg in ("back", "gated") else Nt * Nt
    n, n1, w1, w2 = core_split.units_per_core(units, cores)
    per = [w1 if i < n1 else w2 for i in range(n)]
    starts, b = [], 0
    for p in per:
        starts.append(b)
        b += p
    out = {}
    for walk in ("block", "stride", "rotate"):
        walks = [_walk(walk, i, starts[i], per[i], n) for i in range(n)]
        total = 0
        for j in range(w1):
            hist = [0] * banks
            for i, (first, stride, hi, lo) in enumerate(walks):
                if j >= per[i]:
                    continue
                g = first + j * stride
                if hi != _NO_WRAP and g >= hi:
                    g = lo + (g - hi)
                for p in pages(leg, g, Nt, Ct, Ctw):
                    hist[p % banks] += 1
            total += max(hist)
        out[walk] = total
    return out


# The device numbers, both architectures, from perf/ttx_splitwork/auto_ab.json +
# walk_ab.json (Blackhole qb2 card 1, 11x10, 8 banks) and walk_wh_j10glx02c0.json
# (Wormhole j10glx02 card 0, 8x9, 12 banks). Ratio against `block` on the whole grid,
# so >1 is a win.
MEASURED = [
    ("bh", 110, 8, "fwd", 640, 32, {"stride": 1.438, "rotate": 1.417}),
    ("bh", 110, 8, "fwd", 960, 32, {"stride": 1.128, "rotate": 1.121}),
    ("bh", 110, 8, "back", 960, 32, {"stride": 1.116, "rotate": 1.144}),
    ("bh", 110, 8, "fwd", 512, 32, {"stride": 1.101, "rotate": 0.974}),
    ("bh", 110, 8, "gated", 512, 512, {"stride": 0.964, "rotate": 1.030}),
    ("bh", 110, 8, "back", 512, 128, {"stride": 1.019, "rotate": 1.018}),
    ("wh", 72, 12, "fwd", 640, 32, {"stride": 1.056, "rotate": 1.036}),
    ("wh", 72, 12, "fwd", 960, 32, {"stride": 1.346, "rotate": 1.330}),
    ("wh", 72, 12, "back", 960, 32, {"stride": 1.151, "rotate": 1.300}),
    ("wh", 72, 12, "fwd", 512, 32, {"stride": 1.273, "rotate": 1.267}),
    ("wh", 72, 12, "gated", 512, 512, {"stride": 1.030, "rotate": 1.008}),
    ("wh", 72, 12, "back", 512, 128, {"stride": 1.055, "rotate": 0.980}),
]

TIE = 0.01  # inside this the two walks are the same arm as far as the device could tell


def main():
    rows, right, wrong, ties = [], 0, 0, 0
    print(f"  {'arch':>4} {'leg':>6} {'N':>5} {'C':>4}  {'block':>7}{'stride':>8}{'rotate':>8}"
          f"   model    device   agree")
    for arch, cores, banks, leg, N, C, meas in MEASURED:
        sc = score(leg, N, C, cores, banks)
        # the model predicts time, so the predicted ratio against `block` is block/walk
        pred = {w: sc["block"] / sc[w] for w in ("stride", "rotate")}
        m_pick = max(pred, key=pred.get)
        d_pick = max(meas, key=meas.get)
        tie = abs(meas["stride"] - meas["rotate"]) < TIE
        agree = "tie" if tie else ("yes" if m_pick == d_pick else "NO")
        right += agree == "yes"
        wrong += agree == "NO"
        ties += agree == "tie"
        print(f"  {arch:>4} {leg:>6} {N:>5} {C:>4}  {sc['block']:>7}{sc['stride']:>8}"
              f"{sc['rotate']:>8}   {m_pick:>6}  {d_pick:>6}   {agree}")
        rows.append({"arch": arch, "cores": cores, "banks": banks, "leg": leg, "N": N, "C": C,
                     "cost": sc, "predicted_ratio": pred, "model_pick": m_pick,
                     "device_pick": d_pick, "device_ratio": meas, "tie": tie, "agree": agree})
    print(f"\n  model picks the device's own winner at {right} of {len(MEASURED)}, "
          f"{ties} tie, {wrong} wrong")
    # `block` is never the best walk: the claim the whole change rests on.
    worst = [r for r in rows if r["cost"]["block"] < min(r["cost"]["stride"],
                                                         r["cost"]["rotate"])]
    print(f"  `block` is the cheapest walk in the model at {len(worst)} of {len(rows)} points")
    out = Path(__file__).with_name("bank_model.json")
    out.write_text(json.dumps({"tie_window": TIE, "rows": rows}, indent=1))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
