#!/usr/bin/env python3
"""bcx-p10-mmlay leg 4: the paired round A/B for TT_BIO_MM_LAYOUT.

Device seconds are the claim, so the statistic is the per-round device column
(`taped_s + bwd_s + primal_s`), paired on the ROUND INDEX across the two arms. BindCraft 2's
own per-round shape puts a long forward at rounds 2, 4 and 7 of every process ON BOTH ARMS, so
two raw medians overlap for a reason that is not noise; pairing round k against round k removes
it. The win count is the rank-paired one over the same pairs.
"""
from __future__ import annotations

import json
import pathlib
import statistics as st
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "perf" / "bcx_p10_mmlay" / "out"


def rounds(tag):
    d = json.load(open(OUT / tag / "round_events.json"))
    ev, clk = d["events"], d["aiclk"]
    starts = [e for e in ev if e["kind"] == "round_start"]
    stop = [e for e in ev if e["kind"] == "round_stop"]
    bounds = [e["t0"] for e in starts] + ([stop[0]["t0"]] if stop else [])
    out = []
    for i in range(len(bounds) - 1):
        t0, t1 = bounds[i], bounds[i + 1]
        inside = [e for e in ev if e.get("t0") is not None and e.get("t1") is not None
                  and e["t0"] >= t0 - 1e-6 and e["t1"] <= t1 + 1e-6]
        dev = sum(e["dt"] for e in inside if e["kind"] == "device")
        samples = [(c, l) for t, c, l in clk if t0 <= t <= t1]
        aiclk = sorted(c for c, _ in samples)
        served = starts[i + 1].get("mm_layout", {}).get("served", 0) if i + 1 < len(starts) else None
        out.append({"round": starts[i]["round"], "wall": t1 - t0, "device": dev,
                    "aiclk_med": aiclk[len(aiclk) // 2] if aiclk else None,
                    "aiclk_min": aiclk[0] if aiclk else None,
                    "load1": round(sum(l for _, l in samples) / len(samples), 1) if samples else None,
                    "served_cum": served,
                    "mm_on": d["stamp"]["mm_layout_on"]})
    return out[1:]          # round 1 is BindCraft 2's jit compile


def main():
    off_tags = sys.argv[1].split(",")
    on_tags = sys.argv[2].split(",")
    A = {t: rounds(t) for t in off_tags}
    B = {t: rounds(t) for t in on_tags}
    for name, arms in (("OFF", A), ("ON", B)):
        for tag, rs in arms.items():
            assert all(r["mm_on"] == (name == "ON") for r in rs), f"{tag} is not the {name} arm"
            if name == "ON":
                assert all((r["served_cum"] or 0) > 0 for r in rs[:-1]), f"{tag} served nothing"

    print("%-8s %6s %8s %8s %8s %7s %7s %6s" %
          ("arm", "round", "wall_s", "device_s", "host_s", "aiclk", "aiclkmn", "load1"))
    for tag, rs in list(A.items()) + list(B.items()):
        for r in rs:
            print("%-8s %6d %8.3f %8.3f %8.3f %7s %7s %6s"
                  % (tag, r["round"], r["wall"], r["device"], r["wall"] - r["device"],
                     r["aiclk_med"], r["aiclk_min"], r["load1"]))

    # Pair on (process slot, round index): process i of OFF against process i of ON.
    pairs = []
    for ta, tb in zip(off_tags, on_tags):
        for a, b in zip(A[ta], B[tb]):
            assert a["round"] == b["round"]
            pairs.append((ta, tb, a, b))
    print("\n== paired, round k of process i against round k of process i ==")
    print("%-18s %6s %10s %10s %9s" % ("pair", "round", "off dev_s", "on dev_s", "ratio"))
    ratios, wins = [], 0
    for ta, tb, a, b in pairs:
        rr = a["device"] / b["device"]
        ratios.append(rr)
        wins += rr > 1
        print("%-18s %6d %10.3f %10.3f %9.4fx" % (f"{ta}/{tb}", a["round"], a["device"],
                                                  b["device"], rr))
    off = [a["device"] for _, _, a, _ in pairs]
    on = [b["device"] for _, _, _, b in pairs]
    d = [a - b for a, b in zip(off, on)]
    n = len(d)
    sd = st.stdev(d) if n > 1 else 0.0
    print("\nn pairs %d | OFF device median %.3f s | ON device median %.3f s"
          % (n, st.median(off), st.median(on)))
    print("paired ratio: median %.4fx  mean %.4fx" % (st.median(ratios), st.fmean(ratios)))
    print("paired delta: median %+.3f s  mean %+.3f s  sd %.3f  sem %.3f"
          % (st.median(d), st.fmean(d), sd, sd / n ** 0.5 if n else 0))
    if n > 1 and sd > 0:
        t = st.fmean(d) / (sd / n ** 0.5)
        print("paired t = %.2f on %d df" % (t, n - 1))
    print("rank-paired wins for ON: %d of %d" % (wins, n))
    w = [r["wall"] for _, _, r, _ in pairs], [r["wall"] for _, _, _, r in pairs]
    print("wall median: OFF %.3f s  ON %.3f s  paired ratio median %.4fx"
          % (st.median(w[0]), st.median(w[1]),
             st.median([x / y for x, y in zip(*w)])))
    allclk = [r["aiclk_med"] for _, _, a, b in pairs for r in (a, b) if r["aiclk_med"]]
    allmin = [r["aiclk_min"] for _, _, a, b in pairs for r in (a, b) if r["aiclk_min"]]
    print("AICLK over every timed round: median %s, min %s | load1 %.1f-%.1f"
          % (st.median(allclk), min(allmin),
             min(r["load1"] for _, _, a, b in pairs for r in (a, b) if r["load1"]),
             max(r["load1"] for _, _, a, b in pairs for r in (a, b) if r["load1"])))


if __name__ == "__main__":
    main()
