#!/usr/bin/env python3
"""Gate a c13-matmul-rate session, then decompose it. Refuses to print a table it cannot defend.

Three gates, in this order, and a failure prints the reason and exits nonzero:

  CLOCK    every during-sample at the target, no read error, no gap over 10 ms, AND the sampler's
           span must COVER the measurement's span. The span check is the one that was missing:
           session s1's record read "99.8 % of samples at 1350 MHz" and was still unusable, because
           a record can be clean over an interval far shorter than the thing it certifies.
  CONTROL  the FLOP and byte counts are exact closed forms (no counter is read), the DRAM roof
           reproduces the campaign's 442.9 GB/s, and the dense cube is taken in THIS session under
           the model's own kernel config -- ratios are quoted against that, never a cube from
           another session on another chip.
  A/A      every arm with an _aa twin must agree inside --aa-tol, and the session is refused
           rather than reported if the twin on a decomposition arm blows up.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DRAM_ROOF_GBS = 442.9      # campaign roof, c10-fold-census starved 8192^2 add


def load_clock(p, target, t0=None, t1=None):
    """Clock stats over the MEASUREMENT window, plus the sampler's full span for the cover check.

    Scoring the sampler's whole span is wrong and was wrong here: the sampler necessarily starts
    before the measurement (it has to be up first) and outlives it, so it records the governor's
    idle 800 MHz at both ends. Session s2 scored 6,609 of 69,220 samples at target that way and
    was refused, while every sample inside the 11.1 s it certified read 1350.
    """
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    hdr, all_s = rows[0], [r for r in rows[1:] if "t" in r]
    node = str(hdr["nodes"][0]) if "nodes" in hdr else "3"
    span = (all_s[0]["t"], all_s[-1]["t"])
    s = [r for r in all_s if (t0 is None or r["t"] >= t0) and (t1 is None or r["t"] <= t1)]
    if not s:
        return {"n": 0, "errors": 0, "min": None, "max": None, "at_target": 0,
                "t0": span[0], "t1": span[1], "max_gap_ms": 0.0, "n_span": len(all_s)}
    v = [r[node] for r in s]
    num = [x for x in v if isinstance(x, int)]
    t = [r["t"] for r in s]
    gaps = [b - a for a, b in zip(t, t[1:])]
    return {"n": len(v), "errors": len(v) - len(num), "min": min(num), "max": max(num),
            "at_target": sum(1 for x in num if x == target),
            "t0": span[0], "t1": span[1], "n_span": len(all_s),
            "win_t0": t[0], "win_t1": t[-1],
            "max_gap_ms": 1000.0 * max(gaps) if gaps else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=Path, required=True)
    ap.add_argument("--clock", type=Path, required=True)
    ap.add_argument("--aa-tol", type=float, default=3.0)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    d = json.loads(a.rate.read_text())
    arms, target = d["arms"], d["clock_mhz"]
    fail, warn = [], []

    # ---- GATE 1: clock, including the span check --------------------------------------
    ms, me = d["t_measure_start"], d["t_measure_end"]
    c = load_clock(a.clock, target, ms, me)
    span_ok = c["t0"] <= ms and c["t1"] >= me
    if c["errors"]:
        fail.append("clock: %d read errors" % c["errors"])
    if c["at_target"] != c["n"] - c["errors"]:
        off = c["n"] - c["errors"] - c["at_target"]
        (fail if off > 0.01 * c["n"] else warn).append(
            "clock: %d of %d samples off %d MHz (min %s max %s)"
            % (off, c["n"], target, c["min"], c["max"]))
    if c["max_gap_ms"] > 10.0:
        warn.append("clock: max sample gap %.1f ms" % c["max_gap_ms"])
    if not span_ok:
        fail.append("clock: sampler span [%.1f, %.1f] does not cover measurement [%.1f, %.1f] "
                    "-- short by %.1f s at the head and %.1f s at the tail"
                    % (c["t0"], c["t1"], ms, me, max(0.0, c["t0"] - ms), max(0.0, me - c["t1"])))
    print("CLOCK  %d in-window samples of %d recorded, %d at %d MHz (min %s max %s), %d errors, "
          "max gap %.1f ms; sampler span %.1f s vs measurement %.1f s -> %s"
          % (c["n"], c["n_span"], c["at_target"], target, c["min"], c["max"], c["errors"],
             c["max_gap_ms"], c["t1"] - c["t0"], me - ms,
             "COVERS" if span_ok else "DOES NOT COVER"))

    # ---- GATE 2: controls -------------------------------------------------------------
    ctl = d["controls"]
    exact = {"F2048": 17179869184.0, "F8192": 1099511627776.0, "BW_ADD": 402653184}
    for k, v in exact.items():
        ok = ctl.get(k) == v
        print("CONTROL %-8s = %-16s exact %-16s %s" % (k, ctl.get(k), v, "PASS" if ok else "FAIL"))
        if not ok:
            fail.append("control %s is %s, not the exact %s" % (k, ctl.get(k), v))
    bw = arms.get("ctl_bwadd", {}).get("gbs")
    if bw is None:
        fail.append("control: no DRAM roof arm")
    else:
        dev = 100.0 * (bw - DRAM_ROOF_GBS) / DRAM_ROOF_GBS
        print("CONTROL dram roof  = %.1f GB/s vs campaign %.1f (%+.1f %%) %s"
              % (bw, DRAM_ROOF_GBS, dev, "PASS" if abs(dev) <= 8 else "FAIL"))
        if abs(dev) > 8:
            fail.append("control: DRAM roof %.1f GB/s is %+.1f %% off the campaign roof" % (bw, dev))

    # ---- GATE 3: A/A twins ------------------------------------------------------------
    print("\nA/A twins (tol %.1f %%)" % a.aa_tol)
    for nm in sorted(arms):
        if not nm.endswith("_aa"):
            continue
        base = nm[:-3]
        if base not in arms:
            continue
        x, y = arms[base]["ms_min"], arms[nm]["ms_min"]
        spread = 100.0 * abs(y - x) / min(x, y)
        flag = "ok" if spread <= a.aa_tol else "OVER TOL"
        print("  %-22s %8.4f vs %8.4f ms  %5.2f %%  %s" % (base, x, y, spread, flag))
        if spread > a.aa_tol:
            warn.append("A/A floor on %s is %.2f %%, over the %.1f %% tolerance"
                        % (base, spread, a.aa_tol))

    if fail:
        print("\nREFUSED -- this session cannot be reported:")
        for f in fail:
            print("  - %s" % f)
        return 1

    # ---- the decomposition ------------------------------------------------------------
    cube = arms["ctl_cube8192_ship"]["tflops"]
    cube_roofcfg = arms["ctl_cube8192"]["tflops"]
    print("\nIN-SESSION ROOFS at %d MHz: dense 8192^3 cube %.2f TFLOP/s under the model's kernel "
          "config, %.2f under the roofline script's, DRAM %.1f GB/s"
          % (target, cube, cube_roofcfg, bw))
    print("crossover arithmetic intensity = %.1f FLOP/byte" % (cube * 1e12 / (bw * 1e9)))

    rows = []
    for kn, spec in d["keys"].items():
        F, B = spec["flops"], spec["min_bytes"]
        ai = F / B
        traffic_roof = ai * bw * 1e9 / 1e12
        shape_roof = min(traffic_roof, cube)
        ship = arms["%s_ship" % kn]["tflops"]
        print("\n=== key %s  b=%d M=%d K=%d N=%d  %d calls in the fold"
              % (kn, spec["batch"], spec["m"], spec["k"], spec["n"], spec["calls"]))
        print("  AI %.1f FLOP/byte -> binding roof is %s, %.2f TFLOP/s (%.1f %% of the cube)"
              % (ai, "TRAFFIC" if traffic_roof < cube else "ARITHMETIC", shape_roof,
                 100.0 * shape_roof / cube))
        print("  shipped config %.2f TFLOP/s = %.1f %% of the cube, %.1f %% of the shape's own roof"
              % (ship, 100.0 * ship / cube, 100.0 * ship / shape_roof))
        print("  %-22s %9s %9s %9s %8s" % ("arm", "ms_min", "TFLOP/s", "vs ship", "spread"))
        for nm in sorted(n for n in arms if n.startswith(kn + "_") and not n.endswith("_aa")):
            r = arms[nm]
            print("  %-22s %9.4f %9.2f %8.3fx %7.2f %%"
                  % (nm[len(kn) + 1:], r["ms_min"], r["tflops"],
                     r["tflops"] / ship, r["spread_pct"]))
        rows.append({"key": kn, "ai": ai, "traffic_roof_tflops": traffic_roof,
                     "shape_roof_tflops": shape_roof, "ship_tflops": ship,
                     "frac_of_cube": ship / cube, "frac_of_shape_roof": ship / shape_roof,
                     "calls": spec["calls"], "flops_per_call": F})

    if warn:
        print("\nWARNINGS (reported, not suppressed):")
        for w in warn:
            print("  - %s" % w)
    if a.out:
        a.out.write_text(json.dumps(
            {"clock": c, "cube_ship_tflops": cube, "cube_roofcfg_tflops": cube_roofcfg,
             "dram_gbs": bw, "keys": rows, "warnings": warn}, indent=1))
        print("\nwrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
