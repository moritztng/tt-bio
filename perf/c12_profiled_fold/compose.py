#!/usr/bin/env python3
"""Compose the profiled units into whole-fold device seconds, per device op and per python class.

CPU only, re-runnable on `reduce.py`'s table. Every unit's device ms/call is multiplied by the
fold's OWN calls for that unit, taken from the `counts` phase, so the weights are integers the
instrument cannot inflate.

The per-class split is only emitted for a device op code whose row count and python call count
matched exactly inside the profiled window (`reduce.align`'s length control). A code that did not
match is carried in the totals but reported as UNSPLIT rather than divided on a guess.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

FOLD_S = 14.881          # campaign figure, pinned during-sampled 1350 MHz, quiet box
MHZ = 1350.0
F_FIXED = 3.9830         # c10-fixed-cost, the term that does not move with AICLK

# unit -> calls per fold, from runs/<counts>/counts.json. PairformerLayer is split by variant:
# 264 are built transform_s=True (the trunk and the PairformerModule stack), 16 are the
# MSA-internal transform_s=False ones and they are already inside MSALayer.
CALLS = {"PairformerLayer": 264, "MSALayer": 16, "DiffusionModule": 200,
         "PairConditioningDevice": 1, "RelPosGather": 2, "PairAssemblyDevice": 2}

# measured but not composed: ConfidenceHeadsDevice runs after the diffusion sampler, so reaching it
# needs a whole-fold precursor, which the profiler cannot survive. Bounded by its own inclusive
# host wall from the counts phase instead.
UNMEASURED_BOUND_S = {"ConfidenceHeadsDevice": 0.0231}

CENSUS = {"linear": 4.6604, "matmul": 1.479, "multiply_": 1.751, "add_": 0.8473,
          "add": 0.1403, "multiply": 0.1256, "layer_norm": 1.533, "generic_op": 4.0841}
CENSUS_ARM = {"linear": "grid110 (bare arm was 10.7316 s, 2.30x slower)",
              "matmul": "grid110 (bare arm 2.1174 s)", "multiply_": "bare", "add_": "bare",
              "add": "bare", "multiply": "bare",
              "layer_norm": "bare, published as layer_norm 1.357 + layer_norm_w 0.176",
              "generic_op": "NOT a replay price -- a traffic FLOOR, 3,920 calls, 0.9127 TB"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    T = json.loads(a.table.read_text())
    units = {u["unit"]: u for u in T["units"] if "error" not in u}

    rows, tot = [], 0.0
    for name, calls in CALLS.items():
        u = units.get(name)
        if u is None:
            print("MISSING UNIT %s -- not composed" % name)
            continue
        s = calls * u["device_ms_per_call"] / 1e3
        tot += s
        wall = u.get("synced_wall_ms_per_call") or 0.0
        rows.append({"unit": name, "calls": calls,
                     "device_ms_per_call": u["device_ms_per_call"],
                     "s_per_fold": round(s, 4),
                     "programs_per_call": u["programs_per_call"],
                     "profiled_wall_ms": wall,
                     "in_kernel_pct": round(100 * u["device_ms_per_call"] / wall, 2) if wall else None,
                     "clock_ok": u["clock"]["ok"],
                     "rep_spread_pct": u["rep_control"].get("spread_pct")})
    print("%-24s %7s %12s %10s %9s %8s" % ("unit", "calls", "device ms", "s/fold", "programs", "spread%"))
    for r in rows:
        print("%-24s %7d %12.4f %10.4f %9.1f %8s%s"
              % (r["unit"], r["calls"], r["device_ms_per_call"], r["s_per_fold"],
                 r["programs_per_call"], r["rep_spread_pct"], "" if r["clock_ok"] else "  CLOCK-FAIL"))
    bound = sum(UNMEASURED_BOUND_S.values())
    print("\nSUM device %.4f s = %.1f %% of %.3f s" % (tot, 100 * tot / FOLD_S, FOLD_S))
    print("bounded, not composed: %s <= %.4f s" % (", ".join(UNMEASURED_BOUND_S), bound))
    print("REMAINDER %.4f s  (and %.4f s once the bound above is taken out)"
          % (FOLD_S - tot, FOLD_S - tot - bound))
    print("F = %.4f s exceeds the remainder by %.4f s" % (F_FIXED, F_FIXED - (FOLD_S - tot)))

    code_s: dict = defaultdict(float)
    cls_s: dict = defaultdict(float)
    unsplit: dict = defaultdict(float)
    for name, calls in CALLS.items():
        u = units.get(name)
        if u is None:
            continue
        g = (u.get("align_quality") or {}).get("groups", {})
        for code, v in u["by_op_code"].items():
            code_s[code] += calls * v["ms_per_call"] / 1e3
        for cl, v in u["by_class"].items():
            if cl == "?":
                continue
            cls_s[cl] += calls * v["ms_per_call"] / 1e3
        for code, gv in g.items():
            if not gv.get("matched"):
                unsplit[code] += calls * u["by_op_code"].get(code, {}).get("ms_per_call", 0.0) / 1e3

    print("\nPER DEVICE OP, whole fold")
    print("%-34s %10s %8s" % ("device op", "s/fold", "% fold"))
    for k, v in sorted(code_s.items(), key=lambda kv: -kv[1]):
        print("%-34s %10.4f %7.1f%%" % (k, v, 100 * v / FOLD_S))

    print("\nPER PYTHON CLASS, whole fold, against the census")
    print("%-14s %10s %10s %10s %8s %10s" % ("class", "measured", "census", "signed", "ratio", "Mcycles"))
    for cl in sorted(set(list(cls_s) + list(CENSUS)), key=lambda c: -cls_s.get(c, 0)):
        m = cls_s.get(cl)
        c = CENSUS.get(cl)
        if m is None:
            print("%-14s %10s %10.4f   (no measured rows credited to this name)" % (cl, "-", c))
            continue
        if c is None:
            print("%-14s %10.4f %10s" % (cl, m, "-"))
            continue
        print("%-14s %10.4f %10.4f %+10.4f %8.3fx %10.1f   [%s]"
              % (cl, m, c, m - c, m / c, abs(m - c) * MHZ, CENSUS_ARM.get(cl, "")))

    print("\nUNSPLIT device ops (length control refused the pairing), carried in the totals:")
    for k, v in sorted(unsplit.items(), key=lambda kv: -kv[1]):
        if v > 0:
            print("  %-32s %8.4f s  %.2f %% of the fold" % (k, v, 100 * v / FOLD_S))
    print("  total unsplit %.4f s, %.2f %% of the fold" % (sum(unsplit.values()),
                                                           100 * sum(unsplit.values()) / FOLD_S))
    out = {"fold_s": FOLD_S, "units": rows, "device_total_s": round(tot, 4),
           "coverage_pct": round(100 * tot / FOLD_S, 2),
           "remainder_s": round(FOLD_S - tot, 4),
           "bounded_unmeasured_s": UNMEASURED_BOUND_S,
           "per_device_op_s": {k: round(v, 5) for k, v in sorted(code_s.items(), key=lambda kv: -kv[1])},
           "per_class_s": {k: round(v, 5) for k, v in sorted(cls_s.items(), key=lambda kv: -kv[1])},
           "census": CENSUS, "census_arm": CENSUS_ARM,
           "unsplit_s": {k: round(v, 5) for k, v in unsplit.items() if v > 0}}
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))
        print("\nwrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
