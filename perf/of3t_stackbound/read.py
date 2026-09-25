#!/usr/bin/env python3
"""of3t-stackbound: the ladder read against PREREGISTERED.md, and nothing else.

  read.py LADDER_SCORES.json PD_AA_SHIPA_vs_SHIPB.json > LADDER.json

Every band, T and definition below is copied from the pre-registration (commit 93147d1a9). The
A/A floor is read first; if it is not exactly zero T grows as registered.
"""
from __future__ import annotations

import json
import sys

STACKEXACT_I = 0.3418


def main() -> int:
    sc = json.load(open(sys.argv[1]))
    aa = json.load(open(sys.argv[2]))
    arms = sc["arms"]
    c = {t: arms[t]["x_bar_R1"] for t in arms}
    out = {"what": __doc__.strip().splitlines()[0], "row": "of3t-stackbound",
           "boundary_version": sc["boundary_version"], "bar": sc["bar"],
           "allowance": sc["allowance"], "rest_declaration": sc["rest_declaration"],
           "AA": {k: aa.get(k) for k in ("compared", "bit_identical", "all_bit_identical",
                                         "concatenated_rel_a_vs_b", "cos")}}
    T = 0.02
    c_ship = c["SHIP_A"]
    floor_zero = bool(aa.get("all_bit_identical"))
    if not floor_zero:
        fl = abs(c["SHIP_B"] - c_ship) / abs(c_ship - 1) if c_ship != 1 else float("inf")
        T = max(0.02, 3 * fl)
    out["T"] = T
    ship_clears = c_ship <= 1
    if ship_clears:
        f = {t: c_ship - c[t] for t in c}          # margin in bar units, as registered
        out["closure_definition"] = "SHIP CLEARS: margin c_SHIP - c_arm in bar units"
    else:
        f = {t: 1 - (c[t] - 1) / (c_ship - 1) for t in c}
        out["closure_definition"] = "f = 1 - e/e_SHIP"
    out["x_bar_R1"] = c
    out["x_bar_R0"] = {t: arms[t]["x_bar_R0"] for t in arms}
    out["trunk_over_allowance_R1"] = {t: arms[t]["trunk_over_allowance_R1"] for t in arms}
    out["closure"] = f
    if ship_clears:
        sl = "SHIP CLEARS"
    elif c["SL"] <= 1:
        sl = "CLEARS"
    elif f["SL"] >= T:
        sl = "IMPROVES ONLY"
    else:
        sl = "DOES NOT IMPROVE"
    out["SL_band"] = sl
    out["SL_margin_pct_of_bar"] = 100 * (1 - c["SL"])
    if "SLZ" in f:
        dz = f["SLZ"] - f["SL"]
        out["dZ"] = dz
        out["SLZ_band"] = "ADDS" if dz >= T else ("COSTS" if dz <= -T else "INERT")
        out["SLZ_clears"] = c["SLZ"] <= 1
    if "S" in f and "L" in f:
        I = f["SL"] - f["S"] - f["L"]
        out["I"] = I
        out["dL"] = f["SL"] - f["S"]
        out["I_band"] = ("SUPER-ADDITIVE" if I >= T else
                         ("SUB-ADDITIVE" if I <= -T else "ADDITIVE"))
        out["I_reproduces"] = I >= T
        out["I_reproduces_in_size"] = I >= T and STACKEXACT_I / 2 <= I <= 2 * STACKEXACT_I
    out["trunk"] = {t: {"graded": arms[t]["trunk_vs_upstream_bf16"],
                        "float64": arms[t]["trunk_vs_float64"]} for t in arms}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
