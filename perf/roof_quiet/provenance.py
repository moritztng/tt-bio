#!/usr/bin/env python3
"""Where the 0.7011 scale in ROOF_BUDGET.md comes from, and what it is applied to.

Committed data only, no device. The scale divides the 17.340 s cell of record by the PLAIN
baseline fold of the same session, 24.731 s. The per-unit times it is applied to come from a
different fold in a different phase of that session, the bracketed attrib fold, whose wall is
74.159 s. That the two agree to 0.36 % is a property of the median, not of the method.
"""
from __future__ import annotations

import json
from pathlib import Path

P = Path(__file__).resolve().parents[1] / "roof_budget"
TOP = ["PairformerLayer|1x512x384,1x512x512x128",
       "MSALayer|1x512x512x128,1x1024x512x64", "DiffusionModule|"]


def main() -> int:
    base = json.loads((P / "attrib_512_tip_qb2c2.json").read_text())
    att = json.loads((P / "attrib2_512_tip_qb2c2.json").read_text())["attrib"]
    S = json.loads((P / "roof_budget_512_qb2c2.json").read_text())["summary"]
    sigs = att["sigs"]
    med = sum(sigs[t]["calls"] * sigs[t]["median_ms"] for t in TOP) / 1e3
    tot = sum(sigs[t]["total_ms"] for t in TOP) / 1e3
    wall = att["instrumented_fold_s"]
    cell = S["cell_of_record_s"]

    print("THE PROVENANCE OF THE 0.7011 SCALE")
    print("  numerator    cell of record             %8.3f s" % cell)
    print("  denominator  baseline PLAIN fold median %8.3f s  at loadavg %s"
          % (S["session_fold_s"], ", ".join(S["session_loadavg"])))
    print("  scale                                   %8.4f" % S["cell_scale"])
    print()
    print("THE FOLD THE TIMES ACTUALLY CAME FROM: a different fold, in a different phase")
    print("  attrib instrumented fold wall           %8.3f s" % wall)
    print("  top-3 rebuilt, median x calls           %8.3f s   %.4f of the wall" % (med, med / wall))
    print("  top-3 rebuilt, sum of every call        %8.3f s   %.4f of the wall" % (tot, tot / wall))
    print()
    print("  scale as published    %.3f/%.3f = %.4f" % (cell, S["session_fold_s"], S["cell_scale"]))
    print("  scale self-consistent %.3f/%.3f = %.4f  (denominator = what it is applied to)"
          % (cell, med, cell / med))
    print("  scale vs attrib wall  %.3f/%.3f = %.4f" % (cell, wall, cell / wall))
    print()
    print("THE TWO ESTIMATORS PER TOP UNIT, which is the gap roof-residual-census could not close")
    print("%-46s %6s %10s %10s %7s" % ("unit", "calls", "med x n", "sum", "ratio"))
    for t in TOP:
        m = sigs[t]
        x, y = m["calls"] * m["median_ms"] / 1e3, m["total_ms"] / 1e3
        print("%-46s %6d %10.3f %10.3f %6.2fx" % (t, m["calls"], x, y, y / x))
    print("%-46s %6s %10.3f %10.3f %6.2fx" % ("TOTAL", "", med, tot, tot / med))
    print()
    b = base["baseline_summary"]
    print("AND THE DENOMINATOR'S OWN ERROR BAR: plain folds %s s in that session, spread %.1f %% "
          "of the median." % (b["plain_s"], b["aa_floor_pct"]))
    print("A 5.3 %% crossing cannot be decided against a denominator with a %.1f %% spread."
          % b["aa_floor_pct"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
