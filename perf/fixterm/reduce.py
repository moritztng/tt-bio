#!/usr/bin/env python3
"""Turn fixterm_<tag>.json into the decomposition: components, accounting, residual.

Every per-call number here is a MARGINAL, (t_hi - t_lo) / (hi - lo) over two bracket sizes of
the same arm, so the synced bracket's own launch/drain floor cancels rather than being modelled.
The floor is reported separately from the n-ladder, because it is the first component and the
reason c13's intercept is larger than the fold's own fixed cost.

Least squares on two-parameter fits only; no term is fitted that an arm does not isolate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def fit(xs, ys):
    """y = a + b*x, ordinary least squares, plus the max residual."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    r = max(abs(y - (a + b * x)) for x, y in zip(xs, ys))
    return a, b, r


def main():
    d = json.load(open(sys.argv[1]))
    A = d["arms"]
    out = {"host": d["host"], "tag": d["tag"], "grid": d["grid"], "cores": d["cores"],
           "nodes": d["nodes"], "clock": d["clock"], "clock_mhz": d["clock_mhz"],
           "reps": d["reps"], "refused": sorted(d["refused"])}

    def ms(nm):
        return A[nm]["ms_min"] if nm in A else None

    def marginal(pair):
        """(t_hi - t_lo)/(hi - lo) in ms per call, from the two bracket sizes of one arm."""
        ns = sorted(int(nm.rsplit("_x", 1)[1]) for nm in A
                    if nm.startswith(pair + "_x") and A[nm].get("pair") == pair)
        if len(ns) != 2:
            return None
        lo, hi = ns
        return (A["%s_x%02d" % (pair, hi)]["ms_min"]
                - A["%s_x%02d" % (pair, lo)]["ms_min"]) / (hi - lo)

    # ---- C0: the instrument's own launch/drain floor --------------------------------
    nl = {}
    for base in ("ship", "both_l1"):
        pts = [(A[nm]["nper"], A[nm]["ms_min"]) for nm in A
               if nm.startswith("n") and nm.endswith("_" + base) and nm[1:3].isdigit()
               and A[nm]["inst"] == "synced"]
        if len(pts) >= 3:
            pts.sort()
            L, c, r = fit([p[0] for p in pts], [p[1] for p in pts])
            nl[base] = {"points": pts, "floor_L_ms": L, "marginal_call_ms": c, "max_resid_ms": r,
                        "synced_n1_ms": ms("n01_" + base),
                        "traced_n1_ms": ms("tr01_" + base),
                        "traced_n8_per_call_ms": (ms("tr08_" + base) / 8.0
                                                  if ms("tr08_" + base) else None),
                        "traced_n16_per_call_ms": (ms("tr16_" + base) / 16.0
                                                   if ms("tr16_" + base) else None)}
            if nl[base]["synced_n1_ms"]:
                nl[base]["floor_frac_of_n1"] = L / nl[base]["synced_n1_ms"]
    out["C0_launch_floor"] = nl

    # ---- K ladders on both instruments: the fixed term, reproduced and corrected -----
    ladders = {}
    for base, key in (("ship", "synced_x1"), ("ship", "marginal"), ("both_l1", "marginal"),
                      ("out_l1", "marginal"), ("in_l1", "marginal"), ("ship", "traced")):
        pts = []
        for nm, r in A.items():
            if r.get("kt") is None:
                continue
            if key == "synced_x1" and nm.endswith("_%s_x%02d" % (base, 4)):
                continue
            if key == "traced":
                if nm == "k%02d_%s_tr" % (r["kt"], base):
                    pts.append((r["kt"], r["ms_min"] / r["nper"]))
                continue
        if key == "marginal":
            kts = sorted({r["kt"] for r in A.values() if r.get("kt") is not None})
            for kt in kts:
                m = marginal("k%02d_%s" % (kt, base))
                if m is not None:
                    pts.append((kt, m))
        if len(pts) >= 3:
            pts.sort()
            a, b, r = fit([p[0] for p in pts], [p[1] for p in pts])
            ladders["%s.%s" % (base, key)] = {
                "points": pts, "intercept_ms": a, "slope_ms_per_kt": b, "max_resid_ms": r,
                "at_kt4_ms": a + 4 * b, "fixed_frac_at_kt4": a / (a + 4 * b),
                "breakeven_kt": a / b if b else None}
    # the synced single-bracket ladder, which is the instrument c13 used
    pts = [(A[nm]["kt"], A[nm]["ms_min"]) for nm in A
           if nm.startswith("k") and nm.endswith("_ship") and A[nm].get("kt")]
    if len(pts) >= 3:
        pts.sort()
        a, b, r = fit([p[0] for p in pts], [p[1] for p in pts])
        ladders["ship.synced_x1"] = {"points": pts, "intercept_ms": a, "slope_ms_per_kt": b,
                                     "max_resid_ms": r, "at_kt4_ms": a + 4 * b,
                                     "fixed_frac_at_kt4": a / (a + 4 * b),
                                     "breakeven_kt": a / b if b else None}
    out["K_ladders"] = ladders

    # ---- C1/C2: per-op fixed vs per-output-tile, grid mapping held fixed -------------
    tiles = {}
    for base in ("ship", "both_l1"):
        pts = []
        for b in (1, 2, 4, 8, 16, 32):
            m = marginal("b%02d_%s" % (b, base))
            if m is not None:
                pts.append((b * 16 * 16, m, b))
        if len(pts) >= 3:
            a, s, r = fit([p[0] for p in pts], [p[1] for p in pts])
            tiles["batch." + base] = {"points": [(p[2], p[0], p[1]) for p in pts],
                                      "per_op_fixed_ms": a, "per_out_tile_ms": s,
                                      "max_resid_ms": r, "at_4096_tiles_ms": a + 4096 * s,
                                      "per_op_frac": a / (a + 4096 * s)}
        pts = []
        for nn in (128, 256, 512, 1024, 2048):
            m = marginal("N%04d_%s" % (nn, base))
            if m is not None:
                pts.append((16 * 16 * (nn // 32), m, nn))
        if len(pts) >= 3:
            a, s, r = fit([p[0] for p in pts], [p[1] for p in pts])
            tiles["nladder." + base] = {"points": [(p[2], p[0], p[1]) for p in pts],
                                        "per_op_fixed_ms": a, "per_out_tile_ms": s,
                                        "max_resid_ms": r, "at_4096_tiles_ms": a + 4096 * s,
                                        "per_op_frac": a / (a + 4096 * s)}
    out["C12_per_op_vs_per_tile"] = tiles

    # ---- C3: the DRAM terms, at the intercept and at the slope ----------------------
    dram = {}
    base_l = ladders.get("ship.marginal")
    for nm in ("out_l1", "in_l1", "both_l1"):
        L = ladders.get(nm + ".marginal")
        if L and base_l:
            dram[nm] = {"intercept_ms": L["intercept_ms"],
                        "d_intercept_ms": base_l["intercept_ms"] - L["intercept_ms"],
                        "slope_ms_per_kt": L["slope_ms_per_kt"],
                        "d_slope_ms_per_kt": base_l["slope_ms_per_kt"] - L["slope_ms_per_kt"],
                        "at_kt4_ms": L["at_kt4_ms"],
                        "d_at_kt4_ms": base_l["at_kt4_ms"] - L["at_kt4_ms"]}
    out["C3_dram"] = dram

    # ---- C4/C6: core shape at constant count, and core count ------------------------
    grid = {}
    for base in ("ship", "both_l1"):
        rows = []
        for nm, r in A.items():
            if r.get("gc") and nm.endswith("_%s_x%02d" % (base, 4 if base == "ship" else 2)):
                pair = r["pair"]
                m = marginal(pair)
                if m is not None:
                    rows.append({"gx": r["gx"], "gy": r["gy"], "cores": r["gc"],
                                 "marginal_ms": m})
        rows.sort(key=lambda x: (x["cores"], x["gx"]))
        grid[base] = rows
        by_count = {}
        for x in rows:
            by_count.setdefault(x["cores"], []).append(x)
        grid[base + ".shape_at_fixed_count"] = {
            str(c): {"%dx%d" % (x["gx"], x["gy"]): x["marginal_ms"] for x in v}
            for c, v in by_count.items() if len(v) > 1}
    out["C46_grid"] = grid

    # ---- C5: blocks per output, dest-drain granularity, K-block granularity ----------
    pipe = {}
    for base in ("ship", "both_l1"):
        pipe["out_block_h." + base] = {
            str(A[nm]["obh"]): {"blocks": A[nm]["blocks"], "marginal_ms": marginal(A[nm]["pair"])}
            for nm in A if A[nm].get("obh") is not None
            and nm.endswith("_%s_x%02d" % (base, 4 if base == "ship" else 2))}
        pipe["out_subblock." + base] = {
            "%dx%d" % (A[nm]["sh"], A[nm]["sw"]): marginal(A[nm]["pair"])
            for nm in A if A[nm].get("sw") is not None
            and nm.endswith("_%s_x%02d" % (base, 4 if base == "ship" else 2))}
    pipe["in0_block_w.ship"] = {str(A[nm]["ibw"]): marginal(A[nm]["pair"])
                                for nm in A if A[nm].get("ibw") is not None
                                and nm.endswith("_ship_x04")}
    pipe["flat2d_vs_4d"] = {"flat2d_full_grid": marginal("flat2d_ship"),
                            "flat2d_g8x8": marginal("flat2d_g8x8_ship"),
                            "4d_full_grid": marginal("k04_ship")}
    out["C5_pipeline"] = pipe

    # ---- controls ------------------------------------------------------------------
    ctl = {}
    for nm in ("ctl_cube_ship", "ctl_cube_ship_aa", "ctl_cube_roof", "ctl_bwadd",
               "ctl_bwadd_aa"):
        if nm in A:
            ctl[nm] = {"ms_min": A[nm]["ms_min"], "tflops": A[nm].get("tflops"),
                       "gbs": A[nm].get("gbs"), "spread_pct": A[nm]["spread_pct"]}
    if "ctl_cube_ship" in ctl and "ctl_cube_ship_aa" in ctl:
        ctl["cube_aa_pct"] = 100.0 * abs(ctl["ctl_cube_ship_aa"]["ms_min"]
                                         - ctl["ctl_cube_ship"]["ms_min"]) \
            / ctl["ctl_cube_ship"]["ms_min"]
    if "ctl_bwadd" in ctl and "ctl_bwadd_aa" in ctl:
        ctl["bwadd_aa_pct"] = 100.0 * abs(ctl["ctl_bwadd_aa"]["ms_min"]
                                          - ctl["ctl_bwadd"]["ms_min"]) \
            / ctl["ctl_bwadd"]["ms_min"]
    if "n16_ship" in A and "n16_ship_aa" in A:
        ctl["n16_aa_pct"] = 100.0 * abs(A["n16_ship_aa"]["ms_min"] - A["n16_ship"]["ms_min"]) \
            / A["n16_ship"]["ms_min"]
    ctl["max_spread_pct"] = max(r["spread_pct"] for r in A.values())
    ctl["worst_spread_arm"] = max(A, key=lambda n: A[n]["spread_pct"])
    out["controls"] = ctl

    p = Path(sys.argv[1])
    dest = p.with_name("decomp_%s.json" % d["tag"])
    dest.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))
    print("\nwrote %s" % dest, file=sys.stderr)


if __name__ == "__main__":
    main()
