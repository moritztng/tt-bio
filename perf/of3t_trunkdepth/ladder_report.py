#!/usr/bin/env python3
"""Assemble the seven-depth ladder and run the pre-registered correlation test.

The test, its threshold and its three side conditions were fixed in
`perf/of3t_trunkdepth/PREREGISTERED.md` and pushed before the first arm ran. This script does
not choose them; it applies them and prints which pre-registered branch the data lands on.

POSITIVE needs all three: Spearman |rho| >= 0.929 (exact two-tailed p <= 0.01 at n = 7, with six
candidates screened), the candidate spanning >= 10x across the ladder, and the log-log fit
reproducing >= half the dependent's observed log-spread. The span gate is why a perfect rank
correlation on a variable that moves 2 % does not count.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

RHO_BAR = 0.929
SPAN_BAR = 10.0
FIT_SHARE_BAR = 0.5
SINGLE_TRACK_LEAVES = ("attn_pair_bias", "single_transition")


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            r[order[t]] = avg
        i = j + 1
    return r


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else 0.0


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def ols(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    if sxx == 0:
        return 0.0, my
    b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / sxx
    return b, my - b * mx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="perf/of3t_trunkdepth")
    ap.add_argument("--blocks", default="0,8,16,23,32,40,47")
    ap.add_argument("--out", default="perf/of3t_trunkdepth/LADDER.json")
    a = ap.parse_args()
    D = Path(a.dir)
    ks = [int(x) for x in a.blocks.split(",") if x != ""]
    norms = json.loads((D / "NORMS.json").read_text())["depths"]
    ig = json.loads((D / "IG_LADDER.json").read_text())["scopes"]

    rows = []
    for k in ks:
        sc = json.loads((D / f"SCORE_b{k}.json").read_text())["arms"]
        ours = sc[f"DEV_b{k}"]
        floor = sc["FLOOR_PUREBF16"]
        auto = sc.get("UPSTREAM_BF16AUTO", {})
        e = ours["mass_weighted"]["rel_l2"]
        f = floor["mass_weighted"]["rel_l2"]
        ms = sum(l["share_of_compared_ref_mass"] for l in ours.get("by_leaf", [])
                 if l["leaf"] in SINGLE_TRACK_LEAVES)
        n = norms[str(k)]
        g = ig[f"b{k}"]
        gf = ig.get(f"b{k}_FLOOR", {})
        rows.append({
            "block": k, "n_scored": ours["n_scored"],
            "ours": e, "floor": f, "ours_over_floor": e / f,
            "upstream_bf16auto": auto.get("mass_weighted", {}).get("rel_l2"),
            "zero_model": sc.get("A16_ZERO", {}).get("mass_weighted", {}).get("rel_l2"),
            "instrument_floor": sc.get("INSTRUMENT_FLOOR", {}).get("mass_weighted", {}).get("rel_l2"),
            "break": (sc.get(f"DEV_BREAK_b{k}") or {}).get("mass_weighted", {}).get("rel_l2"),
            "norm_ratio": ours["mass_weighted"]["norm_ratio"],
            "cos": ours["mass_weighted"]["cos"],
            "median_rel_l2": ours["median_rel_l2_over_tensors"],
            "ref_squared_norm": ours["mass_weighted"]["ref_squared_norm"],
            "ds_in_rel": g["ds_in"]["rel_l2"], "ds_in_r": g["ds_in"]["norm_ratio"],
            "ds_in_cos": g["ds_in"]["cos"],
            "dz_in_rel": g["dz_in"]["rel_l2"], "dz_in_r": g["dz_in"]["norm_ratio"],
            "dz_in_cos": g["dz_in"]["cos"],
            "ds_in_cos_floor": (gf.get("ds_in") or {}).get("cos"),
            "dz_in_cos_floor": (gf.get("dz_in") or {}).get("cos"),
            "Ds": 1.0 - g["ds_in"]["cos"], "Dz": 1.0 - g["dz_in"]["cos"],
            "s_in": n["s_in"]["masked"], "z_in": n["z_in"]["masked"],
            "cot_s": n["cot_s_out"]["masked"], "cot_z": n["cot_z_out"]["masked"],
            "s_in_padded": n["s_in"]["padded"], "z_in_padded": n["z_in"]["padded"],
            "single_track_mass_share": ms,
        })

    CAND = [("s_in", "||s_in|| masked"), ("z_in", "||z_in|| masked"),
            ("cot_s", "||dL/ds_out|| masked"), ("cot_z", "||dL/dz_out|| masked"),
            ("single_track_mass_share", "single-track share of reference gradient mass"),
            ("block", "depth index")]
    NORM_CAND = {"s_in", "z_in", "cot_s", "cot_z"}
    DEP = [("ours_over_floor", "OURS / FLOOR, mass-weighted, per depth"),
           ("Ds", "1 - cos(dL/ds_in), the single-track direction deficit")]

    tests = {}
    for dk, dlabel in DEP:
        dv = [r[dk] for r in rows]
        dspread = max(dv) / min(dv) if min(dv) > 0 else float("inf")
        ldv = [math.log10(v) for v in dv]
        lspread = max(ldv) - min(ldv)
        per = {}
        for ck, clabel in CAND:
            cv = [float(r[ck]) for r in rows]
            #: the depth index starts at 0 and a log-log fit needs positives. Shifting the
            #: whole column so its minimum is 1 keeps the ranking, and therefore rho, exact.
            if min(cv) <= 0:
                cv = [v - min(cv) + 1.0 for v in cv]
            rho = spearman(cv, dv)
            span = max(cv) / min(cv) if min(cv) > 0 else float("inf")
            lcv = [math.log10(v) for v in cv]
            b, c0 = ols(lcv, ldv)
            fit = [b * v + c0 for v in lcv]
            fspread = max(fit) - min(fit)
            share = fspread / lspread if lspread > 0 else 0.0
            per[ck] = {"label": clabel, "spearman_rho": rho, "span": span,
                       "loglog_slope": b, "fit_log_spread": fspread,
                       "observed_log_spread": lspread, "fit_share_of_spread": share,
                       "passes_rho": abs(rho) >= RHO_BAR, "passes_span": span >= SPAN_BAR,
                       "passes_fit": share >= FIT_SHARE_BAR,
                       "POSITIVE": bool(abs(rho) >= RHO_BAR and span >= SPAN_BAR
                                        and share >= FIT_SHARE_BAR)}
        norm_win = [c for c in NORM_CAND if per[c]["POSITIVE"]]
        mass_win = per["single_track_mass_share"]["POSITIVE"]
        depth_win = abs(per["block"]["spearman_rho"]) >= RHO_BAR
        if dspread < 3.0:
            branch = "DIFFUSE (the dependent's own spread across the ladder is under 3x)"
        elif norm_win:
            branch = "POSITIVE -- tracks activation norm: " + ", ".join(sorted(norm_win))
        elif mass_win:
            branch = "NEGATIVE-MASS -- tracks the single track's share of reference gradient mass"
        elif depth_win:
            branch = "NEGATIVE-DEPTH -- tracks depth but not norm"
        else:
            branch = "DIFFUSE -- no candidate reaches the pre-registered bar"
        tests[dk] = {"label": dlabel, "spread": dspread, "log_spread": lspread,
                     "candidates": per, "BRANCH": branch}

    out = {"what": __doc__.strip().splitlines()[0],
           "preregistered": "perf/of3t_trunkdepth/PREREGISTERED.md",
           "thresholds": {"spearman_rho": RHO_BAR, "span": SPAN_BAR,
                          "fit_share_of_log_spread": FIT_SHARE_BAR, "n": len(ks)},
           "rows": rows, "tests": tests}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")

    hdr = ("blk  n   ours        floor       o/f      r        cos       ds_in cos  dz_in cos "
           " |s_in|      |z_in|      |ds_out|    |dz_out|   single-mass")
    print(hdr)
    for r in rows:
        print(f"{r['block']:3d} {r['n_scored']:3d} {r['ours']:.4e}  {r['floor']:.4e}  "
              f"{r['ours_over_floor']:7.2f}  {r['norm_ratio']:.5f}  {r['cos']:+.5f}  "
              f"{r['ds_in_cos']:+.6f}  {r['dz_in_cos']:+.6f}  {r['s_in']:.4e}  {r['z_in']:.4e}  "
              f"{r['cot_s']:.4e}  {r['cot_z']:.4e}  {r['single_track_mass_share']*100:9.4f} %")
    for dk, t in tests.items():
        print(f"\n--- dependent: {t['label']}  (spread {t['spread']:.1f}x) ---")
        for ck, c in t["candidates"].items():
            print(f"  {ck:26s} rho {c['spearman_rho']:+.4f}  span {c['span']:10.3g}x  "
                  f"slope {c['loglog_slope']:+.3f}  fit covers {c['fit_share_of_spread']*100:5.1f} % "
                  f" -> {'POSITIVE' if c['POSITIVE'] else '.'}")
        print(f"  BRANCH: {t['BRANCH']}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
