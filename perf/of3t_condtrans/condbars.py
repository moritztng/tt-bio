#!/usr/bin/env python3
"""Our diffusion gradient and upstream 0.5.0's own bf16 gradient, against the same float64 reference.

A27: three denominators, named, because they give three different answers on the same tensor.

  D-F64    ||x - f64|| / ||f64||                      upstream 0.5.0 in float64, the reference
                                                      `sub_boundary.py` saved as S["grad_f64"]
  D-FLOOR  D-F64 with x = upstream 0.5.0's OWN bf16 autocast arm (`floor_bf16.py --policy
           bf16auto`). A26's reachability bar is sqrt(2) x this.
  D-BF16   ||ours - theirs_bf16|| / ||theirs_bf16||

Scored ONLY over the tensors our device arm compares, so the floor and our reading are the same
set. A14 (reference-norm floor 1e-12) is reported, not silently applied, and A16's zero baseline
is measured rather than assumed.

The 30 instances of the leaf are NOT all in the diffusion transformer: 24 are, 3 are in
atom_attn_enc.atom_transformer and 3 in atom_attn_dec.atom_transformer. They are grouped by site.
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import torch

A14_FLOOR = 1e-12
LEAF = "conditioned_transition.layer_norm.layer_norm_s.weight"
SITE = re.compile(r"^(.*)\.blocks\.(\d+)\.(.*)$")


def triple(m, r):
    m = m.reshape(-1).to(torch.float64)
    r = r.reshape(-1).to(torch.float64)
    nr = float(torch.linalg.vector_norm(r))
    nm = float(torch.linalg.vector_norm(m))
    err = float(torch.linalg.vector_norm(m - r))
    cos = float((m @ r) / (nm * nr)) if nm and nr else float("nan")
    return {"rel": err / nr if nr else float("inf"), "err_sq": err * err,
            "r": (nm / nr if nr else float("nan")), "cos": cos,
            "ref_norm": nr, "our_norm": nm, "ref_sq": nr * nr}


def split(k):
    m = SITE.match(k)
    return (m.group(1), int(m.group(2)), m.group(3)) if m else (k.rsplit(".", 1)[0], -1, k)


def agg(rows, key):
    tot = sum(r["err_sq"] for r in rows)
    out = {}
    for r in rows:
        a = out.setdefault(key(r["tensor"]), {"n": 0, "ref_sq": 0.0, "err_sq": 0.0,
                                              "a14_dropped": 0, "min_ref_norm": float("inf")})
        a["n"] += 1
        a["ref_sq"] += r["ref_sq"]
        a["err_sq"] += r["err_sq"]
        a["min_ref_norm"] = min(a["min_ref_norm"], r["ref_norm"])
        a["a14_dropped"] += int(r["ref_norm"] < A14_FLOOR)
    for a in out.values():
        a["rel"] = float(np.sqrt(a["err_sq"] / a["ref_sq"])) if a["ref_sq"] else float("nan")
        a["share_of_the_error_mass"] = a["err_sq"] / tot if tot else 0.0
    return out


def load(p, strip=False):
    d = torch.load(p, map_location="cpu", weights_only=False)
    g = d["grads"] if "grads" in d else d
    if strip:
        g = {(k[len("diffusion_module."):] if k.startswith("diffusion_module.") else k): v
             for k, v in g.items()}
    return {k: v for k, v in g.items() if v is not None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-f64", required=True, help="sub_boundary.pt (reads S['grad_f64'])")
    ap.add_argument("--floor-bf16", required=True)
    ap.add_argument("--floor-f32", required=True)
    ap.add_argument("--ours", required=True, metavar="NAME=PATH", action="append")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    gref = {k: v for k, v in torch.load(a.ref_f64, map_location="cpu",
                                        weights_only=False)["grad_f64"].items() if v is not None}
    gbf = load(a.floor_bf16)
    gf32 = load(a.floor_f32)

    arms = {}
    for spec in a.ours:
        n, _, p = spec.partition("=")
        arms[n] = load(p, strip=True)

    keys = sorted(set(gref) & set(gbf) & set.intersection(*[set(g) for g in arms.values()]))
    rep = {"what": __doc__.strip().splitlines()[0], "leaf": LEAF,
           "A14_floor_on_the_reference_norm": A14_FLOOR,
           "denominators": {
               "D-F64": "||x - upstream_0.5.0_float64|| / ||upstream_0.5.0_float64||",
               "D-FLOOR": "D-F64 with x = upstream 0.5.0's OWN bf16 autocast arm",
               "D-BF16": "||ours - upstream_bf16|| / ||upstream_bf16||"},
           "upstream": "/home/ttuser/of3t_gradients/of3pkg (openfold3 0.5.0), NOT 0.4.3",
           "boundary": a.ref_f64, "tensors": len(keys), "arms": {}}

    fl = [dict(tensor=k, **triple(gbf[k], gref[k])) for k in keys]
    f32 = [dict(tensor=k, **triple(gf32[k], gref[k])) for k in keys]
    fl_by = {r["tensor"]: r for r in fl}
    mw = lambda rows: float(np.sqrt(sum(r["err_sq"] for r in rows)
                                    / sum(r["ref_sq"] for r in rows)))
    rep["floor"] = {"policy": "bf16auto", "scope_mass_weighted_D_FLOOR": mw(fl),
                    "scope_median_D_FLOOR": float(np.median([r["rel"] for r in fl])),
                    "by_leaf": agg(fl, lambda k: split(k)[2]),
                    "by_site": agg(fl, lambda k: split(k)[0])}
    rep["instrument_floor_f32_vs_f64"] = {
        "scope_mass_weighted": mw(f32),
        "scope_median": float(np.median([r["rel"] for r in f32])),
        "worst": max(r["rel"] for r in f32),
        "worst_tensor": max(f32, key=lambda r: r["rel"])["tensor"],
        "leaf_mass_weighted": mw([r for r in f32 if r["tensor"].endswith(LEAF)]),
        "what": "same script, same boundary, outer autocast off. It is both the harness floor "
                "and the break control on the floor instrument: bf16auto must move it."}

    # A16, measured
    zero = [dict(tensor=k, **triple(torch.zeros_like(gref[k]), gref[k])) for k in keys]
    rep["A16_zero_baseline"] = {
        "scope_mass_weighted": mw(zero), "median": float(np.median([r["rel"] for r in zero])),
        "exactly_one": all(abs(r["rel"] - 1.0) < 1e-15 for r in zero),
        "leaf_mass_weighted": mw([r for r in zero if r["tensor"].endswith(LEAF)])}

    # A14 audit on the reference every rel divides by
    rn = [(k, float(torch.linalg.vector_norm(gref[k].reshape(-1).to(torch.float64))))
          for k in keys]
    leaf_rn = [(k, v) for k, v in rn if k.endswith(LEAF)]
    rep["A14"] = {"floor": A14_FLOOR,
                  "tensors_below_the_floor": sum(1 for _, v in rn if v < A14_FLOOR),
                  "scope_min_ref_norm": min(v for _, v in rn),
                  "scope_min_tensor": min(rn, key=lambda x: x[1])[0],
                  "leaf_min_ref_norm": min(v for _, v in leaf_rn),
                  "leaf_min_tensor": min(leaf_rn, key=lambda x: x[1])[0],
                  "leaf_median_ref_norm": float(np.median([v for _, v in leaf_rn]))}

    for n, g in arms.items():
        rows = [dict(tensor=k, **triple(g[k], gref[k])) for k in keys]
        rowsbf = [dict(tensor=k, **triple(g[k], gbf[k])) for k in keys]
        by = {r["tensor"]: r for r in rows}
        bfby = {r["tensor"]: r for r in rowsbf}
        L, LF, LB = agg(rows, lambda k: split(k)[2]), rep["floor"]["by_leaf"], agg(rowsbf, lambda k: split(k)[2])

        leaf_tbl = {}
        for lf in sorted(L, key=lambda k: -L[k]["share_of_the_error_mass"])[:12] + [LEAF]:
            if lf in leaf_tbl:
                continue
            leaf_tbl[lf] = {
                "n": L[lf]["n"],
                "share_of_the_error_mass": L[lf]["share_of_the_error_mass"],
                "mass_share": L[lf]["ref_sq"] / sum(r["ref_sq"] for r in rows),
                "ours_D_F64": L[lf]["rel"], "floor_D_FLOOR": LF[lf]["rel"],
                "ours_over_floor": L[lf]["rel"] / LF[lf]["rel"] if LF[lf]["rel"] else None,
                "ours_D_BF16": LB[lf]["rel"],
                "floor_share_of_ITS_error_mass": LF[lf]["share_of_the_error_mass"],
                "inside_A26_sqrt2": bool(L[lf]["rel"] <= (2 ** 0.5) * LF[lf]["rel"])}

        per_inst = {}
        for k in keys:
            if not k.endswith(LEAF):
                continue
            s, b, _ = split(k)
            o, f, t = by[k], fl_by[k], bfby[k]
            per_inst[f"{s}.{b:02d}"] = {
                "site": s, "block": b,
                "ref_norm_f64": o["ref_norm"],
                "ours_D_F64": o["rel"], "ours_r": o["r"], "ours_cos": o["cos"],
                "floor_D_FLOOR": f["rel"], "floor_r": f["r"], "floor_cos": f["cos"],
                "ours_over_floor": o["rel"] / f["rel"] if f["rel"] else None,
                "ours_D_BF16": t["rel"],
                "inside_A26_sqrt2": bool(o["rel"] <= (2 ** 0.5) * f["rel"]),
                "our_err_sq": o["err_sq"], "floor_err_sq": f["err_sq"]}

        lrows = [r for r in rows if r["tensor"].endswith(LEAF)]
        lfl = [r for r in fl if r["tensor"].endswith(LEAF)]
        lbf = [r for r in rowsbf if r["tensor"].endswith(LEAF)]
        site_ours = agg(lrows, lambda k: split(k)[0])
        site_fl = agg(lfl, lambda k: split(k)[0])
        rep["arms"][n] = {
            "scope_mass_weighted_D_F64": mw(rows),
            "scope_median_D_F64": float(np.median([r["rel"] for r in rows])),
            "scope_over_floor": mw(rows) / rep["floor"]["scope_mass_weighted_D_FLOOR"],
            "scope_inside_A26": bool(mw(rows) <= (2 ** 0.5)
                                     * rep["floor"]["scope_mass_weighted_D_FLOOR"]),
            "LEAF": {
                "n": len(lrows),
                "ours_mass_weighted_D_F64": mw(lrows),
                "floor_mass_weighted_D_FLOOR": mw(lfl),
                "ours_over_floor": mw(lrows) / mw(lfl),
                "A26_bar_sqrt2_x_floor": (2 ** 0.5) * mw(lfl),
                "inside_A26_sqrt2": bool(mw(lrows) <= (2 ** 0.5) * mw(lfl)),
                "ours_mass_weighted_D_BF16": mw(lbf),
                "our_error_mass": sum(r["err_sq"] for r in lrows),
                "floor_error_mass": sum(r["err_sq"] for r in lfl),
                "reference_mass": sum(r["ref_sq"] for r in lrows),
                "share_of_OUR_scope_error_mass":
                    sum(r["err_sq"] for r in lrows) / sum(r["err_sq"] for r in rows),
                "share_of_the_FLOORS_scope_error_mass":
                    sum(r["err_sq"] for r in lfl) / sum(r["err_sq"] for r in fl),
                "mass_share_of_scope":
                    sum(r["ref_sq"] for r in lrows) / sum(r["ref_sq"] for r in rows),
                "instances_inside_A26": sum(1 for v in per_inst.values()
                                            if v["inside_A26_sqrt2"]),
                "by_site": {s: {"n": site_ours[s]["n"], "ours": site_ours[s]["rel"],
                                "floor": site_fl[s]["rel"],
                                "ours_over_floor": site_ours[s]["rel"] / site_fl[s]["rel"],
                                "inside_A26_sqrt2": bool(site_ours[s]["rel"]
                                                         <= (2 ** 0.5) * site_fl[s]["rel"])}
                            for s in sorted(site_ours)},
            },
            "by_leaf": leaf_tbl,
            "leaf_per_instance": per_inst,
        }

    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=1, sort_keys=True)

    for n, v in rep["arms"].items():
        L = v["LEAF"]
        print(f"\n=== arm {n}: scope D-F64 {v['scope_mass_weighted_D_F64']:.6f}  "
              f"floor {rep['floor']['scope_mass_weighted_D_FLOOR']:.6f}  "
              f"x{v['scope_over_floor']:.4f}  A26 {'ok' if v['scope_inside_A26'] else 'FAIL'}")
        print(f"    LEAF n={L['n']}  ours {L['ours_mass_weighted_D_F64']:.6f}  "
              f"floor {L['floor_mass_weighted_D_FLOOR']:.6f}  "
              f"x{L['ours_over_floor']:.4f}  bar {L['A26_bar_sqrt2_x_floor']:.6f}  "
              f"A26 {'ok' if L['inside_A26_sqrt2'] else 'FAIL'}  "
              f"inst inside {L['instances_inside_A26']}/{L['n']}")
        print(f"    leaf share of OUR error mass {L['share_of_OUR_scope_error_mass']*100:.3f} %  "
              f"vs of the FLOOR's {L['share_of_the_FLOORS_scope_error_mass']*100:.3f} %  "
              f"(mass share {L['mass_share_of_scope']*100:.3f} %)")
        for s, d in L["by_site"].items():
            print(f"    {s:34s} n={d['n']:2d} ours {d['ours']:.4f} floor {d['floor']:.4f} "
                  f"x{d['ours_over_floor']:.4f} {'ok' if d['inside_A26_sqrt2'] else 'FAIL'}")
        print(f"\n    {'site.blk':<38} {'ours':>7} {'floor':>7} {'x':>7} {'our r':>7} "
              f"{'our cos':>8} {'fl r':>7} {'fl cos':>8}  A26")
        for k, d in sorted(v["leaf_per_instance"].items(),
                           key=lambda x: -x[1]["our_err_sq"]):
            print(f"    {k:<38} {d['ours_D_F64']:7.4f} {d['floor_D_FLOOR']:7.4f} "
                  f"{d['ours_over_floor']:7.4f} {d['ours_r']:7.3f} {d['ours_cos']:+8.3f} "
                  f"{d['floor_r']:7.3f} {d['floor_cos']:+8.3f}  "
                  f"{'ok' if d['inside_A26_sqrt2'] else 'FAIL'}")
    print(json.dumps({"instrument_floor_f32_vs_f64": rep["instrument_floor_f32_vs_f64"],
                      "A16_zero_baseline": rep["A16_zero_baseline"],
                      "A14": rep["A14"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
