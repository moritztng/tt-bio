#!/usr/bin/env python3
"""of3t-apbleaf: the per-leaf gradient error mass of the 96 `attn_pair_bias.layer_norm_a`
weight and bias tensors, recounted at padded 384.

ABSOLUTE FIRST. A share moves when its denominator collapses, so every row carries its raw
squared error mass before its share, and its share of the REFERENCE mass beside that. The
denominator is declared on every figure and it is the same one the graded trunk artifact uses:
all 2,736 trunk leaf gradients in upstream's parameter space.

THE FLOOR TRAVELS WITH THE ROW. Upstream's own bf16 autocast gradient is scored against the
same float64 on the same tensor, so a row reading 7.4 next to a floor of 3.4 is read as a 2.2x
residue on a regime, and never as a 7.4x defect.

Statistic, `perf/of3t_trunkopclass/census.py`'s and `perf/of3t_wholemodel/agreement.py`'s:

    mass_weighted_rel_l2 = sqrt( sum_k ||arm_k - ref_k||^2 / sum_k ||ref_k||^2 )
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch

PRE = "pairformer_stack.blocks."
LEAF = "attn_pair_bias.layer_norm_a"
CONTROLS = {"ours_vs_f64": 0.8354121633458239,
            "ours_vs_bf16auto": 1.029395337772341,
            "bf16auto_vs_f64": 0.37393839211303687}
CONTROL_TOL = 1e-12
NORM_FLOOR = 1e-12


def load(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(d, dict) and "grads" in d and isinstance(d["grads"], dict):
        d = d["grads"]
    out = {}
    for k, v in d.items():
        if v is None or not hasattr(v, "shape"):
            continue
        kk = k if k.startswith(PRE) else (PRE + k if k.split(".")[0].isdigit() else k)
        if kk.startswith(PRE):
            out[kk] = v.to(torch.float64).reshape(-1)
    return out


def pair(ref, arm, keys):
    d2 = r2 = a2 = dot = 0.0
    for k in keys:
        r, a = ref[k], arm[k]
        d2 += float((a - r).pow(2).sum()); r2 += float(r.pow(2).sum())
        a2 += float(a.pow(2).sum()); dot += float((a * r).sum())
    return d2, r2, a2, dot


def stat(ref, arm, keys, tot_ref_sq, tot_err_sq):
    d2, r2, a2, dot = pair(ref, arm, keys)
    return {"n": len(keys), "abs_err_sq": d2, "abs_err": math.sqrt(d2),
            "ref_sq": r2, "ref_norm": math.sqrt(r2),
            "share_of_error_mass": (d2 / tot_err_sq) if tot_err_sq else None,
            "share_of_reference_mass": (r2 / tot_ref_sq) if tot_ref_sq else None,
            "enrichment": ((d2 / tot_err_sq) / (r2 / tot_ref_sq))
                          if tot_err_sq and tot_ref_sq and r2 else None,
            "mass_weighted_rel_l2": math.sqrt(d2 / r2) if r2 else None,
            "mass_weighted_norm_ratio": math.sqrt(a2 / r2) if r2 else None,
            "mass_weighted_cos": (dot / math.sqrt(a2 * r2)) if a2 > 0 and r2 > 0 else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--f64", required=True)
    ap.add_argument("--bf16auto", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--controls", action="store_true")
    a = ap.parse_args()

    f64, bf16, ours = load(a.f64), load(a.bf16auto), load(a.ours)
    keys = sorted(set(f64) & set(bf16) & set(ours))
    leafkeys = [k for k in keys if LEAF + "." in k]

    R = {"what": __doc__.strip().splitlines()[0], "host": os.uname().nodename,
         "inputs": {"f64": a.f64, "bf16auto": a.bf16auto, "ours": a.ours},
         "denominator": {
             "tensors_scored": len(keys),
             "note": "every share below is over the trunk's own 2,736 leaf gradients in "
                     "upstream's parameter space; the leaf set is a subset of that same set",
             "leaf_tensors": len(leafkeys)},
         "norm_floor_for_relative_stats": NORM_FLOOR}

    for frame, ref, arm in (("ours_vs_bf16auto", bf16, ours),
                            ("ours_vs_f64", f64, ours),
                            ("bf16auto_vs_f64", f64, bf16)):
        td2, tr2, _, _ = pair(ref, arm, keys)
        R.setdefault("trunk", {})[frame] = stat(ref, arm, keys, tr2, td2)
        R.setdefault("leafset", {})[frame] = stat(ref, arm, leafkeys, tr2, td2)
        rows = {}
        for k in leafkeys:
            rows[k] = stat(ref, arm, [k], tr2, td2)
        R.setdefault("by_tensor", {})[frame] = rows
        byb = {}
        for b in sorted({int(k.split(".")[2]) for k in leafkeys}):
            bk = [k for k in leafkeys if int(k.split(".")[2]) == b]
            byb[b] = stat(ref, arm, bk, tr2, td2)
        R.setdefault("by_block", {})[frame] = byb
        print(f"{frame:20s} trunk rel={R['trunk'][frame]['mass_weighted_rel_l2']!r} "
              f"leaf96 share_err={R['leafset'][frame]['share_of_error_mass']:.6%} "
              f"share_ref={R['leafset'][frame]['share_of_reference_mass']:.6%} "
              f"rel={R['leafset'][frame]['mass_weighted_rel_l2']:.6f}", flush=True)

    # the residue over upstream's own bf16 floor, per tensor -- the quantity this row owns
    res = {}
    for k in leafkeys:
        o = R["by_tensor"]["ours_vs_f64"][k]["mass_weighted_rel_l2"]
        u = R["by_tensor"]["bf16auto_vs_f64"][k]["mass_weighted_rel_l2"]
        res[k] = {"ours_vs_f64": o, "upstream_bf16_vs_f64": u,
                  "residue_factor": (o / u) if u else None,
                  "ref_norm": R["by_tensor"]["ours_vs_f64"][k]["ref_norm"],
                  "ours_abs_err": R["by_tensor"]["ours_vs_f64"][k]["abs_err"],
                  "upstream_abs_err": R["by_tensor"]["bf16auto_vs_f64"][k]["abs_err"]}
    R["residue_over_upstream_floor"] = res
    fr = [(v["residue_factor"], k) for k, v in res.items() if v["residue_factor"]]
    fr.sort()
    R["residue_worst"] = [{"param": k, "factor": f} for f, k in fr[::-1][:8]]
    R["residue_best"] = [{"param": k, "factor": f} for f, k in fr[:4]]
    # mass-weighted residue: both sides' absolute error over the same 96 tensors
    oe = math.sqrt(sum(v["ours_abs_err"] ** 2 for v in res.values()))
    ue = math.sqrt(sum(v["upstream_abs_err"] ** 2 for v in res.values()))
    R["residue_mass_weighted"] = {"ours_abs_err": oe, "upstream_abs_err": ue,
                                  "factor": oe / ue if ue else None}
    # worst case per tensor, absolute and relative, located
    w_abs = max(leafkeys, key=lambda k: R["by_tensor"]["ours_vs_f64"][k]["abs_err"])
    w_rel = max((k for k in leafkeys
                 if R["by_tensor"]["ours_vs_f64"][k]["ref_norm"] >= NORM_FLOOR),
                key=lambda k: R["by_tensor"]["ours_vs_f64"][k]["mass_weighted_rel_l2"])
    R["worst_case"] = {
        "by_absolute_error": {"param": w_abs,
                              "abs_err": R["by_tensor"]["ours_vs_f64"][w_abs]["abs_err"],
                              "ref_norm": R["by_tensor"]["ours_vs_f64"][w_abs]["ref_norm"]},
        "by_relative_error": {"param": w_rel,
                              "rel": R["by_tensor"]["ours_vs_f64"][w_rel]["mass_weighted_rel_l2"],
                              "ref_norm": R["by_tensor"]["ours_vs_f64"][w_rel]["ref_norm"]}}

    C, bad = {}, []
    for nm, want in CONTROLS.items():
        got = R["trunk"][nm]["mass_weighted_rel_l2"]
        rd = abs(got - want) / want
        C[nm] = {"published": want, "this_instrument": got, "rel_difference": rd}
        print(f"CONTROL {nm:18s} want {want!r} got {got!r} rel={rd:.3e}")
        if rd > CONTROL_TOL:
            bad.append(nm)
    R["controls"] = C
    R["controls_pass"] = not bad
    json.dump(R, open(a.out, "w"), indent=2)
    if a.controls and bad:
        raise SystemExit(f"CONTROL FAILED for {bad} -- no leaf number may be read")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
