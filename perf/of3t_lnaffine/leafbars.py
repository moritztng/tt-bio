#!/usr/bin/env python3
"""Per leaf and per block, our trunk gradient against BOTH denominators, with A14 applied.

A27: a relative error is meaningless until its denominator is named, and this row has three
that give different answers on the same tensor.

  D-F64    ||ours - f64|| / ||f64||            upstream 0.4.3 in float64
  D-FLOOR  ||theirs_bf16 - f64|| / ||f64||     upstream's OWN bf16, scored the same way. A26's
                                               reachability bar is sqrt(2) x this.
  D-BF16   ||ours - theirs_bf16|| / ||theirs_bf16||   ours against their recipe directly

A14 (reference norm floor 1e-12, the campaign's own constant) is reported rather than silently
applied, because the leaf this row is about is NOT what A14 removes and the distinction decides
whether rel 3.04 is a defect or an artefact.

Note on the error-mass statistic the campaign ranks by. With mass_t = ||ref_t||^2 / S,

    error_mass_t = mass_t * rel_t^2 = ||ours_t - ref_t||^2 / S

the reference norm cancels exactly. So the ranking of leaves by error mass is A14-invariant by
construction; only the per-leaf `rel` recovered from it, sqrt(error_mass / mass), is not.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

A14_FLOOR = 1e-12
BAR_MW = 2.0e-02


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


def leaf_of(k):
    return k.split(".", 3)[3]


def block_of(k):
    return int(k.split(".")[2])


def agg(rows, key):
    """Mass-weighted rel over a group, and the group's share of the total error mass."""
    tot_err = sum(r["err_sq"] for r in rows)
    out = {}
    for r in rows:
        a = out.setdefault(key(r["tensor"]), {"n": 0, "ref_sq": 0.0, "err_sq": 0.0,
                                              "a14_dropped": 0, "min_ref_norm": float("inf")})
        a["n"] += 1
        a["ref_sq"] += r["ref_sq"]
        a["err_sq"] += r["err_sq"]
        a["min_ref_norm"] = min(a["min_ref_norm"], r["ref_norm"])
        if r["ref_norm"] < A14_FLOOR:
            a["a14_dropped"] += 1
    for a in out.values():
        a["rel"] = float(np.sqrt(a["err_sq"] / a["ref_sq"])) if a["ref_sq"] else float("nan")
        a["share_of_the_error_mass"] = a["err_sq"] / tot_err if tot_err else 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-f64", required=True)
    ap.add_argument("--ref-bf16", required=True)
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--out", required=True)
    ap.add_argument("--leaves", default="attn_pair_bias.layer_norm_a.weight,"
                                        "attn_pair_bias.layer_norm_a.bias,"
                                        "single_transition.layer_norm.bias,"
                                        "single_transition.layer_norm.weight,"
                                        "pair_stack.pair_transition.layer_norm.bias,"
                                        "pair_stack.pair_transition.layer_norm.weight")
    a = ap.parse_args()

    f64 = torch.load(a.ref_f64, map_location="cpu", weights_only=False)
    bf16 = torch.load(a.ref_bf16, map_location="cpu", weights_only=False)
    gref = {k: v for k, v in f64["grads"].items() if v is not None}
    gbf = {k: v for k, v in bf16["grads"].items() if v is not None}
    keys = sorted(set(gref) & set(gbf))

    rep = {"what": __doc__.strip().splitlines()[0],
           "A14_floor_on_the_reference_norm": A14_FLOOR,
           "denominators": {
               "D-F64": "||x - upstream_0.4.3_float64|| / ||upstream_0.4.3_float64||",
               "D-FLOOR": "D-F64 with x = upstream 0.4.3's OWN bf16 autocast arm",
               "D-BF16": "||ours - upstream_bf16|| / ||upstream_bf16||"},
           "tensors": len(keys), "arms": {}}

    # ---- the floor, scored exactly as an arm is -----------------------------------------
    fl = [dict(tensor=k, **triple(gbf[k], gref[k])) for k in keys]
    rep["floor"] = {
        "mass_weighted_rel_D_FLOOR":
            float(np.sqrt(sum(r["err_sq"] for r in fl) / sum(r["ref_sq"] for r in fl))),
        "by_leaf": agg(fl, leaf_of), "by_block": agg(fl, block_of)}

    # ---- A14 audit, on the reference every rel in this report divides by -----------------
    ref_rows = [{"tensor": k, "ref_norm": float(torch.linalg.vector_norm(
        gref[k].reshape(-1).to(torch.float64)))} for k in keys]
    dropped = [r for r in ref_rows if r["ref_norm"] < A14_FLOOR]
    tot_err_renorm = None
    a14 = {"floor": A14_FLOOR, "tensors_below_the_floor": len(dropped),
           "their_names": sorted({leaf_of(r["tensor"]) for r in dropped}),
           "worst_ref_norm": min(r["ref_norm"] for r in ref_rows),
           "worst_tensor": min(ref_rows, key=lambda r: r["ref_norm"])["tensor"]}
    rep["A14"] = a14

    for spec in a.arm:
        n, _, p = spec.partition("=")
        g = {k: v for k, v in torch.load(p, map_location="cpu",
                                         weights_only=False)["grads"].items() if v is not None}
        rows64 = [dict(tensor=k, **triple(g[k], gref[k])) for k in keys]
        rowsbf = [dict(tensor=k, **triple(g[k], gbf[k])) for k in keys]
        tot_err_renorm = sum(r["err_sq"] for r in rows64)

        kept = [r for r in rows64 if r["ref_norm"] >= A14_FLOOR]
        mw_all = float(np.sqrt(sum(r["err_sq"] for r in rows64)
                               / sum(r["ref_sq"] for r in rows64)))
        mw_a14 = float(np.sqrt(sum(r["err_sq"] for r in kept) / sum(r["ref_sq"] for r in kept)))
        med_all = float(np.median([r["rel"] for r in rows64]))
        med_a14 = float(np.median([r["rel"] for r in kept]))

        L64, LBF = agg(rows64, leaf_of), agg(rowsbf, leaf_of)
        B64, BBF = agg(rows64, block_of), agg(rowsbf, block_of)
        FL, FB = rep["floor"]["by_leaf"], rep["floor"]["by_block"]

        by_tensor = {r["tensor"]: r for r in rows64}
        bf_by_tensor = {r["tensor"]: r for r in rowsbf}
        fl_by_tensor = {r["tensor"]: r for r in fl}

        want = [x for x in a.leaves.split(",") if x]
        leaf_tbl = {}
        for lf in sorted(L64, key=lambda k: -L64[k]["share_of_the_error_mass"])[:14] + want:
            if lf in leaf_tbl:
                continue
            leaf_tbl[lf] = {
                "n": L64[lf]["n"],
                "share_of_the_error_mass": L64[lf]["share_of_the_error_mass"],
                "mass_share": L64[lf]["ref_sq"] / sum(r["ref_sq"] for r in rows64),
                "min_ref_norm_f64": L64[lf]["min_ref_norm"],
                "a14_dropped": L64[lf]["a14_dropped"],
                "ours_D_F64": L64[lf]["rel"],
                "floor_D_FLOOR": FL[lf]["rel"],
                "ours_over_floor": L64[lf]["rel"] / FL[lf]["rel"] if FL[lf]["rel"] else None,
                "ours_D_BF16": LBF[lf]["rel"],
                "inside_A26_sqrt2": (L64[lf]["rel"] <= (2 ** 0.5) * FL[lf]["rel"]),
            }

        blk_tbl = {}
        for b in sorted(B64, key=lambda k: -B64[k]["share_of_the_error_mass"])[:10]:
            blk_tbl[b] = {
                "share_of_the_error_mass": B64[b]["share_of_the_error_mass"],
                "mass_share": B64[b]["ref_sq"] / sum(r["ref_sq"] for r in rows64),
                "ours_D_F64": B64[b]["rel"], "floor_D_FLOOR": FB[b]["rel"],
                "ours_over_floor": B64[b]["rel"] / FB[b]["rel"] if FB[b]["rel"] else None,
                "ours_D_BF16": BBF[b]["rel"],
                "inside_A26_sqrt2": (B64[b]["rel"] <= (2 ** 0.5) * FB[b]["rel"])}

        # the named tensor, in all three denominators, per block, for the top-error blocks
        NAMED = "attn_pair_bias.layer_norm_a.weight"
        named = {}
        for b in (0, 4, 44, 46, 47, 22, 10):
            k = f"pairformer_stack.blocks.{b}.{NAMED}"
            if k not in by_tensor:
                continue
            o, t, f = by_tensor[k], bf_by_tensor[k], fl_by_tensor[k]
            named[b] = {"f64_norm": o["ref_norm"], "their_bf16_norm": t["ref_norm"],
                        "our_norm": o["our_norm"],
                        "their_bf16_over_f64_norm": t["ref_norm"] / o["ref_norm"],
                        "our_over_f64_norm": o["r"],
                        "ours_D_F64": o["rel"], "cos_vs_f64": o["cos"],
                        "floor_D_FLOOR": f["rel"], "cos_floor_vs_f64": f["cos"],
                        "ours_D_BF16": t["rel"], "cos_ours_vs_their_bf16": t["cos"]}

        rep["arms"][n] = {
            "path": p,
            "scope_mass_weighted_D_F64": mw_all,
            "scope_mass_weighted_D_F64_A14_applied": mw_a14,
            "A14_moves_the_scope_reading_by": mw_a14 - mw_all,
            "scope_median_rel_D_F64": med_all,
            "scope_median_rel_D_F64_A14_applied": med_a14,
            "scope_mass_weighted_D_BF16":
                float(np.sqrt(sum(r["err_sq"] for r in rowsbf)
                              / sum(r["ref_sq"] for r in rowsbf))),
            "scope_over_floor": mw_all / rep["floor"]["mass_weighted_rel_D_FLOOR"],
            "by_leaf": leaf_tbl, "by_block": blk_tbl,
            "attn_pair_bias.layer_norm_a.weight_per_block": named,
            "error_mass_held_by_A14_dropped_tensors":
                sum(r["err_sq"] for r in rows64 if r["ref_norm"] < A14_FLOOR) / tot_err_renorm,
        }

    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=1, sort_keys=True)
    print(json.dumps({"out": a.out, "tensors": len(keys),
                      "A14_dropped": a14["tensors_below_the_floor"],
                      "floor_mw": rep["floor"]["mass_weighted_rel_D_FLOOR"],
                      "arms": {k: {"mw": v["scope_mass_weighted_D_F64"],
                                   "xfloor": v["scope_over_floor"]}
                               for k, v in rep["arms"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
