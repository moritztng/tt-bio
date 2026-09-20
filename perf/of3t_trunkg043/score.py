#!/usr/bin/env python3
"""Score a gradient arm against the 0.4.3 reference, mass-weighted, with both references named.

A23: the headline is mass-weighted and never a count or a median over tensors. The weight of a
tensor is its share of the REFERENCE's squared gradient norm over the compared set, so a tensor
holding a millionth of the gradient cannot move the number by disagreeing:

    rel_mw = sqrt( sum_t w_t rel_t^2 ),   w_t = ||g_ref_t||^2 / sum_u ||g_ref_u||^2

D35: three numbers per tensor, not one. `rel_l2` alone cannot tell 18x too small from 2x too
big; `norm_ratio` and `cos` separate them outright, and `rel^2 = 1 + r^2 - 2 r c` ties the three
together so a transcription error in any of them is visible.

A27: every ratio names how its denominator arm was built. `ours_vs_float64` is against upstream
0.4.3 in float64; `ours_vs_their_bf16` is against upstream 0.4.3 with fp32 parameters under
`torch.autocast('cpu', bfloat16)`, which is their own training recipe. They are different
questions and conflating them has bitten this campaign before.

A16: the zero-gradient baseline is MEASURED, not asserted. A tensor of zeros has r = 0 and
therefore rel = 1 exactly, so every reading is scored against a ceiling of 1.0 and a 0.9 is not
a pass.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

BAR_MW = 2.0e-02
BAR_REACHABLE = BAR_MW * (2 ** 0.5)          # A26
BAR_PER_TENSOR = 5.0e-02


def triple(m, r):
    """rel_l2, norm ratio and cosine of one tensor against the reference."""
    m = m.reshape(-1).to(torch.float64)
    r = r.reshape(-1).to(torch.float64)
    nr = float(torch.linalg.vector_norm(r))
    nm = float(torch.linalg.vector_norm(m))
    rel = float(torch.linalg.vector_norm(m - r) / (nr + 1e-300))
    cos = float((m @ r) / (nm * nr + 1e-300)) if nm and nr else 0.0
    return rel, (nm / nr if nr else float("nan")), cos, nr, nm


def score(ours, ref, keys):
    rows = []
    for k in keys:
        rel, ratio, cos, nr, nm = triple(ours[k], ref[k])
        rows.append({"tensor": k, "rel_l2": rel, "norm_ratio": ratio, "cos": cos,
                     "ref_norm": nr, "our_norm": nm, "ref_sq": nr * nr})
    tot = sum(x["ref_sq"] for x in rows)
    for x in rows:
        x["mass"] = x["ref_sq"] / tot if tot else 0.0
    mw = float(np.sqrt(sum(x["mass"] * x["rel_l2"] ** 2 for x in rows)))
    rows.sort(key=lambda x: -x["rel_l2"])
    by_mass = sorted(rows, key=lambda x: -x["mass"] * x["rel_l2"] ** 2)
    return {
        "mass_weighted_rel_l2": mw,
        "median_rel_l2_over_tensors": float(np.median([x["rel_l2"] for x in rows])),
        "mass_weighted_norm_ratio": float(sum(x["mass"] * x["norm_ratio"] for x in rows)),
        "mass_weighted_cos": float(sum(x["mass"] * x["cos"] for x in rows)),
        "tensors": len(rows),
        "reference_squared_norm": tot,
        "worst_by_rel": rows[0],
        "worst_by_error_mass": by_mass[0],
        "top8_by_error_mass": by_mass[:8],
        "top8_by_rel": rows[:8],
        "over_per_tensor_bar": sum(1 for x in rows if x["rel_l2"] > BAR_PER_TENSOR),
        "over_per_tensor_bar_mass": sum(x["mass"] for x in rows if x["rel_l2"] > BAR_PER_TENSOR),
        "_rows": rows,
    }


def by_leaf(rows):
    """Error mass by LEAF NAME, which is what says whether a reading is one op or the whole block.

    error mass of t = mass_t * rel_t^2, the campaign's own formula. Reported as a share of the
    total, so 'the LayerNorm biases carry it' is a number and not an impression.
    """
    tot = sum(x["mass"] * x["rel_l2"] ** 2 for x in rows) or 1.0
    agg = {}
    for x in rows:
        leaf = x["tensor"].split(".", 3)[3]
        a = agg.setdefault(leaf, {"n": 0, "mass": 0.0, "error_mass": 0.0})
        a["n"] += 1
        a["mass"] += x["mass"]
        a["error_mass"] += x["mass"] * x["rel_l2"] ** 2
    for a in agg.values():
        a["share_of_the_error_mass"] = a["error_mass"] / tot
        a["mass_weighted_rel_l2"] = float(np.sqrt(a["error_mass"] / a["mass"])) if a["mass"] else 0.0
    return dict(sorted(agg.items(), key=lambda kv: -kv[1]["error_mass"])[:20])


def by_block_errormass(rows, nb):
    tot = sum(x["mass"] * x["rel_l2"] ** 2 for x in rows) or 1.0
    out = {}
    for i in range(nb):
        p = f"pairformer_stack.blocks.{i}."
        em = sum(x["mass"] * x["rel_l2"] ** 2 for x in rows if x["tensor"].startswith(p))
        out[i] = em / tot
    return out


def per_block(rows, nb):
    out = {}
    tot = sum(x["ref_sq"] for x in rows)
    for i in range(nb):
        p = f"pairformer_stack.blocks.{i}."
        sel = [x for x in rows if x["tensor"].startswith(p)]
        sq = sum(x["ref_sq"] for x in sel)
        if not sel or not sq:
            continue
        mw = float(np.sqrt(sum(x["ref_sq"] * x["rel_l2"] ** 2 for x in sel) / sq))
        out[i] = {"n": len(sel), "mass_share_of_the_stack": sq / tot,
                  "mass_weighted_rel_l2": mw}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-f64", required=True)
    ap.add_argument("--ref-bf16", required=True)
    ap.add_argument("--ref-f32", default="")
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--blocks", type=int, default=48)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-rows", default="",
                    help="write the full per-tensor table per arm, so the result file can be "
                         "reanalysed without re-running anything")
    a = ap.parse_args()

    f64 = torch.load(a.ref_f64, map_location="cpu", weights_only=False)
    bf16 = torch.load(a.ref_bf16, map_location="cpu", weights_only=False)
    arms = {}
    for spec in a.arm:
        n, _, p = spec.partition("=")
        arms[n] = torch.load(p, map_location="cpu", weights_only=False)

    gref = {k: v for k, v in f64["grads"].items() if v is not None}
    keys = sorted(gref)
    stack_sq = sum(float(torch.linalg.vector_norm(v)) ** 2 for v in gref.values())

    rep = {"what": __doc__.strip().splitlines()[0],
           "bars": {"mass_weighted": BAR_MW, "A26_reachable_sqrt2": BAR_REACHABLE,
                    "per_tensor": BAR_PER_TENSOR},
           "reference": {
               "float64": {"policy": f64["policy"], "tree": f64["tree"],
                           "squared_gradient_norm": stack_sq,
                           "tensors_with_a_gradient": len(gref),
                           "tensors_total": len(f64["grads"]),
                           "how_built": "upstream 0.4.3, every parameter and activation float64, "
                                        "checkpoint upcast once at load, no cast on the path"},
               "their_bf16": {"policy": bf16["policy"],
                              "how_built": "upstream 0.4.3, float32 parameters under "
                                           "torch.autocast('cpu', bfloat16), their own recipe"}},
           "scope": {"compared": len(keys), "of_their_tensors": len(f64["grads"]),
                     "share_of_the_stacks_squared_gradient_norm": 1.0,
                     "note": "every tensor of the stack carries a reference gradient and every "
                             "one is placed, so the compared set is the whole scope and its "
                             "share of the squared gradient norm is 1.0 by construction"},
           "arms": {}}

    # the floor: upstream's own bf16 against upstream's own float64, same boundary, same cotangent
    gbf = {k: v for k, v in bf16["grads"].items() if v is not None}
    floor = score(gbf, gref, keys)
    rows_floor = floor.pop("_rows")
    rep["floor_their_bf16_vs_float64"] = floor
    rep["floor_per_block"] = per_block(rows_floor, a.blocks)
    rep["floor_by_leaf_error_mass"] = by_leaf(rows_floor)
    if a.dump_rows:
        with open(f"{a.dump_rows}.FLOOR.json", "w") as fh:
            json.dump(rows_floor, fh)

    if a.ref_f32:
        f32 = torch.load(a.ref_f32, map_location="cpu", weights_only=False)
        inst = score({k: v for k, v in f32["grads"].items() if v is not None}, gref, keys)
        inst.pop("_rows")
        rep["instrument_floor_their_f32_vs_float64"] = inst

    # A16: a gradient of exact zeros. It must read exactly 1.0, or the ceiling is not where the
    # protocol says it is and no reading below it means what it looks like.
    zero = score({k: torch.zeros_like(gref[k]) for k in keys}, gref, keys)
    zero.pop("_rows")
    rep["A16_zero_gradient_baseline"] = {
        "mass_weighted_rel_l2": zero["mass_weighted_rel_l2"],
        "median_rel_l2_over_tensors": zero["median_rel_l2_over_tensors"],
        "mass_weighted_norm_ratio": zero["mass_weighted_norm_ratio"],
        "exactly_one": zero["mass_weighted_rel_l2"] == 1.0,
        "what": "measured, not asserted: a reading of 0.9 against this is not a pass"}

    for name, d in arms.items():
        g = {k: v for k, v in d["grads"].items() if v is not None}
        miss = [k for k in keys if k not in g]
        v64 = score(g, gref, keys) if not miss else None
        if v64 is None:
            rep["arms"][name] = {"error": f"{len(miss)} reference tensors have no gradient here",
                                 "missing": miss[:8]}
            continue
        rows = v64.pop("_rows")
        vbf = score(g, gbf, keys)
        vbf.pop("_rows")
        rep["arms"][name] = {
            "config": d.get("config"), "permute_cot": d.get("permute_cot", 0),
            "ours_vs_float64": v64,
            "ours_vs_their_bf16": vbf,
            "multiples_of_their_bf16_floor":
                v64["mass_weighted_rel_l2"] / floor["mass_weighted_rel_l2"],
            "inside_the_mass_weighted_bar": v64["mass_weighted_rel_l2"] <= BAR_MW,
            "inside_A26_reachable": v64["mass_weighted_rel_l2"] <= BAR_REACHABLE,
            "worse_than_2p5x_their_bf16":
                v64["mass_weighted_rel_l2"] > 2.5 * floor["mass_weighted_rel_l2"],
            "per_block": per_block(rows, a.blocks),
            "by_leaf_error_mass": by_leaf(rows),
            "block_share_of_the_error_mass": by_block_errormass(rows, a.blocks),
            "forward": {"s_norm": float(d["s"].norm()), "z_norm": float(d["z"].norm())},
        }
        if a.dump_rows:
            with open(f"{a.dump_rows}.{name}.json", "w") as fh:
                json.dump(rows, fh)

    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=2)
    print(json.dumps({
        "floor_mw": floor["mass_weighted_rel_l2"],
        "zero_baseline": rep["A16_zero_gradient_baseline"]["mass_weighted_rel_l2"],
        **{n: {"mw_vs_f64": v.get("ours_vs_float64", {}).get("mass_weighted_rel_l2"),
               "mw_vs_bf16": v.get("ours_vs_their_bf16", {}).get("mass_weighted_rel_l2"),
               "xfloor": v.get("multiples_of_their_bf16_floor")}
           for n, v in rep["arms"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
