#!/usr/bin/env python3
"""P1 failed: upstream's float64 trunk gradient is NOT width-invariant tensor by tensor.

`GROWTH.json` measured the denominator as invariant to 3.2e-15 and then found 0 of 2,736 tensors
bit-identical between the two widths, worst relative 3.7289 at
`pairformer_stack.blocks.26.attn_pair_bias.layer_norm_z.bias`. Both cannot be ignored: a
reference that moves on the tensors carrying the growth would make the growth a comparison of two
references rather than a measurement of our arm.

So this scores the reference against ITSELF across the two widths in the same mass-weighted
metric everything else uses, locates where the movement sits, and prints the movement of the
reference NEXT TO the movement of our arm for each of the top growth carriers. That last column
is the one that decides whether the attribution survives.

The floor's per-tensor behaviour on the same tensors is printed beside it, because `upstream's
own bf16 is flat in width` is a trunk-wide statement and the growth lives in 20 tensors.
"""
import argparse, json, math, os, sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "of3t_trunkg043"))
from score import triple                                                    # noqa: E402

PRE = "pairformer_stack.blocks."


def trunk(p):
    d = torch.load(p, map_location="cpu", weights_only=False)
    g = d["grads"] if isinstance(d, dict) and "grads" in d else d
    return {k: v.to(torch.float64) for k, v in g.items()
            if k.startswith(PRE) and v is not None}


def main() -> int:
    ap = argparse.ArgumentParser()
    for n in ("ref384", "ref64", "floor384", "floor64", "ours384", "ours64", "growth"):
        ap.add_argument("--" + n, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    r3, r6 = trunk(a.ref384), trunk(a.ref64)
    keys = sorted(r3)
    den = sum(float(torch.linalg.vector_norm(r6[k])) ** 2 for k in keys)

    rows = {}
    for k in keys:
        rel, ratio, cos, nr, nm = triple(r3[k], r6[k])
        e2 = float((r3[k].reshape(-1) - r6[k].reshape(-1)).pow(2).sum())
        rows[k] = {"rel_l2": rel, "norm_ratio": ratio, "cos": cos,
                   "ref64_norm": nr, "ref384_norm": nm, "err_sq": e2,
                   "mass": nr * nr / den}
    mw = math.sqrt(sum(rows[k]["err_sq"] for k in keys) / den)

    by_e = sorted(keys, key=lambda k: -rows[k]["err_sq"])
    by_r = sorted(keys, key=lambda k: -rows[k]["rel_l2"])
    out = {"what": __doc__.strip().splitlines()[0],
           "scored_on": os.uname().nodename,
           "REFERENCE_AGAINST_ITSELF_ACROSS_WIDTH": {
               "compared": len(keys),
               "mass_weighted_rel_l2": mw,
               "squared_error_over_squared_norm": sum(rows[k]["err_sq"] for k in keys) / den,
               "median_rel_l2": float(torch.tensor([rows[k]["rel_l2"] for k in keys]).median()),
               "n_over_1e_12": sum(1 for k in keys if rows[k]["rel_l2"] > 1e-12),
               "n_over_1e_6": sum(1 for k in keys if rows[k]["rel_l2"] > 1e-6),
               "n_over_1e_2": sum(1 for k in keys if rows[k]["rel_l2"] > 1e-2),
               "mass_over_1e_6": sum(rows[k]["mass"] for k in keys if rows[k]["rel_l2"] > 1e-6),
               "mass_over_1e_2": sum(rows[k]["mass"] for k in keys if rows[k]["rel_l2"] > 1e-2),
               "top8_by_absolute_movement": [dict(tensor=k, **rows[k]) for k in by_e[:8]],
               "top8_by_relative_movement": [dict(tensor=k, **rows[k]) for k in by_r[:8]]}}

    # the column that decides the attribution: how far the REFERENCE moved on each tensor that
    # carries the growth, against how far OUR ARM moved on the same tensor.
    g = json.load(open(a.growth))
    o3, o6 = trunk(a.ours384), trunk(a.ours64)
    f3, f6 = trunk(a.floor384), trunk(a.floor64)
    carriers = []
    for r in g["TENSORS"]["top_by_growth"][:20]:
        k = r["tensor"]
        d_ours = float((o3[k].reshape(-1) - o6[k].reshape(-1)).pow(2).sum())
        d_ref = rows[k]["err_sq"]
        fr3 = triple(f3[k], r3[k])
        fr6 = triple(f6[k], r6[k])
        carriers.append({
            "tensor": k,
            "share_of_the_growth": r["share_of_the_growth"],
            "mass_share_of_the_trunk": rows[k]["mass"],
            "reference_moved_sq": d_ref,
            "reference_moved_rel": rows[k]["rel_l2"],
            "our_arm_moved_sq": d_ours,
            "our_arm_moved_over_reference_moved": (d_ours / d_ref) if d_ref else float("inf"),
            "ours_rel_64": r["rel_l2_64"], "ours_rel_384": r["rel_l2_384"],
            "ours_norm_ratio_64": r["norm_ratio_64"], "ours_norm_ratio_384": r["norm_ratio_384"],
            "ours_cos_64": r["cos_64"], "ours_cos_384": r["cos_384"],
            "floor_rel_64": fr6[0], "floor_rel_384": fr3[0],
            "floor_norm_ratio_64": fr6[1], "floor_norm_ratio_384": fr3[1],
            "floor_cos_384": fr3[2]})
    out["GROWTH_CARRIERS_WITH_THE_REFERENCES_OWN_MOVEMENT"] = {
        "what": "for each tensor carrying the growth: how far the float64 reference itself moved "
                "with width, how far our arm moved, and what upstream's own bf16 reads at the "
                "same tensor at both widths",
        "rows": carriers,
        "sum_reference_moved_sq_over_sum_our_arm_moved_sq":
            sum(c["reference_moved_sq"] for c in carriers)
            / sum(c["our_arm_moved_sq"] for c in carriers)}

    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({"ref_vs_ref_mw": mw,
                      "n_over_1e_6": out["REFERENCE_AGAINST_ITSELF_ACROSS_WIDTH"]["n_over_1e_6"],
                      "mass_over_1e_6": out["REFERENCE_AGAINST_ITSELF_ACROSS_WIDTH"]["mass_over_1e_6"],
                      "ref_moved_over_ours_moved":
                          out["GROWTH_CARRIERS_WITH_THE_REFERENCES_OWN_MOVEMENT"][
                              "sum_reference_moved_sq_over_sum_our_arm_moved_sq"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
