#!/usr/bin/env python3
"""The FLOOR's width curve at 64, 128, 256 and 384, and the reference's, with no device arm.

of3t-frame384 established the floor is flat between 64 and 384. Two points cannot tell a flat
curve from a curve that leaves and returns, and D175 was refuted for exactly that reason: the
width response is non-monotone and shape-keyed, so nothing here fits a law to two points.

There is no device arm at 128 or 256. This row holds no card, and of3t-padshape's banked
intermediate arms under /tmp/of3t/of3t-padshape no longer exist on qb1 or qb2. So this is the
floor and the reference only, and it is labelled that way everywhere it is quoted.
"""
import argparse, json, math, os, sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "of3t_trunkg043"))
from score import triple                                                    # noqa: E402

PRE = "pairformer_stack.blocks."
# the tensors of3t-widthattr found carrying the growth, so the per-tensor flatness of the floor
# is shown where the growth actually is and not only trunk-wide
CARRIERS = [f"{PRE}4.attn_pair_bias.layer_norm_a.weight",
            f"{PRE}44.attn_pair_bias.layer_norm_a.bias",
            f"{PRE}44.attn_pair_bias.layer_norm_a.weight",
            f"{PRE}0.single_transition.layer_norm.bias",
            f"{PRE}0.single_transition.layer_norm.weight",
            f"{PRE}42.attn_pair_bias.layer_norm_a.bias"]


def trunk(p):
    d = torch.load(p, map_location="cpu", weights_only=False)
    g = d["grads"] if isinstance(d, dict) and "grads" in d else d
    return {k: v.to(torch.float64) for k, v in g.items()
            if k.startswith(PRE) and v is not None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", required=True,
                    metavar="W=F64PATH,BF16PATH")
    ap.add_argument("--base-width", type=int, default=64)
    ap.add_argument("--host", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    pairs = {}
    for spec in a.pair:
        w, _, ps = spec.partition("=")
        f64p, bf16p = ps.split(",")
        pairs[int(w)] = (f64p, bf16p)
    widths = sorted(pairs)

    base = trunk(pairs[a.base_width][0])
    keys = sorted(base)
    den = sum(float(torch.linalg.vector_norm(base[k])) ** 2 for k in keys)

    out = {"what": __doc__.strip().splitlines()[0], "scored_on": os.uname().nodename,
           "host_that_built_every_reference_and_floor_here": a.host,
           "device_arm_at_128_or_256": "none -- this row holds no card and of3t-padshape's "
                                       "banked intermediate arms no longer exist on either box",
           "base_width": a.base_width, "widths": widths, "rows": {}}
    for w in widths:
        f64p, bf16p = pairs[w]
        r = trunk(f64p)
        b = trunk(bf16p)
        ref_sq = sum(float(torch.linalg.vector_norm(r[k])) ** 2 for k in keys)
        floor_e2 = sum(float((b[k].reshape(-1) - r[k].reshape(-1)).pow(2).sum()) for k in keys)
        ref_e2 = sum(float((r[k].reshape(-1) - base[k].reshape(-1)).pow(2).sum()) for k in keys)
        out["rows"][str(w)] = {
            "reference_squared_gradient_norm": ref_sq,
            "reference_vs_base_width_mass_weighted_rel_l2": math.sqrt(ref_e2 / den),
            "floor_mass_weighted_rel_l2": math.sqrt(floor_e2 / ref_sq),
            "floor_squared_gradient_norm": sum(
                float(torch.linalg.vector_norm(b[k])) ** 2 for k in keys),
            "per_carrier_floor_rel_l2": {k: triple(b[k], r[k])[0] for k in CARRIERS},
            "per_carrier_reference_norm": {k: float(torch.linalg.vector_norm(r[k]))
                                           for k in CARRIERS}}
        del r, b

    f = [out["rows"][str(w)]["floor_mass_weighted_rel_l2"] for w in widths]
    out["FLAT"] = {"floor_min": min(f), "floor_max": max(f),
                   "floor_spread_relative": (max(f) - min(f)) / min(f),
                   "reference_squared_norm_spread_relative":
                       (max(out["rows"][str(w)]["reference_squared_gradient_norm"] for w in widths)
                        - min(out["rows"][str(w)]["reference_squared_gradient_norm"] for w in widths))
                       / min(out["rows"][str(w)]["reference_squared_gradient_norm"] for w in widths)}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({w: {"floor": out["rows"][str(w)]["floor_mass_weighted_rel_l2"],
                          "ref_vs_base": out["rows"][str(w)]["reference_vs_base_width_mass_weighted_rel_l2"]}
                      for w in widths} | {"FLAT": out["FLAT"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
