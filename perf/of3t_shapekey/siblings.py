#!/usr/bin/env python3
"""AMENDMENT 1 (D195): is the leaf the width-growth decomposition names actually worse than the
leaves beside it, or is it just where the reference mass sits?

`of3t-widthattr` put 93.80 % of the width growth on LayerNorm affine parameters with
`attn_pair_bias.layer_norm_a.weight` worst at norm_ratio 7.3729 / cos -0.0057. A
difference-of-absolute-errors decomposition points at whatever holds the reference mass, so it
locates the CARRIER. `of3t-ditgap` showed the same instrument naming a leaf that was only 1.0741x
its worst cotangent-sharing sibling.

The cheap question, asked at BOTH widths: divide the named leaf's rel_l2 by each sibling's in the
same `attn_pair_bias` sub-block.

  ~1x their siblings   -> the object is a per-block factor and the candidate sites have to explain
                          a per-block factor, not a leaf
  many times           -> the leaf is the site

Two sibling groups, because they do not read the same cotangent:

  UPSTREAM  `layer_norm_a.*` and `mha.linear_{q,k,v}.*`. The LayerNorm affine gradient is
            `sum_t g_t xhat_t` where `g` is the sum of the three cotangents pulled back through
            q, k and v, and those three weight gradients are built from the same three cotangents
            one step earlier. Same flow.
  DOWNSTREAM `mha.linear_{o,g}.*`, `layer_norm_z.*`, `linear_z.*`. These read the cotangent at the
            sub-block's output or the pair branch, not the one arriving at LayerNorm's input.

Each leaf carries its OWN floor, upstream's bf16 against the same float64 reference at the same
width, so "worse than its siblings" is not read off an unfloored number.

`triple` is imported from of3t-trunkg043's `score.py`, the same definition of rel_l2, norm_ratio
and cos every row above this one uses.
"""
import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "of3t_trunkg043"))
from score import triple                                                    # noqa: E402

PRE = "pairformer_stack.blocks."
SUB = "attn_pair_bias."
NAMED = ("attn_pair_bias.layer_norm_a.weight", "attn_pair_bias.layer_norm_a.bias")
UPSTREAM = ("layer_norm_a.weight", "layer_norm_a.bias", "mha.linear_q.weight",
            "mha.linear_q.bias", "mha.linear_k.weight", "mha.linear_v.weight")


def trunk(p):
    d = torch.load(p, map_location="cpu", weights_only=False)
    g = d["grads"] if isinstance(d, dict) and "grads" in d else d
    return {k: v for k, v in g.items() if k.startswith(PRE) and v is not None}


def leaf_of(k):
    return k.split(".", 3)[3]


def block_of(k):
    return int(k.split(".")[2])


def row(ours, ref, k):
    rel, ratio, cos, rn, on = triple(ours[k], ref[k])
    return {"rel_l2": rel, "norm_ratio": ratio, "cos": cos, "ref_norm": rn, "our_norm": on}


def main():
    ap = argparse.ArgumentParser()
    for n in ("ref384", "ref64", "floor384", "floor64", "ours384", "ours64"):
        ap.add_argument("--" + n, required=True)
    ap.add_argument("--blocks", default="44,4,0",
                    help="the blocks of3t-widthattr put 74.58 % of the growth in")
    ap.add_argument("--floor-host", required=True)
    ap.add_argument("--arm-host", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ref3, ref6 = trunk(a.ref384), trunk(a.ref64)
    fl3, fl6 = trunk(a.floor384), trunk(a.floor64)
    ou3, ou6 = trunk(a.ours384), trunk(a.ours64)
    keys = sorted(set(ref3) & set(ref6) & set(fl3) & set(fl6) & set(ou3) & set(ou6))

    out = {"what": __doc__.strip().splitlines()[0],
           "scored_on": os.uname().nodename, "device_involved": False,
           "hosts": {"floor_and_local_float64_reference_built_on": a.floor_host,
                     "device_arm_built_on": a.arm_host, "scored_on": os.uname().nodename},
           "tensors_scored": len(keys),
           "paths": {"ref384": a.ref384, "ref64": a.ref64, "floor384": a.floor384,
                     "floor64": a.floor64, "ours384": a.ours384, "ours64": a.ours64},
           "sub_blocks": {}, "headline": {}, "per_block_factor": {}}

    want = [int(x) for x in a.blocks.split(",")]
    for blk in want:
        sub = [k for k in keys if block_of(k) == blk and leaf_of(k).startswith(SUB)]
        tab = {}
        for k in sorted(sub):
            leaf = leaf_of(k)[len(SUB):]
            tab[leaf] = {
                "group": "upstream" if leaf in UPSTREAM else "downstream",
                "shape": list(ref3[k].shape),
                "ours_64": row(ou6, ref6, k), "ours_384": row(ou3, ref3, k),
                "floor_64": row(fl6, ref6, k), "floor_384": row(fl3, ref3, k),
            }
            for w in ("64", "384"):
                o, f = tab[leaf][f"ours_{w}"], tab[leaf][f"floor_{w}"]
                o["over_own_floor"] = (o["rel_l2"] / f["rel_l2"]
                                       if f["rel_l2"] else float("inf"))
            tab[leaf]["width_growth_rel_l2"] = (
                tab[leaf]["ours_384"]["rel_l2"] / tab[leaf]["ours_64"]["rel_l2"]
                if tab[leaf]["ours_64"]["rel_l2"] else float("nan"))
            tab[leaf]["width_growth_of_own_floor"] = (
                tab[leaf]["floor_384"]["rel_l2"] / tab[leaf]["floor_64"]["rel_l2"]
                if tab[leaf]["floor_64"]["rel_l2"] else float("nan"))
        out["sub_blocks"][f"block{blk}"] = tab

        # One leaf per sub-block has no usable relative scale, and it is the same one every time:
        # `attn_pair_bias.layer_norm_z.bias`, whose float64 reference gradient is 1e-17 to 1e-19.
        # Divided by that, upstream's OWN bf16 floor reads 1e12-1e13, so taking it as "the worst
        # sibling" would hide every real comparison behind an artifact. The cut is on the REFERENCE
        # being denormal-scale, not on our arm's reading, and it is insensitive: every other leaf
        # in all four sub-blocks has ref_norm >= 4.28e-06, six orders above the cut.
        #
        # A large floor is NOT a reason to exclude a leaf. Several siblings here carry an upstream
        # bf16 floor above 100 % (block 4's `layer_norm_a.bias` is 85.58, `mha.linear_q.weight`
        # 10.39) and that is the regime, not an artifact -- these are gradient leaves on a
        # cancellation-limited reduction. Cutting on the floor would have deleted exactly the
        # siblings the comparison is about.
        excluded = {s: {"floor_rel_l2_64": v["floor_64"]["rel_l2"],
                        "floor_rel_l2_384": v["floor_384"]["rel_l2"],
                        "ref_norm_64": v["floor_64"]["ref_norm"],
                        "ref_norm_384": v["floor_384"]["ref_norm"]}
                    for s, v in tab.items()
                    if min(v["floor_64"]["ref_norm"], v["floor_384"]["ref_norm"]) < 1e-12}
        for s in excluded:
            tab[s]["usable_relative_scale"] = False
        for s in tab:
            tab[s].setdefault("usable_relative_scale", True)

        # the question, at both widths: the named leaf against each sibling in its own group
        h = {"EXCLUDED_no_usable_relative_scale": excluded}
        for nm in NAMED:
            leaf = nm[len(SUB):]
            if leaf not in tab:
                continue
            for w in ("64", "384"):
                mine = tab[leaf][f"ours_{w}"]["rel_l2"]
                for grp in ("upstream", "downstream"):
                    sibs = {s: v[f"ours_{w}"]["rel_l2"] for s, v in tab.items()
                            if s != leaf and v["group"] == grp and v["usable_relative_scale"]}
                    if not sibs:
                        continue
                    worst = max(sibs, key=sibs.get)
                    h[f"{leaf}@{w}_vs_{grp}"] = {
                        "named_rel_l2": mine,
                        "worst_sibling": worst, "worst_sibling_rel_l2": sibs[worst],
                        "named_over_worst_sibling": mine / sibs[worst] if sibs[worst] else float("inf"),
                        "sibling_rel_l2_range": [min(sibs.values()), max(sibs.values())],
                        "n_siblings": len(sibs),
                    }
        out["headline"][f"block{blk}"] = h

        # a per-block factor shows as the SAME width growth on every leaf of the sub-block
        gr = {s: v["width_growth_rel_l2"] for s, v in tab.items()
              if math.isfinite(v["width_growth_rel_l2"]) and v["usable_relative_scale"]}
        up = {s: g for s, g in gr.items() if tab[s]["group"] == "upstream"}
        if gr:
            vals = sorted(gr.values())
            uv = sorted(up.values())
            out["per_block_factor"][f"block{blk}"] = {
                "UPSTREAM_GROUP_width_growth_min": uv[0] if uv else None,
                "UPSTREAM_GROUP_width_growth_max": uv[-1] if uv else None,
                "UPSTREAM_GROUP_spread_max_over_min": (uv[-1] / uv[0]) if uv and uv[0] else None,
                "UPSTREAM_GROUP_n": len(uv),
                "floor_width_growth_by_leaf": {
                    s: v["width_growth_of_own_floor"] for s, v in tab.items()
                    if v["usable_relative_scale"]},
                "width_growth_rel_l2_min": vals[0], "width_growth_rel_l2_max": vals[-1],
                "median": vals[len(vals) // 2],
                "spread_max_over_min": vals[-1] / vals[0] if vals[0] else float("inf"),
                "named_leaves": {nm[len(SUB):]: gr.get(nm[len(SUB):]) for nm in NAMED},
                "by_leaf": dict(sorted(gr.items(), key=lambda kv: -kv[1])),
            }

    # The amendment's actual question in one number per block: the named leaf's excess over its
    # worst usable upstream sibling, at 64 and at 384. If that excess is width-INVARIANT while
    # every leaf's rel_l2 grows, the width growth is a per-block factor and the leaf's excess is a
    # different object.
    out["EXCESS_IS_WIDTH_INVARIANT"] = {}
    for blk in want:
        h = out["headline"].get(f"block{blk}", {})
        for nm in NAMED:
            leaf = nm[len(SUB):]
            a64 = h.get(f"{leaf}@64_vs_upstream")
            a38 = h.get(f"{leaf}@384_vs_upstream")
            if not a64 or not a38:
                continue
            out["EXCESS_IS_WIDTH_INVARIANT"][f"block{blk}.{leaf}"] = {
                "named_over_worst_upstream_sibling_at_64": a64["named_over_worst_sibling"],
                "named_over_worst_upstream_sibling_at_384": a38["named_over_worst_sibling"],
                "how_much_the_excess_itself_moves":
                    a38["named_over_worst_sibling"] / a64["named_over_worst_sibling"]
                    if a64["named_over_worst_sibling"] else float("nan"),
                "sub_block_median_width_growth":
                    out["per_block_factor"].get(f"block{blk}", {}).get("median"),
            }

    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out["EXCESS_IS_WIDTH_INVARIANT"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
