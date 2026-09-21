#!/usr/bin/env python3
"""AMENDMENT 2: scale or wrong transform, per block, and what the answer eliminates.

The amendment offers two branches -- `r ~ 19.5 with cos ~ 1` is a scale and the search is over
a factor; `r ~ 1 with cos ~ 0` is a wrong transform. The per-tensor array says it is NEITHER
uniformly, so the useful statement is the decomposition rather than the label. Writing
`g_dev = alpha * g_ref + e` with `e` orthogonal to `g_ref`:

    alpha = r * cos                 how much of the reference our gradient reproduces
    |e| / |g_ref| = r * sin         the spurious component, in units of the reference

A pure scale is `alpha != 1, e = 0`. A pure wrong transform is `alpha = 0, |e| > 0`. The six
blocks D53 named contain both, and two of them have `alpha` NEGATIVE.

This also runs the cross-leaf table the single-leaf view cannot produce: the same block's other
AdaLN sites, so a block-level fault can be told from an op-level one. No card; the array is the
run's own output.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

MODEL_SQ = 10.279642678524985
BLOCK = re.compile(r"^diffusion_transformer\.blocks\.(\d+)\.")
NAMED = (0, 5, 6, 7, 8, 12)
# The AdaLN sites of a DiT block, and the one non-AdaLN leaf kept as a same-block control.
SITES = ["attention_pair_bias.layer_norm_a.layer_norm_s.weight",
         "attention_pair_bias.layer_norm_a.linear_g.weight",
         "attention_pair_bias.layer_norm_a.linear_g.bias",
         "attention_pair_bias.layer_norm_a.linear_s.weight",
         "conditioned_transition.layer_norm.layer_norm_s.weight",
         "conditioned_transition.layer_norm.linear_g.weight",
         "conditioned_transition.layer_norm.linear_s.weight",
         "attention_pair_bias.linear_z.weight"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tensor", type=Path,
                    default=Path("perf/of3t_rebase/device_gradient_043pt.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("perf/of3t_conditioning/ADALN_SCALE_OR_TRANSFORM.json"))
    a = ap.parse_args()
    rows = json.loads(a.per_tensor.read_text())["per_tensor"]
    by = {}
    for r in rows:
        m = BLOCK.match(r["tensor"])
        if m:
            by[(BLOCK.sub("", r["tensor"]), int(m.group(1)))] = r

    def decomp(r):
        c = r["cos"]
        rr = r["norm_ratio"]
        perp = rr * math.sqrt(max(0.0, 1.0 - c * c))
        return {"block_rel_l2": r["rel_l2"], "r": rr, "cos": c, "alpha": rr * c,
                "perp_rel": perp, "ref_norm": r["ref_norm"],
                "dev_norm": r["device_norm"], "perp_abs": perp * r["ref_norm"]}

    leaf = SITES[0]
    six = {k: decomp(by[(leaf, k)]) for k in NAMED}

    # What the spurious component is too big to be. The whole leaf family's reference mass and
    # the whole MODEL's gradient norm are both smaller than block 8's orthogonal residual, so
    # `e` is not a sum of reference-gradient pieces with bounded coefficients -- the
    # double-counting branch the eliminations record left open dies here, arithmetically.
    fam_sq = sum(by[(leaf, k)]["ref_norm"] ** 2 for k in range(24) if (leaf, k) in by)
    scale_check = {
        "block8_perp_abs": six[8]["perp_abs"],
        "whole_leaf_family_reference_norm": fam_sq ** 0.5,
        "whole_model_reference_norm": MODEL_SQ ** 0.5,
        "block8_perp_over_family_norm": six[8]["perp_abs"] / fam_sq ** 0.5,
        "block8_perp_over_model_norm": six[8]["perp_abs"] / MODEL_SQ ** 0.5,
    }

    table = {}
    for s in SITES:
        present = [(k, by[(s, k)]) for k in range(24) if (s, k) in by]
        if not present:
            continue
        d = {str(k): decomp(r) for k, r in present}
        table[s] = {
            "blocks": len(present),
            "worst_block": max(d, key=lambda k: d[k]["block_rel_l2"]),
            "worst_rel": max(v["block_rel_l2"] for v in d.values()),
            "blocks_with_negative_alpha": sorted(int(k) for k, v in d.items()
                                                 if v["alpha"] < 0),
            "max_abs_perp": max(v["perp_abs"] for v in d.values()),
            "pct_of_model": 100.0 * sum(r["ref_norm"] ** 2 for _, r in present) / MODEL_SQ,
            "per_block": d,
        }

    rep = {"what": "Amendment 2's question answered from the per-tensor array of the run the "
                   "campaign quotes. Host-only; no card opened, nothing re-measured.",
           "answer": "neither branch, and not the same branch in every block",
           "the_six_blocks": six, "spurious_is_too_big_to_be_a_sum": scale_check,
           "cross_leaf_table": table}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1, sort_keys=True, default=str) + "\n")

    print("the six blocks of", leaf)
    print("  blk      rel        r       cos     alpha    perp(rel)  perp(abs)   reading")
    for k in NAMED:
        v = six[k]
        if abs(v["cos"]) < 0.2:
            read = "ORTHOGONAL"
        elif v["alpha"] < 0:
            read = "SIGN-FLIPPED"
        elif v["perp_rel"] < 0.3 * abs(v["alpha"]):
            read = "SCALE"
        else:
            read = "SCALE + spurious"
        print(f"  {k:3d} {v['block_rel_l2']:9.4f} {v['r']:8.4f} {v['cos']:9.5f} "
              f"{v['alpha']:9.4f} {v['perp_rel']:10.4f} {v['perp_abs']:10.5f}   {read}")
    print("\nspurious component vs everything it could be a sum of:")
    for k, v in scale_check.items():
        print(f"  {k:<38} {v:.5f}")
    print("\ncross-leaf, per block (rel):")
    hdr = "  blk " + " ".join(f"{i:>8}" for i in range(24))
    print(hdr)
    for s in SITES:
        if s not in table:
            continue
        d = table[s]["per_block"]
        cells = " ".join((f"{d[str(i)]['block_rel_l2']:8.3f}" if str(i) in d else "       .")
                         for i in range(24))
        print(f"      {cells}   {s}")
    print("\nwrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
