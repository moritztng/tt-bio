#!/usr/bin/env python3
"""of3t-apbleaf: the trunk reading under a substitution at the 96 layer_norm_a tensors.

Two substitutions into the banked device gradient, rescored over the same 2,736 trunk leaf
gradients, the same denominator and the same statistic the graded artifact uses:

    CF_perfect      the 96 replaced by upstream's float64 gradient   the leaf pair's CEILING
    CF_mechanism    the 96 replaced by dW_f64dev                     this row's mechanism exact

The frame is named on every line and it is the one D218 fixed: ours against upstream's own bf16
autocast at padded 384, where the shipped trunk reads 1.0293953377723410 and the in-frame A26
bar is 0.5268825372815341, a factor of 1.9537x. `0.4361680548` and "2.360x" are cross-frame and
withdrawn; they appear nowhere here.

`CF_perfect` bounds `CF_mechanism`: zeroing a tensor's error removes all of it where an arm
removes some, so if the ceiling does not reach the bar then no arm on this leaf set closes
GRADIENTS, whatever the mechanism turns out to be.
"""
from __future__ import annotations

import argparse
import json
import math

import torch

PRE = "pairformer_stack.blocks."
LEAF = "attn_pair_bias.layer_norm_a."
BAR = 0.5268825372815341
SHIPPED = 1.0293953377723410


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


def score(ref, arm, keys):
    d2 = r2 = a2 = dot = 0.0
    for k in keys:
        r, a = ref[k], arm[k]
        d2 += float((a - r).pow(2).sum()); r2 += float(r.pow(2).sum())
        a2 += float(a.pow(2).sum()); dot += float((a * r).sum())
    return {"abs_err": math.sqrt(d2), "ref_norm": math.sqrt(r2),
            "mass_weighted_rel_l2": math.sqrt(d2 / r2),
            "mass_weighted_norm_ratio": math.sqrt(a2 / r2),
            "mass_weighted_cos": dot / math.sqrt(a2 * r2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--f64", required=True)
    ap.add_argument("--bf16auto", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--mech-cf", default="", help="mech.py's --cf-out, the dW_f64dev tensors")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    f64, bf16, ours = load(a.f64), load(a.bf16auto), load(a.ours)
    keys = sorted(set(f64) & set(bf16) & set(ours))
    leafkeys = [k for k in keys if LEAF in k]

    arms = {"SHIPPED": dict(ours)}
    perfect = dict(ours)
    for k in leafkeys:
        perfect[k] = f64[k].clone()
    arms["CF_perfect"] = perfect
    subbed = 0
    if a.mech_cf:
        m = torch.load(a.mech_cf, map_location="cpu", weights_only=False)["dW_f64dev"]
        mech = dict(ours)
        for k in leafkeys:
            if k in m and tuple(m[k].shape) == tuple(ours[k].shape):
                mech[k] = m[k].to(torch.float64).reshape(-1).clone()
                subbed += 1
        arms["CF_mechanism"] = mech

    R = {"what": __doc__.strip().splitlines()[0],
         "frame": "ours against upstream's own bf16 autocast, padded 384 (D218)",
         "bar_in_frame": BAR, "shipped_in_frame": SHIPPED,
         "tensors_scored": len(keys), "leaf_tensors_substituted": len(leafkeys),
         "mechanism_tensors_substituted": subbed, "arms": {}}
    for nm, arm in arms.items():
        e = {}
        for fr, ref in (("ours_vs_bf16auto", bf16), ("ours_vs_f64", f64)):
            e[fr] = score(ref, arm, keys)
            e[fr + "_leaf96"] = score(ref, arm, leafkeys)
        t = e["ours_vs_bf16auto"]["mass_weighted_rel_l2"]
        e["factor_over_bar"] = t / BAR
        e["passes_bar"] = t <= BAR
        e["error_mass_removed_pct"] = 100.0 * (1.0 - (t / SHIPPED) ** 2)
        R["arms"][nm] = e
        print("%-14s trunk(bf16auto)=%.10f  %.4fx bar  passes=%s   trunk(f64)=%.10f"
              % (nm, t, e["factor_over_bar"], e["passes_bar"],
                 e["ours_vs_f64"]["mass_weighted_rel_l2"]))
    json.dump(R, open(a.out, "w"), indent=2)
    print("wrote " + a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
