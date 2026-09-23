#!/usr/bin/env python3
"""of3t-apbleaf: the brief's convergence, crossed at the width this row measures at.

`of3t-trunkblocks` puts 76.4502 % of the trunk's error mass in blocks 44, 4 and 0, and
`of3t-trunkact` puts 56.1174 % of it on the 96 `attn_pair_bias.layer_norm_a` tensors. The
intersection was read off the two axes rather than computed, so this computes it: per block, the
whole block's share of the trunk error mass and the layer_norm_a pair's share of the same
denominator, side by side.
"""
import argparse, json, math, torch

PRE = "pairformer_stack.blocks."
LEAF = "attn_pair_bias.layer_norm_a."


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


ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
ap.add_argument("--ours", required=True)
ap.add_argument("--frame", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
ref, ours = load(a.ref), load(a.ours)
keys = sorted(set(ref) & set(ours))
tot = sum(float((ours[k] - ref[k]).pow(2).sum()) for k in keys)
rtot = sum(float(ref[k].pow(2).sum()) for k in keys)
R = {"what": __doc__.strip().splitlines()[0], "frame": a.frame,
     "tensors": len(keys), "total_error_mass_sq": tot, "total_reference_mass_sq": rtot,
     "by_block": {}}
for b in sorted({int(k.split(".")[2]) for k in keys}):
    bk = [k for k in keys if int(k.split(".")[2]) == b]
    lk = [k for k in bk if LEAF in k]
    be = sum(float((ours[k] - ref[k]).pow(2).sum()) for k in bk)
    le = sum(float((ours[k] - ref[k]).pow(2).sum()) for k in lk)
    R["by_block"][b] = {"n": len(bk), "block_err_sq": be, "leaf_err_sq": le,
                        "block_share_of_trunk_error": be / tot,
                        "leaf_share_of_trunk_error": le / tot,
                        "leaf_share_within_block": (le / be) if be else None}
json.dump(R, open(a.out, "w"), indent=2)
top = sorted(R["by_block"].items(), key=lambda kv: -kv[1]["block_share_of_trunk_error"])[:8]
print("frame %s over %d tensors" % (a.frame, len(keys)))
print("  blk  block_share  leaf_share  leaf_within_block")
for b, v in top:
    print("  %3d   %8.4f%%   %8.4f%%        %7.4f%%"
          % (b, 100 * v["block_share_of_trunk_error"], 100 * v["leaf_share_of_trunk_error"],
             100 * v["leaf_share_within_block"]))
for b in (44, 4, 0, 45):
    v = R["by_block"][b]
    print("  named %3d  block %8.4f%%  leaf %8.4f%%  within %7.4f%%"
          % (b, 100 * v["block_share_of_trunk_error"], 100 * v["leaf_share_of_trunk_error"],
             100 * v["leaf_share_within_block"]))
s3 = sum(R["by_block"][b]["block_share_of_trunk_error"] for b in (44, 4, 0))
l3 = sum(R["by_block"][b]["leaf_share_of_trunk_error"] for b in (44, 4, 0))
print("  blocks 44+4+0 block_share %.4f%%   their layer_norm_a share %.4f%%" % (100 * s3, 100 * l3))
