#!/usr/bin/env python3
"""The two single-track LayerNorm output cotangents, ours against the reference's own.

The left column comes out of the AttentionPairBias backward and the right one out of the
Transition backward, in the same block, on the same track, in the same process. `of3t-bwdaccum`
published this table with the left column 0.98x to 33.30x at cos 0.019 to 0.920 and the right at
cos 1.000; this re-reads it with the repair on, and the right column is the control that must
not move.
"""
import argparse, json, torch


def st(m, r):
    m, r = m.reshape(-1).to(torch.float64), r.reshape(-1).to(torch.float64)
    nr, nm = float(r.norm()), float(m.norm())
    return {"rel": float((m - r).norm() / (nr or 1.0)), "r": nm / nr if nr else float("inf"),
            "cos": float((m @ r) / (nm * nr)) if nm and nr else float("nan"),
            "ref_norm": nr, "our_norm": nm}


ap = argparse.ArgumentParser()
ap.add_argument("--refsites", required=True)
ap.add_argument("--arm", action="append", required=True, metavar="NAME=PATH")
ap.add_argument("--out", required=True)
a = ap.parse_args()

ref = torch.load(a.refsites, map_location="cpu", weights_only=False)["site_cot"]
rep = {"what": __doc__.strip().splitlines()[0], "arms": {}}
LEAF = {"pre_norm_s_weight": "pre_norm_s", "transition_s.norm_weight": "transition_s.norm"}
for spec in a.arm:
    name, _, path = spec.partition("=")
    sites = torch.load(path, map_location="cpu", weights_only=False)["sites"]
    rows = {}
    for s in sites:
        p = s["gamma_path"]
        blk = p.split(".")[1]
        leaf = p.split(".", 2)[2]
        if leaf not in LEAF:
            continue
        key = f"blocks.{blk}.{LEAF[leaf]}"
        if key not in ref or ref[key] is None:
            continue
        rows.setdefault(int(blk), {})[LEAF[leaf]] = st(
            s["g"].reshape(ref[key].shape).to(torch.float64), ref[key])
    rep["arms"][name] = {str(k): v for k, v in sorted(rows.items())}

for name, rows in rep["arms"].items():
    print("==", name)
    print("  blk | attn_pair_bias.layer_norm_a out          | single_transition.layer_norm out")
    for blk in sorted(rows, key=int):
        v = rows[blk]
        A = v.get("pre_norm_s"); T = v.get("transition_s.norm")
        if not A or not T:
            continue
        print("  %3s | ref %9.3e ours %9.3e r %7.3f cos %7.4f | r %6.3f cos %7.4f rel %9.3e"
              % (blk, A["ref_norm"], A["our_norm"], A["r"], A["cos"], T["r"], T["cos"], T["rel"]))
with open(a.out, "w") as fh:
    json.dump(rep, fh, indent=2)
