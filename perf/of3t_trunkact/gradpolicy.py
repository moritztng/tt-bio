#!/usr/bin/env python3
"""Trunk-scope gradient agreement across upstream's own dtype-policy axis.

The GRADIENTS clause compares our trunk against upstream's bf16 AUTOCAST step, which keeps every
layer norm, softmax and reduction in float32. A bf16 device kernel does not have those islands.
This scores our banked trunk gradient against BOTH ends of that axis over the same 2,736
tensors, so the policy's share of the disagreement is measured rather than argued.

Definition, taken from `perf/of3t_wholemodel/agreement.py` so the numbers are comparable:
    mass_weighted_rel_l2 = sqrt( sum ||arm - ref||^2 / sum ||ref||^2 )
CONTROLS: three figures `of3t-frame384` already published for this scope must come back exactly.
"""
from __future__ import annotations

import argparse
import json
import math
import os

PRE = "pairformer_stack.blocks."
FRAME = {   # of3t-frame384, FRAME_N384.json MATCHED, at crop 384 over 2736 tensors
    "ours_vs_f64": 0.8354121633458239,
    "ours_vs_bf16auto": 1.029395337772341,
    "bf16auto_vs_f64": 0.37393839211303687,
}


def load(path, unwrap="grads"):
    import torch
    d = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(d, dict) and unwrap in d and isinstance(d[unwrap], dict):
        d = d[unwrap]
    out = {}
    for k, v in d.items():
        if v is None or not hasattr(v, "shape"):
            continue
        kk = k if k.startswith(PRE) else (PRE + k if k.split(".")[0].isdigit() else k)
        if kk.startswith(PRE):
            out[kk] = v
    return out


def pair(ref, arm):
    import torch
    rows = []
    for k, r in ref.items():
        a = arm.get(k)
        if a is None:
            continue
        r = r.to(torch.float64).reshape(-1)
        a = a.to(torch.float64).reshape(-1)
        rows.append({"param": k, "ref_norm": float(r.norm()), "arm_norm": float(a.norm()),
                     "diff_norm": float((a - r).norm()), "dot": float((a * r).sum())})
    return rows


def stat(rows, label):
    sq = sum(r["ref_norm"] ** 2 for r in rows)
    d2 = sum(r["diff_norm"] ** 2 for r in rows)
    a2 = sum(r["arm_norm"] ** 2 for r in rows)
    dot = sum(r["dot"] for r in rows)
    rels = sorted((r["diff_norm"] / r["ref_norm"]) for r in rows if r["ref_norm"] > 0)
    return {"set": label, "n": len(rows),
            "mass_weighted_rel_l2": math.sqrt(d2 / sq) if sq else None,
            "mass_weighted_norm_ratio": math.sqrt(a2 / sq) if sq else None,
            "mass_weighted_cos": (dot / math.sqrt(a2 * sq)) if a2 > 0 and sq > 0 else None,
            "median_rel_l2_over_tensors": (rels[len(rels) // 2] if rels else None),
            "n_rel_measurable": len(rels),
            "n_over_per_tensor_bar": sum(1 for x in rels if x > 0.05),
            "reference_squared_norm": sq}


def by_group(rows, key):
    g = {}
    for r in rows:
        g.setdefault(key(r), []).append(r)
    return {k: stat(v, k) for k, v in sorted(g.items())}


def leaf_of(r):
    return ".".join(r["param"].split(".")[3:])


def block_of(r):
    return int(r["param"].split(".")[2])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--f64", required=True)
    ap.add_argument("--bf16auto", required=True)
    ap.add_argument("--bf16full", default="")
    ap.add_argument("--ours", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    refs = {"f64": load(a.f64), "bf16auto": load(a.bf16auto)}
    if a.bf16full:
        refs["bf16full"] = load(a.bf16full)
    arms = {}
    for spec in a.ours:
        nm, _, p = spec.partition("=")
        arms[nm] = load(p)

    R = {"what": __doc__.strip().splitlines()[0], "host": os.uname().nodename,
         "inputs": {"f64": a.f64, "bf16auto": a.bf16auto, "bf16full": a.bf16full,
                    "ours": a.ours},
         "n_tensors": {k: len(v) for k, v in list(refs.items()) + list(arms.items())},
         "stats": {}, "by_leaf": {}, "by_block": {}}

    todo = []
    for nm in arms:
        for rn in refs:
            todo.append((f"{nm}_vs_{rn}", refs[rn], arms[nm]))
    for rn in ("bf16auto", "bf16full"):
        if rn in refs:
            todo.append((f"{rn}_vs_f64", refs["f64"], refs[rn]))
    if "bf16full" in refs:
        todo.append(("bf16full_vs_bf16auto", refs["bf16auto"], refs["bf16full"]))

    for label, ref, arm in todo:
        rows = pair(ref, arm)
        R["stats"][label] = stat(rows, label)
        R["by_leaf"][label] = by_group(rows, leaf_of)
        R["by_block"][label] = by_group(rows, block_of)
        print(f"{label:34s} n={R['stats'][label]['n']:5d} "
              f"rel={R['stats'][label]['mass_weighted_rel_l2']!r} "
              f"ratio={R['stats'][label]['mass_weighted_norm_ratio']:.6f} "
              f"cos={R['stats'][label]['mass_weighted_cos']:.6f}", flush=True)

    C = {}
    for k, want in FRAME.items():
        nm = k.replace("ours", next(iter(arms)))
        got = R["stats"].get(nm, {}).get("mass_weighted_rel_l2")
        C[k] = {"published_by_of3t_frame384": want, "this_instrument": got,
                "exact": got == want,
                "rel_difference": (abs(got - want) / want) if got else None}
    R["controls_reproduce_frame384"] = C
    for k, v in C.items():
        print(f"CONTROL {k:20s} want {v['published_by_of3t_frame384']!r} "
              f"got {v['this_instrument']!r} exact={v['exact']} "
              f"rel={v['rel_difference']}")

    if "bf16full" in refs:
        nm = next(iter(arms))
        o = R["stats"][f"{nm}_vs_bf16auto"]["mass_weighted_rel_l2"]
        p = R["stats"]["bf16full_vs_bf16auto"]["mass_weighted_rel_l2"]
        R["policy_share_of_the_trunk_gradient_disagreement"] = {
            "ours_vs_bf16auto": o, "bf16full_vs_bf16auto": p,
            "ours_over_the_no_island_floor": o / p if p else None,
            "what": "bf16full_vs_bf16auto is upstream's OWN stack differing from itself by "
                    "nothing but the fp32 islands. If our arm reads close to it, the trunk's "
                    "gradient disagreement IS the policy and no op of ours carries it."}
        print(json.dumps(R["policy_share_of_the_trunk_gradient_disagreement"], indent=1))

    json.dump(R, open(a.out, "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
