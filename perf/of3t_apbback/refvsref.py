#!/usr/bin/env python3
"""The two float64 references, scored against each other with the campaign's own scorer.

REF_LOCAL is upstream 0.4.3 with every parameter and activation in float64, run on the CAPTURED
c64 boundary with the CAPTURED block-47 cotangent. REF_MODEL is upstream 0.4.3 in float64 run as
the whole model on batch_step003 at crop 384. Neither contains a device op and neither contains a
bf16 rounding, so whatever separates them is the boundary and the crop and nothing else. That
makes this the ceiling on what any repair inside our backward can buy against REF_MODEL.
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.getcwd(), "perf/of3t_trunkg043"))
from score import score, by_leaf

L = torch.load("/home/ttuser/of3t_trunkg043/ref_f64_c64.pt", map_location="cpu", weights_only=False)["grads"]
M = torch.load("/home/ttuser/of3t_refprec/bundle_ref/grads_f64_043.pt", map_location="cpu", weights_only=False)
M = M["grads"] if isinstance(M, dict) and "grads" in M else M
L = {k: v for k, v in L.items() if k.startswith("pairformer_stack.blocks.") and v is not None}
M = {k: v for k, v in M.items() if k.startswith("pairformer_stack.blocks.") and v is not None}
keys = sorted(set(L) & set(M))
s = score(L, M, keys)
rows = s.pop("_rows")
out = {"what": __doc__.strip().splitlines()[0],
       "compared": len(keys),
       "REF_LOCAL_vs_REF_MODEL": {k: s[k] for k in
           ("mass_weighted_rel_l2", "median_rel_l2_over_tensors", "mass_weighted_norm_ratio",
            "mass_weighted_cos", "reference_squared_norm", "worst_by_error_mass")},
       "leaf_error_mass": dict(list(by_leaf(rows).items())[:10])}
per_block = {}
for x in rows:
    b = int(x["tensor"].split(".")[2])
    a = per_block.setdefault(b, {"mass": 0.0, "err": 0.0})
    a["mass"] += x["mass"]; a["err"] += x["mass"] * x["rel_l2"] ** 2
tot = sum(u["err"] for u in per_block.values()) or 1.0
out["per_block_rel"] = {b: (float(np.sqrt(v["err"] / v["mass"])) if v["mass"] else None)
                        for b, v in sorted(per_block.items())}
out["per_block_error_mass_share"] = {b: v["err"] / tot for b, v in sorted(per_block.items())}
json.dump(out, open(sys.argv[1], "w"), indent=2)
print(json.dumps({"mw": s["mass_weighted_rel_l2"], "r": s["mass_weighted_norm_ratio"],
                  "cos": s["mass_weighted_cos"], "n": len(keys)}, indent=1))
print("per-block rel, 47 down to 0:")
print(" ".join("%d:%.3f" % (b, out["per_block_rel"][b]) for b in sorted(out["per_block_rel"], reverse=True)))
