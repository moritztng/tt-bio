#!/usr/bin/env python3
"""Does upstream's OWN bf16 cotangent rot with depth the way ours does?

The depth trend is only ours if the floor does not have it. This scores the cotangent arriving
at each DiT `conditioned_transition.layer_norm.layer_norm_s` for THREE arms against the float64
reference at the same site: ours, upstream's own bf16 autocast, and (as the instrument floor)
nothing -- the reference against itself is zero by construction and is not reported.
"""
import json
import re
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

B = re.compile(r"^diffusion_transformer\.blocks\.(\d+)\.")
LEAF = "conditioned_transition.layer_norm.layer_norm_s.weight"

ref = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["sites"]
bf = torch.load(sys.argv[2], map_location="cpu", weights_only=False)["sites"]
dev = torch.load(sys.argv[3], map_location="cpu", weights_only=False)["sites"]
out = sys.argv[4]


def stat(a, b):
    a, b = a.reshape(-1).double(), b.reshape(-1).double()
    na, nb = float(a.norm()), float(b.norm())
    return {"rel": float((a - b).norm() / nb), "cos": float((a @ b) / (na * nb)),
            "norm_ratio": na / nb}


rows = {}
for nm, r in sorted(ref.items()):
    m = B.match(nm)
    if not m or not nm.endswith(LEAF) or nm not in bf or nm not in dev:
        continue
    k = int(m.group(1))
    Gr = r["G"]
    Gb = bf[nm]["G"]
    pc = dev[nm].get("per_call") or []
    if Gb.shape != Gr.shape or not pc:
        continue
    rpc = pc[0][1].shape[0]
    Gd = torch.cat([g for _, g in pc], dim=0)
    if Gd.shape != Gr.shape:
        continue
    rows[k] = {"ours": stat(Gd, Gr), "floor_bf16": stat(Gb, Gr), "rows": Gr.shape[0],
               "rows_per_call": rpc}

ks = sorted(rows)
oc = np.array([rows[k]["ours"]["cos"] for k in ks])
fc = np.array([rows[k]["floor_bf16"]["cos"] for k in ks])
orl = np.array([rows[k]["ours"]["rel"] for k in ks])
frl = np.array([rows[k]["floor_bf16"]["rel"] for k in ks])
rep = {"what": __doc__.strip().splitlines()[0], "leaf": LEAF, "blocks": len(ks),
       "spearman_cos_vs_block": {"ours": spearmanr(oc, ks).statistic,
                                 "ours_p": spearmanr(oc, ks).pvalue,
                                 "floor_bf16": spearmanr(fc, ks).statistic,
                                 "floor_bf16_p": spearmanr(fc, ks).pvalue},
       "median_rel": {"ours": float(np.median(orl)), "floor_bf16": float(np.median(frl)),
                      "ours_over_floor": float(np.median(orl) / np.median(frl))},
       "median_cos": {"ours": float(np.median(oc)), "floor_bf16": float(np.median(fc))},
       "by_block": rows}
json.dump(rep, open(out, "w"), indent=1, sort_keys=True)

print(f"{'blk':>3} | {'ours rel':>9} {'ours cos':>9} {'ours r':>7} | "
      f"{'floor rel':>10} {'floor cos':>10} {'floor r':>8} | {'x':>7}")
for k in ks:
    o, f = rows[k]["ours"], rows[k]["floor_bf16"]
    print(f"{k:3d} | {o['rel']:9.4f} {o['cos']:+9.3f} {o['norm_ratio']:7.3f} | "
          f"{f['rel']:10.4f} {f['cos']:+10.3f} {f['norm_ratio']:8.3f} | "
          f"{o['rel'] / f['rel']:7.2f}")
print(json.dumps({k: rep[k] for k in ("spearman_cos_vs_block", "median_rel", "median_cos")},
                 indent=1))
