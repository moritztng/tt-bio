#!/usr/bin/env python3
"""The reference mass's concentration per draw and the tensors that decide the bf16-mass field.

Per draw: effective tensor count 1/sum(p^2) of the F64 mass shares; the top tensors with share,
ours / bf16 rel, and ours' norm_ratio and cos (floor vs wrong-computation tell); and the
counterfactual field if every trunk tensor, or each named tensor, is credited as a win.
    topmass.py --draw 1 --draw 2 --draw 3 --draw 4 --out TOPMASS.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "of3t_fullstep64"))
from score import head_of  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--draw", type=int, action="append", required=True)
ap.add_argument("--top", type=int, default=8)
ap.add_argument("--out", required=True)
a = ap.parse_args()
WATCH = ["diffusion_module.diffusion_transformer.blocks.10.attention_pair_bias.layer_norm_a.layer_norm_s.weight",
         "aux_heads.distogram.linear.weight"]
rep = {}
for k in a.draw:
    d = torch.load(f"/home/ttuser/of3t_sigma/scored_s{k}.pt", weights_only=False)
    g, o, b = d["f64"], d["ours"], d["bf16"]
    m = {n: float((v * v).sum()) for n, v in g.items()}
    tot = sum(m.values())
    row = {}
    for n in g:
        ro = (float(((o[n] - g[n]) ** 2).sum()) / m[n]) ** .5
        rb = (float(((b[n] - g[n]) ** 2).sum()) / m[n]) ** .5
        on = float((o[n] * o[n]).sum()) ** .5
        row[n] = {"share": m[n] / tot, "ours": ro, "bf16": rb, "norm_ratio": on / m[n] ** .5,
                  "cos": float((o[n] * g[n]).sum()) / (on * m[n] ** .5) if on else 0.0}
    field = sum(r["share"] for r in row.values() if r["ours"] <= r["bf16"])
    lost_trunk = sum(r["share"] for n, r in row.items() if head_of(n) == "trunk" and r["ours"] > r["bf16"])
    top = sorted(row, key=lambda n: -row[n]["share"])[:a.top]
    rep[f"s{k}"] = {"field": field,
                    "n_eff": 1 / sum(r["share"] ** 2 for r in row.values()),
                    "trunk_share": sum(r["share"] for n, r in row.items() if head_of(n) == "trunk"),
                    "field_if_trunk_all_won": field + lost_trunk,
                    "top": {n: row[n] for n in top},
                    "watch": {n: row[n] for n in WATCH if n in row}}
    r = rep[f"s{k}"]
    print(f"== s{k} field {field:.4f} n_eff {r['n_eff']:.1f} trunk_share {r['trunk_share']:.4f} "
          f"field_if_trunk_all_won {r['field_if_trunk_all_won']:.4f}")
    for n in dict.fromkeys(top + WATCH):
        x = row[n]
        print(f"  {x['share']:.4f} ours {x['ours']:.3f} bf16 {x['bf16']:.3f} nr {x['norm_ratio']:.3f} cos {x['cos']:.4f} {n}")
Path(a.out).write_text(json.dumps(rep, indent=1) + "\n")
