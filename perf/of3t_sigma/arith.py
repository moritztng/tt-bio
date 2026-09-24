#!/usr/bin/env python3
"""What decides draw 3 bf16-mass field: per draw, total F64 mass, trunk rel ours/bf16, and the
lost mass split into named buckets, with the counterfactual field for each bucket credited.
    arith.py --draw 1 --draw 2 --draw 3 --draw 4 --out ARITH.json"""
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "of3t_fullstep64"))
from score import head_of, section_of  # noqa: E402
ap = argparse.ArgumentParser()
ap.add_argument("--draw", type=int, action="append", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
B10 = "diffusion_module.diffusion_transformer.blocks.10.attention_pair_bias.layer_norm_a.layer_norm_s.weight"
DG = "aux_heads.distogram.linear.weight"
rep = {}
for k in a.draw:
    d = torch.load(f"/home/ttuser/of3t_sigma/scored_s{k}.pt", weights_only=False)
    g, o, b = d["f64"], d["ours"], d["bf16"]
    m = {n: float((v * v).sum()) for n, v in g.items()}
    eo = {n: float(((o[n] - g[n]) ** 2).sum()) for n in g}
    eb = {n: float(((b[n] - g[n]) ** 2).sum()) for n in g}
    tot = sum(m.values())
    tr = [n for n in g if head_of(n) == "trunk"]
    lost = {n: m[n] / tot for n in g if eo[n] > eb[n]}
    bucket = {"blocks10_apb_ln_s": [n for n in lost if n == B10],
              "distogram_linear": [n for n in lost if n == DG],
              "trunk": [n for n in lost if n in set(tr)]}
    named = set().union(*bucket.values())
    bucket["other"] = [n for n in lost if n not in named]
    field = 1 - sum(lost.values())
    r = {"total_mass": tot, "field": field,
         "trunk_rel_ours": (sum(eo[n] for n in tr) / sum(m[n] for n in tr)) ** .5,
         "trunk_rel_bf16": (sum(eb[n] for n in tr) / sum(m[n] for n in tr)) ** .5,
         "trunk_mass": sum(m[n] for n in tr),
         "distogram_mass": m[DG],
         "b10": {"share": m[B10] / tot, "ours": (eo[B10] / m[B10]) ** .5, "bf16": (eb[B10] / m[B10]) ** .5},
         "lost": {kk: sum(lost[n] for n in v) for kk, v in bucket.items()}}
    r["field_if"] = {kk: field + v for kk, v in r["lost"].items()}
    r["field_if"]["blocks10+distogram"] = field + r["lost"]["blocks10_apb_ln_s"] + r["lost"]["distogram_linear"]
    rep[f"s{k}"] = r
    print(f"s{k}", json.dumps(r))
    del d, g, o, b
Path(a.out).write_text(json.dumps(rep, indent=1) + "\n")
