#!/usr/bin/env python3
"""of3t-confpfe: the rebuilt float64 reference against the one GO384 was scored with.

    ref_control.py OLD_grads_f64.pt NEW_grads_f64.pt OUT.json

Per parameter prefix: relative L2 difference |new - old| / |old| and the largest per-tensor
figure. Adding pae and pde can only reach what the confidence Pairformer's z output reaches, so
`diffusion_module.` and `aux_heads.experimentally_resolved.` must agree to float64 noise (the
host control: qb2 torch 2.8 against pc torch 2.13) and `aux_heads.pairformer_embedding.` must not.
"""
import json
import sys

import torch

PREFIXES = ("aux_heads.pairformer_embedding.", "aux_heads.experimentally_resolved.",
            "aux_heads.pae.", "aux_heads.pde.", "aux_heads.plddt.", "aux_heads.distogram.",
            "diffusion_module.", "pairformer_stack.", "msa_module.", "input_embedder.")

old, new, out = sys.argv[1:]
a, b = torch.load(old, weights_only=False), torch.load(new, weights_only=False)
res = {}
for pre in PREFIXES:
    num = den = 0.0
    worst, n, none_old, none_new = 0.0, 0, 0, 0
    for k in a:
        if not k.startswith(pre):
            continue
        ga, gb = a[k], b.get(k)
        if ga is None or gb is None:
            none_old += ga is None
            none_new += gb is None
            continue
        n += 1
        d, r = float(((gb - ga) ** 2).sum()), float((ga ** 2).sum())
        num, den = num + d, den + r
        if r > 0:
            worst = max(worst, (d / r) ** 0.5)
    res[pre] = {"n": n, "grad_none_old": none_old, "grad_none_new": none_new,
                "rel": (num / den) ** 0.5 if den else None, "worst_tensor_rel": worst,
                "old_sq": den}
json.dump({"old": old, "new": new, "by_prefix": res}, open(out, "w"), indent=1)
for pre, r in res.items():
    print(f"{pre:40s} n {r['n']:4d} none old/new {r['grad_none_old']}/{r['grad_none_new']} "
          f"rel {r['rel']} worst {r['worst_tensor_rel']:.3e}")
