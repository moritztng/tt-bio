#!/usr/bin/env python3
"""The reduction-order floor: the SAME arm at two thread counts, scored against each other.

`arm2_f32_upstream` is bit-reproducible at a fixed `OMP_NUM_THREADS` and its gradient digest
changes when that count changes, so a byte-identity determinism control is only valid if the
thread count is pinned. This asks the quantitative follow-up: how big is that difference in the
statistic the campaign reports? It is scored arm-against-arm rather than against the float64
reference, so its denominator is the arm's own squared gradient norm and not the campaign's.
"""
import json
import math
import sys

import torch

a_path, b_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
A = torch.load(a_path, map_location="cpu", weights_only=False)
B = torch.load(b_path, map_location="cpu", weights_only=False)

d2 = 0.0
a2 = 0.0
b2 = 0.0
dot = 0.0
rels = []
worst = (None, -1.0)
n_identical = 0
for k, av in A.items():
    bv = B.get(k)
    if av is None or bv is None:
        continue
    a64 = av.to(torch.float64).reshape(-1)
    b64 = bv.to(torch.float64).reshape(-1)
    na = float(torch.linalg.vector_norm(a64))
    nb = float(torch.linalg.vector_norm(b64))
    dd = float(torch.linalg.vector_norm(a64 - b64))
    a2 += na * na
    b2 += nb * nb
    d2 += dd * dd
    dot += float(torch.dot(a64, b64))
    if torch.equal(av, bv):
        n_identical += 1
    if na >= 1e-12:
        r = dd / na
        rels.append(r)
        if r > worst[1]:
            worst = (k, r)

rels.sort()
med = rels[len(rels) // 2] if len(rels) % 2 else 0.5 * (rels[len(rels) // 2 - 1] + rels[len(rels) // 2])
out = {
    "what": "the same fp32 arm at OMP_NUM_THREADS 3 vs 7, scored against each other",
    "a": a_path,
    "b": b_path,
    "n_compared": len(rels),
    "n_bit_identical_tensors": n_identical,
    "mass_weighted_rel_l2": math.sqrt(d2 / a2),
    "median_rel_l2_over_tensors": med,
    "norm_ratio_r": math.sqrt(b2 / a2),
    "cos": dot / math.sqrt(a2 * b2),
    "worst_tensor": worst[0],
    "worst_rel_l2": worst[1],
}
print(json.dumps(out, indent=1))
open(out_path, "w").write(json.dumps(out, indent=1) + "\n")
