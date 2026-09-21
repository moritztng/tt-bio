#!/usr/bin/env python3
"""A/A: two gradient dumps, per tensor, bit-for-bit. Reports how many of the union moved at all
and the worst relative move, so 'identical' is a measurement and not an assertion."""
import argparse, json, sys
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
ap.add_argument("--key", default="grads")
ap.add_argument("--out", default="")
a = ap.parse_args()


def load(p, key):
    d = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(d, dict) and key in d:
        d = d[key]
    return {k: v for k, v in d.items() if v is not None}


A, B = load(a.a, a.key), load(a.b, a.key)
only_a, only_b = sorted(set(A) - set(B)), sorted(set(B) - set(A))
common = sorted(set(A) & set(B))
moved, worst, worst_n = 0, 0.0, None
for n in common:
    x, y = A[n].to(torch.float64).reshape(-1), B[n].to(torch.float64).reshape(-1)
    if x.shape != y.shape:
        moved += 1
        continue
    if not torch.equal(x, y):
        moved += 1
        d = float(torch.linalg.vector_norm(x - y))
        r = d / max(float(torch.linalg.vector_norm(y)), 1e-300)
        if r > worst:
            worst, worst_n = r, n
rep = {"a": a.a, "b": a.b, "n_a": len(A), "n_b": len(B), "n_common": len(common),
       "only_in_a": only_a[:5], "n_only_in_a": len(only_a),
       "only_in_b": only_b[:5], "n_only_in_b": len(only_b),
       "n_moved": moved, "bit_identical": moved == 0 and not only_a and not only_b,
       "worst_rel_move": worst, "worst_tensor": worst_n}
print(json.dumps(rep, indent=2))
if a.out:
    open(a.out, "w").write(json.dumps(rep, indent=2) + "\n")
sys.exit(0)
