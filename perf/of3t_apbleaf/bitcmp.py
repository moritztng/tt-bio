#!/usr/bin/env python3
"""of3t-apbleaf: tensor-by-tensor identity between two gradient files.

An identical SCORE is not an identity -- `of3t-trunkopclass` found a lever that scored the same
and was bit-identical, and the only way to tell those apart is a direct comparison. This is the
A/A determinism floor and the capture-perturbation control, both.
"""
import argparse, json, torch

PRE = "pairformer_stack.blocks."


def load(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(d, dict) and "grads" in d and isinstance(d["grads"], dict):
        d = d["grads"]
    return {(k if k.startswith(PRE) else PRE + k): v
            for k, v in d.items() if v is not None and hasattr(v, "shape")
            and (k.startswith(PRE) or k.split(".")[0].isdigit())}


ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--out", required=True)
z = ap.parse_args()
A, B = load(z.a), load(z.b)
keys = sorted(set(A) & set(B))
moved, worst, worst_k = 0, 0.0, None
for k in keys:
    x, y = A[k].to(torch.float64).reshape(-1), B[k].to(torch.float64).reshape(-1)
    if not torch.equal(x, y):
        moved += 1
        r = float((x - y).norm() / (x.norm() + 1e-300))
        if r > worst:
            worst, worst_k = r, k
R = {"what": __doc__.strip().splitlines()[0], "label": z.label, "a": z.a, "b": z.b,
     "tensors_compared": len(keys), "only_in_a": len(set(A) - set(B)),
     "only_in_b": len(set(B) - set(A)),
     "tensors_that_moved": moved, "bit_identical": moved == 0,
     "worst_relative_move": worst, "worst_tensor": worst_k}
json.dump(R, open(z.out, "w"), indent=2)
print(json.dumps(R))
