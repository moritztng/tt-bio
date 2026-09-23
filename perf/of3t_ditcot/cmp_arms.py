#!/usr/bin/env python3
"""Bit-exact + magnitude comparison of two KCFG arms' dumped device gradients.

A kernel-config pull that changes nothing is INERT; the LoFi control on the same sites must
move the same tensors or no flag reached the kernel. Scored arm-against-arm, so no reference is
involved and D141's architecture mismatch does not touch this number.
"""
import json
import sys

import torch

a_path, b_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
A = torch.load(a_path, map_location="cpu", weights_only=False)
B = torch.load(b_path, map_location="cpu", weights_only=False)
A = A.get("grads", A) if isinstance(A, dict) else A
B = B.get("grads", B) if isinstance(B, dict) else B

keys = sorted(set(A) & set(B))
same = 0
moved = []
for k in keys:
    x, y = A[k], B[k]
    if not (torch.is_tensor(x) and torch.is_tensor(y)) or x.shape != y.shape:
        continue
    if torch.equal(x, y):
        same += 1
        continue
    xf, yf = x.double().reshape(-1), y.double().reshape(-1)
    nx = float(xf.norm())
    moved.append({"tensor": k, "rel": float((yf - xf).norm() / nx) if nx else float("inf"),
                  "max_abs": float((yf - xf).abs().max())})

moved.sort(key=lambda r: -r["rel"])
rep = {"a": a_path, "b": b_path, "compared": len(keys), "bit_identical": same,
       "moved": len(moved), "bit_identical_frac": same / len(keys) if keys else None,
       "max_rel": moved[0]["rel"] if moved else 0.0,
       "top_moved": moved[:15]}
json.dump(rep, open(out, "w"), indent=1)
print(json.dumps({k: v for k, v in rep.items() if k != "top_moved"}, indent=1))
for r in moved[:8]:
    print(f"  {r['rel']:.6e}  {r['tensor']}")
