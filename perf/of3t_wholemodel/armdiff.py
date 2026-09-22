#!/usr/bin/env python3
"""Did the arm flag change the gradient at all, on this scope?

`a-lever-can-fire-and-be-inert` is a standing fleet lesson and this row has both halves of it:
`HOST_F64_SOFTMAX_STATS` says whether the code path was REACHED, and this says whether the
numbers MOVED. Reporting a scope's arm without both is how an arm gets quoted as measured when
it was the shipped arm under another name.

    armdiff.py base.pt arm.pt [--unwrap grads]
"""
import sys
import torch

unwrap = None
args = [a for a in sys.argv[1:]]
if "--unwrap" in args:
    i = args.index("--unwrap"); unwrap = args[i + 1]; del args[i:i + 2]
base, arm = args


def load(p):
    d = torch.load(p, map_location="cpu", weights_only=False)
    if unwrap and unwrap in d:
        d = d[unwrap]
    return {k: v.to(torch.float64).reshape(-1) for k, v in d.items() if v is not None}


a, b = load(base), load(arm)
common = sorted(set(a) & set(b))
n_moved = 0
d2 = 0.0
r2 = 0.0
worst = (0.0, None)
for k in common:
    d = float(torch.linalg.vector_norm(b[k] - a[k]))
    n = float(torch.linalg.vector_norm(a[k]))
    d2 += d * d
    r2 += n * n
    if d != 0.0:
        n_moved += 1
        if n > 0 and d / n > worst[0]:
            worst = (d / n, k)
print(f"{base} -> {arm}")
print(f"  tensors compared {len(common)}, only-in-base {len(set(a)-set(b))}, "
      f"only-in-arm {len(set(b)-set(a))}")
print(f"  tensors that MOVED at all: {n_moved} of {len(common)}")
print(f"  ||arm - base|| / ||base|| over the scope: {(d2 ** 0.5) / (r2 ** 0.5):.6e}")
print(f"  worst per-tensor relative move: {worst[0]:.6e} at {worst[1]}")
print("  VERDICT: " + ("BIT-IDENTICAL -- the arm is inert on this scope"
                       if n_moved == 0 else "the arm moved the gradient"))
