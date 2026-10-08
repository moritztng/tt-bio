"""Compare two coordinate files from bench.py (same seed, two engines or arms).

usage: compare.py A.pt B.pt
Prints torch.equal, max |dA|, and per-sample all-atom RMSD before and after Kabsch superposition.
"""
import sys

import torch


def kabsch_rmsd(P, Q):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = torch.linalg.svd(Pc.t() @ Qc)
    d = torch.sign(torch.det(Vt.t() @ U.t()))
    R = Vt.t() @ torch.diag(torch.tensor([1.0, 1.0, float(d)], dtype=P.dtype)) @ U.t()
    return float(((Pc @ R.t() - Qc) ** 2).sum(-1).mean().sqrt())


a, b = (torch.load(p).double() for p in sys.argv[1:3])
print(f"equal={torch.equal(a, b)} max_abs={float((a - b).abs().max()):.3e} A")
for i in range(a.shape[0]):
    raw = float(((a[i] - b[i]) ** 2).sum(-1).mean().sqrt())
    print(f"sample {i}: rmsd {raw:.4f} A, kabsch {kabsch_rmsd(a[i], b[i]):.4f} A")
