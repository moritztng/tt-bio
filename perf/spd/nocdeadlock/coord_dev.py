"""Kabsch RMSD (Angstrom, all atoms) between two bench.py runs, seed by seed and sample by sample.

usage: python coord_dev.py <run_a> <run_b>   prints per-seed mean/max over samples, the overall mean, and the
seed floor of run_a (seed s vs seed s+1, same run), which is the variation a different seed already causes.
"""
import re, sys
from pathlib import Path

import torch


def kabsch_rmsd(p, q):
    p, q = p.double() - p.double().mean(0), q.double() - q.double().mean(0)
    u, _, vt = torch.linalg.svd(p.T @ q)
    d = torch.sign(torch.linalg.det(u @ vt))
    r = u @ torch.diag(torch.tensor([1.0, 1.0, float(d)], dtype=torch.float64)) @ vt
    return float(((p @ r - q) ** 2).sum(1).mean().sqrt())


def load(run):
    out = {}
    for f in Path(run).glob("coords_*_s*.pt"):
        out[int(re.search(r"_s(\d+)\.pt$", f.name).group(1))] = torch.load(f, weights_only=False)["coords"]
    return out


a, b = load(sys.argv[1]), load(sys.argv[2])
devs = []
for s in sorted(set(a) & set(b)):
    r = [kabsch_rmsd(x, y) for x, y in zip(a[s], b[s])]
    devs += r
    print(f"seed {s}: mean {sum(r) / len(r):.3f} A  max {max(r):.3f} A  (n={len(r)})")
floor = [kabsch_rmsd(a[s][i], a[s + 1][i]) for s in sorted(a) if s + 1 in a for i in range(len(a[s]))]
print(f"A->B mean {sum(devs) / max(1, len(devs)):.3f} A over {len(devs)} samples; "
      f"seed floor (A, s vs s+1) mean {sum(floor) / max(1, len(floor)):.3f} A over {len(floor)}")
