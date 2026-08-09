#!/usr/bin/env python3
"""Compare two fold arms' output coordinates in float64.

The diffusion sampler runs 200 steps from a fixed seed, so any arithmetic change
compounds; the honest numbers are the raw per-atom deviation and the deviation left
after a rigid superposition, which is what a structural comparison would report.

    python3 perf/dit_attn/compare_coords.py perf/dit_attn/fold_X_before.npy \
        perf/dit_attn/fold_X_after.npy
"""
import sys
import numpy as np


def kabsch_rmsd(a, b):
    ac, bc = a - a.mean(0), b - b.mean(0)
    u, _s, vt = np.linalg.svd(ac.T @ bc)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return float(np.sqrt(((ac @ r.T - bc) ** 2).sum(1).mean()))


def main():
    a, b = (np.load(p) for p in sys.argv[1:3])
    assert a.shape == b.shape, (a.shape, b.shape)
    d = a - b
    print(f"n_atoms          {a.shape[0]}")
    print(f"identical        {bool(np.array_equal(a, b))}")
    print(f"raw RMSD         {np.sqrt((d * d).sum(1).mean()):.4f} A")
    print(f"max atom dev     {np.sqrt((d * d).sum(1)).max():.4f} A")
    print(f"aligned RMSD     {kabsch_rmsd(a, b):.4f} A")
    fa, fb = a.ravel(), b.ravel()
    ca, cb = fa - fa.mean(), fb - fb.mean()
    print(f"coord PCC        {float((ca * cb).sum() / (np.linalg.norm(ca) * np.linalg.norm(cb))):.8f}")
    print(f"rmsd/std         {float(np.sqrt((d * d).mean()) / fa.std()):.6f}")


if __name__ == "__main__":
    main()
