#!/usr/bin/env python3
"""Grade probe.py's folds: does the card follow host JAX as closely with dropout on as with it off?

CA RMSD after a Kabsch fit, per predicted state, in Angstrom, float64. With the same key the two
arms draw the same masks, so card-vs-host with dropout on should sit near card-vs-host with dropout
off, and well below what dropout itself moves the host fold by.

    python3 perf/bc2_dropout/grade.py runs/ model_1_ptm
"""
import pathlib
import sys

import numpy as np

CA = 1  # atom37 index


def load(path):
    z = np.load(path)
    ca, metrics = {}, {}
    for s in sorted({k.split("/")[0] for k in z.files}):
        chains = sorted({k.split("/")[1] for k in z.files if k.startswith(s + "/") and "/metric/" not in k})
        xyz = np.concatenate([z[f"{s}/{c}/atoms"][:, CA] for c in chains]).astype(np.float64)
        ok = np.concatenate([z[f"{s}/{c}/atom_mask"][:, CA] for c in chains]) > 0
        ca[s] = (xyz, ok)
        metrics[s] = {k.split("/")[-1]: z[k] for k in z.files if k.startswith(s + "/metric/") and z[k].ndim == 0}
    return ca, metrics


def rmsd(a, b):
    (x, mx), (y, my) = a, b
    m = mx & my
    x, y = x[m] - x[m].mean(0), y[m] - y[m].mean(0)
    u, _, vt = np.linalg.svd(x.T @ y)
    r = u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vt))]) @ vt
    return float(np.sqrt(((x @ r - y) ** 2).sum(1).mean()))


PAIRS = (("device0", "host0", "card vs host, dropout off"),
         ("device1", "host1", "card vs host, dropout on"),
         ("host1", "host0", "what dropout moves the host fold by"),
         ("device1", "device0", "what dropout moves the card fold by"),
         ("device0", "host1", "card without dropout vs host with it (before the fix)"))


def main():
    root, preset = pathlib.Path(sys.argv[1]), sys.argv[2]
    arms = {f"{arm}{d}": load(root / f"{arm}_{preset}_drop{d}_fold.npz") for arm in ("host", "device") for d in (0, 1)}
    for s in arms["host0"][0]:
        print(f"state {s}, CA RMSD in A:")
        for a, b, what in PAIRS:
            print(f"  {rmsd(arms[a][0][s], arms[b][0][s]):7.3f}  {what}")
        for name in sorted(arms["host0"][1][s]):
            print(f"  {name:24s} " + "  ".join(f"{k} {float(arms[k][1][s][name]):.4f}" for k in arms))


if __name__ == "__main__":
    main()
