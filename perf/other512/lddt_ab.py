#!/usr/bin/env python3
"""Superposition-free agreement between two folds of the same sequence.

A global Kabsch RMSD is the wrong instrument on the tiled CDK2 fixtures: the copies are
joined by a floppy linker, so a few degrees of hinge rotation saturates the number while
every domain stays identical (memory cdk2x2-chimeric-fixture-cannot-score-non-bit-exact-parity).
Distance-matrix agreement does not care about the hinge, so it separates "same structure,
different hinge" from "different structure".

    lddt_ab.py <a.cif> <b.cif>
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cif_rmsd import read_atoms  # noqa: E402


def ca(p):
    keys, xyz = read_atoms(Path(p))
    idx = [i for i, k in enumerate(keys) if ".CA." in f".{k[2]}." or k[2] == "CA"]
    return np.array([xyz[i] for i in idx]), [keys[i] for i in idx]


def kabsch(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return float(np.sqrt((((a @ r.T) - b) ** 2).sum(1).mean()))


A, ka = ca(sys.argv[1])
B, kb = ca(sys.argv[2])
assert ka == kb, "different residue sets"
n = len(A)
da = np.linalg.norm(A[:, None] - A[None], axis=-1)
db = np.linalg.norm(B[:, None] - B[None], axis=-1)
m = (da < 15.0) & ~np.eye(n, dtype=bool)
diff = np.abs(da - db)[m]
print(f"CA atoms: {n}")
print(f"local (<15 A) distance agreement: mean |dd| = {diff.mean():.4f} A, median = {np.median(diff):.4f} A")
for t in (0.5, 1.0, 2.0, 4.0):
    print(f"  fraction of local pairs within {t:>3} A: {(diff <= t).mean() * 100:.2f} %")
print(f"lDDT (4-threshold mean): {np.mean([(diff <= t).mean() for t in (0.5, 1.0, 2.0, 4.0)]) * 100:.2f}")
print(f"global CA Kabsch RMSD: {kabsch(A, B):.4f} A")
h = n // 2
print(f"per-half CA Kabsch RMSD: first {h} = {kabsch(A[:h], B[:h]):.4f} A, "
      f"second {n - h} = {kabsch(A[h:], B[h:]):.4f} A")
