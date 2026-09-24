"""Pairwise structure comparisons the matrix table does not make, per model:

  cif_vs_npz   template_cif against template_npz (the two routes on the same template):
               bit-identical coordinates, else CA-RMSD between them
  off_vs_main  template_off here against the matrix's own template_off run on main (a cell that
               folded before this branch): bit-identical, else CA-RMSD

Usage: compare.py <out_root> <matrix_out_root>  -> JSON on stdout
"""
import json
import sys
from pathlib import Path

import gemmi
import numpy as np

OUT, MAIN = Path(sys.argv[1]), Path(sys.argv[2])


def best(model_dir, stem):
    hits = sorted(model_dir.glob(f"*_results_*/structures/{stem}.cif"))
    return hits[0] if hits else None


def ca(path):
    st = gemmi.read_structure(str(path))
    return np.array([r["CA"][0].pos.tolist() for r in st[0][0] if r.find_atom("CA", "*")])


def cmp(a, b):
    if a is None or b is None:
        return {"missing": [str(p) for p in (a, b) if p is None] or True}
    P, Q = ca(a), ca(b)
    if P.shape != Q.shape:
        return {"shape": [P.shape, Q.shape]}
    if np.array_equal(P, Q):
        return {"bit_identical": True, "ca_rmsd": 0.0}
    P, Q = P - P.mean(0), Q - Q.mean(0)
    U, _s, Vt = np.linalg.svd(P.T @ Q)
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return {"bit_identical": False, "ca_rmsd": float(np.sqrt(((P @ R - Q) ** 2).sum(1).mean()))}


res = {}
for d in sorted(p for p in OUT.iterdir() if p.is_dir()):
    m = d.name
    res[m] = {"cif_vs_npz": cmp(best(d, "template_cif"), best(d, "template_npz")),
              "off_vs_main": cmp(best(d, "template_off"), best(MAIN / m, "template_off")),
              "files": {s: str(best(d, s)) for s in ("template_off", "template_npz", "template_cif")}}
json.dump(res, sys.stdout, indent=1)
