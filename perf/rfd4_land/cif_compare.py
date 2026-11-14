"""Compare two design CIFs: byte identity, then coordinates atom for atom.

    python perf/rfd4_land/cif_compare.py A.cif B.cif [--json out.json]

A merge is not allowed to move a design, so the check wanted here is identity, and the
numbers below it (max atom deviation, CA RMSD in place) say how far off it is when it is not.
No superposition: both sides come out of the same conditioning frame, so a real difference
must show in place.
"""
import argparse
import hashlib
import json
import sys

import numpy as np
from biotite.structure.io.pdbx import CIFFile, get_structure

ap = argparse.ArgumentParser()
ap.add_argument("a")
ap.add_argument("b")
ap.add_argument("--json")
g = ap.parse_args()


def load(p):
    arr = get_structure(CIFFile.read(p), model=1)
    return arr


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


A, B = load(g.a), load(g.b)
out = {"a": g.a, "b": g.b, "md5_a": md5(g.a), "md5_b": md5(g.b)}
out["bytes_identical"] = out["md5_a"] == out["md5_b"]
out["n_atoms"] = [A.array_length(), B.array_length()]
if A.array_length() == B.array_length():
    d = np.linalg.norm(A.coord - B.coord, axis=-1)
    ca = (A.atom_name == "CA") & (B.atom_name == "CA")
    out["max_atom_dev_a"] = float(d.max())
    out["rmsd_all_atom_a"] = float(np.sqrt((d ** 2).mean()))
    out["ca_rmsd_a"] = float(np.sqrt((d[ca] ** 2).mean())) if ca.any() else None
    out["n_ca"] = int(ca.sum())
    out["same_sequence"] = bool((A.res_name == B.res_name).all())
else:
    out["max_atom_dev_a"] = None
print(json.dumps(out, indent=1))
if g.json:
    json.dump(out, open(g.json, "w"), indent=1)
sys.exit(0)
