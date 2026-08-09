#!/usr/bin/env python3
"""CA-RMSD and max atom delta between the tuned and untuned arm's CIF, same target and seed.

Both arms fold the same seeded input, so the two structures are in the same frame: the direct
(unsuperposed) numbers are the honest ones. The superposed CA-RMSD is reported alongside because
that is the form W11's control band (1.916 A CA-RMSD / 12.19 A max atom delta) is quoted in.
"""
import sys
import numpy as np


def atoms(path):
    rows, cols, in_loop, hdr = [], [], False, []
    for line in open(path):
        t = line.strip()
        if t.startswith("_atom_site."):
            hdr.append(t.split(".")[1]); in_loop = True; continue
        if in_loop:
            if t.startswith(("ATOM", "HETATM")):
                rows.append(t.split())
            elif rows:
                break
    i = {k: n for n, k in enumerate(hdr)}
    xyz = np.array([[float(r[i["Cartn_x"]]), float(r[i["Cartn_y"]]), float(r[i["Cartn_z"]])]
                    for r in rows])
    names = [r[i["label_atom_id"]] for r in rows]
    return xyz, np.array([n == "CA" for n in names])


def kabsch(a, b):
    ac, bc = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(ac.T @ bc)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    return float(np.sqrt(((ac @ r.T - bc) ** 2).sum(1).mean()))


for lab, p1, p2 in [(sys.argv[i], sys.argv[i + 1], sys.argv[i + 2])
                    for i in range(1, len(sys.argv), 3)]:
    a, ca = atoms(p1)
    b, _ = atoms(p2)
    assert a.shape == b.shape, f"{lab}: atom count differs {a.shape} vs {b.shape}"
    d = np.linalg.norm(a - b, axis=1)
    print(f"{lab:14s} atoms {len(a):5d}  direct: CA-RMSD {np.sqrt((d[ca]**2).mean()):7.4f} A  "
          f"all-atom RMSD {np.sqrt((d**2).mean()):7.4f} A  max delta {d.max():7.4f} A  |  "
          f"superposed CA-RMSD {kabsch(a[ca], b[ca]):7.4f} A")
