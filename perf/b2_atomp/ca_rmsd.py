"""CA RMSD between two mmCIF files after Kabsch: whole chain set and each half.

    python perf/b2_atomp/ca_rmsd.py a.cif b.cif

cdk2x2_512 is bimodal in its second half, so a same-seed read is reported per half.
"""
import sys

import numpy as np


def ca(path):
    hdr, out = [], []
    for line in open(path):
        if line.startswith("_atom_site."):
            hdr.append(line.strip().split(".", 1)[1])
        elif line.startswith(("ATOM", "HETATM")) and hdr:
            d = dict(zip(hdr, line.split()))
            if d.get("label_atom_id") == "CA":
                out.append([float(d["Cartn_x"]), float(d["Cartn_y"]), float(d["Cartn_z"])])
    return np.array(out)


def kabsch_rmsd(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    r = u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vt))]) @ vt
    return float(np.sqrt(((a @ r - b) ** 2).sum(1).mean()))


a, b = ca(sys.argv[1]), ca(sys.argv[2])
h = len(a) // 2
print(f"all {kabsch_rmsd(a, b):.3f}  first-half {kabsch_rmsd(a[:h], b[:h]):.3f}  "
      f"second-half {kabsch_rmsd(a[h:], b[h:]):.3f}  n={len(a)}")
