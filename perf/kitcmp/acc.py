"""Accuracy of one kitcmp arm against another: CA-atom Kabsch RMSD, raw max |dx| (no superposition, 0 = same
coordinates), and mean plDDT of each, on the same fold name. Reads any mmCIF the kits write (atom_site loop).

    python3 perf/kitcmp/acc.py A.cif B.cif        -> one line: rmsd_A, per-half rmsd_A, max_dx_A, plddt_a, plddt_b, n_ca
"""
import sys
import numpy as np


def ca(path):
    cols, xyz, b = [], [], []
    for line in open(path):
        if line.startswith("_atom_site."):
            cols.append(line.split(".", 1)[1].strip())
        elif cols and line.startswith(("ATOM", "HETATM")):
            f = line.split()
            r = dict(zip(cols, f))
            if r.get("label_atom_id", r.get("auth_atom_id")) == "CA":
                xyz.append([float(r["Cartn_x"]), float(r["Cartn_y"]), float(r["Cartn_z"])])
                b.append(float(r.get("B_iso_or_equiv", "nan")))
    return np.array(xyz), np.array(b)


def kabsch(p, q):
    p, q = p - p.mean(0), q - q.mean(0)
    u, _, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(((p @ r - q) ** 2).sum(1).mean()))


a, ba = ca(sys.argv[1]); b, bb = ca(sys.argv[2])
assert len(a) == len(b) > 0, (len(a), len(b))
h = len(a) // 2  # the page fixture is two CDK2 copies in one chain: per-copy RMSD separates fold from copy placement
print(f"rmsd {kabsch(a, b):.3f} A  halves {kabsch(a[:h], b[:h]):.3f}/{kabsch(a[h:], b[h:]):.3f} A  max_dx {np.abs(a - b).max():.4f} A  plddt {np.nanmean(ba):.4f} vs {np.nanmean(bb):.4f}  n_ca {len(a)}")
