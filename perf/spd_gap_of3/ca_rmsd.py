"""CA-RMSD after optimal superposition between two mmCIFs of the same input (one per arm).

    python perf/spd_gap_of3/ca_rmsd.py A.cif B.cif

Reads the _atom_site loop by its own header, so column order does not matter. Prints the RMSD in
Angstrom and the number of CA atoms paired by (chain, residue number)."""
import sys

import numpy as np


def cas(path):
    cols, out = [], {}
    for line in open(path):
        if line.startswith("_atom_site."):
            cols.append(line.split(".", 1)[1].strip())
        elif line.startswith(("ATOM", "HETATM")) and cols:
            f = dict(zip(cols, line.split()))
            if f.get("label_atom_id") == "CA":
                key = (f.get("auth_asym_id", f.get("label_asym_id")), f.get("auth_seq_id", f.get("label_seq_id")))
                out[key] = [float(f["Cartn_x"]), float(f["Cartn_y"]), float(f["Cartn_z"])]
    return out


def kabsch_rmsd(p, q):
    p, q = p - p.mean(0), q - q.mean(0)
    u, _, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(((p @ r - q) ** 2).sum(1).mean()))


if __name__ == "__main__":
    a, b = cas(sys.argv[1]), cas(sys.argv[2])
    keys = sorted(set(a) & set(b))
    print(f"{kabsch_rmsd(np.array([a[k] for k in keys]), np.array([b[k] for k in keys])):.4f} A over {len(keys)} CA")
