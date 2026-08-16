#!/usr/bin/env python3
"""Compare two protenix CIFs: sha of ATOM lines, raw RMSD, Kabsch RMSD, max atom deviation."""
import hashlib
import pathlib
import sys

import numpy as np


def atom_lines(path):
    lines = []
    for ln in pathlib.Path(path).read_text().splitlines():
        if ln.startswith(("ATOM", "HETATM")):
            lines.append(ln)
    return lines


def coords(lines):
    # whitespace-delimited CIF: x, y, z are fields 10, 11, 12 (0-based)
    return np.array([[float(f) for f in l.split()[10:13]] for l in lines])


def kabsch_rmsd(a, b):
    ac = a - a.mean(0)
    bc = b - b.mean(0)
    h = ac.T @ bc
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(((ac @ rot - bc) ** 2).sum(-1).mean()))


def main(pa, pb):
    la, lb = atom_lines(pa), atom_lines(pb)
    sha_a = hashlib.sha256("\n".join(la).encode()).hexdigest()[:16]
    sha_b = hashlib.sha256("\n".join(lb).encode()).hexdigest()[:16]
    print(f"atom_sha  {sha_a}  {sha_b}  {'IDENTICAL' if sha_a == sha_b else 'DIFFERS'}")
    if len(la) != len(lb):
        print(f"atom count differs: {len(la)} vs {len(lb)}")
        return 1
    ca, cb = coords(la), coords(lb)
    diff = np.sqrt(((ca - cb) ** 2).sum(-1))
    print(f"raw_rmsd {float(np.sqrt((diff**2).mean())):.4f} A   max {float(diff.max()):.3f} A")
    print(f"kabsch_rmsd {kabsch_rmsd(ca, cb):.4f} A")
    nd = int((diff > 1e-6).sum())
    print(f"atoms moved (>1e-6 A): {nd}/{len(la)}")
    return 0 if sha_a == sha_b else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
