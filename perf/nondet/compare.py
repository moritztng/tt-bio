#!/usr/bin/env python3
"""Compare two protenix CIFs: sha of ATOM lines, raw RMSD, Kabsch RMSD, max atom deviation.

Parses the mmCIF loop_ header to locate Cartn_x/y/z, so both whitespace-delimited
and column-aligned writers work.
"""
import hashlib
import pathlib
import sys

import numpy as np


def parse(path):
    lines = pathlib.Path(path).read_text().splitlines()
    atom_idx = [i for i, l in enumerate(lines) if l.startswith(("ATOM", "HETATM"))]
    header_end = atom_idx[0] if atom_idx else len(lines)
    cols = [l.strip() for l in lines[:header_end] if l.strip().startswith("_atom_site.")]
    names = {n: i for i, n in enumerate(cols)}
    atom_lines = [lines[i] for i in atom_idx]
    if not {"_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z"} <= set(names):
        # fixed-width fallback: x/y/z at fields 10:13 (older writer)
        xyz = lambda l: [float(f) for f in l.split()[10:13]]
    else:
        ix, iy, iz = (names["_atom_site.Cartn_x"], names["_atom_site.Cartn_y"], names["_atom_site.Cartn_z"])
        xyz = lambda l: [float(l.split()[i]) for i in (ix, iy, iz)]
    return atom_lines, xyz


def kabsch_rmsd(a, b):
    ac = a - a.mean(0)
    bc = b - b.mean(0)
    h = ac.T @ bc
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(((ac @ rot - bc) ** 2).sum(-1).mean()))


def main(pa, pb):
    la, xa = parse(pa)
    lb, xb = parse(pb)
    sha_a = hashlib.sha256("\n".join(la).encode()).hexdigest()[:16]
    sha_b = hashlib.sha256("\n".join(lb).encode()).hexdigest()[:16]
    print(f"atom_sha  {sha_a}  {sha_b}  {'IDENTICAL' if sha_a == sha_b else 'DIFFERS'}")
    if len(la) != len(lb):
        print(f"atom count differs: {len(la)} vs {len(lb)}")
        return 1
    ca = np.array([xa(l) for l in la])
    cb = np.array([xb(l) for l in lb])
    diff = np.sqrt(((ca - cb) ** 2).sum(-1))
    print(f"raw_rmsd {float(np.sqrt((diff**2).mean())):.4f} A   max {float(diff.max()):.3f} A")
    print(f"kabsch_rmsd {kabsch_rmsd(ca, cb):.4f} A")
    nd = int((diff > 1e-6).sum())
    print(f"atoms moved (>1e-6 A): {nd}/{len(la)}")
    return 0 if sha_a == sha_b else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
