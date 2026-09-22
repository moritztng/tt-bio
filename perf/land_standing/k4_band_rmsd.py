#!/usr/bin/env python3
"""CA and all-atom Kabsch RMSD between the K4 A/B's kept CIFs.

Self-checked: base-against-base must read exactly 0.000000 A, which is what says the parser and
the superposition are not inventing a number. Opens no device.
"""
import sys
from pathlib import Path
import numpy as np


def atoms(p: Path):
    """(label, xyz) for every ATOM record, in file order. mmCIF loop, column-indexed by header."""
    cols, rows, inloop = {}, [], False
    for ln in p.read_text().splitlines():
        s = ln.strip()
        if s.startswith("_atom_site."):
            cols[s.split(".", 1)[1]] = len(cols)
            inloop = True
            continue
        if inloop and (s.startswith("ATOM") or s.startswith("HETATM")):
            rows.append(s.split())
        elif inloop and s == "#":
            inloop = False
    need = ["label_atom_id", "label_seq_id", "label_asym_id", "Cartn_x", "Cartn_y", "Cartn_z"]
    ix = [cols[n] for n in need]
    out = []
    for r in rows:
        out.append(((r[ix[2]], r[ix[1]], r[ix[0]]),
                    (float(r[ix[3]]), float(r[ix[4]]), float(r[ix[5]]))))
    return out


def rmsd(a: Path, b: Path, ca_only: bool):
    A, B = atoms(a), atoms(b)
    da, db = dict(A), dict(B)
    keys = [k for k, _ in A if k in db and (not ca_only or k[2] == "CA")]
    assert keys, "no shared atoms"
    P = np.array([da[k] for k in keys], dtype=np.float64)
    Q = np.array([db[k] for k in keys], dtype=np.float64)
    P = P - P.mean(0)
    Q = Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, 1, d]) @ Vt
    return float(np.sqrt(((P @ R - Q) ** 2).sum(1).mean())), len(keys)


root = Path(sys.argv[1])
legs = sorted(d.name for d in root.iterdir() if d.is_dir())
cif = lambda leg: next((root / leg).glob("*.cif"))
base = [l for l in legs if "_base_" in l]
on = [l for l in legs if "_on_" in l]
print(f"legs: {len(base)} base, {len(on)} on")
print("--- CONTROL, base vs base (must be 0.000000):")
for l in base[1:]:
    ca, n1 = rmsd(cif(base[0]), cif(l), True)
    aa, n2 = rmsd(cif(base[0]), cif(l), False)
    print(f"  {base[0]} vs {l:16s} CA {ca:.6f} A ({n1} atoms)  all-atom {aa:.6f} A ({n2})")
print("--- CONTROL, on vs on (must be 0.000000):")
for l in on[1:]:
    ca, _ = rmsd(cif(on[0]), cif(l), True)
    aa, _ = rmsd(cif(on[0]), cif(l), False)
    print(f"  {on[0]} vs {l:18s} CA {ca:.6f} A  all-atom {aa:.6f} A")
print("--- THE MEASUREMENT, base vs on:")
vals = []
for lb in base:
    for lo in on:
        ca, n1 = rmsd(cif(lb), cif(lo), True)
        aa, n2 = rmsd(cif(lb), cif(lo), False)
        vals.append((ca, aa))
        print(f"  {lb} vs {lo:18s} CA {ca:.6f} A ({n1})  all-atom {aa:.6f} A ({n2})")
ca = [v[0] for v in vals]
aa = [v[1] for v in vals]
print(f"\nCA       min {min(ca):.6f}  max {max(ca):.6f}  mean {sum(ca)/len(ca):.6f} A")
print(f"ALL-ATOM min {min(aa):.6f}  max {max(aa):.6f}  mean {sum(aa)/len(aa):.6f} A")
print("BAR 0.60 A; 512 aa seed floor 1.84 A")
