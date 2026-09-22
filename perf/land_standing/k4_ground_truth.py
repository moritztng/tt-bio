#!/usr/bin/env python3
"""Score both K4 arms against the DEPOSITED structure, not against each other.

The A/B only established that the structure MOVED 0.2266 A. That is not the same as whether it
got worse: the base arm is the incumbent route, not ground truth. `cdk2x2_298.yaml`'s own header
says the target is CDK2 apo, PDB 1HCL, 298 aa one chain, and `perf/fused_sdpa/cifs/1hcl.cif` is
the deposited structure, so the ground-truth comparison costs no device time.

Both arms are scored over the IDENTICAL atom set (residues present in the native and in both
models), so an atom-count difference cannot flatter either one.
"""
import sys
from pathlib import Path
import numpy as np


def ca(p: Path, chain=None):
    """{label_seq_id: xyz} for CA atoms of the (single) polymer chain."""
    cols, rows, inloop = {}, [], False
    for ln in p.read_text().splitlines():
        s = ln.strip()
        if s.startswith("_atom_site."):
            cols[s.split(".", 1)[1]] = len(cols)
            inloop = True
            continue
        if inloop and s.startswith("ATOM"):
            rows.append(s.split())
        elif inloop and s == "#":
            inloop = False
    ix = {n: cols[n] for n in ("label_atom_id", "label_seq_id", "label_asym_id",
                               "Cartn_x", "Cartn_y", "Cartn_z")}
    out = {}
    for r in rows:
        if r[ix["label_atom_id"]] != "CA":
            continue
        if chain and r[ix["label_asym_id"]] != chain:
            continue
        try:
            sid = int(r[ix["label_seq_id"]])
        except ValueError:
            continue
        out.setdefault(sid, (float(r[ix["Cartn_x"]]), float(r[ix["Cartn_y"]]),
                             float(r[ix["Cartn_z"]])))
    return out


def rmsd_tm(P, Q):
    """Kabsch CA-RMSD and TM-score over the paired coordinates, superposed on all of them."""
    n = len(P)
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, 1, d]) @ Vt
    dev = np.sqrt(((Pc @ R - Qc) ** 2).sum(1))
    d0 = 1.24 * (n - 15) ** (1 / 3) - 1.8 if n > 21 else 0.5
    return float(np.sqrt((dev ** 2).mean())), float((1.0 / (1.0 + (dev / d0) ** 2)).mean())


native_p, root = Path(sys.argv[1]), Path(sys.argv[2])
legs = sorted(d.name for d in root.iterdir() if d.is_dir())
cif = lambda leg: next((root / leg).glob("*.cif"))
base_leg = next(l for l in legs if "_base_" in l)
on_leg = next(l for l in legs if "_on_" in l)

nat = ca(native_p)
chains = sorted({c for c in [None]})
b, o = ca(cif(base_leg)), ca(cif(on_leg))
print(f"native {native_p.name}: {len(nat)} CA   base: {len(b)} CA   on: {len(o)} CA")
shared = sorted(set(nat) & set(b) & set(o))
print(f"scored over {len(shared)} shared residues (identical atom set for both arms)")
assert len(shared) > 200, f"only {len(shared)} shared residues; numbering may not line up"

N = np.array([nat[i] for i in shared], dtype=np.float64)
B = np.array([b[i] for i in shared], dtype=np.float64)
O = np.array([o[i] for i in shared], dtype=np.float64)

rb, tb = rmsd_tm(B, N)
ro, to = rmsd_tm(O, N)
rc, tc = rmsd_tm(B, O)
print("\n--- CA vs the DEPOSITED structure (lower RMSD / higher TM is better)")
print(f"  base (incumbent, K4 off)   CA-RMSD {rb:.6f} A   TM {tb:.6f}")
print(f"  on   (K4)                  CA-RMSD {ro:.6f} A   TM {to:.6f}")
print(f"  delta (on - base)          {ro - rb:+.6f} A      TM {to - tb:+.6f}")
print(f"  -> K4 is {'CLOSER to native' if ro < rb else 'FURTHER from native'}")
print(f"\n--- for reference, the arms against each other: CA {rc:.6f} A, TM {tc:.6f}")
