#!/usr/bin/env python3
"""Where does the pair-projection arm-to-arm deviation live: spread over the chain, or a few loops?

Global all-atom Kabsch is one number and on a 298 aa monomer with flexible loops it cannot tell a
uniform numerical drift from two residues taking a different rotamer. Reuses read_atoms/kabsch_rmsd
from perf/other512/cif_rmsd.py so the superposition is the same one the page string uses.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "perf" / "other512"))
from cif_rmsd import kabsch_rmsd, read_atoms  # noqa: E402


def superpose(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    P0, Q0 = P - pc, Q - qc
    V, S, Wt = np.linalg.svd(P0.T @ Q0)
    d = np.sign(np.linalg.det(V @ Wt))
    return P0 @ (V @ np.diag([1.0, 1.0, d]) @ Wt), Q0


def main():
    a, b = Path(sys.argv[1]), Path(sys.argv[2])
    ka, xa = read_atoms(a)
    kb, xb = read_atoms(b)
    assert ka == kb, "atom identity differs"
    Pa, Qb = superpose(xa, xb)
    dev = np.linalg.norm(Pa - Qb, axis=1)
    print(f"  atoms {len(dev)}  global all-atom RMSD {kabsch_rmsd(xa, xb):.6f} A")

    # key order in cif_rmsd.read_atoms: asym, seq, atom, comp (whichever are present)
    seq = np.array([int(k[1]) if k[1].isdigit() else -1 for k in ka])
    isca = np.array([k[2] == "CA" for k in ka])
    print(f"  CA-only RMSD {kabsch_rmsd(xa[isca], xb[isca]):.6f} A over {isca.sum()} CA")
    print(f"  per-atom deviation after global superposition: "
          f"median {np.median(dev):.4f}  p90 {np.percentile(dev, 90):.4f}  max {dev.max():.4f}")

    per = {}
    for s, d in zip(seq, dev):
        per.setdefault(s, []).append(d)
    res = sorted(((np.mean(v), s, len(v)) for s, v in per.items()), reverse=True)
    frac = sum(1 for m, _, _ in res if m > 1.0) / len(res)
    print(f"  residues with mean deviation > 1.0 A: {sum(1 for m,_,_ in res if m>1.0)}/{len(res)}"
          f" ({100*frac:.1f}%)")
    print("  worst 12 residues (mean dev A, seq id, n atoms):")
    for m, s, n in res[:12]:
        print(f"    {m:8.3f}  seq {s:4d}  {n} atoms")
    # a trimmed RMSD: drop the worst 5% of atoms, which is what a loop-flip story predicts recovers
    keep = dev <= np.percentile(dev, 95)
    print(f"  RMSD over the best 95% of atoms: {np.sqrt((dev[keep]**2).mean()):.6f} A")


if __name__ == "__main__":
    main()
