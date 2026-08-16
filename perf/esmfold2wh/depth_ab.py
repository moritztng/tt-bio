#!/usr/bin/env python3
"""What an MSA-depth cap costs, scored on two real CIFs rather than asserted.

    depth_ab.py <full-depth.cif> <capped.cif>

Reports all-atom Kabsch RMSD and the mean/min pLDDT each arm wrote. The parser and the
superposition are `perf/other512/cif_rmsd.py`'s, so these numbers sit on the same scale as the
rest of this lineage: Kabsch over ALL atoms, equal weights.

A cap is only defensible if the RMSD it costs is small against the spread the model already has
between neighbouring lengths, so run it at a length where BOTH depths fold (640 aa), not only at
the length that needs the cap.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "other512"))
from cif_rmsd import read_atoms  # noqa: E402


def plddt(p: Path):
    """The B-factor column of the atom_site loop, which is where the CIF carries pLDDT."""
    lines = p.read_text().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_":
            j, cols = i + 1, []
            while j < len(lines) and lines[j].strip().startswith("_atom_site."):
                cols.append(lines[j].strip())
                j += 1
            if cols and "_atom_site.B_iso_or_equiv" in cols:
                k = cols.index("_atom_site.B_iso_or_equiv")
                vals = []
                while j < len(lines) and not lines[j].strip().startswith(("#", "loop_", "_")):
                    f = lines[j].split()
                    if len(f) > k:
                        try:
                            vals.append(float(f[k]))
                        except ValueError:
                            pass
                    j += 1
                return np.array(vals)
            i = j
        else:
            i += 1
    return np.array([])


def kabsch(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return float(np.sqrt((((a @ r.T) - b) ** 2).sum(1).mean()))


def main():
    full, capped = Path(sys.argv[1]), Path(sys.argv[2])
    ka, xa = read_atoms(full)
    kb, xb = read_atoms(capped)
    shared = [k for k in ka if k in set(kb)]
    if not shared:
        raise SystemExit("no atoms in common -- the two CIFs are not the same target")
    ia = {k: n for n, k in enumerate(ka)}
    ib = {k: n for n, k in enumerate(kb)}
    A = xa[[ia[k] for k in shared]]
    B = xb[[ib[k] for k in shared]]
    pa, pb = plddt(full), plddt(capped)
    print(f"atoms: full={len(ka)} capped={len(kb)} shared={len(shared)}")
    print(f"all-atom Kabsch RMSD: {kabsch(A, B):.4f} A")
    for tag, v in (("full", pa), ("capped", pb)):
        if v.size:
            print(f"pLDDT {tag}: mean={v.mean():.3f} min={v.min():.3f}")


if __name__ == "__main__":
    main()
