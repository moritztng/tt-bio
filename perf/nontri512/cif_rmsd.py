#!/usr/bin/env python3
"""Reconcile the two parity figures this task carries for `_UNFUSED_SILU`, without re-folding.

`state/fold-nontriangle-below-4x.md` carried "+0.0030 A kabsch_rmsd" and read it as negligible.
`state/protenix-trunk--y-silu-lowering.md` H4 killed the same change at "4.133 A all-atom RMSD
against a 0.000 A A/A self-RMSD, on a 0.1136 A envelope bound".

Both arms of the 512 aa fold A/B wrote their CIFs and they are still on disk, so the number can be
measured directly rather than argued. Kabsch superposition over ALL atoms, which is what tt-bio's
own `kabsch_rmsd` does (memory `tt-bio-kabsch-rmsd-mislabeled-ca-rmsd`: the name says CA, the
function is all-atom).
"""
import itertools
import sys
from pathlib import Path

import numpy as np


def read_atoms(p: Path):
    """Parse the mmCIF atom_site loop. Returns (keys, coords[N,3])."""
    lines = p.read_text().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_":
            j, cols = i + 1, []
            while j < len(lines) and lines[j].strip().startswith("_atom_site."):
                cols.append(lines[j].strip())
                j += 1
            if cols:
                idx = {c: k for k, c in enumerate(cols)}
                need = ["_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z"]
                if all(n in idx for n in need):
                    keycols = [c for c in ("_atom_site.label_asym_id", "_atom_site.label_seq_id",
                                           "_atom_site.label_atom_id", "_atom_site.label_comp_id")
                               if c in idx]
                    keys, xyz = [], []
                    while j < len(lines):
                        s = lines[j].strip()
                        if not s or s.startswith("#") or s.startswith("loop_") or s.startswith("_"):
                            break
                        f = s.split()
                        if len(f) < len(cols):
                            break
                        keys.append(tuple(f[idx[c]] for c in keycols))
                        xyz.append([float(f[idx[n]]) for n in need])
                        j += 1
                    return keys, np.asarray(xyz, dtype=np.float64)
            i = j
        else:
            i += 1
    raise SystemExit(f"no _atom_site loop in {p}")


def kabsch_rmsd(P, Q):
    """RMSD after optimal rigid superposition. All atoms, equal weights."""
    P = P - P.mean(0)
    Q = Q - Q.mean(0)
    V, S, Wt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    P = P @ (V @ D @ Wt)
    return float(np.sqrt(((P - Q) ** 2).sum() / len(P)))


def main():
    root = Path(sys.argv[1])
    cifs = sorted(root.glob("*/cdk2x2_512.cif"))
    arms = {}
    for c in cifs:
        arms[c.parent.name] = c
    print(f"{len(arms)} folds on disk: {', '.join(sorted(arms))}\n")

    data = {}
    for name, p in sorted(arms.items()):
        k, x = read_atoms(p)
        data[name] = (k, x)
        print(f"  {name:22s} {len(x)} atoms")

    ref_keys = data[sorted(data)[0]][0]
    for name, (k, _) in data.items():
        if k != ref_keys:
            raise SystemExit(f"atom identity differs in {name} -- cannot compare by order")
    print(f"\n  atom identity identical across all {len(data)} folds "
          f"({len(ref_keys)} atoms), so the comparison is atom-for-atom\n")

    def arm_of(n):
        return "int_usilu" if "usilu" in n else "int"

    print("  pairwise all-atom Kabsch RMSD, A:")
    rows = []
    for a, b in itertools.combinations(sorted(data), 2):
        r = kabsch_rmsd(data[a][1], data[b][1])
        kind = "A/A" if arm_of(a) == arm_of(b) else "A/B"
        rows.append((kind, a, b, r))
        print(f"    {kind}  {a:22s} vs {b:22s}  {r:10.6f}")

    aa = [r for k, _, _, r in rows if k == "A/A"]
    ab = [r for k, _, _, r in rows if k == "A/B"]
    print(f"\n  A/A self-RMSD  : max {max(aa):.6f} A  over {len(aa)} pairs")
    print(f"  A/B int vs usilu: min {min(ab):.6f}  max {max(ab):.6f} A  over {len(ab)} pairs")
    print(f"\n  envelope bound this org uses: 0.1136 A")
    print(f"  y-silu-lowering measured     : 4.133 A (298 aa)")


if __name__ == "__main__":
    main()
