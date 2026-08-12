#!/usr/bin/env python3
"""All-atom Kabsch RMSD between the CIFs each arm of a fold A/B wrote.

Same parser and same superposition as `perf/nontri512/cif_rmsd.py` (commit 26f26116), so the
numbers are comparable across this lineage: Kabsch over ALL atoms, equal weights, which is what
tt-bio's own `kabsch_rmsd` does (memory `tt-bio-kabsch-rmsd-mislabeled-ca-rmsd`: the name says CA,
the function is all-atom).

    cif_rmsd.py <dir-of-arm-dirs> [--ref on]

Each arm dir is `<size>_<arm>_<run>` and holds one CIF. Every pair is reported; the `on` vs `on`
pair is the A/A floor and must be 0.000000 -- anything else means the instrument is broken and no
other number in the run is interpretable.
"""
import argparse
import itertools
import re
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
                    return keys, np.asarray(xyz, dtype=np.float64), keycols
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
    P = P @ (V @ np.diag([1.0, 1.0, d]) @ Wt)
    return float(np.sqrt(((P - Q) ** 2).sum() / len(P)))


def arm_of(dirname):
    m = re.match(r"^(\d+)_(.+)_(\d+)$", dirname)
    return (m.group(1), m.group(2)) if m else (None, dirname)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--size", default=None)
    ap.add_argument("--ref", default="on")
    # One `cif/` tree accumulates arms from every task that ever ran the harness, and arms from
    # different MODELS have different atom counts, which aborts the atom-for-atom check. Name the
    # arms the comparison is about.
    ap.add_argument("--arms", default=None, help="comma-separated arm names to include")
    a = ap.parse_args()

    data = {}
    for d in sorted(a.root.iterdir()):
        if not d.is_dir():
            continue
        size, arm = arm_of(d.name)
        if a.size and size != a.size:
            continue
        if a.arms and arm not in a.arms.split(","):
            continue
        cifs = sorted(d.glob("*.cif"))
        if not cifs:
            continue
        k, x, keycols = read_atoms(cifs[0])
        data[d.name] = (arm, k, x, keycols)
        print(f"  {d.name:26s} {len(x):6d} atoms  {cifs[0].name}")
    if len(data) < 2:
        raise SystemExit("need at least two folds on disk")

    ref_keys = data[sorted(data)[0]][1]
    for name, (_, k, _, _) in data.items():
        if k != ref_keys:
            raise SystemExit(f"atom identity differs in {name} -- cannot compare by order")
    print(f"\n  atom identity identical across all {len(data)} folds ({len(ref_keys)} atoms), "
          f"so the comparison is atom-for-atom\n")

    # CA-only as well as all-atom. Ask 4649 fixes its merge threshold on the Kabsch **CA** RMSD, and
    # a backbone-only superposition is the number this lineage has always quoted beside the all-atom
    # one (`opendde-512aa-deep-perf` §12.4: 1.2527 A CA, 2.5379 A all-atom).
    keycols = data[sorted(data)[0]][3]
    ca = None
    if "_atom_site.label_atom_id" in keycols:
        j = keycols.index("_atom_site.label_atom_id")
        ca = np.array([k[j] == "CA" for k in ref_keys])
        print(f"  CA atoms: {int(ca.sum())} of {len(ref_keys)}\n")

    print("  pairwise Kabsch RMSD, A  (all-atom | CA-only):")
    rows = []
    for x, y in itertools.combinations(sorted(data), 2):
        r = kabsch_rmsd(data[x][2], data[y][2])
        rca = kabsch_rmsd(data[x][2][ca], data[y][2][ca]) if ca is not None else float("nan")
        kind = "A/A" if data[x][0] == data[y][0] else "A/B"
        rows.append((kind, x, y, r, rca))
        print(f"    {kind}  {x:26s} vs {y:26s}  {r:11.6f} | {rca:11.6f}")

    aa = [(r, rca) for k, _, _, r, rca in rows if k == "A/A"]
    if aa:
        print(f"\n  A/A structural floor: max {max(v[0] for v in aa):.6f} A all-atom, "
              f"{max(v[1] for v in aa):.6f} A CA, over {len(aa)} pair(s)")
    print(f"\n  vs the `{a.ref}` arm  (all-atom | CA-only):")
    refs = [n for n, v in data.items() if v[0] == a.ref]
    for name, (arm, _, x, _) in sorted(data.items()):
        if arm == a.ref:
            continue
        vals = [kabsch_rmsd(x, data[r][2]) for r in refs]
        vca = [kabsch_rmsd(x[ca], data[r][2][ca]) for r in refs] if ca is not None else [float("nan")]
        print(f"    {arm:16s} {min(vals):9.6f} .. {max(vals):9.6f} | "
              f"{min(vca):9.6f} .. {max(vca):9.6f} A")


if __name__ == "__main__":
    main()
