#!/usr/bin/env python3
"""Kabsch RMSD between the arms of `fused_key_ab_fixed.py`, with the A/A floor beside it.

Reuses `perf/other512/cif_rmsd.py`'s parser and superposition unchanged, so the numbers are
comparable across that lineage. Two metrics are reported per pair because the campaign's bar and
opendde's published cell are quoted in different ones: ALL-ATOM Kabsch (what tt-bio's own
`kabsch_rmsd` computes, despite the CA in its name) and CA-only Kabsch.

The same-arm pairs are the A/A floor. main-vs-main must be 0.000000 -- the arm is deterministic
and a nonzero floor means no other number in the run is interpretable.

    arm_rmsd.py perf/allm_safety/out --model opendde --size 512
"""
import argparse
import itertools
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "perf" / "other512"))
import cif_rmsd as CR  # noqa: E402


def coords(p: Path, ca_only: bool):
    cols, rows = CR.atom_site_table(p)
    xi, yi, zi = cols["Cartn_x"], cols["Cartn_y"], cols["Cartn_z"]
    ai = cols.get("label_atom_id", cols.get("auth_atom_id"))
    out = []
    for r in rows:
        if ca_only and ai is not None and r[ai].strip().strip('"') != "CA":
            continue
        out.append((float(r[xi]), float(r[yi]), float(r[zi])))
    return np.asarray(out, dtype=np.float64)


def kabsch(A, B):
    A = A - A.mean(0)
    B = B - B.mean(0)
    U, S, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, 1.0, d]) @ Vt
    return float(np.sqrt(((A @ R.T if False else (B @ R.T) - A) ** 2).sum(1).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--size", type=int, default=512)
    a = ap.parse_args()

    legs = {}
    pat = re.compile(rf"^cif_{re.escape(a.model)}_{a.size}_leg(\d+)_(\w+)$")
    for d in sorted(a.root.iterdir()):
        m = pat.match(d.name)
        if not m:
            continue
        cifs = sorted(d.glob("*.cif")) or sorted(d.glob("*.pdb"))
        if cifs:
            legs[(int(m.group(1)), m.group(2))] = cifs[0]
    if len(legs) < 2:
        print(f"only {len(legs)} leg(s) found under {a.root} -- nothing to score")
        return 1

    print(f"{a.model} {a.size} aa -- Kabsch RMSD between arms, Angstrom")
    print(f"  bar 0.60 A   seed floor 1.84 A (re-running with a different seed moves the "
          f"structure this far)")
    print(f"  {len(legs)} legs: " + ", ".join(f"leg{i}={arm}" for (i, arm) in sorted(legs)))
    print()
    print(f"  {'pair':28s} {'all-atom':>10s} {'CA-only':>10s}  {'n_atoms':>8s}")
    aa_floor = {"main": [], "cand": []}
    cross = []
    for (i1, a1), (i2, a2) in itertools.combinations(sorted(legs), 2):
        P, Q = legs[(i1, a1)], legs[(i2, a2)]
        allv = kabsch(coords(P, False), coords(Q, False))
        cav = kabsch(coords(P, True), coords(Q, True))
        n = len(coords(P, False))
        tag = f"leg{i1}{a1[:4]} vs leg{i2}{a2[:4]}"
        same = a1 == a2
        print(f"  {tag:28s} {allv:10.6f} {cav:10.6f}  {n:8d}   {'A/A' if same else 'A/B'}")
        (aa_floor[a1] if same else cross).append(allv)
    print()
    for arm, v in aa_floor.items():
        if v:
            print(f"  A/A floor {arm:5s}: max {max(v):.6f} A over {len(v)} same-arm pair(s)")
    if cross:
        print(f"  A/B cross    : max {max(cross):.6f} A over {len(cross)} pair(s)")
        print(f"  => {'INSIDE' if max(cross) < 0.60 else 'OUTSIDE'} the 0.60 A bar; "
              f"{max(cross) / 1.84:.4f}x the 1.84 A seed floor")
    return 0


if __name__ == "__main__":
    sys.exit(main())
