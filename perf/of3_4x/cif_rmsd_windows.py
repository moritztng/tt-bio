#!/usr/bin/env python3
"""Windowed all-atom Kabsch RMSD: separate a rigid-body hinge from local destruction.

A single global number cannot tell the two apart. cdk2x2 is a chimera of two copies of the same
domain, so a change that only rotates one copy relative to the other reports a large global RMSD
with every window internally perfect. Superpose each window on its own and the two cases separate:

    global large + windows small  -> rigid-body hinge, local geometry intact
    global large + windows large  -> the change destroyed structure

Same parser and same superposition as perf/other512/cif_rmsd.py, so global numbers are comparable.

    cif_rmsd_windows.py <ref.cif> <arm.cif> [--window 64]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "other512"))
from cif_rmsd import read_atoms, kabsch_rmsd  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", type=Path)
    ap.add_argument("arm", type=Path)
    ap.add_argument("--window", type=int, default=64)
    a = ap.parse_args()

    kr, xr = read_atoms(a.ref)
    ka, xa = read_atoms(a.arm)
    if kr != ka:
        raise SystemExit("atom identity differs -- cannot compare by order")

    # residue index per atom, from the atom_site key tuple (asym_id, seq_id, ...)
    res = [k[:2] for k in kr]
    order, seen = [], {}
    for r in res:
        if r not in seen:
            seen[r] = len(order)
            order.append(r)
    ridx = np.array([seen[r] for r in res])
    n_res = len(order)

    print(f"  {len(kr)} atoms, {n_res} residues, window {a.window}")
    print(f"  global                 {kabsch_rmsd(xa, xr):11.6f} A")
    for name, lo, hi in (("first half", 0, n_res // 2), ("second half", n_res // 2, n_res)):
        m = (ridx >= lo) & (ridx < hi)
        print(f"  {name:22s} {kabsch_rmsd(xa[m], xr[m]):11.6f} A  ({m.sum()} atoms)")

    print(f"\n  per-{a.window}-residue window, each superposed independently:")
    vals = []
    for lo in range(0, n_res, a.window):
        m = (ridx >= lo) & (ridx < lo + a.window)
        if m.sum() < 12:
            continue
        r = kabsch_rmsd(xa[m], xr[m])
        vals.append(r)
        print(f"    res {lo:4d}-{min(lo + a.window, n_res) - 1:4d}  {r:11.6f} A  ({m.sum()} atoms)")
    if vals:
        print(f"\n  windows: min {min(vals):.6f}  median {float(np.median(vals)):.6f}  "
              f"max {max(vals):.6f} A")


if __name__ == "__main__":
    main()
