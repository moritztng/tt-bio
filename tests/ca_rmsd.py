#!/usr/bin/env python3
"""Seed-paired Ca-RMSD between two predicted CIFs (or pred vs ground truth).

Superposes on matched CA atoms (by chain+seqid) and reports RMSD in Angstrom.
Used to compare the resident-trunk branch against main on a confident target:
per-op PCC can hide a fold regression, so we fold a known structure with the
same seed on both and compare the seed-paired CA-RMSD (eager noise sets the bar).
"""
from __future__ import annotations

import argparse
import sys

import gemmi


def ca_atoms(path: str) -> dict:
    st = gemmi.read_structure(path)
    st.remove_alternative_conformations()
    out = {}
    model = st[0]
    for chain in model:
        for res in chain:
            ca = res.find_atom("CA", "*")
            if ca is not None:
                out[(chain.name, res.seqid.num)] = ca.pos
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    args = ap.parse_args()
    a = ca_atoms(args.a)
    b = ca_atoms(args.b)
    keys = sorted(set(a) & set(b))
    if not keys:
        # fall back to positional matching within chain order
        av = list(a.values())
        bv = list(b.values())
        n = min(len(av), len(bv))
        pa = [av[i] for i in range(n)]
        pb = [bv[i] for i in range(n)]
    else:
        pa = [a[k] for k in keys]
        pb = [b[k] for k in keys]
    if not pa:
        print("NO_MATCH")
        sys.exit(1)
    sup = gemmi.superpose_positions(pa, pb)
    print(f"n_ca={len(pa)} rmsd={sup.rmsd:.4f}")


if __name__ == "__main__":
    main()
