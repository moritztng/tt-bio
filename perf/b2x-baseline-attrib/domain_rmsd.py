#!/usr/bin/env python3
"""Where two folds of cdk2x2_512 differ: per chain, and at the hinge between them.

The 512 aa fixture is CDK2 fused to a truncated copy of itself, and its inter-domain hinge is
unconstrained (memory ``cdk2x2-chimeric-fixture-cannot-score-non-bit-exact-parity``). A whole-
structure RMSD on it therefore reads the hinge angle, not the numerics. Superposing each chain on
itself separates the two: a per-chain RMSD near zero with a large whole-structure RMSD is the
hinge moving, and a per-chain RMSD of the same size as the whole is the fold itself changing.

Also prints the mean B-factor, which the CIF writer fills with plDDT, so two folds can be compared
on confidence even when no plDDT was recorded alongside the file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read(p: Path):
    lines = p.read_text().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_":
            j, cols = i + 1, []
            while j < len(lines) and lines[j].strip().startswith("_atom_site."):
                cols.append(lines[j].strip())
                j += 1
            if cols and "_atom_site.Cartn_x" in cols:
                idx = {c: k for k, c in enumerate(cols)}
                rows = []
                while j < len(lines) and not lines[j].startswith("#"):
                    f = lines[j].split()
                    if len(f) >= len(cols):
                        rows.append(f)
                    j += 1
                key, xyz, bf = [], [], []
                for f in rows:
                    key.append((f[idx["_atom_site.label_asym_id"]],
                                f[idx["_atom_site.label_seq_id"]],
                                f[idx["_atom_site.label_atom_id"]]))
                    xyz.append([float(f[idx["_atom_site.Cartn_x"]]),
                                float(f[idx["_atom_site.Cartn_y"]]),
                                float(f[idx["_atom_site.Cartn_z"]])])
                    bf.append(float(f[idx["_atom_site.B_iso_or_equiv"]]))
                return key, np.asarray(xyz), np.asarray(bf)
            i = j
        i += 1
    raise SystemExit(f"no atom_site loop in {p}")


def kabsch(A, B):
    A = A - A.mean(0)
    B = B - B.mean(0)
    V, _S, Wt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ Wt
    return float(np.sqrt((((B @ R.T) - A) ** 2).sum(1).mean()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--arm", type=Path, required=True)
    ap.add_argument("--label", default="pair")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    ka, xa, ba = read(a.ref)
    kb, xb, bb = read(a.arm)
    ia = {k: i for i, k in enumerate(ka)}
    common = [k for k in kb if k in ia]
    ib = {k: i for i, k in enumerate(kb)}
    A = np.asarray([xa[ia[k]] for k in common])
    B = np.asarray([xb[ib[k]] for k in common])

    res = {"label": a.label, "ref": str(a.ref), "arm": str(a.arm),
           "n_matched": len(common),
           "whole_allatom_rmsd_A": round(kabsch(A, B), 4),
           "mean_bfactor_ref": round(float(ba.mean()), 4),
           "mean_bfactor_arm": round(float(bb.mean()), 4),
           "per_chain": {}}
    for ch in sorted({k[0] for k in common}):
        sel = [i for i, k in enumerate(common) if k[0] == ch]
        ca = [i for i in sel if common[i][2] == "CA"]
        res["per_chain"][ch] = {
            "n_atoms": len(sel),
            "allatom_rmsd_A": round(kabsch(A[sel], B[sel]), 4),
            "ca_rmsd_A": round(kabsch(A[ca], B[ca]), 4) if ca else None,
        }
    print(json.dumps(res, indent=1))
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
