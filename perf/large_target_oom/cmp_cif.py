#!/usr/bin/env python3
"""Fold-level numerical-equivalence check for the OOM workstream: compare two tt-bio CIFs.

Reports CA Kabsch RMSD, max CA deviation, and the PCC of per-residue pLDDT (B-factor column).
Both files come from the same tt-bio writer, so the _atom_site column order is parsed from the
header rather than assumed.

Usage: cmp_cif.py <a.cif> <b.cif>
"""
import sys

import numpy as np


def parse_cif(path):
    with open(path) as fh:
        lines = fh.read().splitlines()
    cols, rows, in_loop = [], [], False
    for i, ln in enumerate(lines):
        if ln.startswith("_atom_site."):
            cols.append(ln.split()[0])
            in_loop = True
        elif in_loop and cols and (ln.startswith("ATOM") or ln.startswith("HETATM")):
            parts = ln.split()
            if len(parts) < len(cols):
                parts = " ".join(lines[i : i + 2]).split()  # rare line wrap
            rows.append(parts)
        elif in_loop and cols and not ln.startswith("#"):
            if rows:
                break
    idx = {c: k for k, c in enumerate(cols)}
    out = {}
    for p in rows:
        if p[idx["_atom_site.label_atom_id"]] != "CA":
            continue
        key = (p[idx["_atom_site.label_asym_id"]], int(p[idx["_atom_site.label_seq_id"]]))
        out[key] = (
            np.array([float(p[idx["_atom_site.Cartn_x"]]), float(p[idx["_atom_site.Cartn_y"]]),
                      float(p[idx["_atom_site.Cartn_z"]])]),
            float(p[idx["_atom_site.B_iso_or_equiv"]]),
        )
    return out


def kabsch_rmsd(P, Q):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    C = Pc.T @ Qc
    V, _, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    U = V @ np.diag([1, 1, d]) @ Wt
    diff = Pc @ U - Qc
    return float(np.sqrt((diff**2).sum(1).mean())), float(np.sqrt((diff**2).sum(1)).max())


def main(a, b):
    ca_a, ca_b = parse_cif(a), parse_cif(b)
    common = sorted(set(ca_a) & set(ca_b))
    if len(common) < 10:
        sys.exit(f"only {len(common)} common CA atoms -- wrong pair of files?")
    P = np.array([ca_a[k][0] for k in common])
    Q = np.array([ca_b[k][0] for k in common])
    plddt_a = np.array([ca_a[k][1] for k in common])
    plddt_b = np.array([ca_b[k][1] for k in common])
    rmsd, maxdev = kabsch_rmsd(P, Q)
    pcc = float(np.corrcoef(plddt_a, plddt_b)[0, 1])
    print(f"common_CA={len(common)} kabsch_rmsd_A={rmsd:.4f} max_ca_dev_A={maxdev:.4f} "
          f"plddt_pcc={pcc:.6f} plddt_meanabs_delta={np.abs(plddt_a - plddt_b).mean():.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
