#!/usr/bin/env python3
"""RMSD between two design CIFs, direct and after a rigid fit.

    python perf/pxdesign/cif_rmsd.py A.cif B.cif [--chain A] [--json]

Used to answer one question: when the reference stack moves (torch 2.3.1/cu121 -> 2.7.1/cu128, the
step that reaches a B200), does PXDesign-d still return the same design for the same seed?

Direct RMSD is the primary number. Both designs are conditioned on the same target in the same
frame, so the binder coordinates are directly comparable and a rigid fit would hide a real
displacement. The fitted number is reported beside it because a pure global rotation would be a
different fact than a different fold, and Kabsch is not transitive, so only ever compare a pair.

Read the pair against two scales, or it means nothing: the same-stack same-seed repeat (which must
be 0.0) and the same-stack different-seed pair (the fixture's own diversity).
"""
import argparse
import json
import math
import sys


def atom_site(path):
    fields, rows, in_loop = [], [], False
    for line in open(path).read().splitlines():
        ls = line.strip()
        if ls.startswith("_atom_site."):
            fields.append(ls.split(".", 1)[1].split()[0])
            in_loop = True
            continue
        if not in_loop:
            continue
        if not ls or ls.startswith("#") or ls.startswith("loop_") or ls.startswith("_"):
            if rows:
                break
            continue
        parts = ls.split()
        if len(parts) == len(fields):
            rows.append(parts)
        elif rows:
            break
    return {f: i for i, f in enumerate(fields)}, rows


def coords(path, chain=None):
    idx, rows = atom_site(path)

    def col(*names):
        for n in names:
            if n in idx:
                return idx[n]
        return None

    cx, cy, cz = col("Cartn_x"), col("Cartn_y"), col("Cartn_z")
    ch, rs, an = col("label_asym_id", "auth_asym_id"), col("label_seq_id", "auth_seq_id"), \
        col("label_atom_id", "auth_atom_id")
    out = []
    for r in rows:
        if chain is not None and ch is not None and r[ch] != chain:
            continue
        key = (r[ch] if ch is not None else "?", r[rs] if rs is not None else "?",
               r[an] if an is not None else "?")
        out.append((key, (float(r[cx]), float(r[cy]), float(r[cz]))))
    return out


def rmsd(a, b):
    n = len(a)
    return math.sqrt(sum((p[i] - q[i]) ** 2 for p, q in zip(a, b) for i in range(3)) / n)


def kabsch_rmsd(a, b):
    import numpy as np
    P, Q = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    P = P - P.mean(0)
    Q = Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return float(np.sqrt(((P @ R.T - Q) ** 2).sum() / len(P)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--chain", default=None, help="restrict to one chain, e.g. the binder")
    ap.add_argument("--json", action="store_true")
    o = ap.parse_args()

    A, B = coords(o.a, o.chain), coords(o.b, o.chain)
    ka, kb = [k for k, _ in A], [k for k, _ in B]
    res = {"a": o.a, "b": o.b, "chain": o.chain, "n_atom_a": len(A), "n_atom_b": len(B),
           "atom_keys_match": ka == kb}
    if not A or ka != kb:
        res["ok"] = False
        res["why"] = "atom identity differs, so no RMSD is defined between these two files"
        print(json.dumps(res, indent=1) if o.json else res["why"])
        return 2
    xa, xb = [v for _, v in A], [v for _, v in B]
    res["rmsd_direct_A"] = round(rmsd(xa, xb), 4)
    res["rmsd_fitted_A"] = round(kabsch_rmsd(xa, xb), 4)
    res["max_atom_dev_A"] = round(max(math.dist(p, q) for p, q in zip(xa, xb)), 4)
    res["identical"] = res["rmsd_direct_A"] == 0.0
    res["ok"] = True
    if o.json:
        print(json.dumps(res, indent=1))
    else:
        print("%s vs %s%s: %d atoms, direct RMSD %.4f A, fitted %.4f A, max atom dev %.4f A"
              % (o.a, o.b, " chain " + o.chain if o.chain else "", len(xa),
                 res["rmsd_direct_A"], res["rmsd_fitted_A"], res["max_atom_dev_A"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
