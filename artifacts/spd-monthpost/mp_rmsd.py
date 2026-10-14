"""mp_rmsd.py A.cif B.cif ...pairs: CA RMSD (A) after Kabsch, over CA atoms present in both, keyed by (chain, residue)."""
import sys
import numpy as np


def ca(path):
    cols, out = [], {}
    for ln in open(path):
        if ln.startswith("_atom_site."):
            cols.append(ln.split(".", 1)[1].strip())
        elif ln.startswith(("ATOM", "HETATM")) and cols:
            f = ln.split()
            r = dict(zip(cols, f))
            if r.get("label_atom_id") == "CA":
                out[(r.get("label_asym_id"), r.get("label_seq_id"))] = [float(r["Cartn_x"]), float(r["Cartn_y"]), float(r["Cartn_z"])]
    return out


def rmsd(a, b):
    k = sorted(set(a) & set(b))
    P, Q = np.array([a[i] for i in k], float), np.array([b[i] for i in k], float)
    assert np.isfinite(P).all() and np.isfinite(Q).all(), "non-finite coordinates"
    P, Q = P - P.mean(0), Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(U @ Vt))
    S[-1] *= d
    return float(np.sqrt(max((P ** 2).sum() + (Q ** 2).sum() - 2 * S.sum(), 0) / len(k))), len(k)


for x, y in zip(sys.argv[1::2], sys.argv[2::2]):
    a, b = ca(x), ca(y)
    r, n = rmsd(a, b)
    per = {c: rmsd({k: v for k, v in a.items() if k[0] == c}, {k: v for k, v in b.items() if k[0] == c})[0]
           for c in sorted({k[0] for k in a})}
    print(f"{r:.3f} A over {n} CA, per chain " + " ".join(f"{c}={v:.3f}" for c, v in per.items()) + f"  {x}  {y}")
