"""Score the ab.sh runs: accuracy vs crystal, and the atomworks effect beside the seed floor.

Usage: python score.py <runs_root> <out.json>
For each input: CA-RMSD (and a Kabsch-refined TM-score, a lower bound on TM-align) of the
top-ranked sample against the crystal where one exists, per arm and seed; then the top-ranked
structure of 2.2.1 vs 3.0.4 at the same seed (the atomworks effect) next to seed 0 vs seed 1 within
one tree (the seed floor): CA-RMSD, all-heavy-atom RMSD and, for a ligand/glycan, the non-protein
heavy-atom RMSD after superposing on protein CA.
"""
import json
import sys
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
REPO = Path(__file__).resolve().parents[2]
CRYSTAL = {"prot": REPO / "examples/ground_truth_structures/prot.cif",
           "prot_glycan": REPO / "examples/ground_truth_structures/prot.cif"}
ARMS, SEEDS = ("aw221", "aw304"), (0, 1)


def load(p):
    a = pdbx.get_structure(pdbx.CIFFile.read(str(p)), model=1)
    return a[a.element != "H"]


def kabsch(P, Q):
    """R, t so that P @ R + t ~ Q."""
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, 1, d]) @ Vt
    return R, qc - pc @ R


def rmsd(P, Q):
    return float(np.sqrt(((P - Q) ** 2).sum(1).mean()))


def ca(a):
    return a[struc.filter_amino_acids(a) & (a.atom_name == "CA")]


def tm(P, Q):
    L = len(Q)
    d0 = 1.24 * np.cbrt(max(L, 19) - 15) - 1.8
    R, t = kabsch(P, Q)
    d = np.linalg.norm(P @ R + t - Q, axis=1)
    best = float((1 / (1 + (d / d0) ** 2)).sum() / L)
    for _ in range(20):  # TM-style refinement on the close core
        core = d < max(d0, 4.0)
        if core.sum() < 3:
            break
        R, t = kabsch(P[core], Q[core])
        d = np.linalg.norm(P @ R + t - Q, axis=1)
        best = max(best, float((1 / (1 + (d / d0) ** 2)).sum() / L))
    return best


def pair(a, b):
    """Superpose b onto a on protein CA; RMSD over CA, all matched heavy atoms, non-protein heavy atoms."""
    ib = {k: i for i, k in enumerate(zip(b.chain_id, b.res_id, b.atom_name))}
    common = [(i, ib[k]) for i, k in enumerate(zip(a.chain_id, a.res_id, a.atom_name)) if k in ib]
    ia, jb = np.array([c[0] for c in common]), np.array([c[1] for c in common])
    A, B = a.coord[ia], b.coord[jb]
    prot = struc.filter_amino_acids(a)[ia]
    isca = prot & (a.atom_name[ia] == "CA")
    R, t = kabsch(B[isca], A[isca])
    B2 = B @ R + t
    out = {"ca": rmsd(B2[isca], A[isca]), "heavy": rmsd(B2, A), "n_heavy": int(len(ia))}
    if (~prot).any():
        out["nonprotein"] = rmsd(B2[~prot], A[~prot])
        out["n_nonprotein"] = int((~prot).sum())
    return out


res = {}
for d in sorted({p.name.rsplit("_", 2)[0] for p in ROOT.iterdir() if p.is_dir()}):
    top = {}
    for arm in ARMS:
        for s in SEEDS:
            run = ROOT / f"{d}_{arm}_s{s}"
            ok = (run / "rc").exists() and (run / "rc").read_text().strip() == "0"
            cifs = list(run.rglob(f"{d}.cif")) if ok else []
            top[(arm, s)] = load(cifs[0]) if cifs else None
    r = {"crystal": {}, "pairs": {}}
    if d in CRYSTAL:
        x = ca(load(CRYSTAL[d]))
        for (arm, s), a in top.items():
            if a is None:
                continue
            p = ca(a[a.chain_id == a.chain_id[0]])
            n = min(len(p), len(x))
            P, Q = p.coord[:n], x.coord[:n]
            R, t = kabsch(P, Q)
            r["crystal"][f"{arm}_s{s}"] = {"ca_rmsd": rmsd(P @ R + t, Q), "tm_kabsch": tm(P, Q), "n_ca": int(n)}

    def P(k1, k2):
        a, b = top.get(k1), top.get(k2)
        return pair(a, b) if a is not None and b is not None else None

    for s in SEEDS:
        r["pairs"][f"atomworks_effect_s{s}"] = P(("aw221", s), ("aw304", s))
    for arm in ARMS:
        r["pairs"][f"seed_floor_{arm}"] = P((arm, 0), (arm, 1))
    res[d] = r
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
