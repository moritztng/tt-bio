#!/usr/bin/env python3
"""Boltz-2 --fast vs full-precision parity.

Compares two prediction runs of the SAME inputs with the SAME seed — one with
``--fast`` (block-fp8) and one without (bfloat16) — reporting per-structure and
per-chain Kabsch RMSD, coordinate PCC, and confidence-metric deltas.

Both runs share input, seed and atom ordering, so atoms pair 1:1 by
(chain, seqid, residue, atom, altloc). NOTE: the TT diffusion pipeline is not
bit-deterministic run-to-run even at fixed seed, so interpret fast-vs-full
against a full-vs-full(repeat) noise floor and full-vs-full(other-seed) spread.
Global multi-chain RMSD is dominated by inter-chain placement; per-chain RMSD
isolates each chain's internal fold fidelity.

Usage: boltz2_fast_parity.py FULL_RESULT_DIR FAST_RESULT_DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gemmi
import numpy as np

CONF_KEYS = ["confidence_score", "ptm", "iptm", "complex_plddt",
             "complex_iplddt", "complex_pde", "complex_ipde"]


def load_atoms(path: Path):
    st = gemmi.read_structure(str(path))
    recs = {}
    for chain in st[0]:
        for res in chain:
            for atom in res:
                recs[(chain.name, res.seqid.num, res.name, atom.name, atom.altloc)] = \
                    np.array([atom.pos.x, atom.pos.y, atom.pos.z])
    return recs


def kabsch(A: np.ndarray, B: np.ndarray):
    ca, cb = A.mean(0), B.mean(0)
    A0, B0 = A - ca, B - cb
    U, _, Vt = np.linalg.svd(A0.T @ B0)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    A_al = (R @ A0.T).T
    rmsd = float(np.sqrt(((A_al - B0) ** 2).sum() / len(A)))
    return rmsd, A_al, B0


def tm_score(pred_al: np.ndarray, ref: np.ndarray) -> float:
    """TM-score (Zhang-Skolnick) under a FIXED 1:1 CA alignment.

    ``pred_al`` is the predicted CA coords already Kabsch-aligned onto ``ref``;
    both are [N,3] over the same matched atoms. TM-score in [0,1] (1 = identical).
    Uses the standard d0 = 1.24*(L-15)^(1/3) - 1.8 (clamped to >= 0.5) and the
    per-residue deviation after superposition. The alignment is fixed (identical
    atoms pair 1:1), so this is the fixed-alignment TM-score, the right metric
    for implementation parity (no alignment search needed).
    """
    L = len(pred_al)
    if L < 1:
        return 0.0
    Lnorm = max(L, 1)
    d0 = 1.24 * (Lnorm - 15) ** (1.0 / 3.0) - 1.8
    if d0 < 0.5:
        d0 = 0.5
    dev = np.sqrt(((pred_al - ref) ** 2).sum(1))
    return float(np.sum(1.0 / (1.0 + (dev / d0) ** 2)) / Lnorm)


def lddt(pred_al: np.ndarray, ref: np.ndarray, cutoff: float = 15.0) -> float:
    """CA-lDDT (Mariani et al.) under a FIXED 1:1 alignment.

    ``pred_al`` is the predicted CA coords already Kabsch-aligned onto ``ref``
    (alignment does NOT change pairwise distances, so lDDT is alignment-
    invariant — we pass the aligned coords only for convenience). For each
    residue, neighbors within ``cutoff`` A in the REFERENCE structure are
    preserved contacts; per-residue lDDT averages the fraction preserved at the
    four standard thresholds (0.5, 1, 2, 4 A). lDDT in [0,1] (1 = identical
    local structure). Returns 0.0 for <2 residues.
    """
    L = len(ref)
    if L < 2:
        return 0.0
    dref = np.sqrt(((ref[:, None, :] - ref[None, :, :]) ** 2).sum(-1))
    dpred = np.sqrt(((pred_al[:, None, :] - pred_al[None, :, :]) ** 2).sum(-1))
    thr = (0.5, 1.0, 2.0, 4.0)
    per_res = np.zeros(L)
    for i in range(L):
        nb = np.where((dref[i] < cutoff) & (np.arange(L) != i))[0]
        if len(nb) == 0:
            per_res[i] = 1.0
            continue
        dd = np.abs(dref[i, nb] - dpred[i, nb])
        per_res[i] = float(np.mean([np.mean(dd < t) for t in thr]))
    return float(per_res.mean())


def compare_structure(full_cif: Path, fast_cif: Path):
    full, fast = load_atoms(full_cif), load_atoms(fast_cif)
    keys = [k for k in full if k in fast]
    n_full, n_fast, n = len(full), len(fast), len(keys)
    if n == 0:
        return {"n_matched": 0, "n_full": n_full, "n_fast": n_fast, "per_chain": {}}
    F = np.array([full[k] for k in keys])
    G = np.array([fast[k] for k in keys])
    rmsd, G_al, F0 = kabsch(G, F)
    pcc = float(np.corrcoef(G_al.flatten(), F0.flatten())[0, 1])
    dev = np.sqrt(((G_al - F0) ** 2).sum(1))
    per_chain = {}
    for cid in sorted({k[0] for k in keys}):
        ck = [k for k in keys if k[0] == cid]
        if len(ck) < 3:
            continue
        cr, cg_al, cf0 = kabsch(np.array([fast[k] for k in ck]),
                                np.array([full[k] for k in ck]))
        per_chain[cid] = {"n": len(ck), "rmsd": cr,
                          "pcc": float(np.corrcoef(cg_al.flatten(), cf0.flatten())[0, 1])}
    _tm = tm_score(G_al, F0)
    _lddt = lddt(G_al, F0)
    return {"n_matched": n, "n_full": n_full, "n_fast": n_fast,
            "kabsch_rmsd": rmsd, "coord_pcc": pcc,
            "tm_score": _tm, "lddt": _lddt,
            "dev_max": float(dev.max()), "dev_med": float(np.median(dev)),
            "per_chain": per_chain}


def load_results(d: Path):
    return {r["id"]: r for r in json.load(open(d / "results.json"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("full_dir")
    ap.add_argument("fast_dir")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    full_dir, fast_dir = Path(args.full_dir), Path(args.fast_dir)
    full_res, fast_res = load_results(full_dir), load_results(fast_dir)
    ids = [i for i in full_res if i in fast_res]
    tag = f" [{args.label}]" if args.label else ""

    geo = {}
    print(f"### Geometry{tag} (B aligned onto A)\n")
    print("| target | n_atoms | global RMSD (Å) | global PCC | per-chain RMSD (Å) | per-chain PCC |")
    print("|---|---|---|---|---|---|")
    for i in ids:
        s = compare_structure(full_dir / "structures" / f"{i}.cif",
                              fast_dir / "structures" / f"{i}.cif")
        geo[i] = s
        if s["n_matched"] == 0:
            print(f"| {i} | NO MATCH full={s['n_full']} fast={s['n_fast']} | | | | |")
            continue
        pc = s["per_chain"]
        pcr = " ".join(f"{c}:{pc[c]['rmsd']:.2f}" for c in pc)
        pcp = " ".join(f"{c}:{pc[c]['pcc']:.3f}" for c in pc)
        print(f"| {i} | {s['n_matched']} | {s['kabsch_rmsd']:.3f} | {s['coord_pcc']:.4f} | {pcr} | {pcp} |")

    print(f"\n### Confidence deltas{tag} (B − A)\n")
    print("| target | " + " | ".join(CONF_KEYS) + " |")
    print("|" + "---|" * (len(CONF_KEYS) + 1))
    for i in ids:
        f, g = full_res[i], fast_res[i]
        cells = [f"{g[k]-f[k]:+.4f}" if k in f and k in g and isinstance(f[k], (int, float)) else "—"
                 for k in CONF_KEYS]
        print(f"| {i} | " + " | ".join(cells) + " |")

    print("\n<!-- JSON " + json.dumps({"geometry": geo}) + " -->")


if __name__ == "__main__":
    main()
