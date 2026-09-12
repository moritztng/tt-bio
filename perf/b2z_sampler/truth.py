#!/usr/bin/env python3
"""The step curve against experiment, for the two panel targets that have a deposited structure.

Everything else in this workstream scores an arm against the 200-step fold, which answers "did the
answer change" but not "did it get worse". Two panel members have ground truth committed next to
their fixture -- `prot` is PDB 7ROA and `ubq` is 1UBQ -- so for those two the question can be asked
properly: CA lDDT and CA-RMSD against the crystal structure, at every step count.

Residues are matched by sequence, not by numbering: a deposited chain starts where the crystal
had density, which is rarely residue 1, and mmCIF label_seq_id and PDB resSeq do not agree. The
lDDT thresholds, inclusion radius and superposition are imported from `analyze.py` so the numbers
sit on the same scale as the rest of the tables.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import INCLUSION_A, LDDT_THRESHOLDS, kabsch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "perf" / "other512"))
from cif_rmsd import read_atoms  # noqa: E402

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M",
}


def ca_chains_cif(p: Path) -> dict[str, tuple[str, np.ndarray]]:
    keys, xyz = read_atoms(p)
    out: dict[str, list] = {}
    for k, x in zip(keys, xyz):
        asym, seq, atom, comp = k
        if atom != "CA" or comp not in THREE_TO_ONE:
            continue
        out.setdefault(asym, []).append((seq, THREE_TO_ONE[comp], x))
    return {c: ("".join(r[1] for r in v), np.asarray([r[2] for r in v])) for c, v in out.items()}


def ca_chains_pdb(p: Path) -> dict[str, tuple[str, np.ndarray]]:
    out: dict[str, list] = {}
    seen: set = set()
    for ln in p.read_text().splitlines():
        if not ln.startswith(("ATOM  ", "HETATM")) or ln[12:16].strip() != "CA":
            continue
        comp, ch, res = ln[17:20].strip(), ln[21], ln[22:27]
        alt = ln[16]
        if comp not in THREE_TO_ONE or alt not in (" ", "A") or (ch, res) in seen:
            continue
        seen.add((ch, res))
        out.setdefault(ch, []).append(
            (THREE_TO_ONE[comp], [float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]))
    return {c: ("".join(r[0] for r in v), np.asarray([r[1] for r in v])) for c, v in out.items()}


def ca_chains(p: Path):
    return ca_chains_pdb(p) if p.suffix.lower() == ".pdb" else ca_chains_cif(p)


def score(truth: Path, pred: Path) -> dict:
    """CA lDDT and CA-RMSD of pred against truth, over the residues the two share."""
    tc, pc = ca_chains(truth), ca_chains(pred)
    best = None
    for _tk, (tseq, txyz) in tc.items():
        for _pk, (pseq, pxyz) in pc.items():
            sm = difflib.SequenceMatcher(None, tseq, pseq, autojunk=False)
            blocks = [b for b in sm.get_matching_blocks() if b.size]
            n = sum(b.size for b in blocks)
            if best is None or n > best[0]:
                ti = np.concatenate([np.arange(b.a, b.a + b.size) for b in blocks]) if n else None
                pi = np.concatenate([np.arange(b.b, b.b + b.size) for b in blocks]) if n else None
                best = (n, txyz[ti], pxyz[pi], len(tseq), len(pseq))
    n, A, B, nt, np_ = best
    if n < 10:
        raise SystemExit(f"only {n} residues match between {truth.name} and {pred.name}")
    da = np.linalg.norm(A[:, None] - A[None], axis=-1)
    db = np.linalg.norm(B[:, None] - B[None], axis=-1)
    m = (da < INCLUSION_A) & ~np.eye(n, dtype=bool)
    diff = np.abs(da - db)[m]
    return {"n_matched": n, "n_truth": nt, "n_pred": np_,
            "lddt_ca_vs_truth": round(float(np.mean([(diff <= t).mean()
                                                     for t in LDDT_THRESHOLDS]) * 100), 3),
            "rmsd_ca_vs_truth_A": round(kabsch(A, B), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--cifdir", type=Path, required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = []
    pat = re.compile(rf"^{re.escape(a.target)}__s(\d+)_r(\d+)_seed(\d+)__")
    for p in sorted(a.cifdir.glob(f"{a.target}__*")):
        m = pat.match(p.name)
        if not m or "_model_" in p.name:
            continue
        rows.append({"steps": int(m[1]), "recycles": int(m[2]), "seed": int(m[3]),
                     **score(a.truth, p)})
    rows.sort(key=lambda r: (-r["recycles"], -r["steps"], r["seed"]))
    a.out.write_text(json.dumps({"target": a.target, "truth": str(a.truth), "rows": rows},
                                indent=1))
    for r in rows:
        print(f"  s{r['steps']:<4} r{r['recycles']} seed{r['seed']}  "
              f"lDDT_vs_truth={r['lddt_ca_vs_truth']:6.2f}  "
              f"RMSD_vs_truth={r['rmsd_ca_vs_truth_A']:6.3f} A  (n={r['n_matched']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
