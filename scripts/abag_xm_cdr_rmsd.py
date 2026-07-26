#!/usr/bin/env python3
"""Phase 4 label: per-CDR RMSD (the last missing Phase 4 label script).

For each CDR loop (H1, H2, H3 on the heavy chain; L1, L2, L3 on the light chain if
present), Kabsch-superpose the model's CDR CA atoms onto the native's and report
the post-superposition RMSD. Numbering is IMGT (via ANARCI); CDR ranges are the
IMGT definitions:
  VH: CDR1 27-38, CDR2 56-65, CDR3 105-117
  VL: CDR1 27-38, CDR2 56-65, CDR3 105-117
Model and native derive each antibody chain sequence from the same fold YAML, so
residue k (0-based along the chain) has the same IMGT number in both -- the CDR
CA sets are paired by sequence index, no alignment step.

Usage:
    PYTHONPATH=<worktree> python3 scripts/abag_xm_cdr_rmsd.py <model.cif> <native.cif> <fold.yaml> [--out json]
"""
import argparse, json, math, sys
from pathlib import Path
import gemmi
import numpy as np

THREE_TO_ONE = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
    "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V","MSE":"M","SEC":"U","PYL":"O","UNK":"X"}
CDR_RANGES = {"H1": (27, 38), "H2": (56, 65), "H3": (105, 117),
              "L1": (27, 38), "L2": (56, 65), "L3": (105, 117)}


def _seq_of(chain):
    return "".join(THREE_TO_ONE.get(r.name.strip().upper(), "X")
                     for r in chain.first_conformer())


def _load(path):
    st = gemmi.read_structure(str(path))
    chains = {}
    for model in st:
        for ch in model:
            chains[ch.name] = ch
        break
    return chains


def _find_chain(chains, want_seq):
    for name, ch in chains.items():
        if _seq_of(ch) == want_seq:
            return name
    return None


def _imgt_numbers(seq):
    """Return list of IMGT residue numbers (int), one per actual residue position
    (gaps - dropped so len == len(seq))."""
    from anarci import run_anarci
    r0, r1, r2, r3 = run_anarci([("H", seq)], scheme="imgt",
                                  ncpu=2, bit_score_threshold=40)
    if not r1 or not r1[0] or not r1[0][0]:
        return None
    numbering = r1[0][0][0]  # tuple[0] = list of ((imgt_num, ins), aa) over IMGT positions
    return [num[0][0] for num in numbering if num[1] != "-"]


def _cdr_cas(chain, imgt_nums, cdr_lo, cdr_hi):
    """CA positions of residues whose IMGT number is in [cdr_lo, cdr_hi]."""
    cas = []
    for k, res in enumerate(chain.first_conformer()):
        if k >= len(imgt_nums):
            break
        num = imgt_nums[k]
        if cdr_lo <= num <= cdr_hi:
            for at in res:
                if at.name == "CA":
                    cas.append([at.pos.x, at.pos.y, at.pos.z])
                    break
    return np.asarray(cas, dtype=np.float64) if cas else np.empty((0, 3))


def _kabsch_rmsd(P, Q):
    """RMSD of P onto Q after optimal Kabsch superposition (CA sets, same length)."""
    if len(P) < 3 or len(P) != len(Q):
        return None
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    # aligned_i = R @ (P_i - P.mean) + Q.mean  ==  (P_i - P.mean) @ R.T + Q.mean
    # diff_i = aligned_i - Q_i = (Pc @ R.T) - Qc
    diff = Pc @ R.T - Qc
    return float(np.sqrt((diff * diff).sum(axis=1).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("native")
    ap.add_argument("yaml")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import yaml as _yaml
    ydoc = _yaml.safe_load(Path(a.yaml).read_text())
    yseqs = {s["protein"]["id"]: s["protein"]["sequence"]
               for s in ydoc["sequences"] if s.get("protein", {}).get("id") in ("H", "L")}

    m_chains = _load(a.model)
    n_chains = _load(a.native)
    out = {"model": a.model, "native": a.native, "yaml": a.yaml, "cdrs": {}}
    for yid in ("H", "L"):
        seq = yseqs.get(yid)
        if not seq:
            continue
        mc = _find_chain(m_chains, seq)
        nc = _find_chain(n_chains, seq)
        if not mc or not nc:
            continue
        imgt = _imgt_numbers(seq)
        if imgt is None or len(imgt) != len(seq):
            out["cdrs"][f"{yid}1"] = None
            continue
        for cdr_name, (lo, hi) in CDR_RANGES.items():
            if not cdr_name.startswith(yid):
                continue
            m_cas = _cdr_cas(m_chains[mc], imgt, lo, hi)
            n_cas = _cdr_cas(n_chains[nc], imgt, lo, hi)
            rmsd = _kabsch_rmsd(m_cas, n_cas) if len(m_cas) >= 3 and len(m_cas) == len(n_cas) else None
            out["cdrs"][cdr_name] = round(rmsd, 6) if rmsd is not None else None
    print(json.dumps(out, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
