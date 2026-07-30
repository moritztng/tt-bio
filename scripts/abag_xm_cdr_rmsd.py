#!/usr/bin/env python3
"""Phase 4 label: per-CDR RMSD (IMGT, via ANARCI).

For each CDR loop (H1/H2/H3 on heavy; L1/L2/L3 on light if present), Kabsch-
superpose the model's CDR CA atoms onto the native's and report the post-
superposition RMSD. Handles full-length H/L (variable+constant): anarci only
numbers the variable region, so we map IMGT numbers to chain residues via the
anarci query_start offset. Model and native share the fold-YAML sequence, so
residue k (0-based along the chain) has the same IMGT number in both -- the CDR
CA sets are paired by sequence index, no alignment step.

Usage:
    PYTHONPATH=<worktree> python3 scripts/abag_xm_cdr_rmsd.py <model.cif> <native.cif> <fold.yaml> [--out json]
"""
import argparse, contextlib, json, math, sys
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
    """Return (chain_name, offset): where want_seq starts in that chain, or (None, None).

    Same reason as in abag_xm_interface_lddt: the native routinely resolves residues the YAML
    construct does not, so requiring exact equality matched the model and never the native, and
    every CDR came back null for those targets. The offset matters here rather than being a
    formality -- it feeds _cdr_cas's query_start, so the IMGT numbering lands on the right
    residues instead of being shifted off the CDR.

    Third pass (closeout census class): equal-length near-identity unique match for
    natives carrying a few point differences from the YAML construct; same rule as
    abag_xm_interface_lddt._find_chain (mismatches <= max(1, 5% of length), unique).
    """
    for name, ch in chains.items():
        if _seq_of(ch) == want_seq:
            return name, 0
    for name, ch in chains.items():
        off = _seq_of(ch).find(want_seq)
        if off >= 0:
            return name, off
    max_mm = max(1, int(0.05 * len(want_seq)))
    hits = [name for name, ch in chains.items()
            if len(_seq_of(ch)) == len(want_seq)
            and sum(a != b for a, b in zip(_seq_of(ch), want_seq)) <= max_mm]
    if len(hits) == 1:
        return hits[0], 0
    return None, None


def _imgt_numbers(seq, chain_type):
    """Return (imgt_list, query_start) where imgt_list[k] is the IMGT number of
    chain residue (query_start + k), for the variable region anarci numbered.
    Returns (None, 0) if anarci found no hit."""
    from anarci import run_anarci
    # ANARCI prints species-limit warnings to stdout; callers parse this
    # script's stdout as JSON, so keep the channel clean (9lwc census class).
    with contextlib.redirect_stdout(sys.stderr):
        r0, r1, r2, r3 = run_anarci([(chain_type, seq)], scheme="imgt",
                                    ncpu=2, bit_score_threshold=40)
    if not r1 or not r1[0] or not r1[0][0]:
        return None, 0
    numbering = r1[0][0][0]
    imgt_list = [num[0][0] for num in numbering if num[1] != "-"]
    # query_start from the best hit (r2[0][0]); default 0 if absent
    qs = 0
    try:
        if r2 and r2[0] and r2[0][0]:
            qs = int(r2[0][0].get("query_start", 0))
    except Exception:
        qs = 0
    return imgt_list, qs


def _cdr_cas(chain, imgt_list, query_start, cdr_lo, cdr_hi):
    """CA positions of chain residues whose IMGT number is in [cdr_lo, cdr_hi].
    Residue j (0-based in chain) maps to imgt_list[j - query_start] when in range."""
    cas = []
    for j, res in enumerate(chain.first_conformer()):
        k = j - query_start
        if k < 0 or k >= len(imgt_list):
            continue
        num = imgt_list[k]
        if cdr_lo <= num <= cdr_hi:
            for at in res:
                if at.name == "CA":
                    cas.append([at.pos.x, at.pos.y, at.pos.z])
                    break
    return np.asarray(cas, dtype=np.float64) if cas else np.empty((0, 3))


def _kabsch_rmsd(P, Q):
    if len(P) < 3 or len(P) != len(Q):
        return None
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
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
        mc, m_off = _find_chain(m_chains, seq)
        nc, n_off = _find_chain(n_chains, seq)
        if not mc or not nc:
            continue
        imgt_list, qs = _imgt_numbers(seq, yid)
        if not imgt_list:
            out["cdrs"][f"{yid}1"] = None
            continue
        for cdr_name, (lo, hi) in CDR_RANGES.items():
            if not cdr_name.startswith(yid):
                continue
            # qs is where the IMGT-numbered region starts inside the YAML sequence; m_off/n_off
            # are where that YAML sequence starts inside each actual chain. Both are needed, or
            # a native with extra leading residues gets its CDRs read at the wrong offset.
            m_cas = _cdr_cas(m_chains[mc], imgt_list, qs + m_off, lo, hi)
            n_cas = _cdr_cas(n_chains[nc], imgt_list, qs + n_off, lo, hi)
            rmsd = _kabsch_rmsd(m_cas, n_cas) if len(m_cas) >= 3 and len(m_cas) == len(n_cas) else None
            out["cdrs"][cdr_name] = round(rmsd, 6) if rmsd is not None else None
    print(json.dumps(out, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
