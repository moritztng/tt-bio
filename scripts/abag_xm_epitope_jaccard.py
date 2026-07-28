#!/usr/bin/env python3
"""Phase 4 label: epitope Jaccard.

Epitope = the set of antigen residues that contact any antibody residue (any heavy-atom
distance < 8 A, the standard DockQ/interface cutoff). Reported as Jaccard similarity
between the model's epitope residue-index set and the native's, where residue indices
are 0-based positions along the antigen amino-acid sequence (which is identical between
model and native -- both derive it from the same fold YAML chain A -- so index i in
the model antigen maps to index i in the native antigen without an alignment step).

Usage:
    PYTHONPATH=<worktree> python3 scripts/abag_xm_epitope_jaccard.py <model.cif> <native.cif> <fold.yaml> [--out json]

Model CIF chain IDs (Protenix writes A,B,C,... in input order; boltz2/opendde similarly) need
not match the native (A,H,L here) -- the antigen is identified by matching its sequence to
the fold YAML's chain A sequence. Antibody chains = the YAML's H and L chains (matched into
the CIF by sequence). Sanity: raises if the matched antigen residue count != len(YAML A seq).
"""
import argparse, json, sys
from pathlib import Path
import gemmi

CONTACT_A = 8.0  # heavy-atom interface cutoff (DockQ convention)


THREE_TO_ONE = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
    "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V","MSE":"M","SEC":"U","PYL":"O","UNK":"X"}

def _seq_of(chain):
    # gemmi make_one_letter_sequence() returns "" for some protenix CIFs (non-standard
    # residue naming / polymer typing), so build the sequence from residue names directly.
    seq = []
    for res in chain.first_conformer():
        seq.append(THREE_TO_ONE.get(res.name.strip().upper(), "X"))
    return "".join(seq)


def _ca_by_residue(chain):
    """Return list of (residue_index, gemmi.Point) for polymer residues, 0-based along seq."""
    out = []
    for res in chain.first_conformer():
        if res.name not in ("ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
                            "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL","MSE","SEC","PYL","UNK"):
            continue
        ca = res.sole_residue("CA") if False else None
        for at in res:
            if at.name == "CA":
                ca = at.pos
                break
        if ca is not None:
            out.append((len(out), ca))
    return out


def _contact_set(ag_cas, ab_cas):
    """Antigen residue indices with any CA within CONTACT_A of any antibody CA."""
    ag_idx = set()
    for i, pa in ag_cas:
        for j, pb in ab_cas:
            if pa.dist(pb) < CONTACT_A:
                ag_idx.add(i)
                break
    return ag_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("native")
    ap.add_argument("yaml")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import yaml as _yaml
    ydoc = _yaml.safe_load(Path(a.yaml).read_text())
    yseqs = {}
    for s in ydoc["sequences"]:
        p = s.get("protein", {})
        if p.get("id") in ("A", "H", "L"):
            yseqs[p["id"]] = p.get("sequence") or ""

    def _find_chain(cif_path, want_seq):
        st = gemmi.read_structure(str(cif_path))
        best, best_len = None, -1
        for model in st:
            for ch in model:
                seq = _seq_of(ch)
                # match by exact equality, else longest common substring (handles X/unknowns)
                if seq == want_seq:
                    return ch.name, seq
                # quick prefix overlap for near-matches
                overlap = sum(1 for x, y in zip(seq, want_seq) if x == y)
                if overlap > best_len:
                    best_len, best = overlap, ch.name
            break
        return best, None

    # identify antigen + antibody chains in model and native by sequence
    m_ag = _find_chain(a.model, yseqs.get("A", ""))
    n_ag = _find_chain(a.native, yseqs.get("A", ""))
    ab_seqs = [yseqs.get("H", ""), yseqs.get("L", "")]
    m_ab = [_find_chain(a.model, s) for s in ab_seqs]
    n_ab = [_find_chain(a.native, s) for s in ab_seqs]
    m_ab = [c for c in m_ab if c]
    n_ab = [c for c in n_ab if c]

    def _load(path, chname):
        st = gemmi.read_structure(str(path))
        for model in st:
            for ch in model:
                if ch.name == chname:
                    return ch
            break
        return None

    m_ag_ch = _load(a.model, m_ag[0]) if m_ag else None
    n_ag_ch = _load(a.native, n_ag[0]) if n_ag else None
    m_ab_chs = [_load(a.model, c) for c, _ in m_ab]
    n_ab_chs = [_load(a.native, c) for c, _ in n_ab]

    # sanity: antigen residue count == len(YAML A)
    if m_ag_ch and yseqs.get("A"):
        nres = sum(1 for _ in m_ag_ch.first_conformer()
                    if _.name in ("ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
                                  "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL","UNK"))
        if nres != len(yseqs["A"]):
            raise SystemExit(f"model antigen residue count {nres} != YAML A len {len(yseqs['A'])}")

    m_ag_cas = _ca_by_residue(m_ag_ch) if m_ag_ch else []
    n_ag_cas = _ca_by_residue(n_ag_ch) if n_ag_ch else []
    m_ab_cas = []
    for ch in m_ab_chs:
        m_ab_cas += _ca_by_residue(ch)
    n_ab_cas = []
    for ch in n_ab_chs:
        n_ab_cas += _ca_by_residue(ch)

    m_epi = _contact_set(m_ag_cas, m_ab_cas)
    n_epi = _contact_set(n_ag_cas, n_ab_cas)
    union = m_epi | n_epi
    inter = m_epi & n_epi
    jac = (len(inter) / len(union)) if union else 1.0
    out = {"model": a.model, "native": a.native, "yaml": a.yaml,
           "model_antigen_chain": m_ag, "native_antigen_chain": n_ag,
           "model_epitope_size": len(m_epi), "native_epitope_size": len(n_epi),
           "epitope_intersection": len(inter), "epitope_union": len(union),
           "epitope_jaccard": round(jac, 6)}
    print(json.dumps(out, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
