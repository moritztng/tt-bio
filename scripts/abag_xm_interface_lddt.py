#!/usr/bin/env python3
"""Phase 4 label: interface lDDT (from-scratch; ost/OpenStructure not installed).

lDDT (local Distance Difference Test, Mariani et al. 2013, Bioinformatics 29(20):2722)
per-residue: for each residue i in the model, consider all atoms in OTHER residues
within cutoff_radius (15 A) of any atom of residue i. For each such considered
atom pair (a in residue i, b in residue j != i), score it by |d_model(a,b) -
d_native(a',b')| < threshold, where a', b' are the same atoms in the residue-mapped
native. lDDT(i) = mean over thresholds {0.5,1,2,4} A of the fraction of considered
pairs passing the threshold. Interface lDDT = mean of lDDT(i) over the antigen
interface residues (antigen residues with any CA within 8 A of an antibody CA,
same definition as epitope Jaccard).

Residue mapping: model and native derive each chain sequence from the same fold YAML,
so model residue index k (0-based along the chain sequence) maps to native residue
index k of the SAME chain (identified by exact sequence match). No alignment step.

Usage:
    PYTHONPATH=<worktree> python3 scripts/abag_xm_interface_lddt.py <model.cif> <native.cif> <fold.yaml> [--out json]
"""
import argparse, json, math, sys
from pathlib import Path
import gemmi

THREE_TO_ONE = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
    "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V","MSE":"M","SEC":"U","PYL":"O","UNK":"X"}
CUTOFF_RADIUS = 15.0
THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
CONTACT_A = 8.0


def _seq_of(chain):
    return "".join(THREE_TO_ONE.get(r.name.strip().upper(), "X")
                     for r in chain.first_conformer())


def _atoms_by_residue(chain):
    """Return list of residues; each residue = (residue_index, [(atom_name, gemmi.Point)])."""
    out = []
    for res in chain.first_conformer():
        atoms = [(at.name, at.pos) for at in res]
        if atoms:
            out.append((len(out), atoms))
    return out


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

    Exact equality is too strict for the native. A deposited structure routinely resolves
    residues the YAML construct does not carry -- 9k6j's antigen is 181 residues in the YAML
    and the model, 224 in the native -- so the antigen matched the model and could never match
    the native, and interface lDDT was skipped for the whole target. Seven of 48 labelled
    targets failed exactly this way.

    Every observed case is an exact CONTIGUOUS substring (extra residues at the C-terminus,
    offset 0), so a substring search is sufficient and stays unambiguous; anything needing gapped
    alignment is deliberately still a miss rather than a guess. The offset is returned rather
    than assumed zero so residue indices can be shifted instead of taken as identical.

    Third pass (closeout census class): the deposited native can carry a few point
    differences from the YAML construct (9ly2/9ly3/9lz2: ~5 mismatches on 283-344 aa;
    9mz8: 1 on a 16-mer). Accept an equal-length near-identity match -- mismatches
    <= max(1, 5% of length) -- ONLY when it is unique across chains; a wrong-chain
    guess is worse than a null.
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


def _lddt_per_residue(model_residues, native_residues, model_to_native_idx):
    """model_to_native_idx: model_residue_index -> native_residue_index (same for matched chains).
    Returns {model_residue_index: lddt}."""
    # build per-residue atom lists with positions; precompute native atom positions by (native_res_idx, atom_name)
    native_atoms = {}  # (nres_idx, atom_name) -> Point
    for nres_idx, atoms in native_residues:
        for an, pos in atoms:
            native_atoms[(nres_idx, an)] = pos

    lddt = {}
    for mres_idx, atoms in model_residues:
        nres_idx = model_to_native_idx.get(mres_idx, mres_idx)
        # considered atom pairs: atoms b in OTHER model residues within CUTOFF_RADIUS of any atom a of mres
        # collect candidate neighbor residues first (by any atom within cutoff)
        # The cutoff test already computes d_model, and |d_model - d_native| does not depend on the
        # threshold -- so keep the distance from the test and reduce each pair to a single delta,
        # once. The previous form recomputed both distances and both native lookups for every one of
        # the four thresholds, i.e. 8 distance evaluations and 8 dict lookups per pair instead of 2
        # and 2. This stage measured 4.35 s/sample against 0.28-0.46 s for its sibling per-sample
        # stages, which is 39% of a fold's label cost.
        #
        # Bit-identical, not merely close: the same two gemmi .dist() calls on the same inputs, and
        # the same abs() and comparison. Nothing is reordered or approximated -- only stopped from
        # being repeated.
        deltas = []
        for a_name, a_pos in atoms:
            na = native_atoms.get((nres_idx, a_name))
            for other_mres_idx, other_atoms in model_residues:
                if other_mres_idx == mres_idx:
                    continue
                other_nres_idx = model_to_native_idx.get(other_mres_idx, other_mres_idx)
                for b_name, b_pos in other_atoms:
                    d_model = a_pos.dist(b_pos)
                    if d_model > CUTOFF_RADIUS:
                        continue
                    # A pair whose native counterpart is missing was counted as "considered" but
                    # never scored, so it must still gate the None result below the same way.
                    nb = native_atoms.get((other_nres_idx, b_name))
                    if na is None or nb is None:
                        deltas.append(None)
                        continue
                    deltas.append(abs(d_model - na.dist(nb)))
        if not deltas:
            lddt[mres_idx] = None
            continue
        scored = [d for d in deltas if d is not None]
        total = len(scored)
        per_thr = [sum(1 for d in scored if d < thr) / total if total else 0.0
                   for thr in THRESHOLDS]
        lddt[mres_idx] = sum(per_thr) / len(per_thr)
    return lddt


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
               for s in ydoc["sequences"] if s.get("protein", {}).get("id") in ("A", "H", "L")}

    m_chains = _load(a.model)
    n_chains = _load(a.native)
    # identify each YAML chain in model and native by exact sequence match
    chain_map = {}  # yaml_id -> (model_chain_name, native_chain_name)
    for yid, seq in yseqs.items():
        mc, m_off = _find_chain(m_chains, seq)
        nc, n_off = _find_chain(n_chains, seq)
        if mc and nc:
            chain_map[yid] = (mc, nc, m_off, n_off)
    if "A" not in chain_map:
        raise SystemExit("antigen (YAML A) not found in model/native by sequence")

    # per-chain residue lists + index mapping (model idx -> native idx, identity since same seq)
    m_res = {yid: _atoms_by_residue(m_chains[v[0]]) for yid, v in chain_map.items()}
    n_res = {yid: _atoms_by_residue(n_chains[v[1]]) for yid, v in chain_map.items()}

    # interface antigen residues (CA-CA < CONTACT_A to any antibody chain)
    ag_name = chain_map["A"][0]
    ab_names = [chain_map[k][0] for k in ("H", "L") if k in chain_map]
    m_ag_cas = [(i, next((p for n, p in ats if n == "CA"), None))
                for i, ats in m_res["A"] if any(n == "CA" for n, _ in ats)]
    m_ab_cas = []
    for yid in ("H", "L"):
        if yid not in m_res:
            continue
        for i, ats in m_res[yid]:
            for n, p in ats:
                if n == "CA":
                    m_ab_cas.append((i, p))
    interface = set()
    for i, pa in m_ag_cas:
        if pa is None:
            continue
        for j, pb in m_ab_cas:
            if pa.dist(pb) < CONTACT_A:
                interface.add(i)
                break

    # lDDT over the antigen chain (A), restricted to interface residues for interface lDDT
    # residue mapping: identity within the antigen chain (same sequence)
    # Model index -> native index. Identity only holds when both matched the YAML sequence at
    # the same offset; when the native carries extra leading residues the shift is essential,
    # or every residue would be scored against the wrong one.
    _, _, m_off_a, n_off_a = chain_map["A"]
    ag_lddt = _lddt_per_residue(
        m_res["A"], n_res["A"],
        {i: i - m_off_a + n_off_a for i in range(len(m_res["A"]))})
    iface_lddt = [v for i, v in ag_lddt.items() if i in interface and v is not None]
    mean_iface = sum(iface_lddt) / len(iface_lddt) if iface_lddt else None
    all_lddt = [v for v in ag_lddt.values() if v is not None]
    mean_all = sum(all_lddt) / len(all_lddt) if all_lddt else None

    out = {"model": a.model, "native": a.native, "yaml": a.yaml,
           "antigen_chain": chain_map["A"][0],
           "n_interface_residues": len(interface),
           "interface_lddt": round(mean_iface, 6) if mean_iface is not None else None,
           "mean_antigen_lddt": round(mean_all, 6) if mean_all is not None else None}
    print(json.dumps(out, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
