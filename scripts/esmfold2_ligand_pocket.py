#!/usr/bin/env python3
"""Did the ligand land in the RIGHT pocket? Predicted structure vs the deposited cocrystal.

The parity metrics for a co-fold -- pLDDT, distogram, coordinate RMSD -- are all
dominated by the L**2 protein block, so a ligand parked in solvent barely moves them.
This scores placement directly, and against the truth rather than against another
prediction:

  pocket_recall     of the protein residues within `--cutoff` of the ligand in the
                    DEPOSITED structure, the fraction also in contact in the prediction
  pocket_precision  the other direction: predicted contacts that are real
  jaccard           |intersection| / |union| of the two residue sets
  min_contact_A     closest ligand-protein heavy-atom distance in the prediction
                    (a large value with an empty contact set is the solvent failure)

It compares residue SETS, not coordinates, so it needs no superposition and no
resolved-atom bookkeeping: a prediction that folds the protein slightly differently but
grips the ligand with the same residues still scores 1.0, which is the biologically
meaningful statement. Sequence offset between the deposited chain and the folded
sequence is resolved by aligning the one-letter sequences.

  python3 scripts/esmfold2_ligand_pocket.py --pred out/fkg_ligand.cif \\
      --truth 1fkg.cif --ligand SB3
"""
from __future__ import annotations

import argparse
import json

import gemmi

_AA3 = gemmi


def _one_letter(res_name: str) -> str:
    info = _AA3.find_tabulated_residue(res_name)
    return info.one_letter_code.upper() if info and info.is_amino_acid() else "X"


def polymer_residues(st: gemmi.Structure):
    """[(chain_name, residue)] for every amino-acid residue, in file order."""
    out = []
    for ch in st[0]:
        for res in ch:
            info = _AA3.find_tabulated_residue(res.name)
            if info and info.is_amino_acid():
                out.append((ch.name, res))
    return out


def ligand_atoms(st: gemmi.Structure, code: str | None):
    """Heavy atoms of the ligand: the named CCD component, or every non-polymer,
    non-water heavy atom when no code is given (a predicted CIF may label it LIG)."""
    atoms = []
    for ch in st[0]:
        for res in ch:
            info = _AA3.find_tabulated_residue(res.name)
            is_poly = bool(info and (info.is_amino_acid() or info.is_nucleic_acid()))
            if is_poly or res.name in ("HOH", "DOD"):
                continue
            if code and res.name.upper() != code.upper():
                continue
            atoms += [a for a in res if a.element != gemmi.Element("H")]
    return atoms


def contact_indices(st: gemmi.Structure, code: str | None, cutoff: float):
    """(sequence-position set in contact, one-letter sequence, min contact distance).

    Positions index the structure's own polymer order, so the caller aligns them.
    """
    poly = polymer_residues(st)
    seq = "".join(_one_letter(r.name) for _c, r in poly)
    lig = ligand_atoms(st, code)
    if not lig:
        raise SystemExit(f"no ligand atoms found (code={code!r})")
    hit, best = set(), float("inf")
    for i, (_c, res) in enumerate(poly):
        for a in res:
            if a.element == gemmi.Element("H"):
                continue
            for b in lig:
                d = a.pos.dist(b.pos)
                best = min(best, d)
                if d <= cutoff:
                    hit.add(i)
                    break
            else:
                continue
            break
    return hit, seq, best


def align_offset(truth_seq: str, pred_seq: str) -> int:
    """Index in `pred_seq` where `truth_seq` starts (best identity over all offsets).

    The deposited chain is usually a contiguous, possibly gapped window of the sequence
    that was folded; an exact substring wins outright, otherwise take the offset with the
    most matching positions. Refuses a poor match rather than scoring the wrong residues.
    """
    at = pred_seq.find(truth_seq)
    if at >= 0:
        return at
    best_off, best_score = 0, -1
    for off in range(-len(truth_seq) + 1, len(pred_seq)):
        score = sum(1 for i, c in enumerate(truth_seq)
                    if 0 <= off + i < len(pred_seq) and pred_seq[off + i] == c)
        if score > best_score:
            best_off, best_score = off, score
    ident = best_score / len(truth_seq)
    if ident < 0.7:
        raise SystemExit(f"deposited and predicted sequences do not match "
                         f"({ident:.0%} identity at the best offset) -- wrong target?")
    return best_off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="predicted structure (cif/pdb)")
    ap.add_argument("--truth", required=True, help="deposited cocrystal (cif/pdb)")
    ap.add_argument("--ligand", default=None,
                    help="CCD code of the ligand in the DEPOSITED file (e.g. SB3)")
    ap.add_argument("--cutoff", type=float, default=4.5, help="contact cut-off, Angstrom")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    truth = gemmi.read_structure(args.truth)
    truth.remove_alternative_conformations()
    pred = gemmi.read_structure(args.pred)

    t_hit, t_seq, t_min = contact_indices(truth, args.ligand, args.cutoff)
    p_hit, p_seq, p_min = contact_indices(pred, None, args.cutoff)
    off = align_offset(t_seq, p_seq)
    t_mapped = {i + off for i in t_hit if 0 <= i + off < len(p_seq)}

    inter = t_mapped & p_hit
    union = t_mapped | p_hit
    m = {
        "pred": args.pred, "truth": args.truth, "ligand": args.ligand,
        "cutoff_A": args.cutoff, "seq_offset": off,
        "truth_pocket_residues": len(t_mapped), "pred_pocket_residues": len(p_hit),
        "shared_residues": len(inter),
        "pocket_recall": round(len(inter) / len(t_mapped), 4) if t_mapped else None,
        "pocket_precision": round(len(inter) / len(p_hit), 4) if p_hit else 0.0,
        "jaccard": round(len(inter) / len(union), 4) if union else None,
        "truth_min_contact_A": round(t_min, 3), "pred_min_contact_A": round(p_min, 3),
        "truth_pocket_seq": "".join(p_seq[i] for i in sorted(t_mapped)),
        "pred_pocket_seq": "".join(p_seq[i] for i in sorted(p_hit)),
    }
    print(json.dumps(m, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(m, f, indent=2)


if __name__ == "__main__":
    main()
