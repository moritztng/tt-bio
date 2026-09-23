#!/usr/bin/env python3
"""Seed-paired Ca-RMSD between two predicted CIFs (or pred vs ground truth).

Superposes on matched CA atoms (by chain+seqid) and reports RMSD in Angstrom.
Used to compare the resident-trunk branch against main on a confident target:
per-op PCC can hide a fold regression, so we fold a known structure with the
same seed on both and compare the seed-paired CA-RMSD (eager noise sets the bar).

`--by-sequence` matches chains by their sequence instead of their name, and tries
every assignment among identical copies, keeping the lowest RMSD. Two models (or a
model and the PDB) rarely agree on chain letters for a homo-oligomer, and a swapped
copy would otherwise read as a 30 A error. Residues are keyed by their position in
the full sequence (label_seq_id), so a crystal structure with unmodelled loops lines
up with a prediction that has every residue.
"""
from __future__ import annotations

import argparse
import itertools
import sys

import gemmi


def ca_atoms(path: str) -> dict:
    st = gemmi.read_structure(path)
    st.remove_alternative_conformations()
    out = {}
    model = st[0]
    for chain in model:
        for res in chain:
            ca = res.find_atom("CA", "*")
            if ca is not None:
                out[(chain.name, res.seqid.num)] = ca.pos
    return out


def ca_chains(path: str, sequences: dict[str, str] | None = None) -> dict:
    """{chain name: (sequence, {residue index: CA position})} for the protein chains.

    The residue index is label_seq_id when the file has it, else the author number.
    The sequence is the entity's full one-letter sequence when the file declares one
    (mmCIF from the PDB does), else what the modelled residues spell."""
    st = gemmi.read_structure(path)
    st.remove_alternative_conformations()
    st.setup_entities()
    full = {}
    for ent in st.entities:
        if ent.entity_type == gemmi.EntityType.Polymer and ent.full_sequence:
            seq = gemmi.one_letter_code([gemmi.Entity.first_mon(m) for m in ent.full_sequence])
            for sub in ent.subchains:
                full[sub] = seq.upper()
    out = {}
    for chain in st[0]:
        cas, spelled = {}, []
        for res in chain:
            info = gemmi.find_tabulated_residue(res.name)
            if info is None or not info.is_amino_acid():
                continue
            ca = res.find_atom("CA", "*")
            if ca is None:
                continue
            idx = res.label_seq if res.label_seq is not None else res.seqid.num
            cas[idx] = ca.pos
            spelled.append(info.one_letter_code.upper())
        if not cas:
            continue
        sub = chain[0].subchain if len(chain) else ""
        seq = full.get(sub) or "".join(spelled)
        if chain.name in out:            # same author chain split over subchains
            out[chain.name][1].update(cas)
        else:
            out[chain.name] = (seq, cas)
    return out


def _same_chain(sa: str, sb: str) -> bool:
    """A prediction spells its whole sequence; a crystal entity also does. Tolerate a
    prediction whose chain spells only what the other declares (identical strings) or
    a modelled subset of it."""
    return sa == sb or (len(sa) != len(sb) and (sa in sb or sb in sa))


def matched_ca(a: dict, b: dict, max_assignments: int = 5040):
    """Every CA-pair assignment of b's chains onto a's, grouped by identical sequence.

    Yields (positions_a, positions_b, mapping) for each assignment. Chains of `a` with
    no sequence match in `b` are skipped, never guessed."""
    groups: dict[str, list[str]] = {}
    for name, (seq, _) in a.items():
        groups.setdefault(seq, []).append(name)
    options = []
    for seq, names_a in groups.items():
        names_b = [n for n, (sb, _) in b.items() if _same_chain(seq, sb)]
        if len(names_b) < len(names_a):
            names_a = names_a[:len(names_b)]
        if not names_a:
            continue
        options.append([list(zip(names_a, p)) for p in itertools.permutations(names_b, len(names_a))])
    for n, combo in enumerate(itertools.product(*options)):
        if n >= max_assignments:
            break
        mapping = [pair for grp in combo for pair in grp]
        pa, pb = [], []
        for na, nb in mapping:
            ca, cb = a[na][1], b[nb][1]
            for k in sorted(set(ca) & set(cb)):
                pa.append(ca[k])
                pb.append(cb[k])
        yield pa, pb, mapping


def best_rmsd(a: dict, b: dict):
    """(rmsd, n_ca, mapping, positions_a, positions_b) for the lowest-RMSD assignment."""
    best = None
    for pa, pb, mapping in matched_ca(a, b):
        if len(pa) < 3:
            continue
        r = gemmi.superpose_positions(pa, pb).rmsd
        if best is None or r < best[0]:
            best = (r, len(pa), mapping, pa, pb)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--by-sequence", action="store_true",
                    help="match chains by sequence, trying every assignment of identical copies")
    args = ap.parse_args()
    if args.by_sequence:
        best = best_rmsd(ca_chains(args.a), ca_chains(args.b))
        if best is None:
            print("NO_MATCH")
            sys.exit(1)
        print(f"n_ca={best[1]} rmsd={best[0]:.4f} chains={best[2]}")
        return
    a = ca_atoms(args.a)
    b = ca_atoms(args.b)
    keys = sorted(set(a) & set(b))
    if not keys:
        # fall back to positional matching within chain order
        av = list(a.values())
        bv = list(b.values())
        n = min(len(av), len(bv))
        pa = [av[i] for i in range(n)]
        pb = [bv[i] for i in range(n)]
    else:
        pa = [a[k] for k in keys]
        pb = [b[k] for k in keys]
    if not pa:
        print("NO_MATCH")
        sys.exit(1)
    sup = gemmi.superpose_positions(pa, pb)
    print(f"n_ca={len(pa)} rmsd={sup.rmsd:.4f}")


if __name__ == "__main__":
    main()
