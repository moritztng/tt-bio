#!/usr/bin/env python3
"""Generate a synthetic template mmCIF (+ a yaml referencing it) for OOM testing.

The template chain A reproduces the query sequence so the parser's
query<->template alignment is full-length, and lays the residues out as a
plausible (non-physical) extended chain with N/CA/C/CB atoms. Coordinates need
not be physical -- this exists only to exercise the device-resident template
injection path at large L without OOM.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gemmi

# minimal 1-letter -> 3-letter; query sequences use the 20 canonical AAs.
AA3 = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY",
    "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN",
    "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER", "T": "THR", "V": "VAL",
    "W": "TRP", "Y": "TYR",
}


def read_query(yaml_path: Path) -> str:
    """Pull the single-chain protein sequence out of a sweep yaml (no yaml dep)."""
    for line in yaml_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("sequence:"):
            return line.split("sequence:", 1)[1].strip()
    raise ValueError(f"no sequence in {yaml_path}")


def build_template(seq: str) -> gemmi.Structure:
    st = gemmi.Structure()
    st.name = "synth_tmpl"
    st.spacegroup_hm = "P 1"
    st.cell = gemmi.UnitCell(1, 1, 1, 90, 90, 90)
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    for i, aa in enumerate(seq):
        res = gemmi.Residue()
        res.name = AA3.get(aa.upper(), "GLY")
        res.seqid = gemmi.SeqId(i + 1, " ")
        x = 3.8 * i  # CA spacing along x; extended chain
        for name, dx, dy, dz in (
            ("N", -0.5, 0.8, 0.0),
            ("CA", 0.0, 0.0, 0.0),
            ("C", 0.7, 0.9, 0.0),
            ("CB", 0.0, -0.8, 1.2),
        ):
            if res.name == "GLY" and name == "CB":
                continue
            atom = gemmi.Atom()
            atom.name = name
            atom.element = gemmi.Element(name[0])
            atom.pos = gemmi.Position(x + dx, dy, dz)
            atom.occ = 1.0
            atom.b_iso = 50.0
            res.add_atom(atom)
        chain.add_residue(res)
    model.add_chain(chain)
    st.add_model(model)
    st.setup_entities()
    # Populate each polymer entity's SEQRES (full_sequence) from its residues so
    # the parser (which reads entity.full_sequence) sees the full-length sequence.
    for entity in st.entities:
        if entity.entity_type == gemmi.EntityType.Polymer:
            names = [res.name for res in seq_to_residues(seq)]
            entity.full_sequence = names
    st.assign_label_seq_id()
    return st


def seq_to_residues(seq: str):
    out = []
    for aa in seq:
        r = gemmi.Residue()
        r.name = AA3.get(aa.upper(), "GLY")
        out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-yaml", type=Path, required=True, help="sweep yaml to read the query from")
    ap.add_argument("--out-cif", type=Path, required=True)
    ap.add_argument("--out-yaml", type=Path, required=True, help="yaml with template + msa")
    args = ap.parse_args()

    seq = read_query(args.in_yaml)
    st = build_template(seq)
    args.out_cif.parent.mkdir(parents=True, exist_ok=True)
    st.make_mmcif_document().write_file(str(args.out_cif))

    # msa line (if any) carried from the source yaml so it stays offline
    msa_line = ""
    for line in args.in_yaml.read_text().splitlines():
        if line.strip().startswith("msa:"):
            msa_line = "      " + line.strip() + "\n"
    yaml = (
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        f"      sequence: {seq}\n"
        f"{msa_line}"
        "templates:\n"
        f"  - cif: {args.out_cif.resolve().as_posix()}\n"
    )
    args.out_yaml.write_text(yaml)
    print(f"wrote {args.out_cif} ({len(seq)} res) and {args.out_yaml}")


if __name__ == "__main__":
    main()
