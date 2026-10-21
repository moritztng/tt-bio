"""Chemistry check: does a generated design carry the input's chemistry unchanged?

    python perf/rfd4/cond_chem/chem_check.py --input <annotated input cif> \
        --generated <generated cif> [--comps SAM,DA,DC,...] [--out json]

Geometry is graded in Angstrom with a seed floor beside it (``perf/rfd4/e2e/grade.py``).
Chemistry is not: a design that puts the right atoms in nearly the right place with the wrong
element, the wrong bond order or the wrong formal charge is wrong, not imprecise. This script
is the hard-stop half of the grade, and it reads the written CIF, not an in-memory array,
because the file is what the user gets.

Three tables, all restricted to the non-protein components (ligands, nucleotides, glycans):

* ``_atom_site.type_symbol``       element, per atom
* ``_atom_site.pdbx_formal_charge`` formal charge, per atom
* ``_chem_comp_bond``              bond order and aromatic flag, per component

Atoms are keyed by (label_comp_id, auth_seq_id, label_atom_id) so a chain renaming between the
input and the design does not read as a chemistry change. Any difference is a failure.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

#: The 20 + UNK protein residues and water. Everything else is "chemistry the protein path
#: never exercises" and is what this script grades.
PROTEIN_COMPS = frozenset(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL UNK HOH".split()
)

#: An expandable segment is written as one placeholder residue named `<min>-<max>` holding a
#: single element-X atom, and the design replaces it with the residues it sampled. It is a
#: length sentinel, not a chemical component, so its disappearance is the writer working.
_SENTINEL = re.compile(r"^\d+-\d+$")


def is_chemistry(comp: str) -> bool:
    return comp not in PROTEIN_COMPS and not _SENTINEL.match(comp)


def read_block(path):
    import biotite.structure.io.pdbx as pdbx

    f = pdbx.CIFFile.read(str(path))
    return f.block


def _col(cat, name):
    return cat[name].as_array(str) if name in cat else None


def atom_chemistry(block, comps=None, hydrogens=False):
    """``{(comp, seq, atom): (element, charge)}`` for every non-protein atom.

    Hydrogens are excluded by default: RFD4's atom37 vocabulary is heavy-atom only, so a
    design never carries them even when the input did. They are counted, not compared.
    """
    cat = block["atom_site"]
    comp = _col(cat, "label_comp_id")
    n_h = [0]
    seq = _col(cat, "auth_seq_id")
    atom = _col(cat, "label_atom_id")
    elem = _col(cat, "type_symbol")
    charge = _col(cat, "pdbx_formal_charge")
    out = {}
    for i in range(len(comp)):
        c = comp[i]
        if c not in comps if comps is not None else not is_chemistry(c):
            continue
        if not hydrogens and elem[i] in ("H", "D"):
            n_h[0] += 1
            continue
        out[(c, seq[i], atom[i])] = (elem[i], _charge(charge[i] if charge is not None else "?"))
    return out, n_h[0]


def _charge(v):
    """CIF formal charge as an int; `?`/`.`/empty all mean neutral."""
    v = (v or "").strip()
    if v in ("", "?", "."):
        return 0
    try:
        return int(float(v))
    except ValueError:
        return v


def bond_chemistry(block, comps=None):
    """``{(comp, atom_a, atom_b): (value_order, aromatic)}``, atom pair unordered."""
    if "chem_comp_bond" not in block:
        return {}
    cat = block["chem_comp_bond"]
    comp = _col(cat, "comp_id")
    a1 = _col(cat, "atom_id_1")
    a2 = _col(cat, "atom_id_2")
    order = _col(cat, "value_order")
    arom = _col(cat, "pdbx_aromatic_flag")
    out = {}
    for i in range(len(comp)):
        c = comp[i]
        if c not in comps if comps is not None else not is_chemistry(c):
            continue
        key = (c,) + tuple(sorted((a1[i], a2[i])))
        out[key] = (order[i], arom[i] if arom is not None else "?")
    return out


def diff(ref, got, label):
    """Differences between two chemistry tables, as a list of readable rows."""
    bad = []
    for k in sorted(set(ref) | set(got), key=lambda t: tuple(map(str, t))):
        r, g = ref.get(k), got.get(k)
        if r != g:
            bad.append({"table": label, "key": list(map(str, k)), "input": r, "generated": g})
    return bad


def check(input_cif, generated_cif, comps=None, hydrogens=False):
    ref, got = read_block(input_cif), read_block(generated_cif)
    (atoms_ref, h_ref), (atoms_got, h_got) = atom_chemistry(ref, comps, hydrogens), atom_chemistry(got, comps, hydrogens)
    bonds_ref, bonds_got = bond_chemistry(ref, comps), bond_chemistry(got, comps)
    bad = diff(atoms_ref, atoms_got, "atom_site") + diff(bonds_ref, bonds_got, "chem_comp_bond")
    comps_seen = sorted({k[0] for k in atoms_ref})
    return {
        "input": str(input_cif),
        "generated": str(generated_cif),
        "components": comps_seen,
        "atoms_checked": len(atoms_ref),
        "hydrogens_dropped": {"input": h_ref, "generated": h_got},
        "bonds_checked": len(bonds_ref),
        "charged_atoms": sorted(
            [list(map(str, k)) + [v[1]] for k, v in atoms_ref.items() if v[1] not in (0, "0")],
            key=lambda r: r[0],
        ),
        "differences": bad,
        "verdict": "PASS" if not bad and atoms_ref else ("EMPTY" if not atoms_ref else "FAIL"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="the annotated CIF the model read")
    ap.add_argument("--generated", required=True, help="the design the model wrote")
    ap.add_argument("--comps", default=None,
                    help="comma-separated component ids to grade; default is every non-protein one")
    ap.add_argument("--with-hydrogens", action="store_true",
                    help="compare hydrogens too; off because atom37 is heavy-atom only")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    comps = frozenset(a.comps.split(",")) if a.comps else None
    res = check(a.input, a.generated, comps, a.with_hydrogens)
    text = json.dumps(res, indent=2)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text + "\n")
    print(text)
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
