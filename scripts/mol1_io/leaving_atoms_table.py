"""Build tt_bio/data/ccd_leaving_atoms.json.gz from Mol-1's components.cif (weights row ``mol1-components``).

    python scripts/mol1_io/leaving_atoms_table.py <components.cif> tt_bio/data/ccd_leaving_atoms.json.gz

CCD code -> the heavy atoms `_chem_comp_atom.pdbx_leaving_atom_flag` marks, for the codes that have any, read with
biotite as upstream's ccd.get_component_atom_array reads them. A code the `mols` library lacks is built from the
model's own CCD table (protenix_data.CcdTable), which carries no leaving flags; this supplies them. The shipped table
is from components.cif sha256 64f53e4d72c8f919d9d2afd34183df7c0b6211929dfe0eeb865fad82cd1374d5.
"""
import gzip
import json
import sys

import biotite.structure.io.pdbx as pdbx

if __name__ == "__main__":
    cif = pdbx.CIFFile.read(sys.argv[1])
    table = {}
    for code in cif.keys():
        atoms = cif[code].get("chem_comp_atom")
        if atoms is None:
            continue
        flag, name, element = (atoms[f].as_array() for f in ("pdbx_leaving_atom_flag", "atom_id", "type_symbol"))
        leaving = [str(n) for n, f, e in zip(name, flag, element) if f == "Y" and e not in ("H", "D")]
        if leaving:
            table[code] = leaving
    with gzip.open(sys.argv[2], "wt") as fh:
        json.dump(table, fh, separators=(",", ":"), sort_keys=True)
    print(len(table), "components")
