"""Build tt_bio/data/ccd_chem_comp.json.gz from Mol-1's components.cif (weights row ``mol1-components``).

    python scripts/mol1_io/chem_comp_table.py <components.cif> tt_bio/data/ccd_chem_comp.json.gz

Keeps the five `_chem_comp` fields upstream's CIFWriter copies into a prediction, read with biotite's as_item() as
upstream reads them. The shipped table is from components.cif sha256 64f53e4d72c8f919d9d2afd34183df7c0b6211929dfe0eeb865fad82cd1374d5.
"""
import gzip
import json
import sys

import biotite.structure.io.pdbx as pdbx

FIELDS = ("type", "name", "pdbx_synonyms", "formula", "formula_weight")

if __name__ == "__main__":
    cif = pdbx.CIFFile.read(sys.argv[1])
    table = {code: [cif[code]["chem_comp"][f].as_item() for f in FIELDS] for code in cif.keys()}
    with gzip.open(sys.argv[2], "wt") as fh:
        json.dump(table, fh, separators=(",", ":"), sort_keys=True)
    print(len(table), "components")
