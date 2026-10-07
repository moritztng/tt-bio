#!/usr/bin/env python3
"""Fetch each reference mmCIF and write a two-chain tt-bio YAML (chain A = target, B = binder).

    python build_inputs.py set.tsv <data dir>

Sequences are the full deposited entity sequences (_entity_poly.pdbx_seq_one_letter_code_can),
so unmodelled tails are folded too, as a user would submit them. Writes <data>/ref/<pdb>.cif and
<data>/inputs/<pdb>.yaml.
"""
import csv, sys, urllib.request
from pathlib import Path

import gemmi
import yaml

tsv, data = sys.argv[1], Path(sys.argv[2])
(data / "ref").mkdir(parents=True, exist_ok=True)
(data / "inputs").mkdir(parents=True, exist_ok=True)
for row in csv.DictReader(open(tsv), delimiter="\t"):
    pdb = row["pdb"]
    cif = data / "ref" / f"{pdb}.cif"
    if not cif.exists():
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb}.cif", cif)
    block = gemmi.cif.read(str(cif)).sole_block()
    ep = block.find("_entity_poly.", ["entity_id", "pdbx_seq_one_letter_code_can", "pdbx_strand_id"])
    seq = {}
    for r in ep:
        for ch in gemmi.cif.as_string(r[2]).split(","):
            seq[ch] = "".join(gemmi.cif.as_string(r[1]).split())
    out = {"version": 1, "sequences": [
        {"protein": {"id": "A", "sequence": seq[row["target"]]}},
        {"protein": {"id": "B", "sequence": seq[row["binder"]]}}]}
    (data / "inputs" / f"{pdb}.yaml").write_text(yaml.safe_dump(out, sort_keys=False))
    print(pdb, len(seq[row["target"]]), len(seq[row["binder"]]))
