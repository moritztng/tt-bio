#!/usr/bin/env python3
"""The accuracy set as one stock Protenix 2.0.0 input JSON, MSAs from the pc cache.

    python make_json.py <data dir> <msa dir on the box> > acc.json

One item per complex, named by PDB id, chains in the order target (A), binder (B).
"""
import json, sys
from pathlib import Path

import yaml

data, msa = Path(sys.argv[1]), sys.argv[2]
items = []
for yml in sorted((data / "inputs").glob("*.yaml")):
    chains = [s["protein"] for s in yaml.safe_load(yml.read_text())["sequences"]]
    items.append({"name": yml.stem, "sequences": [
        {"proteinChain": {"sequence": c["sequence"], "count": 1,
                          "pairedMsaPath": f"{msa}/{yml.stem}/{c['id']}.paired.a3m",
                          "unpairedMsaPath": f"{msa}/{yml.stem}/{c['id']}.unpaired.a3m"}} for c in chains]})
json.dump(items, sys.stdout, indent=1)
