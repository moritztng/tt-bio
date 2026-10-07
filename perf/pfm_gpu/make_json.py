#!/usr/bin/env python3
"""Stock Protenix 2.0.0 input JSON for the PFM timing arms.

    python make_json.py <out.json> <items> <name>=<yaml>:<msa dir on box>[:unpaired] ...

Each input becomes <items> identical entries <name>_r0 .. <name>_r{items-1}, so one process folds it
several times on one loaded model: r0 is the first fold of that shape (compile, graph capture), r1.. warm.
':unpaired' hands Protenix only <chain>.unpaired.a3m, which is how the Wormhole Galaxy run's MSA was
counted (depth 5,889 = 1,911 + 3,978 unpaired rows, no paired block).
"""
import json, sys
import yaml

out, n, specs = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
items = []
for spec in specs:
    name, rest = spec.split("=", 1)
    yml, msa, *flag = rest.split(":")
    seqs = []
    for s in yaml.safe_load(open(yml))["sequences"]:
        c = s["protein"]
        e = {"sequence": c["sequence"], "count": 1, "unpairedMsaPath": f"{msa}/{c['id']}.unpaired.a3m"}
        if "unpaired" not in flag:
            e["pairedMsaPath"] = f"{msa}/{c['id']}.paired.a3m"
        seqs.append({"proteinChain": e})
    items += [{"name": f"{name}_r{i}", "sequences": seqs} for i in range(n)]
json.dump(items, open(out, "w"), indent=1)
print(out, len(items), "items")
