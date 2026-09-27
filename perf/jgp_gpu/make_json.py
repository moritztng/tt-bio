#!/usr/bin/env python3
"""complex730 as a stock Protenix 2.0.0 input: two proteinChain entries with our cached MSA.

    python make_json.py <complex730.yaml> <msa dir on the box> <n items> > complex730.json

N copies of the same complex under names r0..r{N-1}, so one process folds it N times on one loaded model:
r0 carries the CUDA/cuBLAS/cuEquivariance first-call cost, r1.. are the warm reps.
"""
import json, sys
import yaml

yml, msa, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
chains = [s["protein"] for s in yaml.safe_load(open(yml))["sequences"]]
seqs = [{"proteinChain": {"sequence": c["sequence"], "count": 1,
                          "pairedMsaPath": f"{msa}/{c['id']}.paired.a3m",
                          "unpairedMsaPath": f"{msa}/{c['id']}.unpaired.a3m"}} for c in chains]
json.dump([{"name": f"r{i}", "sequences": seqs} for i in range(n)], sys.stdout, indent=1)
