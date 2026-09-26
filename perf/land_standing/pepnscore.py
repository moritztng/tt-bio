#!/usr/bin/env python3
"""CA-RMSD of each arm against the DEPOSITED 3B34, using the instrument that already exists.

`of3_score_ref.ca_map` pairs by label_seq_id -- the prediction numbers the 891-residue entity
1..891 as 3B34 does -- drops unresolved residues and asserts residue identity per pair. A first
version of this file re-implemented that with gemmi's default auth numbering and its own identity
assertion caught the mismatch at residue 5. Reusing the validated one is both correct and less
code.

Output path is an ARGUMENT, unlike the scorer this borrows from, which hardcodes its own and so
cannot be run twice without destroying the first result.
"""
import gzip
import json
import sys
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np

WT = Path("/home/ttuser/.coworker/wt/land-standing")
sys.path.insert(0, str(WT / "perf" / "fused_sdpa"))
sys.path.insert(0, str(WT / "perf" / "other512"))
from of3_score_ref import ca_map, gt_rmsd, kabsch_rmsd  # noqa: E402

with tempfile.NamedTemporaryFile("w", suffix=".cif", delete=False) as fp:
    fp.write(gzip.open(WT / "perf/land_standing/fixtures/3b34.cif.gz", "rt").read())
GT = ca_map(Path(fp.name))
PAIRS = [(i, i) for i in range(1, 892)]

out_path = sys.argv[1]
legs = {}
for spec in sys.argv[2:]:
    name, path = spec.split("=", 1)
    legs[name] = Path(path)

rep = {"deposited": "3B34", "gt_ca_rmsd_A": {}, "n_matched": None, "pairwise_ca_rmsd_A": {}}
for name, cif in sorted(legs.items()):
    r, n = gt_rmsd(cif, GT, PAIRS)
    rep["gt_ca_rmsd_A"][name] = round(r, 6)
    rep["n_matched"] = n
for a, b in combinations(sorted(legs), 2):
    A, B = ca_map(legs[a]), ca_map(legs[b])
    k = sorted(set(A) & set(B))
    rep["pairwise_ca_rmsd_A"]["%s|%s" % (a, b)] = round(
        kabsch_rmsd(np.array([A[i][1] for i in k]), np.array([B[i][1] for i in k])), 6)

print(json.dumps(rep, indent=1))
Path(out_path).write_text(json.dumps(rep, indent=1))
print("wrote %s" % out_path)
