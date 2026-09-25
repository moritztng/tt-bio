"""openbind on PepN (3B34), narrow-q off vs on at three seeds: CA RMSD to the deposited structure,
arm-to-arm and seed-to-seed. Positions pair by label_seq_id (the prediction numbers the 891-residue
entity 1..891, as 3B34 does); unresolved residues drop out, residue identity is asserted per pair.
Float64 Kabsch.
"""
import gzip
import json
import sys
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT / "perf" / "fused_sdpa"))
sys.path.insert(0, str(WT / "perf" / "other512"))
from of3_score_ref import ca_map, gt_rmsd, kabsch_rmsd  # noqa: E402

OUT = WT / "perf/land_standing/out/narrowq_openbind_pepn"
with tempfile.NamedTemporaryFile("w", suffix=".cif", delete=False) as fp:
    fp.write(gzip.open(WT / "perf/land_standing/fixtures/3b34.cif.gz", "rt").read())
GT = ca_map(Path(fp.name))
PAIRS = [(i, i) for i in range(1, 892)]

legs = {d.name: next(iter(sorted(d.rglob("*.cif")))) for d in sorted(OUT.glob("s*_o*"))
        if any(d.rglob("*.cif"))}
rep = {"gt_ca_rmsd_A": {}, "n_matched": None, "plddt": {}, "pairwise_ca_rmsd_A": {}}
for name, cif in legs.items():
    r, n = gt_rmsd(cif, GT, PAIRS)
    rep["gt_ca_rmsd_A"][name] = round(r, 6); rep["n_matched"] = n
    conf = sorted(cif.parent.glob("confidence_*.json"))
    rep["plddt"][name] = json.loads(conf[0].read_text()).get("complex_plddt") if conf else None
for a, b in combinations(sorted(legs), 2):
    A, B = ca_map(legs[a]), ca_map(legs[b])
    k = sorted(set(A) & set(B))
    rep["pairwise_ca_rmsd_A"][f"{a}|{b}"] = round(kabsch_rmsd(np.array([A[i][1] for i in k]),
                                                              np.array([B[i][1] for i in k])), 6)
print(json.dumps(rep, indent=1))
(OUT / "score.json").write_text(json.dumps(rep, indent=1))
