"""openbind at 896 aa, narrow-q off vs on at two seeds: distance to 1HCL and to each other.

cdk2x2_896 is CDK2 (1HCL, 298 aa) tiled three times plus two residues, so each full copy scores
against the deposited structure. The scorer asserts residue identity per matched position, which
is what validates the tiling. Pairwise arm-to-arm RMSD is float64 Kabsch over all 896 CA.
"""
import json
import sys
from itertools import combinations
from pathlib import Path

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT / "perf" / "fused_sdpa"))
sys.path.insert(0, str(WT / "perf" / "other512"))
from of3_score_ref import ca_map, gt_rmsd, kabsch_rmsd  # noqa: E402
import numpy as np  # noqa: E402

OUT = WT / "perf/land_standing/out/narrowq_openbind_acc"
GT = ca_map(WT / "perf/fused_sdpa/cifs/1hcl.cif")
COPIES = {f"copy{k + 1}": [(298 * k + j, j) for j in range(1, 299)] for k in range(3)}


def plddt(cif):
    js = sorted(cif.parent.glob("confidence_*.json"))
    return json.loads(js[0].read_text()).get("complex_plddt") if js else None


legs = {}
for d in sorted(OUT.glob("s*_o*")):
    cif = next(iter(sorted(d.rglob("*.cif"))), None)
    if cif:
        legs[d.name] = cif
rep = {"gt_ca_rmsd_A": {}, "pairwise_ca_rmsd_A": {}, "plddt": {}}
for name, cif in legs.items():
    rep["gt_ca_rmsd_A"][name] = {c: round(gt_rmsd(cif, GT, p)[0], 6) for c, p in COPIES.items()}
    rep["gt_ca_rmsd_A"][name]["mean"] = round(float(np.mean(list(rep["gt_ca_rmsd_A"][name].values()))), 6)
    rep["plddt"][name] = plddt(cif)
for a, b in combinations(sorted(legs), 2):
    A, B = ca_map(legs[a]), ca_map(legs[b])
    k = sorted(set(A) & set(B))
    rep["pairwise_ca_rmsd_A"][f"{a}|{b}"] = round(kabsch_rmsd(np.array([A[i][1] for i in k]),
                                                              np.array([B[i][1] for i in k])), 6)
print(json.dumps(rep, indent=1))
(OUT / "score.json").write_text(json.dumps(rep, indent=1))
