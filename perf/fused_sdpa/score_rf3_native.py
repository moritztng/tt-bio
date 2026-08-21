#!/usr/bin/env python3
"""RF3 298 aa arms scored against the experimental structure (1HCL), which no RNG
argument reaches, plus arm-vs-arm to prove the instrument reproduces the prior pass.

Host only, no device. Reuses perf/of3_ref/score.py`s ca_map/gt_rmsd and
perf/other512/cif_rmsd.py`s Kabsch, so the numbers are comparable across the lineage.
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "perf" / "other512"))
sys.path.insert(0, str(HERE))
from cif_rmsd import kabsch_rmsd, read_atoms
from of3_score_ref import ca_map, gt_rmsd

CIFS = HERE / "cifs"
ARMS = ["def", "hifi", "gln", "newdefault", "lev"]
gt = ca_map(CIFS / "1hcl.cif")
print(f"1HCL resolved CA: {len(gt)}  seq_id range {min(gt)}..{max(gt)}")

pred0 = ca_map(CIFS / "rf3_298_def.cif")
print(f"RF3 pred CA: {len(pred0)}  seq_id range {min(pred0)}..{max(pred0)}")
shared = sorted(set(pred0) & set(gt))
ident = [(i, pred0[i][0], gt[i][0]) for i in shared if pred0[i][0] != gt[i][0]]
print(f"shared seq_ids: {len(shared)}   identity mismatches: {len(ident)}  {ident[:5]}")

pairs = [(i, i) for i in shared]
out = {"gt": "1hcl.cif", "n_gt_ca": len(gt), "n_shared_ca": len(shared),
       "identity_mismatches": len(ident), "arms": {}}
for a in ARMS:
    p = CIFS / f"rf3_298_{a}.cif"
    if not p.exists():
        continue
    r, n = gt_rmsd(p, gt, pairs)
    ka, xa = read_atoms(p)
    kd, xd = read_atoms(CIFS / "rf3_298_def.cif")
    ca_p = ca_map(p)
    A = np.array([ca_p[i][1] for i in shared]); B = np.array([pred0[i][1] for i in shared])
    out["arms"][a] = {"gt_ca_rmsd_A": round(r, 6), "n_ca": n,
                      "allatom_vs_def_A": round(kabsch_rmsd(xa, xd), 6) if ka == kd else None,
                      "ca_vs_def_A": round(kabsch_rmsd(A, B), 6),
                      "atom_keys_match_def": ka == kd, "n_atoms": len(ka)}
    print(f"  {a:12s} GT_CA {r:9.6f} A ({n} CA)   vs_def all-atom "
          f"{out['arms'][a]['allatom_vs_def_A']} "
          f"CA {out['arms'][a]['ca_vs_def_A']}")
(HERE / "rf3_native_298.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out["arms"], indent=1))
