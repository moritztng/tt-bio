#!/usr/bin/env python3
"""The verdict: RF3's two triangle-attention arms scored against the 1HCL crystal, five seeds each.

The lever was previously rejected on 1.9335 A CA *from the shipped default*, which says the
structure moved, not that it got worse, and the shipped arm is the one under suspicion. cdk2x2_298
is CDK2 1:1, so it has an experimental answer that no RNG or basin argument reaches. Distance to
that answer is one-signed and it is the same fixed third point for both arms.

Decision rule, pre-registered in state/fused-sdpa-adopt_PLAN.md section 4 BEFORE this ran:

    margin = median G(hifi) - median G(def),  G = CA RMSD to 1HCL
    S(arm) = that arm's own max - min across the five seeds

    margin > +0.25 A and margin > max(S(def), S(hifi))  ->  FLOOR, the fused arm is the worse fold
    margin < -0.25 A and |margin| > max(S(def), S(hifi))  ->  the incumbent was wrong, adopt
    otherwise                                           ->  FLOOR on the narrow ground that the
                                                             fixture cannot separate the arms

Host only, no device. Reuses perf/of3_ref/score.py's ca_map/gt_rmsd and perf/other512/cif_rmsd.py's
Kabsch unmodified, so the numbers are comparable across the whole lineage.
"""
import json, statistics, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "perf" / "other512"))
sys.path.insert(0, str(HERE))
from cif_rmsd import kabsch_rmsd, read_atoms          # noqa: E402
from of3_score_ref import ca_map, gt_rmsd             # noqa: E402

SEEDS_DIR = HERE / "seeds"
ARMS = ["def", "hifi"]
THRESHOLD_A = 0.25

gt = ca_map(HERE / "cifs" / "1hcl.cif")
print(f"1HCL resolved CA: {len(gt)}  seq_id range {min(gt)}..{max(gt)}")


def arm_folds(arm):
    """(index, seed, cif) for every kept warm fold of an arm, in run order."""
    out = []
    for d in sorted((SEEDS_DIR / arm).glob("f*_seed*")):
        i, sd = d.name[1:].split("_seed")
        cif = d / "cdk2x2_298.cif"
        if cif.exists():
            out.append((int(i), int(sd), cif))
    return sorted(out)


anchor = None
res = {"gt": "1hcl.cif", "n_gt_ca": len(gt), "threshold_A": THRESHOLD_A, "arms": {}}
for arm in ARMS:
    folds = arm_folds(arm)
    if not folds:
        print(f"{arm}: no folds yet")
        continue
    pred0 = ca_map(folds[0][2])
    shared = sorted(set(pred0) & set(gt))
    ident = [i for i in shared if pred0[i][0] != gt[i][0]]
    assert not ident, f"{arm}: residue identity mismatch at {ident[:5]} -- the parse is wrong"
    pairs = [(i, i) for i in shared]
    if anchor is None:
        anchor = (shared, pred0)
        print(f"shared seq_ids: {len(shared)}   identity mismatches: 0")

    per = []
    for i, sd, cif in folds:
        r, n = gt_rmsd(cif, gt, pairs)
        ca_p = ca_map(cif)
        A = np.array([ca_p[j][1] for j in anchor[0]])
        B = np.array([anchor[1][j][1] for j in anchor[0]])
        keys_a, xa = read_atoms(cif)
        keys_d, xd = read_atoms(folds[0][2] if arm == "def" else arm_folds("def")[0][2])
        per.append({"i": i, "seed": sd, "gt_ca_rmsd_A": round(r, 6), "n_ca": n,
                    "ca_vs_def_seed0_A": round(kabsch_rmsd(A, B), 6),
                    "allatom_vs_def_seed0_A": (round(kabsch_rmsd(xa, xd), 6)
                                               if keys_a == keys_d else None),
                    "cif_sha_dir": cif.parent.name})
        print(f"  {arm:5s} f{i} seed{sd}  GT_CA {r:9.6f} A   "
              f"CA vs def-seed0 {per[-1]['ca_vs_def_seed0_A']:9.6f}")

    # The A/A control is the repeated seed: same seed, same arm, two folds in one process.
    by_seed = {}
    for p in per:
        by_seed.setdefault(p["seed"], []).append(p["gt_ca_rmsd_A"])
    aa = {str(s): v for s, v in by_seed.items() if len(v) > 1}
    aa_spread = max((max(v) - min(v) for v in aa.values()), default=None)

    # one reading per distinct seed for the median, so a repeated seed does not get two votes
    uniq = [by_seed[s][0] for s in sorted(by_seed)]
    res["arms"][arm] = {
        "folds": per, "n_seeds": len(uniq),
        "median_gt_ca_A": round(statistics.median(uniq), 6),
        "min_gt_ca_A": round(min(uniq), 6), "max_gt_ca_A": round(max(uniq), 6),
        "seed_spread_A": round(max(uniq) - min(uniq), 6),
        "aa_control_seeds": sorted(aa), "aa_control_spread_A": aa_spread,
    }
    a = res["arms"][arm]
    print(f"  -> {arm}: median {a['median_gt_ca_A']:.6f} A over {a['n_seeds']} seeds, "
          f"range {a['min_gt_ca_A']:.6f}..{a['max_gt_ca_A']:.6f}, spread {a['seed_spread_A']:.6f}, "
          f"A/A spread {a['aa_control_spread_A']}")

if len(res["arms"]) == 2:
    d, h = res["arms"]["def"], res["arms"]["hifi"]
    margin = h["median_gt_ca_A"] - d["median_gt_ca_A"]
    spread = max(d["seed_spread_A"], h["seed_spread_A"])
    if margin > THRESHOLD_A and margin > spread:
        verdict, why = "FLOOR", "fused arm is the worse fold, margin clears threshold and spread"
    elif margin < -THRESHOLD_A and -margin > spread:
        verdict, why = "ADOPT", "fused arm is the better fold, margin clears threshold and spread"
    else:
        verdict, why = "FLOOR_NARROW", ("fixture cannot separate the arms: margin is inside the "
                                        "threshold or inside the seed spread")
    res["verdict"] = {"margin_A": round(margin, 6), "max_seed_spread_A": round(spread, 6),
                      "verdict": verdict, "why": why}
    print(f"\nmargin (hifi - def) = {margin:+.6f} A   max seed spread {spread:.6f} A"
          f"   threshold {THRESHOLD_A} A\nVERDICT: {verdict} -- {why}")

(HERE / "rf3_seeds_298.json").write_text(json.dumps(res, indent=1) + "\n")
print("wrote", HERE / "rf3_seeds_298.json")
