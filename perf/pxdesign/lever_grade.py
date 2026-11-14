"""Grade a PXDesign lever arm against a base arm, paired by seed (both from perf/pxdesign/lever_ab.py).

    python perf/pxdesign/lever_grade.py BASE_DIR ARM_DIR [--floor OTHER_BASE_DIR]

A design has no ground truth, so the grade is the charter's normal-mode one transposed: the same-seed binder
deviation (all binder atoms, both already in the input target's frame, no further fit) against the 0.60 A kill
bar, beside the seed floor (the median deviation between two different seeds of the base arm), plus paired
per-seed deltas of what makes a backbone usable: the target-fit residual, CA-CA bond error, CA clashes, radius of
gyration and CA contacts with the target. `--floor` adds the A/A same-seed deviation between two base runs.

A diffusion trajectory re-rolls on any numeric change, so a same-seed deviation near the seed floor says the arm
drew a different design, not a worse one. `pooled` therefore compares each quality metric's distribution across
all seeds unpaired (mean difference, Welch 95 % interval); `pass_pooled` holds when every interval covers 0 or
leans the good way (lower fit residual, bond error and clashes; more contacts).
"""
import argparse, json, statistics
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("base", type=Path)
ap.add_argument("arm", type=Path)
ap.add_argument("--floor", type=Path)
a = ap.parse_args()


def load(d):
    return torch.load(d / "designs.pt"), json.loads((d / "arm.json").read_text())


def rmsd(x, y):
    return float((x - y).pow(2).sum(-1).mean().sqrt())


def quality(d, fit):
    ca = d["binder"][d["binder_ca"]]
    bond = (ca[1:] - ca[:-1]).norm(dim=-1)
    dm = torch.cdist(ca, ca)
    n = len(ca)
    far = torch.triu(torch.ones(n, n, dtype=torch.bool), 3)
    contacts = (torch.cdist(ca, d["target"]) < 8.0).any(-1).sum()
    return {"fit_rmsd": fit, "ca_bond_err": float((bond - 3.8).abs().mean()),
            "ca_clash": int((dm[far] < 3.0).sum()), "rg": float((ca - ca.mean(0)).norm(dim=-1).pow(2).mean().sqrt()),
            "contact_res": int(contacts)}


B, bj = load(a.base)
A, aj = load(a.arm)
fitB = {d["seed"]: d["fit_rmsd"] for d in bj["designs"] if not d["cold"]}
fitA = {d["seed"]: d["fit_rmsd"] for d in aj["designs"] if not d["cold"]}
seeds = sorted(set(B) & set(A))
same = [rmsd(B[s]["binder"], A[s]["binder"]) for s in seeds]
floor = [rmsd(B[s]["binder"], B[t]["binder"]) for s, t in zip(seeds, seeds[1:])]
out = {"base": str(a.base), "arm": str(a.arm), "levers": aj["levers"], "bf16": aj["bf16"], "seeds": seeds,
       "same_seed_binder_rmsd": [round(x, 3) for x in same], "same_seed_median": round(statistics.median(same), 3),
       "same_seed_max": round(max(same), 3), "seed_floor_median": round(statistics.median(floor), 3),
       "warm_median_s": {"base": bj["warm_median_s"], "arm": aj["warm_median_s"]}}
if a.floor:
    F, _ = load(a.floor)
    aa = [rmsd(B[s]["binder"], F[s]["binder"]) for s in seeds if s in F]
    out["aa_same_seed_median"] = round(statistics.median(aa), 3)
qB = {s: quality(B[s], fitB[s]) for s in seeds}
qA = {s: quality(A[s], fitA[s]) for s in seeds}
for k in qB[seeds[0]]:
    delta = [qA[s][k] - qB[s][k] for s in seeds]
    out[f"d_{k}"] = {"mean": round(statistics.mean(delta), 4), "min": round(min(delta), 4),
                     "max": round(max(delta), 4), "base_mean": round(statistics.mean(qB[s][k] for s in seeds), 4)}
BETTER = {"fit_rmsd": -1, "ca_bond_err": -1, "ca_clash": -1, "rg": 0, "contact_res": 1}
out["pooled"], ok = {}, True
for k, sign in BETTER.items():
    xa, xb = [qA[s][k] for s in seeds], [qB[s][k] for s in seeds]
    d = statistics.mean(xa) - statistics.mean(xb)
    half = 1.96 * (statistics.variance(xa) / len(xa) + statistics.variance(xb) / len(xb)) ** 0.5
    out["pooled"][k] = [round(d - half, 4), round(d, 4), round(d + half, 4)]
    worse = (d - half > 0) if sign < 0 else (d + half < 0) if sign > 0 else False
    ok &= not worse
out["pass_pose"] = out["same_seed_median"] <= 0.60
out["pass_pooled"] = ok
print(json.dumps(out, indent=1))
