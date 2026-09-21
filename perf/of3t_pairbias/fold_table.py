"""The accuracy table: is the repaired pair bias reliably worse, or is it inside seed scatter?

Reads the two arms' fold records and reports, per target and then pooled:

  accuracy   Ca-RMSD against the DEPOSITED structure, rank 0 (what a user receives) and
             best-of-5 (what the sampler reaches). The question is "closer to the reference",
             not "the number moved";
  lever      arm against arm at a MATCHED seed, which is what the change does to a structure;
  seed floor same arm, different seed, every pair -- the variation we already accept and ship.
             Each target gets its OWN floor; a floor borrowed from another target is not one;
  A/A        same arm, same seed, twice. Anything but 0.000 means the comparison is noisy at
             the bottom and every number above it is suspect.

The decision rule the brief sets: the repair is refused only if it is RELIABLY worse ACROSS
targets and seeds. So the pooled line reports the paired per-seed accuracy difference, the sign
count, and a two-sided exact sign-test p, rather than one target's mean.

    python3 perf/of3t_pairbias/fold_table.py
"""
import argparse
import itertools
import json
import math
import os
import statistics

import torch


def kabsch(a, b):
    a, b = torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)
    a, b = a - a.mean(0), b - b.mean(0)
    u, s, vt = torch.linalg.svd(a.T @ b)
    d = torch.sign(torch.linalg.det(u @ vt))
    r = u @ torch.diag(torch.tensor([1.0, 1.0, d], dtype=torch.float64)) @ vt
    return float(torch.sqrt(((a @ r - b) ** 2).sum(1).mean()))


def sign_test_p(diffs):
    """Two-sided exact sign test on the paired differences (ties dropped)."""
    nz = [d for d in diffs if d != 0.0]
    n, k = len(nz), sum(1 for d in nz if d > 0)
    if n == 0:
        return 1.0, 0, 0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail), n, sum(1 for d in nz if d > 0)


ap = argparse.ArgumentParser()
ap.add_argument("--ship", default="perf/of3t_pairbias/folds_ship_c3.json,"
                                  "perf/of3t_pairbias/folds_ship_c2.json")
ap.add_argument("--fix", default="perf/of3t_pairbias/folds_fix_c3.json,"
                                 "perf/of3t_pairbias/folds_fix_c2.json")
ap.add_argument("--out", default="perf/of3t_pairbias/fold_table.json")
a = ap.parse_args()

def load(paths):
    """One arm can be split over cards, one card per target. Merge on target and keep which card
    each target came from: a target whose two arms landed on different cards is a comparison this
    row would have to throw away."""
    parts = [json.load(open(q)) for q in paths.split(",")]
    out = dict(card={}, targets={}, rows=[], wiring=parts[0]["wiring"])
    for part in parts:
        out["rows"] += part["rows"]
        for t, meta in part["targets"].items():
            if any(r["target"] == t for r in part["rows"]):
                out["targets"][t] = meta
                out["card"][t] = part["card"]
        assert part["wiring"] == out["wiring"], "files of one arm disagree on their wiring"
    return out


data = {arm: load(p) for arm, p in (("ship", a.ship), ("fix", a.fix))}
rows = {arm: {(r["target"], r["seed"], r["repeat"]): r for r in d["rows"]}
        for arm, d in data.items()}
targets = [t for t in data["ship"]["targets"]
           if any(k[0] == t for k in rows["ship"]) and any(k[0] == t for k in rows["fix"])]
for _t, _c in data["ship"]["card"].items():
    assert data["fix"]["card"].get(_t) == _c, f"{_t}: the two arms landed on different cards"
    print(f"card {_c}: {_t}, both arms")
print(f"wiring: ship {data['ship']['wiring']}\n        fix  {data['fix']['wiring']}")

report = {"targets": {}, "pooled": {}}
paired = []
for t in targets:
    meta = data["ship"]["targets"][t]
    seeds = sorted({k[1] for k in rows["ship"] if k[0] == t and k[2] == 0}
                   & {k[1] for k in rows["fix"] if k[0] == t and k[2] == 0})
    ent = {"pdb_id": meta["pdb_id"], "chain": meta["chain"], "n_res": meta["n_res"],
           "msa_rows": meta["msa_rows"], "blurb": meta["blurb"], "seeds": seeds, "arms": {}}
    print(f"\n=== {t} ({meta['pdb_id']}_{meta['chain']}, {meta['n_res']} res, MSA "
          f"{meta['msa_rows']} rows, {meta['blurb']}) -- {len(seeds)} seeds")
    for arm in ("ship", "fix"):
        r0 = [rows[arm][(t, s, 0)]["rank0"] for s in seeds]
        bt = [rows[arm][(t, s, 0)]["best"] for s in seeds]
        ent["arms"][arm] = dict(
            n=len(seeds), rank0=r0, rank0_mean=statistics.fmean(r0),
            rank0_median=statistics.median(r0), rank0_max=max(r0),
            best_mean=statistics.fmean(bt),
            plddt=statistics.fmean(rows[arm][(t, s, 0)]["plddt"] for s in seeds),
            ptm=statistics.fmean(rows[arm][(t, s, 0)]["ptm"] for s in seeds),
            ranked_is_best=sum(rows[arm][(t, s, 0)]["best_index"] == 0 for s in seeds))
        e = ent["arms"][arm]
        print(f"  {arm:4s} rank0 mean {e['rank0_mean']:.3f} median {e['rank0_median']:.3f} worst "
              f"{e['rank0_max']:.3f} | best-of-5 {e['best_mean']:.3f} | pLDDT {e['plddt']:.4f} "
              f"pTM {e['ptm']:.4f}")
    d = [rows["fix"][(t, s, 0)]["rank0"] - rows["ship"][(t, s, 0)]["rank0"] for s in seeds]
    ent["accuracy_delta"] = dict(per_seed=d, mean=statistics.fmean(d),
                                 median=statistics.median(d),
                                 worse=sum(1 for x in d if x > 0), n=len(d))
    paired += [(t, s, x) for s, x in zip(seeds, d)]
    lev = [kabsch(rows["ship"][(t, s, 0)]["ca_xyz"][rows["ship"][(t, s, 0)]["best_index"]],
                  rows["fix"][(t, s, 0)]["ca_xyz"][rows["fix"][(t, s, 0)]["best_index"]])
           for s in seeds]
    ent["lever"] = dict(per_seed=lev, mean=statistics.fmean(lev), median=statistics.median(lev),
                        min=min(lev), max=max(lev))
    floors = {}
    for arm in ("ship", "fix"):
        pr = [kabsch(rows[arm][(t, x, 0)]["ca_xyz"][rows[arm][(t, x, 0)]["best_index"]],
                     rows[arm][(t, y, 0)]["ca_xyz"][rows[arm][(t, y, 0)]["best_index"]])
              for x, y in itertools.combinations(seeds, 2)]
        floors[arm] = dict(n_pairs=len(pr), mean=statistics.fmean(pr),
                           median=statistics.median(pr), min=min(pr), max=max(pr))
    ent["seed_floor"] = floors
    ent["lever_over_floor"] = ent["lever"]["mean"] / floors["ship"]["mean"] if floors["ship"]["mean"] else None
    print(f"  accuracy delta (fix - ship), rank 0: mean {ent['accuracy_delta']['mean']:+.3f} A  "
          f"worse on {ent['accuracy_delta']['worse']}/{len(d)} seeds")
    print(f"  lever      mean {ent['lever']['mean']:.3f} A  range {min(lev):.3f}-{max(lev):.3f}")
    print(f"  seed floor ship mean {floors['ship']['mean']:.3f} A ({floors['ship']['n_pairs']} "
          f"pairs, {floors['ship']['min']:.3f}-{floors['ship']['max']:.3f}) | fix mean "
          f"{floors['fix']['mean']:.3f} A")
    print(f"  lever / own floor: {ent['lever_over_floor']:.2f}x")
    report["targets"][t] = ent

aa = []
for arm in ("ship", "fix"):
    for (t, s, rep), r in rows[arm].items():
        if rep == 1 and (t, s, 0) in rows[arm]:
            b = rows[arm][(t, s, 0)]
            aa.append(dict(arm=arm, target=t, seed=s,
                           rmsd=kabsch(r["ca_xyz"][r["best_index"]],
                                       b["ca_xyz"][b["best_index"]])))
report["aa_floor"] = aa
for x in aa:
    print(f"\nA/A {x['arm']} {x['target']} seed {x['seed']}: {x['rmsd']:.6f} A")

d = [x for _, _, x in paired]
p, n, worse = sign_test_p(d)
report["pooled"] = dict(n=len(d), mean=statistics.fmean(d), median=statistics.median(d),
                        worse=sum(1 for x in d if x > 0), sign_test_p=p, n_nonzero=n,
                        per_seed=[dict(target=t, seed=s, delta=x) for t, s, x in paired],
                        by_target={t: report["targets"][t]["accuracy_delta"]["mean"]
                                   for t in targets})
print(f"\nPOOLED rank-0 accuracy, fix - ship, over {len(d)} matched folds on {len(targets)} "
      f"targets: mean {statistics.fmean(d):+.3f} A  median {statistics.median(d):+.3f} A  "
      f"fix worse on {sum(1 for x in d if x > 0)}/{len(d)}  two-sided sign test p = {p:.4f}")
print("per-target means (fix - ship): " + "  ".join(
    f"{t} {report['targets'][t]['accuracy_delta']['mean']:+.3f}" for t in targets))

os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(report, open(a.out, "w"), indent=1)
print("\nwrote", a.out)
