"""End-to-end Angstrom effect of the pair-bias split on a real OpenFold3 fold.

Three numbers, and only the three together mean anything:

  accuracy   Ca-RMSD of each arm's ranked-best structure against the 1UBQ experimental
             structure -- the test is "closer to the reference", not "the number moved";
  lever      arm-vs-arm Ca-RMSD at a MATCHED seed, which is what the change does;
  seed floor same arm, different seed, which is the variation we already accept.

A lever below its own target's seed floor is inside noise; above it, it is a finding. The
512 aa cell's 1.84 A floor is not inherited here -- ubiquitin is 76 aa and gets its own.

    python3 perf/of3t_pairbias/fold_rmsd.py --root /tmp/of3t/of3t-pairbias/fold
"""
import argparse
import itertools
import json
import os
import statistics

import gemmi

OUT = "perf/of3t_pairbias/fold_rmsd.json"
GT = "examples/ground_truth_structures/ubiquitin.pdb"


def ca(path):
    st = gemmi.read_structure(path)
    st.remove_alternative_conformations()
    return {(c.name, r.seqid.num): r.find_atom("CA", "*").pos
            for c in st[0] for r in c if r.find_atom("CA", "*") is not None}


def rmsd(a, b):
    keys = sorted(set(a) & set(b))
    if not keys:
        av, bv = list(a.values()), list(b.values())
        n = min(len(av), len(bv))
        keys = None
        pa, pb = av[:n], bv[:n]
    else:
        pa, pb = [a[k] for k in keys], [b[k] for k in keys]
    return gemmi.superpose_positions(pa, pb).rmsd, len(pa)


ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/tmp/of3t/of3t-pairbias/fold")
ap.add_argument("--samples", type=int, default=5)
a = ap.parse_args()

gt = ca(GT)
runs = {}
for d in sorted(os.listdir(a.root)):
    sd = os.path.join(a.root, d, "openfold3_results_ubq", "structures")
    if not os.path.isdir(sd):
        continue
    files = [os.path.join(sd, "ubq.cif")] + [
        os.path.join(sd, f"ubq_model_{i}.cif") for i in range(1, a.samples)]
    runs[d] = [ca(f) for f in files if os.path.exists(f)]

report = {"ground_truth": GT, "n_ca_gt": len(gt), "runs": {}, "lever": {}, "seed_floor": {}}
print(f"ground truth {GT}: {len(gt)} CA\n")
print("accuracy against 1UBQ (rank 0 is the structure a user receives)")
for name, models in sorted(runs.items()):
    vs = [rmsd(m, gt)[0] for m in models]
    report["runs"][name] = dict(n_samples=len(vs), rank0=vs[0], best=min(vs),
                                median=statistics.median(vs), all=vs)
    print(f"  {name:10s} rank0 {vs[0]:6.3f} A   best {min(vs):6.3f} A   "
          f"median {statistics.median(vs):6.3f} A   n={len(vs)}")

print("\nlever: shipped vs split, same seed, rank 0")
for seed in sorted({n.split("_s")[1] for n in runs}):
    s, f = f"ship_s{seed}", f"fix_s{seed}"
    if s in runs and f in runs:
        v, n = rmsd(runs[s][0], runs[f][0])
        allv = [rmsd(runs[s][i], runs[f][i])[0] for i in range(min(len(runs[s]), len(runs[f])))]
        report["lever"][seed] = dict(rank0=v, n_ca=n, per_sample=allv)
        print(f"  seed {seed}: rank0 {v:6.3f} A   per-sample {[f'{x:.3f}' for x in allv]}")

print("\nseed floor: same arm, different seed, rank 0")
for arm in ("ship", "fix"):
    seeds = sorted(n.split("_s")[1] for n in runs if n.startswith(arm + "_s"))
    pairs = {f"{x}v{y}": rmsd(runs[f"{arm}_s{x}"][0], runs[f"{arm}_s{y}"][0])[0]
             for x, y in itertools.combinations(seeds, 2)}
    if pairs:
        report["seed_floor"][arm] = dict(pairs=pairs, min=min(pairs.values()),
                                         max=max(pairs.values()),
                                         mean=statistics.fmean(pairs.values()))
        print(f"  {arm:5s} " + "  ".join(f"{k} {v:6.3f} A" for k, v in pairs.items())
              + f"   mean {statistics.fmean(pairs.values()):6.3f} A")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(report, open(OUT, "w"), indent=1)
print("\nwrote", OUT)
