"""End-to-end Angstrom effect of the pair-bias split on a real OpenFold3 fold.

Four numbers, and only the four together mean anything:

  accuracy   Ca-RMSD of each arm's structure against the 1UBQ experimental structure --
             the test is "closer to the reference", not "the number moved". Reported both
             as rank 0 (what a user receives) and best-of-5 (what the sampler can reach);
  lever      arm-vs-arm Ca-RMSD at a MATCHED seed, which is what the change does;
  seed floor same arm, different seed, which is the variation we already accept. Below the
             floor a lever is noise, above it a finding, and the bare Angstrom says neither.
             ubiquitin is 76 aa and gets its OWN floor; the 512 aa cell's 1.84 A is not it;
  ranking    plDDT and confidence_score against the RMSD they are meant to rank, because a
             change that leaves the sampler alone can still move which sample is served.

    python3 perf/of3t_pairbias/fold_rmsd.py
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
    pa, pb = ([a[k] for k in keys], [b[k] for k in keys]) if keys else (
        list(a.values()), list(b.values()))
    n = min(len(pa), len(pb))
    return gemmi.superpose_positions(pa[:n], pb[:n]).rmsd, n


ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/tmp/of3t/of3t-pairbias/fold")
ap.add_argument("--samples", type=int, default=5)
a = ap.parse_args()

gt = ca(GT)
runs, conf = {}, {}
for d in sorted(os.listdir(a.root)):
    base = os.path.join(a.root, d, "openfold3_results_ubq")
    sd = os.path.join(base, "structures")
    if not os.path.isdir(sd):
        continue
    files = [os.path.join(sd, "ubq.cif")] + [
        os.path.join(sd, f"ubq_model_{i}.cif") for i in range(1, a.samples)]
    runs[d] = [ca(f) for f in files if os.path.exists(f)]
    conf[d] = json.load(open(os.path.join(base, "results.json")))[0]

arms = ("ship", "fix")
seeds = sorted({int(n.split("_s")[1]) for n in runs})
report = {"ground_truth": GT, "n_ca_gt": len(gt), "seeds": seeds, "runs": {},
          "lever": {}, "seed_floor": {}, "summary": {}}
print(f"ground truth {GT}: {len(gt)} CA   seeds {seeds}\n")
print("accuracy against 1UBQ")
for name in sorted(runs, key=lambda n: (n.split("_s")[0], int(n.split("_s")[1]))):
    vs = [rmsd(m, gt)[0] for m in runs[name]]
    c = conf[name]
    report["runs"][name] = dict(n_samples=len(vs), rank0=vs[0], best=min(vs),
                                median=statistics.median(vs), all=vs,
                                plddt=c["plddt"], confidence_score=c["confidence_score"],
                                ranked_is_best=abs(vs[0] - min(vs)) < 1e-9)
    print(f"  {name:8s} rank0 {vs[0]:6.3f} A  best {min(vs):6.3f} A  median "
          f"{statistics.median(vs):6.3f} A  plddt {c['plddt']:.4f}  "
          f"{'ranked=best' if abs(vs[0] - min(vs)) < 1e-9 else 'ranked>best'}")

for arm in arms:
    ns = [n for n in runs if n.startswith(arm + "_s")]
    r0 = [report["runs"][n]["rank0"] for n in ns]
    bst = [report["runs"][n]["best"] for n in ns]
    report["summary"][arm] = dict(
        n=len(ns), rank0_mean=statistics.fmean(r0), rank0_median=statistics.median(r0),
        rank0_max=max(r0), best_mean=statistics.fmean(bst), best_median=statistics.median(bst),
        plddt_mean=statistics.fmean(report["runs"][n]["plddt"] for n in ns),
        ranked_is_best=sum(report["runs"][n]["ranked_is_best"] for n in ns))
    s = report["summary"][arm]
    print(f"  -> {arm:5s} n={s['n']}  rank0 mean {s['rank0_mean']:.3f} median "
          f"{s['rank0_median']:.3f} worst {s['rank0_max']:.3f} | best-of-5 mean "
          f"{s['best_mean']:.3f} | plddt {s['plddt_mean']:.4f} | ranked==best "
          f"{s['ranked_is_best']}/{s['n']}")

print("\nlever: shipped vs split, same seed")
lv = []
for s in seeds:
    sh, fx = f"ship_s{s}", f"fix_s{s}"
    if sh in runs and fx in runs:
        v, n = rmsd(runs[sh][0], runs[fx][0])
        per = [rmsd(runs[sh][i], runs[fx][i])[0] for i in range(min(len(runs[sh]), len(runs[fx])))]
        report["lever"][s] = dict(rank0=v, n_ca=n, per_sample=per)
        lv.append(v)
        print(f"  seed {s}: rank0 {v:6.3f} A   per-sample " + " ".join(f"{x:.3f}" for x in per))
if lv:
    report["lever"]["rank0_mean"] = statistics.fmean(lv)
    report["lever"]["rank0_median"] = statistics.median(lv)
    print(f"  -> lever rank0 mean {statistics.fmean(lv):.3f} A  median "
          f"{statistics.median(lv):.3f} A  range {min(lv):.3f}-{max(lv):.3f} A")

print("\nseed floor: same arm, different seed (every pair)")
for arm in arms:
    ss = sorted(int(n.split("_s")[1]) for n in runs if n.startswith(arm + "_s"))
    pairs = {f"{x}v{y}": rmsd(runs[f"{arm}_s{x}"][0], runs[f"{arm}_s{y}"][0])[0]
             for x, y in itertools.combinations(ss, 2)}
    if not pairs:
        continue
    vals = list(pairs.values())
    report["seed_floor"][arm] = dict(pairs=pairs, n_pairs=len(vals), min=min(vals),
                                     max=max(vals), mean=statistics.fmean(vals),
                                     median=statistics.median(vals))
    print(f"  {arm:5s} n={len(vals)} pairs  mean {statistics.fmean(vals):6.3f} A  median "
          f"{statistics.median(vals):6.3f} A  range {min(vals):.3f}-{max(vals):.3f} A")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(report, open(OUT, "w"), indent=1)
print("\nwrote", OUT)
