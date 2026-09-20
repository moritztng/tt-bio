"""The four arms, the rule comparison and the seed floor, from the captured folds.

Four arms out of two fold campaigns, because a selection-rule change does not move the
diffusion samples:

    shipped   trunk scale_pair_bias=False, samples ranked by the shipped rule
    D1        trunk scale_pair_bias=True,  samples ranked by the shipped rule
    D10       the SHIPPED samples, re-ranked by the candidate rule
    D1+D10    the D1 samples, re-ranked by the candidate rule

Every arm is scored on rank 0 (what a user is served) and best-of-5 (what the sampler can
reach), and the seed floor is measured on this target in this session -- same arm, different
seed, rank-0 structure against rank-0 structure, every seed pair. An Angstrom delta without
that floor beside it cannot be told from re-running with another seed.

`--signals` answers deliverable 1 directly: the Spearman rank correlation of every head
output against the true RMSD, pooled over all samples of all seeds, which says whether the
head's ORDERING is wrong or whether the rule reads the wrong signal out of a head that orders
fine.

Structure files are written by rank under the SHIPPED rule, so a re-ranked arm's rank-0
structure is found by mapping sample index -> file through that order.

    python3 perf/of3t_confhead/analyze.py --root ~/of3t_confhead_out/fold
"""
import argparse
import glob
import itertools
import json
import os
import statistics

from rules import RULES        # same directory; the candidate set, fixed before the numbers

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.path.expanduser("~/of3t_confhead_out/fold"))
ap.add_argument("--rule", default="plddt", help="the candidate rule the D10 arms use")
ap.add_argument("--out", default="perf/of3t_confhead/analyze.json")
a = ap.parse_args()


def load(arm):
    runs = {}
    for p in sorted(glob.glob(os.path.join(a.root, f"{arm}_s*", "samples.json"))):
        d = json.load(open(p))
        runs[d["seed"]] = {"dir": os.path.dirname(p), "s": d["per_sample"],
                           "aiclk": d["aiclk_during"], "wall": d["wall_s"],
                           "scale": d["trunk_scale_pair_bias"]}
    return runs


def shipped_order(samples):
    """Sample indices in the order the shipped rule ranks them; index 0 is `<stem>.cif`."""
    return sorted(range(len(samples)), key=lambda i: -samples[i]["rank_score"])


def pick(samples, rule):
    return max(range(len(samples)), key=lambda i: RULES[rule](samples[i]))


def cif_for(run, k):
    """The written structure file holding sample k."""
    r = shipped_order(run["s"]).index(k)
    stem = "ubq.cif" if r == 0 else f"ubq_model_{r}.cif"
    return os.path.join(run["dir"], "openfold3_results_ubq", "structures", stem)


def ca(path):
    import gemmi
    st = gemmi.read_structure(path)
    st.remove_alternative_conformations()
    return [r.find_atom("CA", "*").pos for c in st[0] for r in c
            if r.find_atom("CA", "*") is not None]


def pair_rmsd(p, q):
    import gemmi
    x, y = ca(p), ca(q)
    n = min(len(x), len(y))
    return gemmi.superpose_positions(x[:n], y[:n]).rmsd


def spearman(xs, ys):
    n = len(xs)
    rx = {i: r for r, i in enumerate(sorted(range(n), key=lambda i: xs[i]))}
    ry = {i: r for r, i in enumerate(sorted(range(n), key=lambda i: ys[i]))}
    d = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - 6 * d / (n * (n * n - 1))


ship, fix = load("ship"), load("fix")
seeds = sorted(set(ship) & set(fix))
rep = {"root": a.root, "rule": a.rule, "seeds_complete": seeds,
       "seeds_ship_only": sorted(set(ship) - set(fix)),
       "seeds_fix_only": sorted(set(fix) - set(ship)), "arms": {}}
print(f"seeds with BOTH arms: {seeds}   ship-only {rep['seeds_ship_only']}   "
      f"fix-only {rep['seeds_fix_only']}")

# clocks and the flags, asserted rather than assumed
for name, runs in (("ship", ship), ("fix", fix)):
    want = (name == "fix")
    bad = [s for s, r in runs.items() if r["scale"] is not want]
    assert not bad, f"{name} arm has wrong trunk flag on seeds {bad}"
clks = sorted({c for r in list(ship.values()) + list(fix.values())
               for c in ([r["aiclk"]["median"]] if r["aiclk"]["median"] else [])})
rep["aiclk_medians_recorded"] = clks
print(f"trunk flags correct in both arms; per-fold AICLK medians recorded: {clks}")

ARMS = {"shipped": (ship, "shipped"), "D1": (fix, "shipped"),
        "D10": (ship, a.rule), "D1+D10": (fix, a.rule)}
print(f"\n{'arm':9s} {'rank 0':>18} {'best of 5':>12} {'picks best':>11} {'worst seed':>11}")
for arm, (runs, rule) in ARMS.items():
    r0, bst, hits = [], [], 0
    for s in seeds:
        sm = runs[s]["s"]
        k = pick(sm, rule)
        r0.append(sm[k]["rmsd_ca"])
        b = min(x["rmsd_ca"] for x in sm)
        bst.append(b)
        hits += abs(sm[k]["rmsd_ca"] - b) < 1e-9
    rep["arms"][arm] = {"n": len(seeds), "rank0_mean": statistics.fmean(r0),
                        "rank0_median": statistics.median(r0), "rank0_max": max(r0),
                        "best_mean": statistics.fmean(bst), "picks_best": hits,
                        "rank0": r0}
    v = rep["arms"][arm]
    print(f"{arm:9s} {v['rank0_mean']:8.3f} A mean  {v['best_mean']:8.3f} A  "
          f"{hits:>6}/{len(seeds)}  {v['rank0_max']:8.3f} A")

# seed floor, measured here, on the rank-0 structure each arm actually serves
print("\nseed floor, this target, this session: rank 0 against rank 0, every seed pair")
rep["seed_floor"] = {}
for arm, (runs, rule) in ARMS.items():
    paths = {s: cif_for(runs[s], pick(runs[s]["s"], rule)) for s in seeds}
    if not all(os.path.exists(p) for p in paths.values()):
        print(f"  {arm:9s} structures missing, skipped")
        continue
    vals = [pair_rmsd(paths[x], paths[y]) for x, y in itertools.combinations(seeds, 2)]
    rep["seed_floor"][arm] = {"n_pairs": len(vals), "mean": statistics.fmean(vals),
                              "median": statistics.median(vals),
                              "min": min(vals), "max": max(vals)}
    f = rep["seed_floor"][arm]
    print(f"  {arm:9s} n={f['n_pairs']:3d}  mean {f['mean']:6.3f} A  median {f['median']:6.3f}"
          f"  range {f['min']:.3f}-{f['max']:.3f} A")

# which signal orders the samples? pooled over every sample of every seed
print("\nhead signal against the truth, Spearman over all samples (positive = ranks better "
      "structures higher)")
rep["signals"] = {}
SIGNS = {"plddt": 1, "plddt_ca": 1, "plddt_min": 1, "ptm": 1, "rank_score": 1,
         "pae_mean": -1, "pae_offdiag_mean": -1, "pde_mean": -1, "gpde": -1,
         "resolved_mean": 1}
for name, runs in (("shipped", ship), ("D1", fix)):
    rep["signals"][name] = {}
    row = []
    for sig, sign in SIGNS.items():
        rhos = [spearman([-sign * x[sig] for x in runs[s]["s"]],
                         [x["rmsd_ca"] for x in runs[s]["s"]]) for s in seeds]
        m = statistics.fmean(rhos)
        se = statistics.stdev(rhos) / len(rhos) ** 0.5 if len(rhos) > 1 else float("nan")
        rep["signals"][name][sig] = {"mean": m, "se": se, "per_seed": rhos}
        row.append((m, se, sig))
    print(f"  {name}:")
    for m, se, sig in sorted(row, reverse=True):
        print(f"    {sig:18s} {m:+.3f} +- {se:.3f}")

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(rep, open(a.out, "w"), indent=1)
print("\nwrote", a.out)
