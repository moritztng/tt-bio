"""D10's premise, re-read off the data that was already stored.

`perf/of3t_pairbias/fold_rmsd.json` keeps, per arm and seed, the five per-sample Ca-RMSDs in
the ORDER THE HEAD RANKED THEM -- index 0 is what the user is served. That is enough to ask
whether the head's ORDERING got worse with D1's trunk correction, which is what D10 asserts,
without touching a card.

It did not, and the opposite claim is not supported either. A 5-sample ordering has two
one-sided marginals and they disagree here, so the script reports BOTH plus the symmetric
statistic:

  predicted rank of the truly-best sample   ship 3.56, fix 2.22  (random 3.0)
  true rank of the sample the head PICKED   ship 2.67, fix 3.33  (random 3.0)
  Spearman rho, predicted vs true order     ship 0.178, fix 0.267 (random 0.0)

Quote either marginal alone and the head looks better, or worse, with the fix. The symmetric
statistic says neither: both arms order their own samples barely above chance, and at nine
seeds the difference between 0.178 and 0.267 is inside one standard error. D10's "the head is
more confident and no better at ranking" is right about the second half and its cause is not
a calibration regression.

What DOES change, and it is large, is the sample set the same weak ordering is applied to:
the within-seed spread triples, 0.284 -> 0.971 A, because the corrected trunk's samples go
bimodal. A near-random pick over a tight set costs 0.102 A; over a bimodal one it costs
0.616 A. So D10 is a SELECTION problem on a harder, better sample set, and "restore the old
calibration" would throw away the mode that holds every structure better than the shipped arm
has ever reached (0.560 A against a shipped best of 0.646 A).

    python3 perf/of3t_confhead/ordering.py
"""
import json
import math
import os
import statistics

SRC = "perf/of3t_pairbias/fold_rmsd.json"
OUT = "perf/of3t_confhead/ordering.json"
ARMS = ("ship", "fix")
SEEDS = range(1, 10)

d = json.load(open(SRC))
rep = {"source": SRC, "convention": "rank is 1-indexed; with 5 samples random = 3.0",
       "arms": {}, "basins": {}}

def _mean_se(xs):
    """Mean and its standard error. Nine seeds is few enough that a difference quoted
    without one cannot be told from the sampling noise of the nine."""
    m = statistics.fmean(xs)
    se = statistics.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else float("nan")
    return m, se


for arm in ARMS:
    rows = {s: d["runs"][f"{arm}_s{s}"]["all"] for s in SEEDS}
    picks, rank_of_best, rank_of_pick, regret, spread, r0, best, rho = ([] for _ in range(8))
    for s, v in rows.items():
        n = len(v)
        order = sorted(range(n), key=lambda i: v[i])            # true order, best first
        true_rank = {i: r + 1 for r, i in enumerate(order)}
        picks.append(true_rank[0] == 1)
        rank_of_best.append(order.index(min(range(n), key=lambda i: v[i])) + 0)  # placeholder
        rank_of_best[-1] = v.index(min(v)) + 1                  # PREDICTED rank of the best
        rank_of_pick.append(true_rank[0])                       # TRUE rank of the head's pick
        regret.append(v[0] - min(v))
        spread.append(max(v) - min(v))
        r0.append(v[0]); best.append(min(v))
        dd = sum((i + 1 - true_rank[i]) ** 2 for i in range(n))
        rho.append(1 - 6 * dd / (n * (n * n - 1)))
    n_s = len(rows[1])
    a = {"n_seeds": len(rows), "n_samples": n_s, "picks_best": sum(picks),
         "random_rank": (n_s + 1) / 2,
         "mean_regret": statistics.fmean(regret),
         "mean_within_seed_spread": statistics.fmean(spread),
         "mean_rank0": statistics.fmean(r0), "mean_best": statistics.fmean(best),
         "per_seed_predicted_rank_of_best": rank_of_best,
         "per_seed_true_rank_of_pick": rank_of_pick,
         "per_seed_spearman": [round(x, 4) for x in rho]}
    for key, xs in (("predicted_rank_of_best", rank_of_best),
                    ("true_rank_of_pick", rank_of_pick), ("spearman", rho)):
        a[f"{key}_mean"], a[f"{key}_se"] = _mean_se(xs)
    rep["arms"][arm] = a

print(f"{'arm':6s} {'picks best':>10}  {'pred rank of best':>18}  {'true rank of pick':>18}"
      f"  {'spearman':>14}  {'regret':>8}  {'spread':>8}")
for arm in ARMS:
    a = rep["arms"][arm]
    print(f"{arm:6s} {a['picks_best']:>7}/9  "
          f"{a['predicted_rank_of_best_mean']:>10.2f} +- {a['predicted_rank_of_best_se']:.2f}  "
          f"{a['true_rank_of_pick_mean']:>10.2f} +- {a['true_rank_of_pick_se']:.2f}  "
          f"{a['spearman_mean']:>6.3f} +- {a['spearman_se']:.3f}  "
          f"{a['mean_regret']:>8.3f}  {a['mean_within_seed_spread']:>8.3f}")
print(f"random  1.8/9        {rep['arms']['fix']['random_rank']:.2f}              "
      f"{rep['arms']['fix']['random_rank']:.2f}           0.000           --        --")
_sp = [rep["arms"][a]["spearman_mean"] for a in ARMS]
_se = (rep["arms"]["ship"]["spearman_se"] ** 2 + rep["arms"]["fix"]["spearman_se"] ** 2) ** 0.5
rep["spearman_delta"] = {"fix_minus_ship": _sp[1] - _sp[0], "se": _se}
print(f"-> ordering quality, fix minus ship: {_sp[1] - _sp[0]:+.3f} +- {_se:.3f} Spearman. "
      f"The two marginals disagree in sign; the symmetric statistic separates neither arm "
      f"from the other, nor convincingly from chance.")

# The sample distribution: one mode or two?
print("\nthe 45 samples per arm, pooled and sorted -- where the mass sits")
for arm in ARMS:
    vals = sorted(x for s in SEEDS for x in d["runs"][f"{arm}_s{s}"]["all"])
    gaps = [(vals[i + 1] - vals[i], vals[i], vals[i + 1]) for i in range(len(vals) - 1)]
    g, lo, hi = max(gaps)
    below = [v for v in vals if v <= lo]
    rep["basins"][arm] = {
        "n": len(vals), "min": vals[0], "max": vals[-1],
        "largest_gap": g, "gap_between": [lo, hi],
        "n_below_gap": len(below), "mean_below_gap": statistics.fmean(below),
        "mean_above_gap": statistics.fmean(v for v in vals if v > lo),
    }
    b = rep["basins"][arm]
    print(f"  {arm:5s} n={b['n']} range {b['min']:.3f}-{b['max']:.3f} A  largest gap "
          f"{b['largest_gap']:.3f} A between {lo:.3f} and {hi:.3f}  "
          f"({b['n_below_gap']} samples below it, mean {b['mean_below_gap']:.3f} A)")

# The one decision that costs Angstrom: which mode does the head serve from?
print("\nthe head's pick against the worst mode (samples above the pooled largest gap)")
rep["worst_mode"] = {}
for arm in ARMS:
    lo = rep["basins"][arm]["gap_between"][0]
    rows = [d["runs"][f"{arm}_s{s}"]["all"] for s in SEEDS]
    n_bad = rep["basins"][arm]["n"] - rep["basins"][arm]["n_below_gap"]
    p = n_bad / rep["basins"][arm]["n"]
    served_bad = sum(v[0] > lo for v in rows)
    # P(at least this many bad picks) if the head selected uniformly at random
    tail = sum(math.comb(len(rows), k) * p ** k * (1 - p) ** (len(rows) - k)
               for k in range(served_bad, len(rows) + 1))
    rep["worst_mode"][arm] = {"cut": lo, "n_in_mode": n_bad, "frac_of_samples": p,
                              "served_from_mode": served_bad, "n_seeds": len(rows),
                              "p_at_least_this_many_if_random": tail}
    print(f"  {arm:5s} the mode above {lo:.3f} A holds {n_bad}/{rep['basins'][arm]['n']} "
          f"samples ({p:.0%}), and the head serves from it on {served_bad}/{len(rows)} seeds "
          f"-- P = {tail:.4f} if it picked at random")

# Selection-policy costs, stated as policy and not as accuracy
rep["policies"] = {}
for arm in ARMS:
    rows = [d["runs"][f"{arm}_s{s}"]["all"] for s in SEEDS]
    rep["policies"][arm] = {
        "head_rank0": statistics.fmean(v[0] for v in rows),
        "oracle_best_of_5": statistics.fmean(min(v) for v in rows),
        "random_of_5": statistics.fmean(statistics.fmean(v) for v in rows),
        "oracle_headroom": statistics.fmean(v[0] for v in rows) - statistics.fmean(min(v) for v in rows),
    }
print("\nselection policies over the SAME five samples (no extra model work):")
for arm in ARMS:
    p = rep["policies"][arm]
    print(f"  {arm:5s} head picks {p['head_rank0']:.3f} A | random {p['random_of_5']:.3f} A | "
          f"perfect selector {p['oracle_best_of_5']:.3f} A | headroom {p['oracle_headroom']:.3f} A")

# The seed floor this target carries, recomputed here so the deltas above have it beside them
rep["seed_floor"] = {a: d["seed_floor"][a] for a in ARMS if a in d.get("seed_floor", {})}
for arm in ARMS:
    if arm in rep["seed_floor"]:
        f = rep["seed_floor"][arm]
        print(f"  seed floor, {arm:5s}: {f['mean']:.3f} A mean over {f['n_pairs']} pairs "
              f"({f['min']:.3f}-{f['max']:.3f})")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(rep, open(OUT, "w"), indent=1)
print("\nwrote", OUT)
