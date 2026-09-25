"""Is D10 a calibration regression, or a weak selector that only now has consequences?

D10 records the confidence head as having had "its calibration changed by the corrected trunk".
`perf/of3t_pairbias/fold_rmsd.json` already carries, per seed and arm, the five per-sample RMSDs
in the model's OWN ranked order. That is enough to measure the ordering directly.
"""
import json, statistics as st, sys
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "perf/of3t_pairbias/fold_rmsd.json"))
arms, out = {}, {}
for k, r in d["runs"].items():
    arms.setdefault(k.rsplit("_s", 1)[0], []).append((k, r))
for arm, items in sorted(arms.items()):
    regrets, ranks, spreads = [], [], []
    for k, r in sorted(items):
        a = r["all"]
        regrets.append(r["rank0"] - r["best"])
        ranks.append(a.index(min(a)))
        spreads.append(max(a) - min(a))
    out[arm] = {"seeds": len(items), "picks_best": ranks.count(0),
                "mean_rank_of_best": st.mean(ranks), "mean_regret": st.mean(regrets),
                "mean_within_seed_spread": st.mean(spreads),
                "mean_best": st.mean(r["best"] for _, r in items),
                "mean_rank0": st.mean(r["rank0"] for _, r in items),
                "mean_plddt": st.mean(r["plddt"] for _, r in items)}
    print(arm, json.dumps(out[arm], indent=1))
out["random_baseline_mean_rank_of_best"] = 2.0
out["note"] = ("5 samples per seed, so a uniformly random selector picks the best 1 time in 5 "
               "and sits at mean rank 2.0.")
json.dump(out, open("d10_ordering.json", "w"), indent=1)
