#!/usr/bin/env python3
"""Pair an ABBA timing run: each `on` fold against the mean of its two bracketing `base` folds.

`fold_cond.py` reports a ratio of medians, which on a box whose load drifts across the session
charges the drift to whichever arm sat in the slow half. The bracketed pair cancels a linear
drift inside one rep, and the same estimator run on the two BASE positions is the A/A floor it
has to clear -- the floor and the headline must come out of the same estimator or the margin is
not a margin (`b2z2-redteam-v3`).
"""
import json, statistics as st, sys
from pathlib import Path

runs = json.loads(Path(sys.argv[1]).read_text())["runs"]
warm = [r for r in runs if not r["cold"]]
reps = {}
for r in warm:
    reps.setdefault(r["rep"], {})[r["pos"]] = r

pairs, floor = [], []
for rep, byp in sorted(reps.items()):
    if set(byp) != {0, 1, 2}:
        continue
    b = (byp[0]["fold_s"] + byp[2]["fold_s"]) / 2
    pairs.append(b / byp[1]["fold_s"])
    # A/A on the same estimator: the two base folds against their own mean, taken as the
    # ratio one bracket arm would read if it were the lever.
    floor.append(byp[0]["fold_s"] / byp[2]["fold_s"])

out = {
    "n_pairs": len(pairs),
    "paired_ratio_median": round(st.median(pairs), 5),
    "paired_ratio_mean": round(st.fmean(pairs), 5),
    "positive": sum(p > 1.0 for p in pairs),
    "pairs": [round(p, 5) for p in pairs],
    "aa_floor_bracket_median": round(st.median([max(f, 1 / f) for f in floor]), 5),
    "aa_floor_pairs": [round(f, 5) for f in floor],
    "median_fold_s": {a: round(st.median([r["fold_s"] for r in warm if r["arm"] == a]), 3)
                      for a in sorted({r["arm"] for r in warm})},
    "digests": sorted({r["sha256"] for r in warm}),
}
print(json.dumps(out, indent=1))
