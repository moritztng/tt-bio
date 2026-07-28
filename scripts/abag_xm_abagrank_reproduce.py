"""Reproduce ABAG-Rank on its own bundled example and score it against the truth in the h5.

The README publishes no expected numbers, so the obvious acceptance bar would be mechanical
("it ran"). It can be numeric instead: examples/8b9v_filtered.h5 stores a per-sample
``abag_dockq``, so each released checkpoint can be compared against the baseline that actually
matters for this campaign -- AF3's own ``ranking_score``.

Tie handling is the trap. ``af3_ranking_score`` in that h5 is rounded to 2 decimals, giving 6
distinct values over 33 samples with 5 tied at the maximum, so "the top-1 pick" is not defined
by the score alone. Selecting with ``list.index(max(...))`` silently resolves the tie by CSV row
order, which differs per run and made AF3's top-1 loss look like 0.000 in one run and 0.110 in
another from identical scores. Top-1 loss here is therefore the EXPECTED loss under random
tie-breaking (the mean over the tied set), reported with the best/worst bracket.

Usage: run from an ABAG-Rank checkout after both run_inference.py invocations, passing the
two output dirs.
"""
import ast
import csv
import statistics
import sys
import pathlib

import h5py
from scipy.stats import spearmanr

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
RUNS = sys.argv[2:] or ["inf_main", "inf_noesm"]
TARGET = "8b9v"

truth = {}
with h5py.File(ROOT / "examples" / f"{TARGET}_filtered.h5", "r") as f:
    g = f[TARGET]
    for k in g:
        if "sample-" in k and "abag_dockq" in g[k]:
            truth[k] = float(g[k]["abag_dockq"][()])
if not truth:
    sys.exit("no abag_dockq ground truth found in the bundled h5")


def norm(name):
    """CSV writes b'seed-10_sample-1'; the h5 key is seed-10_seed-10_sample-1."""
    s = name.strip()
    if s[:2] in ("b'", 'b"'):
        s = ast.literal_eval(s).decode()
    seed = s.split("_")[0]
    return s if s.startswith(f"{seed}_{seed}") else f"{seed}_{s}"


def expected_top1_loss(score, gt):
    """Expected (oracle - selected) under random tie-breaking, plus the best/worst bracket."""
    best = max(score.values())
    tied = [k for k in score if score[k] == best]
    oracle = max(gt.values())
    picks = [gt[k] for k in tied]
    return (oracle - statistics.mean(picks), oracle - max(picks), oracle - min(picks), len(tied))


gts = dict(truth)
oracle = max(gts.values())
print("target %s | %d samples | abag_dockq min %.3f median %.3f max %.3f"
      % (TARGET, len(gts), min(gts.values()),
         statistics.median(gts.values()), oracle))

af3 = None
for run in RUNS:
    csv_path = ROOT / run / f"{TARGET}_ranked_by_model.csv"
    if not csv_path.exists():
        print("%-10s MISSING %s" % (run, csv_path)); continue
    rows = list(csv.DictReader(open(csv_path)))
    pred, a3, unmatched = {}, {}, 0
    for r in rows:
        k = norm(r["sample_name"])
        if k not in gts:
            unmatched += 1; continue
        pred[k] = float(r["model_predicted_dockq"])
        a3[k] = float(r["af3_ranking_score"])
    af3 = af3 or a3
    ks = list(pred)
    rho = spearmanr([pred[k] for k in ks], [gts[k] for k in ks]).correlation
    mean_l, best_l, worst_l, nties = expected_top1_loss(pred, gts)
    print("\n%-10s n=%d unmatched=%d nan=%d" % (run, len(ks), unmatched,
                                                sum(1 for v in pred.values() if v != v)))
    print("   spearman(model_predicted_dockq, abag_dockq) = %+.3f" % rho)
    print("   top-1 loss = %.3f (ties=%d, bracket %.3f..%.3f)" % (mean_l, nties, best_l, worst_l))

if af3:
    ks = list(af3)
    rho = spearmanr([af3[k] for k in ks], [gts[k] for k in ks]).correlation
    mean_l, best_l, worst_l, nties = expected_top1_loss(af3, gts)
    print("\n%-10s BASELINE TO BEAT" % "af3_score")
    print("   distinct score values: %d over %d samples" % (len(set(af3.values())), len(af3)))
    print("   spearman(af3_ranking_score, abag_dockq) = %+.3f" % rho)
    print("   top-1 loss = %.3f (ties=%d, bracket %.3f..%.3f)" % (mean_l, nties, best_l, worst_l))
