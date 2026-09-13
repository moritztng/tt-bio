#!/usr/bin/env python3
"""Read a fold A/B written by `baseline_attrib.py --phases baseline` and rule on it.

One file per (arm, round). The rounds exist because a single round charges whichever arm led it
for the box's own drift; reversing the lead between rounds is what separates the lever from the
drift, so the per-round ratios are printed as well as the pooled one. The A/A floor is the same
arm measured against itself across the two rounds, and a ratio inside that floor is not a result.

Every fold's CIF digest is pooled too. These are layout levers: a correct one is bit-exact, so a
second digest here is a stop signal rather than something to score in Angstrom.

    ab_analyse.py [--path TMPL] [--arms A,B] [--rounds N]

TMPL takes %(arm)s and %(round)d. Defaults to D1's pass-1 files.
"""
import argparse, itertools, json, math, statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("--path", default="/home/ttuser/scratch/uod/ab_%(arm)s_r%(round)d.json")
ap.add_argument("--arms", default="base,patched", help="slow arm first")
ap.add_argument("--rounds", type=int, default=2)
a = ap.parse_args()
ARMS = a.arms.split(",")
ROUNDS = list(range(a.rounds))

data = {}
for arm in ARMS:
    for r in ROUNDS:
        d = json.load(open(a.path % {"arm": arm, "round": r}))
        pl = [x["fold_s"] for x in d["baseline"] if x["arm"] == "plain"]
        ins = [x["fold_s"] for x in d["baseline"] if x["arm"] == "instr"]
        digs = {v for x in d["baseline"] for v in x["cifs"].values()}
        data[(arm, r)] = (pl, ins, digs, d["cold_s"], d["env"]["git_head"][:8])
        print("%-8s r%d head=%s clk=%s cold=%6.3f plain=%s instr=%s" % (
            arm, r, d["env"]["git_head"][:8], d["env"].get("aiclk_mhz", "?").strip(),
            d["cold_s"], ["%.3f" % v for v in pl], ["%.3f" % v for v in ins]))

digs = set().union(*[v[2] for v in data.values()])
print("\ndistinct CIF sha256 across all %d timed folds: %d  %s" % (
    sum(len(v[0]) + len(v[1]) for v in data.values()), len(digs), [d[:16] for d in digs]))

pool = {x: sorted(itertools.chain(*[data[(x, r)][0] for r in ROUNDS])) for x in ARMS}
for x in ARMS:
    print("%-8s plain folds n=%d  median %.4f  min %.4f  max %.4f  spread %.3f %%" % (
        x, len(pool[x]), st.median(pool[x]), min(pool[x]), max(pool[x]),
        100 * (max(pool[x]) / min(pool[x]) - 1)))
slow, fast = ARMS
mb, mp = st.median(pool[slow]), st.median(pool[fast])
print("\nRATIO (pooled plain medians): %.5fx   delta %.4f s" % (mb / mp, mb - mp))

floors = []
for x in ARMS:
    ms = [st.median(data[(x, r)][0]) for r in ROUNDS]
    f = max(ms) / min(ms)
    floors.append(f)
    print("A/A %-8s %s -> %.5fx" % (x, " vs ".join("%.4f" % m for m in ms), f))
print("worst-case A/A floor %.5fx" % max(floors))

print("\nper-round ratios (%s/%s):" % (slow, fast))
for r in ROUNDS:
    b, p = st.median(data[(slow, r)][0]), st.median(data[(fast, r)][0])
    print("  r%d  %.4f / %.4f = %.5fx" % (r, b, p, b / p))
print("\nevery %s fold vs every %s fold: %d of %d pairs have %s slower" % (
    slow, fast,
    sum(1 for x in pool[slow] for y in pool[fast] if x > y),
    len(pool[slow]) * len(pool[fast]), slow))

# The A/A number above is a RANGE over the round medians, and a range grows with the number of
# rounds by construction: adding rounds to resolve a small effect also inflates the floor it has
# to clear. That made it the right yardstick for D1`s two-round design and the wrong one here.
# The design is PAIRED -- both arms run back to back inside a round and the lead alternates -- so
# the drift-immune statistic is the per-round ratio, and the question is whether it is one-sided.
signs = [st.median(data[(slow, r)][0]) > st.median(data[(fast, r)][0]) for r in ROUNDS]
k, n = sum(signs), len(ROUNDS)
# Two-sided sign test against "the lead alternates, so a drift is equally likely either way".
p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n)
rat = [st.median(data[(slow, r)][0]) / st.median(data[(fast, r)][0]) for r in ROUNDS]
print("\npaired over rounds: %s slower in %d of %d  sign-test p=%.4f" % (slow, k, n, p))
print("per-round ratio mean %.5fx  min %.5fx  max %.5fx" % (
    st.fmean(rat), min(rat), max(rat)))
