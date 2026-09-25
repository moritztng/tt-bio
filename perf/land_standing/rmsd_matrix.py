#!/usr/bin/env python3
"""Is the dividing-k lever's structural move distinguishable from re-seeding?

CA-RMSD between every pair of folds at 832 tokens, split into WITHIN-arm pairs (seed noise) and
ACROSS-arm pairs (seed noise + whatever the lever does). A single "lever moved it X A, the seed
floor is Y A" is a point estimate against a point estimate; with several folds per arm the
question becomes whether the across-arm distances are drawn from the same distribution as the
within-arm ones, which an exact permutation over the arm labels answers without assuming
normality.
"""
import itertools
import json
import pathlib
import sys

sys.path.insert(0, "/home/ttuser/.coworker/wt/land-standing")
sys.path.insert(0, "/home/ttuser/.coworker/wt/land-standing/perf/of3t_rankunify")
from ca_rmsd import ca_rmsd                                            # noqa: E402

RUNS = json.loads(sys.argv[1])            # {"on:s0": path, ...}
names = sorted(RUNS)
d = {}
for a, b in itertools.combinations(names, 2):
    d[(a, b)] = ca_rmsd(RUNS[a], RUNS[b])

arm = {n: n.split(":")[0] for n in names}
within = [v for (a, b), v in d.items() if arm[a] == arm[b]]
across = [v for (a, b), v in d.items() if arm[a] != arm[b]]
mean = lambda xs: sum(xs) / len(xs)                                    # noqa: E731
obs = mean(across) - mean(within)

# Exact permutation over which folds are labelled "on". The statistic is recomputed from the
# SAME distance matrix under each relabelling, so nothing is refolded and nothing is assumed.
n_on = sum(1 for n in names if arm[n] == "on")
count = tot = 0
for on in itertools.combinations(names, n_on):
    lab = {n: ("on" if n in on else "off") for n in names}
    w = [v for (a, b), v in d.items() if lab[a] == lab[b]]
    x = [v for (a, b), v in d.items() if lab[a] != lab[b]]
    if not w or not x:
        continue
    tot += 1
    if abs(mean(x) - mean(w)) >= abs(obs) - 1e-12:
        count += 1

for (a, b), v in sorted(d.items()):
    print(f"  {a:8s} vs {b:8s}  {v:8.4f}  {'within' if arm[a] == arm[b] else 'ACROSS'}")
print(f"\nwithin-arm (seed only) n={len(within)} mean {mean(within):.4f} "
      f"range {min(within):.4f}..{max(within):.4f}")
print(f"across-arm (seed+lever) n={len(across)} mean {mean(across):.4f} "
      f"range {min(across):.4f}..{max(across):.4f}")
print(f"observed separation {obs:+.4f} A")
print(f"exact permutation p = {count}/{tot} = {count / tot:.4f}")
out = pathlib.Path("/home/ttuser/.coworker/wt/land-standing/perf/land_standing/out/of3_832"
                   "/rmsd_matrix.json")
out.write_text(json.dumps({"runs": RUNS,
                           "pairs": {f"{a}|{b}": v for (a, b), v in d.items()},
                           "within_mean": mean(within), "across_mean": mean(across),
                           "within": within, "across": across,
                           "separation_A": obs, "perm_p": count / tot, "perm_n": tot}, indent=1))
print(f"wrote {out}")
