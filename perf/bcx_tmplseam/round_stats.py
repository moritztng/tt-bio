#!/usr/bin/env python3
"""The A/B's statistics, recomputed from round_ab.json so the doc's numbers are reproducible.

The row's first reading quoted a ratio of medians (0.98x) and called the arms not separated. The
ratio of MEANS is 1.04x -- the other way. Both are in this file because the disagreement is the
result: an effect whose sign depends on the statistic is not an effect at n=5 per arm. The
permutation test is exact (252 splits), not asymptotic, which matters at this n.
"""
import itertools
import json
import pathlib
import statistics as st
import sys

src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                   pathlib.Path(__file__).with_name("runs") / "round_ab_artifacts" / "round_ab.json")
rounds = json.loads(src.read_text())["rounds"]
timed = [r for r in rounds if r["round"] > 2]          # 1 and 2 are the two compiles
off = [r["sequence_gradients_s"] for r in timed if not r["template_on_device"]]
on = [r["sequence_gradients_s"] for r in timed if r["template_on_device"]]

obs = st.mean(off) - st.mean(on)
pool = off + on
hits = tot = 0
for c in itertools.combinations(range(len(pool)), len(off)):
    a = [pool[i] for i in c]
    b = [pool[i] for i in range(len(pool)) if i not in c]
    tot += 1
    hits += abs(st.mean(a) - st.mean(b)) >= abs(obs) - 1e-12

seq = [(r["template_on_device"], r["sequence_gradients_s"]) for r in timed]
fwd = [seq[i][1] - seq[i + 1][1] for i in range(0, len(seq) - 1, 2)]
bwd = [seq[i + 2][1] - seq[i + 1][1] for i in range(0, len(seq) - 2, 2)]

out = {
    "n_per_arm": {"off": len(off), "on": len(on)},
    "off": {"all": off, "median": st.median(off), "mean": round(st.mean(off), 4),
            "min": min(off), "max": max(off)},
    "on": {"all": on, "median": st.median(on), "mean": round(st.mean(on), 4),
           "min": min(on), "max": max(on)},
    "ratio_of_medians_off_over_on": round(st.median(off) / st.median(on), 4),
    "ratio_of_means_off_over_on": round(st.mean(off) / st.mean(on), 4),
    "sign_of_effect_depends_on_statistic": (st.median(off) - st.median(on)) * (st.mean(off) - st.mean(on)) < 0,
    "permutation": {"statistic": "difference in means, seconds", "observed": round(obs, 4),
                    "two_sided_p": round(hits / tot, 4), "splits": tot, "exact": True},
    "adjacent_pairs": {"off_minus_following_on": [round(x, 3) for x in fwd],
                       "mean": round(st.mean(fwd), 3),
                       "following_off_minus_on": [round(x, 3) for x in bwd],
                       "mean_other_direction": round(st.mean(bwd), 3)},
    "on_range_nested_in_off_range": min(on) >= min(off) and max(on) <= max(off),
}
print(json.dumps(out, indent=1))
