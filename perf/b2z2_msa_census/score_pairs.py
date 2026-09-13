"""Paired ABBA derivation for the PWA cell run: the A/B and the A/A floor of the same estimator.

`max/min` over an arm pools the box's drift across the whole run, which on a loaded box is a range
statistic and not the floor the paired estimator actually faces. Within a rep the two `off` folds
bracket the two `on` folds, so `off_first` vs `off_last` is an A/A pair drawn exactly the way the
A/B pairs are. Both are reported; so is the rank separation, which is the statistic that does not
care about drift at all.
"""
import json, statistics as st, sys
from itertools import combinations
from math import comb
from pathlib import Path

p = Path(sys.argv[1])
d = json.loads(p.read_text())
timed = [r for r in d["runs"] if not r["warmup"]]
rep = {}
for r in timed:
    rep.setdefault(r["rep"], []).append(r)

out = {"doc": __doc__}
for key in ("fold_s", "msa_track_s"):
    ab, aa = [], []
    for i in sorted(rep):
        r = rep[i]
        assert [x["arm"] for x in r] == ["off", "on", "on", "off"], [x["arm"] for x in r]
        off, on = [r[0][key], r[3][key]], [r[1][key], r[2][key]]
        ab += [round(o / n, 5) for o, n in zip(off, on)]
        aa += [round(max(off) / min(off), 5), round(max(on) / min(on), 5)]
    offs = [r[key] for r in timed if r["arm"] == "off"]
    ons = [r[key] for r in timed if r["arm"] == "on"]
    sep = max(ons) < min(offs)
    n = len(offs)
    out[key] = {
        "median_off_s": round(st.median(offs), 4), "median_on_s": round(st.median(ons), 4),
        "ab_pairs": ab, "ab_median": round(st.median(ab), 5),
        "ab_positive": f"{sum(1 for x in ab if x > 1)}/{len(ab)}",
        "aa_pairs": aa, "aa_median": round(st.median(aa), 5), "aa_worst": round(max(aa), 5),
        "rank_separated": sep,
        "separation_p": round(1 / comb(2 * n, n), 5) if sep else None,
        "pooled_range_off": round(max(offs) / min(offs), 5),
    }
d["derived"] = out
p.write_text(json.dumps(d, indent=1))
for k, v in out.items():
    if k == "doc":
        continue
    print(f"{k}: off {v['median_off_s']} on {v['median_on_s']} | A/B {v['ab_median']} "
          f"({v['ab_positive']} positive) | A/A median {v['aa_median']} worst {v['aa_worst']} | "
          f"rank-separated {v['rank_separated']} p={v['separation_p']} | "
          f"pooled off range {v['pooled_range_off']}")
