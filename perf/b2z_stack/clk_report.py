#!/usr/bin/env python3
"""Slice a sysfs clock trace to the folds that ran inside it."""
import csv, json, statistics as st, sys
from collections import defaultdict

trace, foldjson, chip = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = defaultdict(list)
with open(trace) as f:
    for r in csv.DictReader(f):
        if not r["aiclk_mhz"]:
            continue
        rows[int(r["chip"])].append((float(r["t"]), int(r["aiclk_mhz"]),
                                     int(r["power_uw"] or 0), int(r["temp_mdegc"] or 0)))
fold = json.load(open(foldjson))
print(f"trace: {sum(len(v) for v in rows.values())} samples, chips {sorted(rows)}")


def stats(lo, hi, c):
    s = [x for x in rows[c] if lo <= x[0] <= hi]
    if not s:
        return None
    clk = [x[1] for x in s]
    pw = [x[2] / 1e6 for x in s]
    tp = [x[3] / 1e3 for x in s]
    return dict(n=len(s), clk_min=min(clk), clk_med=st.median(clk), clk_max=max(clk),
                clk_p05=sorted(clk)[max(0, int(0.05 * len(clk)))],
                W_med=round(st.median(pw), 1), W_max=round(max(pw), 1),
                C_med=round(st.median(tp), 1), C_max=round(max(tp), 1))


for f in fold["folds"]:
    for c in (chip,):
        s = stats(f["epoch_start"], f["epoch_end"], c)
        print(f"{f['tag']:6s} {f['wall_s']:7.3f}s chip{c}  AICLK min/p05/med/max "
              f"{s['clk_min']}/{s['clk_p05']}/{s['clk_med']:.0f}/{s['clk_max']} MHz  "
              f"power {s['W_med']}/{s['W_max']} W  temp {s['C_med']}/{s['C_max']} C  n={s['n']}")
warm = [f for f in fold["folds"] if f["tag"] != "cold"]
lo, hi = warm[0]["epoch_start"], warm[-1]["epoch_end"]
for c in sorted(rows):
    s = stats(lo, hi, c)
    print(f"ALL-WARM chip{c}: AICLK min/p05/med/max {s['clk_min']}/{s['clk_p05']}/{s['clk_med']:.0f}/"
          f"{s['clk_max']}  power {s['W_med']}/{s['W_max']} W  temp {s['C_med']}/{s['C_max']} C")
allclk = [x[1] for x in rows[chip] if lo <= x[0] <= hi]
hist = {}
for v in allclk:
    hist[v] = hist.get(v, 0) + 1
print("AICLK histogram over the warm folds, chip %d:" % chip,
      {k: f"{100*v/len(allclk):.1f}%" for k, v in sorted(hist.items())})
