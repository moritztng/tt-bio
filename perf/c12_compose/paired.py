#!/usr/bin/env python3
"""The session's real A/A floor: the paired per-rep spread, not the difference of two medians.

`fold_compose.py` reports its A/A floor as `median(base@pos0) - median(base@pos4)`. On session s2
that read 0.1080 s and it is wrong by a factor of 2.6, because taking medians first cancels the
within-rep load swings that are exactly the noise the floor is supposed to measure. The same
session's per-rep base-vs-base differences are -0.143, +0.013, -1.048, -0.435, +0.689 s: a 1.74 s
spread, paired sd 0.635 s, se 0.284 s, 95% CI +/-0.79 s.

So the floor to compare an effect against is the A/A arm's own paired confidence interval. A lever
whose paired CI overlaps the A/A arm's paired CI is not resolved by that session, whatever the
medians say. This script prints both, per rep and pooled, and it exists because s2 looked like a
clean NO-GO on medians and is in fact a session that cannot resolve its own pre-registered effect.

Each arm is differenced against its OWN rep's base mean (pos0 and pos4), which is what makes the
read paired and removes the monotone load ramp that destroyed session s1.

    paired.py            # reads perf/c12_compose/out/s2.json
"""
import json, statistics as st
d = json.load(open("perf/c12_compose/out/s2.json"))
runs = [r for r in d["runs"] if not r["cold"]]
reps = sorted({r["rep"] for r in runs})
print("per-rep, each arm against that rep's OWN base mean (pos0,pos4) -- removes the load ramp")
print("%4s %7s %7s %7s %9s %9s %9s %9s" % ("rep","baseA","baseB","bmean","silu","hoist","both","A/A"))
acc = {"silu": [], "hoist": [], "both": []}
aa = []
for rp in reps:
    g = {(r["arm"], r["pos"]): r["fold_s"] for r in runs if r["rep"] == rp}
    b0, b4 = g[("base", 0)], g[("base", 4)]
    bm = (b0 + b4) / 2
    row = []
    for a in ("silu", "hoist", "both"):
        v = [x for (k, p), x in g.items() if k == a][0]
        acc[a].append(bm - v)
        row.append(bm - v)
    aa.append(b0 - b4)
    print("%4d %7.3f %7.3f %7.3f %+9.4f %+9.4f %+9.4f %+9.4f" % (rp, b0, b4, bm, row[0], row[1], row[2], b0 - b4))
print()
for a in ("silu", "hoist", "both"):
    v = acc[a]; m = st.mean(v); sd = st.stdev(v); se = sd / len(v) ** 0.5
    print("%6s paired mean %+.4f s  sd %.4f  se %.4f  95%% CI [%+.4f, %+.4f]" % (a, m, sd, se, m - 2.776 * se, m + 2.776 * se))
m = st.mean(aa); sd = st.stdev(aa); se = sd / len(aa) ** 0.5
print("%6s paired mean %+.4f s  sd %.4f  se %.4f  95%% CI [%+.4f, %+.4f]" % ("A/A", m, sd, se, m - 2.776 * se, m + 2.776 * se))
print()
s = st.mean(acc["silu"]); h = st.mean(acc["hoist"]); b = st.mean(acc["both"])
print("sum of singles %+.4f s   measured both %+.4f s   both/sum %.3f" % (s + h, b, b / (s + h)))
print("harness summary:", json.dumps(d.get("summary", {}), indent=1)[:900])
