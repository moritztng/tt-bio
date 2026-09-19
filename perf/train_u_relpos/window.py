#!/usr/bin/env python3
"""The A/B over one uniform settled window, with each arm's wall-clock span printed so it can be
checked against the host's cotenancy and the streamed AICLK log."""
import json, statistics as st, sys, time
from pathlib import Path

BASE = Path("/home/ttuser/.coworker/wt/train-u-relpos-ondevice/perf/train_u_relpos")
LO, HI = int(sys.argv[1]), int(sys.argv[2])
Z = lambda t: time.strftime("%H:%M:%SZ", time.gmtime(t))
res = {}
for arm in ("A1", "B1", "A2", "B2"):
    for rk in (0, 1):
        rows = [json.loads(l) for l in open(BASE / arm / f"history-rank{rk}.jsonl")]
        rows = [r for r in rows if LO <= r["step"] <= HI]
        t2t = [b["t"] - a["t"] for a, b in zip(rows, rows[1:]) if b["step"] == a["step"] + 1]
        res[(arm, rk)] = dict(n=len(rows), data=st.mean(r["data"] for r in rows),
                              wall=st.mean(r["wall"] for r in rows),
                              t2t=st.mean(t2t), t2tmed=st.median(t2t),
                              span=f"{Z(rows[0]['t'])}-{Z(rows[-1]['t'])}",
                              loss=rows[-1]["loss"], dig=rows[-1]["digest"][:12])
        r = res[(arm, rk)]
        print(f"{arm} rank{rk}  steps {LO}-{HI} n={r['n']}  {r['span']}  data {r['data']:.3f}  "
              f"wall {r['wall']:.3f}  t2t {r['t2t']:.3f} (med {r['t2tmed']:.3f})  "
              f"loss {r['loss']:.9f} digest {r['dig']}")
print()
for rk in (0, 1):
    a = [res[(x, rk)] for x in ("A1", "A2")]
    b = [res[(x, rk)] for x in ("B1", "B2")]
    am, bm = st.mean(x["t2t"] for x in a), st.mean(x["t2t"] for x in b)
    print(f"rank{rk}  data {st.mean(x['data'] for x in a):.3f} -> {st.mean(x['data'] for x in b):.3f}"
          f"   wall {st.mean(x['wall'] for x in a):.3f} -> {st.mean(x['wall'] for x in b):.3f}"
          f"   cadence {am:.3f} -> {bm:.3f} s = {100 * (am / bm - 1):+.1f} %")
    print(f"        per-arm cadence  A {a[0]['t2t']:.3f} {a[1]['t2t']:.3f}   "
          f"B {b[0]['t2t']:.3f} {b[1]['t2t']:.3f}")
