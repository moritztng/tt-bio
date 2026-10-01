#!/usr/bin/env python3
"""Summarise enq_round.py's enqueue.json per seam over warm rounds (3+)."""
import collections, json, pathlib, statistics as st, sys
o = pathlib.Path(sys.argv[1])
r = json.loads((o / "enqueue.json").read_text())
d = json.loads((o / "round_events.json").read_text())
starts = sorted(e["t0"] for e in d["events"] if e["kind"] == "round_start")
stop = [e["t0"] for e in d["events"] if e["kind"] == "round_stop"]
warm = [x for x in r if x["t"] > starts[2]]
g = collections.defaultdict(list)
for x in warm:
    g[x["seam"]].append(x)
for s, xs in sorted(g.items()):
    m = lambda k: st.median(x[k] for x in xs)
    print(f"{s:20s} n={len(xs):2d} wall {m('wall'):.3f} enqueue {m('enqueue'):.3f} "
          f"blocked {m('blocked'):.3f} readbacks {m('readbacks'):.0f}")
nr = len(starts) - 2
print(f"per warm round ({nr}): seam wall {sum(x['wall'] for x in warm) / nr:.3f} s, "
      f"enqueue {sum(x['enqueue'] for x in warm) / nr:.3f} s")
t1 = stop[0] if stop else starts[-1]
c = sorted(x[1] for x in d["aiclk"] if starts[2] <= x[0] <= t1)
print("AICLK during warm rounds: med", c[len(c) // 2], "min", c[0], "n", len(c))
