"""Turn a probe trace into a per-leg attribution table.

Reads the enter/exit JSONL the probe wrote and prints, per span, exclusive time
(own time minus the time its instrumented children held), so the columns add up
instead of double-counting nesting.
"""
import collections
import json
import sys

path = sys.argv[1]
rows = []
for line in open(path):
    try:
        r = json.loads(line)
    except Exception:
        continue
    if r.get("ev") in ("in", "out"):
        rows.append(r)

# Match enter/exit by index, keeping the nesting order per pid.
open_stack = collections.defaultdict(list)
spans = []          # (op, t_in, t_out, depth, parent_i, info)
by_i = {}
for r in rows:
    pid = r["pid"]
    if r["ev"] == "in":
        parent = open_stack[pid][-1] if open_stack[pid] else None
        s = {"op": r["op"], "i": r["i"], "t_in": r["t"], "pce_in": r.get("pce"),
             "depth": len(open_stack[pid]), "parent": parent,
             "info": {k: v for k, v in r.items()
                      if k not in ("ev", "op", "i", "t", "pid", "pce")}}
        by_i[r["i"]] = s
        spans.append(s)
        open_stack[pid].append(r["i"])
    else:
        s = by_i.get(r["i"])
        if s is None:
            continue
        s["t_out"] = r["t"]
        s["dt"] = r.get("dt", r["t"] - s["t_in"])
        s["pce_out"] = r.get("pce")
        if open_stack[pid] and open_stack[pid][-1] == r["i"]:
            open_stack[pid].pop()

done = [s for s in spans if "dt" in s]
child_time = collections.defaultdict(float)
for s in done:
    if s["parent"] is not None:
        child_time[s["parent"]] += s["dt"]

print("== timeline (depth-indented, s) ==")
for s in done if len(done) < 400 else [x for x in done if x["depth"] < 4]:
    excl = s["dt"] - child_time[s["i"]]
    info = {k: v for k, v in s["info"].items() if v not in (None, False)}
    print("%8.2f  %s%-26s dt=%8.3f excl=%8.3f pce %s->%s %s"
          % (s["t_in"], "  " * s["depth"], s["op"], s["dt"], excl,
             s["pce_in"], s["pce_out"], info if info else ""))

print()
print("== totals by op (inclusive / exclusive / n) ==")
agg = collections.defaultdict(lambda: [0.0, 0.0, 0])
for s in done:
    a = agg[s["op"]]
    a[0] += s["dt"]
    a[1] += s["dt"] - child_time[s["i"]]
    a[2] += 1
for op, (inc, exc, n) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
    print("%-26s incl=%9.3f excl=%9.3f n=%d" % (op, inc, exc, n))
