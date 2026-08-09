#!/usr/bin/env python3
"""Aggregate a pf_block_ops.py record dump by (op, call site) and split it into T3's slice."""
import collections
import json
import sys

d = json.load(open(sys.argv[1]))
R = d["records"]
tot = d["sum_s"]
wall = d["block_wall_s"]
print("block_wall_ms", round(wall * 1e3, 3), "sum_ms", round(tot * 1e3, 3),
      "coverage", round(100 * tot / wall, 1))

g = collections.OrderedDict()
for r in R:
    k = (r["op"], r["site"])
    e = g.setdefault(k, {"n": 0, "s": 0.0, "chain": r["chain"], "ins": collections.Counter(),
                         "out": collections.Counter()})
    e["n"] += 1
    e["s"] += r["s"]
    e["ins"][tuple("x".join(map(str, i["shape"])) + ":" + i["buf"][:3] for i in r["in"])] += 1
    if r["out"]:
        e["out"]["x".join(map(str, r["out"]["shape"])) + ":" + r["out"]["buf"][:3]] += 1

rows = sorted(g.items(), key=lambda kv: -kv[1]["s"])
hdr = "%-26s %-11s %4s %9s %8s %6s  %s" % ("op", "site", "n", "us/call", "ms/blk", "%sum", "chain")
print(hdr)
for (op, site), e in rows:
    print("%-26s %-11s %4d %9.1f %8.3f %6.2f  %s"
          % (op, site, e["n"], e["s"] / e["n"] * 1e6, e["s"] * 1e3, 100 * e["s"] / tot,
             ">".join(e["chain"][1:])))

print("\n=== shapes for the biggest 25 rows ===")
for (op, site), e in rows[:25]:
    print(op, site, "n=%d" % e["n"])
    for k, c in e["ins"].most_common(4):
        print("   in ", c, list(k))
    for k, c in e["out"].most_common(3):
        print("   out", c, k)
