#!/usr/bin/env python3
"""Reads one perf/bc4_gates/chain.sh directory: gradient digests per arm and iteration, raw gate calls in the
instrumented iteration, the census synced total and its per-op delta (mean of the two base arms minus fix),
AICLK median and load over each arm.   python perf/bc4_gates/read_chain.py <chain dir> [out.json]"""
import collections, json, statistics, sys
from pathlib import Path

D = Path(sys.argv[1])
arms = [a for a in ("a1", "b1", "a2") if (D / a / "census.json").exists()]
spans = {}
for line in (D / "card").read_text().splitlines():
    w = line.split()
    if len(w) >= 3 and w[1] in ("start",) and w[0] in arms:
        spans.setdefault(w[0], [int(w[2]), None])
    if len(w) >= 3 and w[1].startswith("rc=") and w[0] in arms:
        spans[w[0]][1] = int(w[3])
clk = [l.split("\t") for l in (D / "aiclk.tsv").read_text().splitlines() if l.count("\t") == 2]
doc = dict(dir=str(D), arms={})
ops = {}
for a in arms:
    c = json.loads((D / a / "census.json").read_text())
    r = json.loads((D / f"{a}.json").read_text())
    t0, t1 = spans.get(a, [0, 1 << 62])
    cs = [int(x[1]) for x in clk if t0 <= int(x[0]) <= (t1 or 1 << 62)]
    ld = [float(x[2]) for x in clk if t0 <= int(x[0]) <= (t1 or 1 << 62)]
    at = str(r["iterations"])
    gates = r["per_iteration"].get(at, {})
    doc["arms"][a] = dict(
        route_arm=r["arm"], gsha=[u["gsha"] for u in c["uninstrumented"]],
        uninstrumented_s=[round(u["s"], 3) for u in c["uninstrumented"]],
        synced_total_s=round(c["synced_total_s"], 4), instrumented_s=round(c["iteration_s"], 3),
        aiclk_census=c.get("aiclk"), aiclk_median=statistics.median(cs) if cs else None,
        aiclk_min=min(cs) if cs else None, load1_median=statistics.median(ld) if ld else None,
        gates_at_instrumented_iteration={k: v for k, v in gates.items() if v.get("taped")})
    agg = collections.Counter()
    n = collections.Counter()
    for o in c["ops"]:
        agg[(o["name"], o["origin"])] += o["synced_s"]
        n[(o["name"], o["origin"])] += o["calls"]
    ops[a] = (agg, n)
if "b1" in ops and len(ops) == 3:
    keys = set().union(*(ops[a][0] for a in ops))
    rows = []
    for k in keys:
        base = (ops["a1"][0][k] + ops["a2"][0][k]) / 2
        nb = (ops["a1"][1][k] + ops["a2"][1][k]) / 2
        rows.append(dict(op=k[0], origin=k[1], base_s=round(base, 4), fix_s=round(ops["b1"][0][k], 4),
                         delta_s=round(base - ops["b1"][0][k], 4), base_calls=nb, fix_calls=ops["b1"][1][k]))
    rows.sort(key=lambda x: -abs(x["delta_s"]))
    doc["op_delta_top"] = rows[:30]
    doc["synced_delta_s"] = round((doc["arms"]["a1"]["synced_total_s"] + doc["arms"]["a2"]["synced_total_s"]) / 2
                                  - doc["arms"]["b1"]["synced_total_s"], 4)
    doc["base_aa_synced_spread_s"] = round(abs(doc["arms"]["a1"]["synced_total_s"] - doc["arms"]["a2"]["synced_total_s"]), 4)
s = json.dumps(doc, indent=1)
print(s)
if len(sys.argv) > 2:
    Path(sys.argv[2]).write_text(s)
