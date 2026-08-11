#!/usr/bin/env python3
"""Is the census's per-site total dominated by one outlier invocation?

bench() holds `reps` extra outputs live beside the block, so the same op measured early in the
block (when L1 still holds the trimul/tri_att working set) can drop down the reps ladder and be
timed under memory pressure that the same call does not see 5 invocations later. A site total
that is then multiplied by 480 carries that outlier 480 times.
"""
import json, sys, statistics, collections

CEN = sys.argv[1] if len(sys.argv) > 1 else "perf/bigswing/ops_pv2_512_fast_r4_qb2c0.json"
INV = 480
cen = json.load(open(CEN))
by = collections.defaultdict(list)
for r in cen["records"]:
    if r.get("s") is None:
        continue
    key = (r["op"], r.get("site", "?"), tuple(map(int, r["out"]["shape"])) if r.get("out") else ())
    by[key].append((r["i"], r["s"], r.get("reps_used")))

tot_excess = 0.0
rows = []
for key, v in by.items():
    if len(v) < 4:
        continue
    v.sort()
    ss = [x[1] for x in v]
    med = statistics.median(ss)
    mx = max(ss)
    if mx < 3 * med or mx * 1e3 < 0.2:
        continue
    excess = sum(s - med for s in ss if s > 3 * med)
    tot_excess += excess
    rows.append((excess, key, len(v), med, mx, [x[2] for x in v]))

print("census %s   block wall %.3f ms" % (CEN, cen["block_wall_s"] * 1e3))
print("sites with an invocation > 3x the site median, n >= 4:")
print("%-16s %-22s %-24s %3s %10s %10s %10s %s" % (
    "op", "site", "out", "n", "median ms", "max ms", "excess s/fold", "reps_used"))
for excess, (op, site, out), n, med, mx, reps in sorted(rows, reverse=True):
    print("%-16s %-22s %-24s %3d %10.4f %10.4f %10.3f %s" % (
        op, site, str(list(out))[:24], n, med * 1e3, mx * 1e3, excess * INV, reps))
print()
print("total excess if every site is priced at its median instead of its sum: %.3f s/fold"
      % (tot_excess * INV))
print("census sum %.3f ms/block = %.3f s/fold; median-priced %.3f s/fold" % (
    cen["sum_s"] * 1e3, cen["sum_s"] * INV, (cen["sum_s"] - tot_excess) * INV))
