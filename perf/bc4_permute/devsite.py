#!/usr/bin/env python3
"""Device kernel time per (outermost ttnn verb, call site) from a perf/bc4_devgap devprof capture.
    python perf/bc4_permute/devsite.py <devprof.json.gz> <iteration> [out.json]
A program is named by the span holding its op id (runtime_id >> 10); unmapped programs are reported apart.
"""
import bisect, collections, gzip, json, sys

d = json.load(gzip.open(sys.argv[1]))
it = int(sys.argv[2])
C = {c: i for i, c in enumerate(d["program_cols"])}
sp = sorted(d["spans"])  # a replayed program keeps its capture iteration's op id
starts = [s[0] for s in sp]
agg = collections.defaultdict(lambda: [0, 0.0])
for p in d["programs"]:
    if p[C["iteration"]] != it:
        continue
    oid = p[C["runtime_id"]] >> 10
    i = bisect.bisect_right(starts, oid) - 1
    key = (sp[i][2], sp[i][3]) if i >= 0 and sp[i][0] <= oid < sp[i][1] else ("unmapped", "")
    agg[key][0] += 1
    agg[key][1] += p[C["k_ns"]] * 1e-9
calls = collections.Counter((s[2], s[3]) for s in sp if s[4] == it)
rows = [dict(verb=k[0], site=k[1], calls=calls[k], programs=a[0], device_ms=round(a[1] * 1e3, 3))
        for k, a in sorted(agg.items(), key=lambda kv: -kv[1][1])]
print("kernel s/it", round(sum(r["device_ms"] for r in rows) / 1e3, 4))
for r in rows:
    if r["verb"] in ("permute", "transpose", "concat", "to_layout") or "permute" in r["site"] or "transpose" in r["site"]:
        print("%-10s %-50s calls %5d progs %5d %8.2f ms" % (r["verb"], r["site"], r["calls"], r["programs"], r["device_ms"]))
if len(sys.argv) > 3:
    json.dump(dict(source=sys.argv[1], iteration=it, sha=d.get("sha"), chip=d.get("chip"), rows=rows), open(sys.argv[3], "w"), indent=1)
