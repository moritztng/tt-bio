"""Attribute device-idle gaps of an lpx-census run to the ttnn call that follows them and the one before.

usage: gaps.py RUNDIR FOLD [TOP]      e.g. gaps.py ~/lpx/census/r2 full 40

Reads RUNDIR/{sig,ops,progs}_FOLD.jsonl (census.py's format). A gap is the idle time between the end of
one device program and the start of the next, both inside one profiler flush batch (a gap across a flush
is the census's own sync + profiler read and is reported separately). Each gap is charged to the call
whose program starts after it, keyed by (op, call site, region), and also to the (previous op -> next op)
pair, which is what tells a host loop from a host sync.
"""
import json, sys
from collections import defaultdict
from pathlib import Path

run, fold = Path(sys.argv[1]), sys.argv[2]
top = int(sys.argv[3]) if len(sys.argv) > 3 else 40
sig = {}
for line in open(run / f"sig_{fold}.jsonl"):
    d = json.loads(line)
    if "op" in d:
        sig[d["sig"]] = (d["op"].replace("ttnn.", ""), d["site"][0] if d["site"] else "?", d["reg"])
owner, cyc = {}, {}
for line in open(run / f"ops_{fold}.jsonl"):
    s, i0, i1, c, st = json.loads(line)
    for i in range(i0, i1):
        owner[i] = s
        cyc[i] = (c, st)
progs = []
for line in open(run / f"progs_{fold}.jsonl"):
    p = json.loads(line)
    if p[-2] is None:
        continue
    progs.append((p[1], p[-2], p[-1], p[0] >> 10, p[3]))   # batch, start, end, op id, kernel ns
progs.sort(key=lambda p: (p[0], p[1]))
by_next, by_pair, by_phase = defaultdict(lambda: [0.0, 0]), defaultdict(lambda: [0.0, 0]), defaultdict(float)
kern = idle = cross = 0.0
hist = defaultdict(lambda: [0.0, 0])
prev = None
for p in progs:
    kern += (p[4] or 0) * 1e-9
    if prev is not None and prev[0] == p[0]:
        g = max(0, p[1] - prev[2]) * 1e-9
        idle += g
        n = sig.get(owner.get(p[3]), ("<unhooked>", "?", "?"))
        q = sig.get(owner.get(prev[3]), ("<unhooked>", "?", "?"))
        by_next[n][0] += g; by_next[n][1] += 1
        by_pair[(q[0] + " @" + q[1], n[0] + " @" + n[1])][0] += g
        by_pair[(q[0] + " @" + q[1], n[0] + " @" + n[1])][1] += 1
        by_phase[n[2].split("/")[0] + "/" + (n[2].split("/")[2] if n[2].count("/") >= 2 else "")] += g
        b = "<10us" if g < 1e-5 else "<100us" if g < 1e-4 else "<1ms" if g < 1e-3 else "<10ms" if g < 1e-2 else ">=10ms"
        hist[b][0] += g; hist[b][1] += 1
    elif prev is not None:
        cross += max(0, p[1] - prev[2]) * 1e-9
    prev = p
span = (progs[-1][2] - progs[0][1]) * 1e-9
print(f"programs {len(progs)}  span {span:.2f}s  kernel(sum) {kern:.2f}s  idle-in-batch {idle:.2f}s  "
      f"across-flush {cross:.2f}s")
print("gap size histogram:", {k: (round(v[0], 2), v[1]) for k, v in sorted(hist.items())})
print("\nby phase/region:")
for k, v in sorted(by_phase.items(), key=lambda x: -x[1])[:15]:
    print(f"  {v:8.2f}s  {k}")
print(f"\ntop {top} by next call (gap charged to the call that starts after it):")
for k, (g, n) in sorted(by_next.items(), key=lambda x: -x[1][0])[:top]:
    print(f"  {g:8.3f}s {n:7d} {1e3 * g / n:8.3f}ms  {k[0]:28s} {k[1]:45s} {k[2][-45:]}")
print(f"\ntop {top} (previous -> next) pairs:")
for k, (g, n) in sorted(by_pair.items(), key=lambda x: -x[1][0])[:top]:
    print(f"  {g:8.3f}s {n:7d} {1e3 * g / n:8.3f}ms  {k[0][:60]:60s} -> {k[1][:60]}")
