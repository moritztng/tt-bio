#!/usr/bin/env python3
"""DRAM bytes per `MSALayer.__call__`, counted off a graph capture instead of the profiler.

Why not the profiler's `ops_perf` columns, which the census used: finding 7 of
`b2z2-msa-layer-census` is that the capture reports `1x1x32x32` for a class of in-place
`BinaryNg` ADDs, so 14 programs a call contribute exactly ZERO bytes to every model built off
that CSV while really moving ~200 MB each. A graph capture does not have that hole -- it sees
the buffer, and the buffer is the right unit anyway (`ttnn-graph-byte-count-must-dedupe-buffer-
not-tensor-id`: ttnn hands a reshape or a slice a fresh tensor id over the SAME buffer, so
deduping by tensor id charges a metadata view a full DRAM read it never performs).

The counting rule is the red team's, unchanged: per program, each operand buffer once. A
buffer is READ by every top-level op that consumes it and is not the op that allocated it, and
WRITTEN once by the op that allocated it. Totals are DRAM only; L1 is reported beside them
because the whole point of a residency lever is to move traffic from one to the other.

    python3 perf/b2z2_msa_move/bytes_move.py ops_wh_c9.graph.json.gz --out bytes_wh_c9.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "perf" / "b2x_difflayer"))

from itemize import top_level_spans                                           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("graph")
ap.add_argument("--out", required=True)
ap.add_argument("--label", default="")
a = ap.parse_args()

op = gzip.open if a.graph.endswith(".gz") else open
with op(a.graph, "rt") as fh:
    nodes = json.load(fh)

by_counter = {n["counter"]: n for n in nodes}
ops, owner = top_level_spans(nodes)

buf_size, buf_kind, buf_alloc = {}, {}, {}
for n in nodes:
    if n["node_type"] == "buffer":
        p = n.get("params") or {}
        buf_size[n["counter"]] = int(p.get("size", 0) or 0)
        buf_kind[n["counter"]] = str(p.get("type", ""))
        buf_alloc[n["counter"]] = None
for n in nodes:
    if n["node_type"] != "buffer_allocate":
        continue
    for c in (n.get("connections") or []):
        if c in buf_size:
            buf_alloc[c] = owner.get(n["counter"])

tensors_of = defaultdict(list)
for b in buf_size:
    for c in (by_counter[b].get("connections") or []):
        if by_counter.get(c, {}).get("node_type") == "tensor":
            tensors_of[b].append(c)

rd = defaultdict(float)
wr = defaultdict(float)
rd_l1 = defaultdict(float)
wr_l1 = defaultdict(float)
for b, size in buf_size.items():
    seen = set()
    for t in tensors_of[b]:
        for c in (by_counter[t].get("connections") or []):
            o = owner.get(c)
            if o is not None:
                seen.add(o)
    seen.discard(buf_alloc.get(b))
    dram = buf_kind[b] == "DRAM"
    for i in seen:
        (rd if dram else rd_l1)[i] += size
    if buf_alloc.get(b) is not None:
        (wr if dram else wr_l1)[buf_alloc[b]] += size

agg = defaultdict(lambda: {"n": 0, "dram_rd": 0.0, "dram_wr": 0.0, "l1_rd": 0.0, "l1_wr": 0.0})
for i, o in enumerate(ops):
    e = agg[o["name"]]
    e["n"] += 1
    e["dram_rd"] += rd[i]
    e["dram_wr"] += wr[i]
    e["l1_rd"] += rd_l1[i]
    e["l1_wr"] += wr_l1[i]

tot = {k: sum(v[k] for v in agg.values()) for k in ("dram_rd", "dram_wr", "l1_rd", "l1_wr")}
out = {
    "label": a.label, "graph": Path(a.graph).name, "n_top_level_ops": len(ops),
    "totals_MB": {k: round(v / 1e6, 3) for k, v in tot.items()},
    "dram_total_GB": round((tot["dram_rd"] + tot["dram_wr"]) / 1e9, 4),
    "by_op": {k: {"n": v["n"], **{kk: round(vv / 1e6, 3) for kk, vv in v.items() if kk != "n"}}
              for k, v in sorted(agg.items(), key=lambda kv: -(kv[1]["dram_rd"] + kv[1]["dram_wr"]))},
}
Path(a.out).write_text(json.dumps(out, indent=1))

print(f"{len(ops)} top-level ttnn ops")
print(f"DRAM read  {tot['dram_rd']/1e6:10.1f} MB")
print(f"DRAM write {tot['dram_wr']/1e6:10.1f} MB")
print(f"DRAM TOTAL {out['dram_total_GB']:10.4f} GB   (the currency a clone roof is measured in)")
print(f"L1   read  {tot['l1_rd']/1e6:10.1f} MB   write {tot['l1_wr']/1e6:10.1f} MB")
print()
print(f"{'op':<40}{'n':>4}{'dram_rd':>10}{'dram_wr':>10}{'l1_rd':>9}{'l1_wr':>9}")
for k, v in out["by_op"].items():
    if v["dram_rd"] + v["dram_wr"] + v["l1_rd"] + v["l1_wr"] < 1.0:
        continue
    print(f"{k:<40}{v['n']:>4}{v['dram_rd']:>10.1f}{v['dram_wr']:>10.1f}{v['l1_rd']:>9.1f}{v['l1_wr']:>9.1f}")
