#!/usr/bin/env python3
"""What `fusion_pairs.py`'s candidate rule cannot see: producer -> in-place-consumer chains.

`census.py:split_io` returns `[(b, frac) for b in ins if b not in outs_s], outs`. For an in-place
op such as `ttnn.multiply_(o, g)` the destination `o` is in BOTH `ins` and `outs`, so it is dropped
from the read list and recorded as a write only. A buffer the matmul wrote and the multiply then
consumed in place therefore has n_w=2, n_r=0, and `fusion_pairs.py`'s `n_w != 1 or n_r != 1` test
throws it out. The gate operand survives, the accumulator does not.

That is the conservative direction, but it is not zero, and `tt_bio/trimul_tail.py` -- a fused
kernel that already ships -- accounts a saving on exactly this class. This file counts it.
"""
import gzip
import importlib.util
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CENSUS = os.path.join(ROOT, "perf", "b2z2_byte_floor", "census.py")

src = open(CENSUS).read().rstrip()[: -len("main()")]
C = importlib.util.module_from_spec(importlib.util.spec_from_loader("cr", loader=None))
C.__file__ = CENSUS
exec(compile(src, CENSUS, "exec"), C.__dict__)


def site(r):
    fns = [t.split(":", 1)[1] for t in r["tags"]]
    fns = [f for f in fns if f != "__call__"]
    return fns[-1] if fns else r["owner"]


tr = json.load(gzip.open(sys.argv[1] if len(sys.argv) > 1 else
                         os.path.join(ROOT, "perf/b2z2_byte_floor/out/trace_512_wh_c10.json.gz"), "rt"))
rows, bufs = tr["rows"], tr["buffers"]

# raw per-buffer events, and separately the events split_io HIDES (in-place destinations)
ev = defaultdict(list)
hidden = defaultdict(list)          # buf -> [(i, 'inplace')]
for i, r in enumerate(rows):
    if r["op"] in C.NON_OPS:
        continue
    reads, writes = C.split_io(r)
    if r["op"] in C.VIEWS and set(writes) & {b for b, _ in reads}:
        continue
    ins = list(dict.fromkeys(r["in"]))
    for b in writes:
        if b in ins:                       # read-modify-write the rule reports as write-only
            hidden[b].append(i)
    for b, f in reads:
        ev[b].append((i, "r", f))
    for b in writes:
        ev[b].append((i, "w", 1.0))

dram_total = 0.0
for b, es in ev.items():
    v = bufs[b]
    if v["where"] == "DRAM":
        dram_total += sum(f for _, k, f in es if k == "r") * v["bytes"] + \
                      sum(1 for _, k, _ in es if k == "w") * v["bytes"]

extra, seen = [], set()
for b, ins_at in hidden.items():
    v = bufs[b]
    if v["where"] != "DRAM":
        continue
    es = sorted(ev[b])
    n_w = sum(1 for _, k, _ in es if k == "w")
    n_r = sum(1 for _, k, _ in es if k == "r")
    # exactly one real producer write, then exactly one in-place consumer, nothing else
    if n_r != 0 or n_w != 2 or len(ins_at) != 1:
        continue
    wi = [i for i, k, _ in es if k == "w"]
    prod_i, cons_i = wi[0], ins_at[0]
    if prod_i >= cons_i or wi[1] != cons_i:
        continue
    p, c = rows[prod_i], rows[cons_i]
    extra.append({"bytes": v["bytes"], "shape": v["shape"],
                  "producer": p["op"], "producer_site": site(p),
                  "consumer": c["op"], "consumer_site": site(c),
                  "removable_b": 2.0 * v["bytes"]})

by = defaultdict(lambda: {"n": 0, "removable_b": 0.0})
for x in extra:
    k = (x["producer_site"], x["producer"], x["consumer_site"], x["consumer"])
    by[k]["n"] += 1
    by[k]["removable_b"] += x["removable_b"]

tot = sum(x["removable_b"] for x in extra)
print("=" * 100)
print("PRODUCER -> IN-PLACE CONSUMER: the class fusion_pairs.py's n_w==1 and n_r==1 test excludes")
print(f"512 tokens, {tr['arch']}; rules imported from census.py")
print("=" * 100)
print(f"  block DRAM traffic                 {dram_total/1e6:9.1f} MB")
print(f"  hidden single-use accumulators     {len(extra):4d} allocations, {tot/1e6:.1f} MB "
      f"removable = {100*tot/dram_total:.2f} % of the block")
print()
print(f"{'removable MB':>12} {'n':>4}  {'producer':<46} -> {'consumer':<40}")
for k, v in sorted(by.items(), key=lambda kv: -kv[1]["removable_b"]):
    if v["removable_b"] / 1e6 < 1.0:
        continue
    print(f"{v['removable_b']/1e6:>12.1f} {v['n']:>4}  "
          f"{k[1].replace('ttnn.','')+' @'+k[0]:<46} -> {k[3].replace('ttnn.','')+' @'+k[2]:<40}")
json.dump({"block_dram_b": dram_total, "hidden_removable_b": tot,
           "allocations": sorted(extra, key=lambda x: -x["removable_b"])},
          open(os.path.join(HERE, "inplace_gap.json"), "w"), indent=1)
print(f"\nwrote {os.path.join(HERE,'inplace_gap.json')}")
