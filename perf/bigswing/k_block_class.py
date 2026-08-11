#!/usr/bin/env python3
"""How much of the 512 aa block is a matmul that ttnn gave in0_block_w = 1 or 2.

ttnn picks the K-block width in create_matmul_1d_systolic_array_program_config
(k_tiles_per_core = div_up(k_tiles, num_cores), then shrunk to divide k_tiles) and in
get_mcast_1d_config (Kt % 2 == 0 ? 2 : 1). At c_z = 256, Kt = 8 against 110 cores, so every
matmul that reaches those factories runs a Kt-deep contraction one tile at a time.
This prices that class against the census, per site, with the split's own bucket attached.
"""
import json, sys, collections

CENSUS = sys.argv[1] if len(sys.argv) > 1 else "perf/bigswing/ops_pv2_512_fast_r4_qb2c0.json"
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "perf/bigswing/split_512_fast_r4_qb2c0.json"
CORES = 110
INV = 480

cen = json.load(open(CENSUS))
spl = json.load(open(SPLIT))
bucket = {(r["op"], r["site"], r["i"]): r["bucket"] for r in spl["rows"]}

MM = {"matmul", "linear", "minimal_matmul"}
tiles = lambda x: -(-int(x) // 32)


def shape_of(t):
    return [int(v) for v in t["shape"]]


rows = []
for i, r in enumerate(cen["records"]):
    op = r.get("op", "")
    if op not in MM or not r.get("in") or r.get("out") is None:
        continue
    a, b = shape_of(r["in"][0]), shape_of(r["in"][1])
    if len(a) < 2 or len(b) < 2:
        continue
    batch_a = 1
    for d in a[:-2]:
        batch_a *= d
    batch_b = 1
    for d in b[:-2]:
        batch_b *= d
    mt, kt, nt = batch_a * tiles(a[-2]), tiles(a[-1]), tiles(b[-1])
    s = r.get("s")
    rows.append(dict(i=r.get("i", i), op=op, site=r.get("site", "?"), s=(s or 0.0),
                     timed=s is not None, m=mt, k=kt, n=nt, batch_b=batch_b,
                     out_buf=r["out"]["buf"], in_buf=r["in"][0]["buf"],
                     bucket=bucket.get((op, r.get("site", "?"), r.get("i", i)), "?")))

by_site = collections.defaultdict(lambda: dict(s=0.0, n=0, k=set(), m=set(), nn=set(), buckets=set(), out=set()))
tot_mm = tot_narrow_k = 0.0
for r in rows:
    tot_mm += r["s"]
    if r["k"] < CORES and r["batch_b"] == 1:
        tot_narrow_k += r["s"]
        d = by_site[(r["op"], r["site"])]
        d["s"] += r["s"]; d["n"] += 1
        d["k"].add(r["k"]); d["m"].add(r["m"]); d["nn"].add(r["n"])
        d["buckets"].add(r["bucket"]); d["out"].add(r["out_buf"])

print("census %s" % CENSUS)
print("block wall %.3f ms   fold %.3f s" % (cen["block_wall_s"] * 1e3, cen["block_wall_s"] * INV))
print("matmul-class rows %d  timed %d" % (len(rows), sum(r["timed"] for r in rows)))
print("all matmul-class time      %8.3f ms/block  %7.3f s/fold" % (tot_mm * 1e3, tot_mm * INV))
print("  of which Kt < %d cores, in1 unbatched: %8.3f ms/block  %7.3f s/fold  (%.1f %% of block)"
      % (CORES, tot_narrow_k * 1e3, tot_narrow_k * INV, 100 * tot_narrow_k / cen["block_wall_s"]))
print()
hdr = "%-16s %-22s %3s %8s %8s  %6s %3s %4s  %-6s %s"
print(hdr % ("op", "site", "n", "ms/blk", "s/fold", "m", "k", "n", "out", "bucket"))
for (op, site), d in sorted(by_site.items(), key=lambda kv: -kv[1]["s"]):
    print(hdr % (op, site, d["n"], "%8.3f" % (d["s"] * 1e3), "%8.3f" % (d["s"] * INV),
                 ",".join(map(str, sorted(d["m"])))[:6], ",".join(map(str, sorted(d["k"])))[:3],
                 ",".join(map(str, sorted(d["nn"])))[:4], "/".join(sorted(d["out"])),
                 "/".join(sorted(d["buckets"]))))
