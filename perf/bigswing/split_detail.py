#!/usr/bin/env python3
"""Detail behind the three-way split: shapes on the big rows, per-body FLOP, untimed-row pricing.

The untimed rows are the §10b failure mode -- an excluded matmul absorbed as a zero made four
earlier go/no-go verdicts unsound -- so they get priced at a floor here rather than ignored.
"""
import json
import sys
from collections import defaultdict

sys.path.insert(0, "perf/bigswing")
from three_way_split import ROOFS, body_of, owner_map, read_roof_bytes, row_flop, tb  # noqa: E402

sp = json.load(open(sys.argv[1]))
cen = json.load(open(sys.argv[2]))
own = owner_map("tt_bio/tenstorrent.py")
byi = {r["i"]: r for r in cen["records"]}

print("TOP 14 ROWS, with shapes")
for r in sp["rows"][:14]:
    c = byi[r["i"]]
    ins = "; ".join("{}/{}".format(i["shape"], i["buf"]) for i in (c.get("in") or []))
    o = c.get("out") or {}
    print("{:7.3f}ms {:<26} {:<10} {:6.2f}TF ae{:.2f} be{:.2f} rd{:6.1f} w{:6.1f}MB {:<12} {}".format(
        1e3 * r["s"], r["op"], r["bucket"], r["tflops"], r["arith_eff"], r["byte_eff"],
        r["rd_MB"], r["w_MB"], r["body"], r["site"]))
    print("          in: {}  out {}/{}".format(ins[:130], o.get("shape"), o.get("buf")))

print("\nPER-BODY FLOP per fold (480 invocations)")
bf, bt = defaultdict(float), defaultdict(float)
for c in cen["records"]:
    if c.get("error") or not c.get("s"):
        continue
    b = body_of(own, [c.get("site")] + list(c.get("chain") or []))
    bf[b] += row_flop(c) * 480
    bt[b] += c["s"] * 480
for b in sorted(bf, key=lambda x: -bt[x]):
    print("  {:<18} {:.4e} FLOP  {:8.3f} s  {:6.2f} TF/s".format(
        b, bf[b], bt[b], bf[b] / bt[b] / 1e12 if bt[b] else 0))
f2 = sum(bf[b] for b in ("trimul", "tri_att"))
t2 = sum(bt[b] for b in ("trimul", "tri_att"))
print("  TWO BODIES         {:.4e} FLOP  {:8.3f} s  {:6.2f} TF/s   (pf_floor census: 5.2880e14)".format(
    f2, t2, f2 / t2 / 1e12))

print("\nUNTIMED ROWS -- excluded from the sum, priced at a per-row floor")
tot = 0.0
for c in cen["records"]:
    if not c.get("error"):
        continue
    f = row_flop(c)
    rd, rl1 = read_roof_bytes(c)
    w = tb(c.get("out"))
    wroof = ROOFS["w_matmul_dram"] if c["op"] in ("matmul", "linear", "minimal_matmul") \
        else ROOFS["w_unary_dram"]
    fl = max(f / 40.40e12, rd / ROOFS["r_dram"], w / wroof)
    tot += fl
    b = body_of(own, [c.get("site")] + list(c.get("chain") or []))
    o = c.get("out") or {}
    print("  {:<15} {:<10} out {}/{} flop {:.3e} floor {:6.3f}ms  {}  {}".format(
        c["op"], b, o.get("shape"), o.get("buf"), f, 1e3 * fl, c["site"], c["error"][:55]))
print("  SUM of untimed per-row floors: {:.3f} ms/block = {:.3f} s/fold (LOWER bound on their cost)"
      .format(1e3 * tot, tot * 480))
