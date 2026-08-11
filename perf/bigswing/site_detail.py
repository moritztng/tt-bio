#!/usr/bin/env python3
import json, sys, collections
cen = json.load(open("perf/bigswing/ops_pv2_512_fast_r4_qb2c0.json"))
INV = 480
want = set(sys.argv[1:]) or {"tenstorrent.py:2528", "tenstorrent.py:2539", "tenstorrent.py:2550"}
tiles = lambda x: -(-int(x) // 32)
for r in cen["records"]:
    if r.get("site") not in want:
        continue
    a = [int(v) for v in r["in"][0]["shape"]]
    b = [int(v) for v in r["in"][1]["shape"]]
    o = [int(v) for v in r["out"]["shape"]]
    ba = 1
    for d in a[:-2]:
        ba *= d
    s = r.get("s")
    flop = 2.0 * ba * a[-2] * a[-1] * b[-1]
    print("%-22s i=%-4s %-14s a=%-22s b=%-12s out=%-22s %s/%s  s=%s  TF/s=%s  mt=%d kt=%d nt=%d" % (
        r["site"], r.get("i"), r.get("op"), a, b, o, r["in"][0]["buf"], r["out"]["buf"],
        ("%.4f ms" % (s * 1e3)) if s is not None else "UNTIMED",
        ("%.2f" % (flop / s / 1e12)) if s else "-", ba * tiles(a[-2]), tiles(a[-1]), tiles(b[-1])))
tot = collections.defaultdict(float)
for r in cen["records"]:
    if r.get("site") in want and r.get("s"):
        tot[r["site"]] += r["s"]
for k, v in tot.items():
    print("TOTAL %-22s %8.3f ms/block  %7.3f s/fold" % (k, v * 1e3, v * INV))
