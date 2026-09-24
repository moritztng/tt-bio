#!/usr/bin/env python3
"""The census linear rows before and after, from two census.py JSONs. No device.

    rows.py <census before.json> <census after.json>

Every `linear`-verb matmul signature, per pair unit (K=2 calls minus K=1 calls), its unit time
and us/call, in both runs, matched on (phase, weight shape, transpose): the input shape is the
thing this branch changes, so it cannot be the key.
"""
import re, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hallgrad"))
from census_table import analyze, b_device  # noqa: E402
import json


def linear_rows(blob):
    out = {}
    for r in analyze(blob)["rows"]:
        if r["origin"] != "linear" or r["op"] not in ("linear", "matmul"):
            continue
        shapes = re.findall(r"\[[^\]]*\]", r["sig"])
        tb = "transpose_b=True" in r["sig"]
        k = (r["phase"], r["op"], shapes[1], tb)
        out[k] = (shapes[0], r["count"], r["time"] * 1e3, r["t_call"] * 1e6)
    return out


a, b = (json.load(open(p)) for p in sys.argv[1:3])
ra, rb = linear_rows(a), linear_rows(b)
ba, bb = b_device(a), b_device(b)
print(f"b before {ba:.5f} s, after {bb:.5f} s")
print("| phase | op | weight | transpose_b | input before | input after | calls | us/call before | us/call after | ms before | ms after | x |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
ta = tb_ = 0.0
for k in sorted(set(ra) | set(rb), key=lambda k: -ra.get(k, (0, 0, 0, 0))[2]):
    x, y = ra.get(k), rb.get(k)
    if not x or not y:
        print(f"| {k} | unmatched: before {x}, after {y} |")
        continue
    ta += x[2]; tb_ += y[2]
    print(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {x[0]} | {y[0]} | {x[1]}/{y[1]} | {x[3]:.0f} | {y[3]:.0f} | "
          f"{x[2]:.2f} | {y[2]:.2f} | {x[3] / y[3]:.2f} |")
print(f"linear-verb matmuls per unit: {ta:.2f} ms before ({ta / ba / 10:.1f} % of b), {tb_:.2f} ms after, saving {ta - tb_:.2f} ms")
