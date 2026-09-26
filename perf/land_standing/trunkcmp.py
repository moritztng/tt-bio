#!/usr/bin/env python3
"""Grade TT_BIO_TRIATT_DIVIDING_K on the TRUNK, where there is no sampler and no seed floor.

Two OFF runs are the control: the trunk is deterministic given its inputs, so they must agree to
zero. Anything the ON run moves beyond that is the lever, measured without the diffusion sampler
amplifying it into a different structure.
"""
import json
import math
import sys


def load(path):
    rec = json.loads(open(path).readline())
    return {t["i"]: t for t in rec["tensors"]}


def rel_l2(a, b):
    num = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    den = math.sqrt(sum(y * y for y in b))
    return num / den if den else float("nan")


off1, on, off2 = (load(p) for p in sys.argv[1:4])
NAME = {0: "s  [1,832,384]", 1: "z  [1,832,832,128]"}
print(f"{'tensor':22s} {'OFF vs OFF (control)':>22s} {'ON vs OFF (the lever)':>22s}")
rows = {}
for i in sorted(off1):
    c = rel_l2(off1[i]["probe"], off2[i]["probe"])
    e = rel_l2(on[i]["probe"], off1[i]["probe"])
    rows[i] = (c, e)
    print(f"{NAME.get(i, str(i)):22s} {c:22.3e} {e:22.3e}")

print("\nglobal scalars (whole tensor, not the probe):")
for i in sorted(off1):
    for k in ("rms", "sum", "absmax"):
        a, b, c = off1[i][k], on[i][k], off2[i][k]
        same = "identical" if a == c else f"OFF pair differ by {abs(a - c):.6g}"
        print(f"  {NAME.get(i, i):22s} {k:7s} off={a:.10g}  on={b:.10g}  [{same}]")

ctrl = max(r[0] for r in rows.values())
eff = max(r[1] for r in rows.values())
print(f"\ncontrol (OFF vs OFF) max rel L2 = {ctrl:.3e}")
print(f"lever   (ON vs OFF)  max rel L2 = {eff:.3e}")
if ctrl == 0:
    print("the trunk is bit-deterministic across runs, so the instrument has NO floor to clear")
print(f"VERDICT: the lever moves the trunk by {eff:.3e} relative, against a control of {ctrl:.3e}")
