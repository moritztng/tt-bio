#!/usr/bin/env python3
"""Side-by-side firing table for two models, plus the bucket each lever falls in."""
import json
import sys
from pathlib import Path

b = json.load(open(sys.argv[1]))
p = json.load(open(sys.argv[2]))
B = {r["flag"]: r for r in b["rows"]}
P = {r["flag"]: r for r in p["rows"]}

print("meta boltz2      :", {k: v for k, v in b.items() if k not in ("rows",)})
print()
print("meta protenix-v2 :", {k: v for k, v in p.items() if k not in ("rows",)})
print()
hdr = "%-28s %-9s %9s %9s %9s %9s  %s" % (
    "lever", "resolved", "b2_serv", "b2_decl", "px_serv", "px_decl", "bucket")
print(hdr)
print("-" * len(hdr))
rows = []
for flag in B:
    rb, rp = B[flag], P.get(flag, {})
    if rb.get("state") == "not-imported" and rp.get("state") == "not-imported":
        continue
    bs, bd = rb.get("served"), rb.get("declined")
    ps, pd = rp.get("served"), rp.get("declined")
    res = str(rb.get("resolved") or rp.get("resolved"))[:9]

    def n(x):
        return x if isinstance(x, int) else -1
    if n(bs) > 0 and n(ps) > 0:
        bucket = "1 both"
    elif n(bs) > 0 and n(ps) <= 0:
        bucket = "2 B2-ONLY" + (" (px declines %s)" % pd if n(pd) > 0 else " (px never reaches)")
    elif n(bs) <= 0 and n(ps) > 0:
        bucket = "2R PX-ONLY" + (" (b2 declines %s)" % bd if n(bd) > 0 else " (b2 never reaches)")
    elif n(bs) <= 0 and n(ps) <= 0 and (n(bd) > 0 or n(pd) > 0):
        bucket = "3 neither serves"
    else:
        bucket = "3 untouched"
    rows.append((flag, res, bs, bd, ps, pd, bucket))
    print("%-28s %-9s %9s %9s %9s %9s  %s" % (flag, res, bs, bd, ps, pd, bucket))

print()
print("protenix transition channels:", p.get("transition_channels"))
print("BH transition raise bound c <=", p.get("bh_transition_rows_max_c"))
print("boltz2 transition channels:", b.get("transition_channels"))
print()
for label, d in (("boltz2", B), ("protenix-v2", P)):
    for flag, r in d.items():
        if r.get("rejects"):
            print(label, flag, "rejects:", r["rejects"])
