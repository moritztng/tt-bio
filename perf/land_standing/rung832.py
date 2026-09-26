#!/usr/bin/env python3
"""Why does 832 not get a single-chunk k, when 288 does?

The verdict doc's cheapest unblock is "a single-chunk k at 832". At 288 the dividing ladder
serves on its first rung, (288, 288), and that rung is more accurate than the fall-back it
replaces. At 832 the fold picks (416, 416) -- two k chunks -- so the first rung must be refused.
This prints each rung and its refusal so the reason is read rather than assumed.
"""
import json

for f in ("n256_288", "n352_704", "n832_864"):
    d = json.load(open("perf/land_standing/out/khole_fixed/%s.json" % f))
    for r in d["rows"]:
        a = r["arms"]["dividing"]
        print("\nn=%d pad=%d  k_ladder=%s  served=%s"
              % (r["n"], r["padded"], r.get("dividing_k"), a["served"]))
        for rung in a.get("rungs", []):
            print("   q=%-5s k=%-5s kv_bf=%s divides=%-5s served=%s"
                  % (rung.get("q_chunk"), rung.get("k_chunk"), rung.get("kv_bf"),
                     rung.get("k_divides"), rung.get("served")))
        if a.get("rejects"):
            print("   rejects: %s" % a["rejects"])
        if a.get("l1_refusals"):
            print("   l1_refusals: %s" % a["l1_refusals"])
