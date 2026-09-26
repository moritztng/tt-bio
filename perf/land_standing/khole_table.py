#!/usr/bin/env python3
"""The per-call float64 grading, read across lengths so the shipped route is its own reference.

The question the trunk comparison cannot answer is whether the route the dividing-k lever
unlocks at 832 is a WORSE kernel or merely a different one. These files already hold the answer:
each row grades both arms against a float64 reference at one call, so a length where the SHIPPED
arm serves gives the distance main already accepts.
"""
import json

for f in ("n256_288", "n352_704", "n832_864"):
    d = json.load(open("perf/land_standing/out/khole_fixed/%s.json" % f))
    for r in d["rows"]:
        for arm in ("shipped", "dividing"):
            a = r["arms"][arm]
            rel = a.get("rel_vs_f64") or {}
            print("n=%5d pad=%5d %-9s served=%-5s shipped_k=%-5s k_ladder=%-18s "
                  "rel_vs_f64(after_bias)=%s"
                  % (r["n"], r["padded"], arm, a["served"], r.get("shipped_k"),
                     r.get("dividing_k"), rel.get("scale_after_bias")))
