#!/usr/bin/env python3
"""Every statistic the report needs for one section, across every arm."""
import json
import sys

d = json.load(open(sys.argv[1]))
want = sys.argv[2]
for arm, p in d["pairs"].items():
    for s in p["sets"]:
        if s["set"] == want:
            print(f"{arm:<48}{s['mass_weighted_rel_l2']:>13.6e}"
                  f"{s['mass_weighted_norm_ratio']:>10.4f}{(s[chr(39)+chr(39)] if False else (s["mass_weighted_cos"] if s["mass_weighted_cos"] is not None else float("nan"))):>10.6f}"
                  f"  med {s['median_rel_l2_over_tensors']:.4e}"
                  f"  over-bar {s['n_over_per_tensor_bar']}/{s['n']}"
                  f"  worst {s['worst_rel_l2']:.4e} {s['worst_tensor'].split('diffusion_module.')[-1]}")
