#!/usr/bin/env python3
"""Per-size ON/OFF summary of a head-major-qkv fold A/B, with the A/A floor beside every signal."""
import json
import statistics as st
import sys

for path in sys.argv[1:]:
    d = json.load(open(path))
    by = {}
    print(f"### {path}")
    for r in d["runs"]:
        if "error" in r:
            print("  ERR", r)
            continue
        hm = r.get("head_major_qkv", {})
        by.setdefault((r["size"], r["arm"]), []).append(r)
        print(f'  {r["size"]:4d} {r["arm"]:9s} fold {r["fold_s"]:8.3f}  blk {r["block_wall_ms"]:10.2f}'
              f'  triatt {r.get("triatt_wall_ms"):10.2f} /{r.get("triatt_calls")}'
              f'  plddt {r["plddt"]}  sha {list(r["cif_sha256"].values())}'
              f'  served {hm.get("served")}/{hm.get("declined")}  load {r["loadavg"][0]}')
    print()
    for size in sorted({k[0] for k in by}):
        off, on = by[(size, "nohmqkv")], by[(size, "hmqkv")]

        def med(rs, k):
            return st.median([r[k] for r in rs])

        def aa(rs, k):
            return abs(rs[0][k] - rs[1][k])

        for k, unit in (("fold_s", "s"), ("block_wall_ms", "ms"), ("triatt_wall_ms", "ms")):
            o, n = med(off, k), med(on, k)
            print(f'  n={size:4d} {k:15s} {o:10.3f} -> {n:10.3f} {unit:2s}  {o / n:.4f}x'
                  f'  delta {o - n:9.3f}  A/A {aa(off, k):.3f} / {aa(on, k):.3f}'
                  f'  signal/noise {abs(o - n) / max(aa(off, k), aa(on, k), 1e-9):6.1f}x')
        print()
