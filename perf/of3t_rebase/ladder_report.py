#!/usr/bin/env python3
"""The block-boundary ladder, three columns per boundary, as amendments 18 and 19 specify.

Run the crop-64 arm at each captured block first, then point this at the arms. It reports, per
boundary:

  (1) the cotangent norm ratio PER TRACK  -- the D37 check. If this is not ~1.000 the arm at
      that boundary is not trustworthy and nothing else on its row should be read.
  (2) the gradient norm ratio PER TRACK   -- the factor itself. Pair is 1.000 at blocks 0 and 23
      and 1.1832 at block 47; the ladder says where it turns on.
  (3) the REFERENCE gradient squared-norm total at that boundary, over the FULL scope (A20:
      compared plus unreached), beside the activation and cotangent norms.

The three outcomes were registered before the ladder existed:
  * the factor turns on where the gradient TOTAL crosses a threshold while activations stay flat
        -> a dtype-range story, and it names the dtype;
  * it turns on at ONE boundary with no regime change in either variable
        -> a code path at that boundary, located;
  * it climbs smoothly with neither
        -> accumulation, and both standing mechanism stories are wrong.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

A14 = 1e-12


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="instrument_a_bundle_LADDER_block*.json")
    ap.add_argument("--dir", type=Path, default=Path("perf/of3t_rebase"))
    ap.add_argument("--out", type=Path, default=Path("perf/of3t_rebase/ladder_report.json"))
    a = ap.parse_args()

    files = sorted(a.dir.glob(a.glob),
                   key=lambda p: int("".join(c for c in p.stem.split("block")[-1]
                                             if c.isdigit()) or 0))
    if not files:
        print(f"no arms matching {a.dir}/{a.glob} yet -- run the crop-64 arm at each captured "
              f"block first")
        return 1
    rows = []
    for f in files:
        d = json.loads(f.read_text())
        blk = d["block"]
        kept = [r for r in d["per_parameter"]
                if r["ref_norm"] >= A14 and r["norm_ratio"] is not None]
        pair = [r["norm_ratio"] for r in kept if r["key"].startswith("pair_stack")]
        sing = [r["norm_ratio"] for r in kept
                if not r["key"].startswith("pair_stack") and r["norm_ratio"] < 10]
        cot = d.get("cotangent_on_device", {})
        rows.append({
            "block": blk,
            "cot_ratio_s": cot.get("ratio_s"), "cot_ratio_z": cot.get("ratio_z"),
            "grad_ratio_single": st.median(sing) if sing else None,
            "grad_ratio_pair": st.median(pair) if pair else None,
            "n_single": len(sing), "n_pair": len(pair),
            "ref_grad_sq_full_scope": d["reach"]["squared_norm_full_scope"],
            "ref_grad_sq_compared": d["reach"]["squared_norm_compared"],
            "s_norm": d["probe"]["s_norm"], "z_norm": d["probe"]["z_norm"],
            "cot_s_norm": d["probe"]["cot_s_norm"], "cot_z_norm": d["probe"]["cot_z_norm"],
            "median_rel": d["summary"]["median"],
        })
    hdr = (f"{'blk':>4} {'cot r_s':>9} {'cot r_z':>9} | {'grad r single':>13} "
           f"{'grad r pair':>11} | {'ref grad sq':>12} {'s_norm':>11} {'z_norm':>11}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        f2 = lambda v, w=9, p=6: (f"{v:>{w}.{p}f}" if isinstance(v, float) else f"{'--':>{w}}")
        print(f"{r['block']:>4} {f2(r['cot_ratio_s'])} {f2(r['cot_ratio_z'])} | "
              f"{f2(r['grad_ratio_single'], 13, 4)} {f2(r['grad_ratio_pair'], 11, 4)} | "
              f"{r['ref_grad_sq_full_scope']:>12.4e} {r['s_norm']:>11.4e} "
              f"{r['z_norm']:>11.4e}")
    bad = [r["block"] for r in rows
           if r["cot_ratio_z"] is not None and abs(r["cot_ratio_z"] - 1) > 1e-3]
    if bad:
        print(f"\nWARNING: cotangent ratio away from 1 at blocks {bad} -- those rows are not "
              f"trustworthy and the D37 check has to be resolved before reading them.")
    a.out.write_text(json.dumps({"rows": rows, "a14_floor": A14}, indent=1) + "\n")
    print("\n->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
