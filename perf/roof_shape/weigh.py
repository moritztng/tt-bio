#!/usr/bin/env python3
"""Join the shape-honest rates to the FLOP census and read off the arithmetic floor.

The floor is a sum of times, not an average of rates, so the aggregate is the FLOP-weighted
HARMONIC mean of the per-class fractions:

    effective rate = covered FLOP / sum_i (FLOP_i / rate_i)
    mean fraction  = effective rate / the same session's dense cube

The arithmetic mean of the fractions is printed beside it because it is the number a reader
expects, and it is 1.5x too optimistic here. Use the harmonic one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# census shape (batch, M, K, N) -> (label, [arms that compute it; the fastest wins])
# Two census rows share one arm where the fold shares one kernel: the triangle attention's
# QK^T and AV are one fused SDPA call, and both 128->128 output projections are the same matmul.
CLASSES = [
    ("TriangleMultiplication in-proj  [g_a|g_b|p_a|p_b|g_out]", [(1, 262144, 128, 640)],
     ["trimul_in_whole", "trimul_in_h64", "trimul_in_h64_cg", "trimul_in_flat"]),
    ("DiffusionTransformer token linear 768x768", [(1, 512, 768, 768)],
     ["dit_768x768", "dit_768x768_l1"]),
    # `triatt_in_shipped` is `triatt_qkv.qkvgb_heads`, the generic_op the fold actually issues for
    # this shape; the three stock `ttnn.linear` arms below it are what the class was priced by
    # until `perf/roof_triatt_rate/` measured the two side by side and the shipped kernel came out
    # 1.65x ahead. Same for the SDPA and out-projection rows. A roofs file without the shipped
    # arms is unaffected: `class_rates` takes the max over the arms it finds.
    ("TriangleAttention in-proj  [q|k|v|g|bias]", [(1, 262144, 128, 544)],
     ["triatt_in_whole", "triatt_in_h64", "triatt_in_flat", "triatt_in_shipped"]),
    ("TriangleMultiplication triangle product", [(128, 512, 512, 512)], ["trimul_einsum"]),
    ("TriangleAttention fused SDPA  (QK^T and AV)",
     [(2048, 512, 32, 512), (2048, 512, 512, 32)],
     ["triatt_sdpa_q32", "triatt_sdpa_q64", "triatt_sdpa_q128", "triatt_sdpa_q256",
      "triatt_sdpa_q512", "triatt_sdpa_q128_k128", "triatt_sdpa_q256_k256",
      "triatt_sdpa_shipped"]),
    ("pair Transition fc1 / fc2", [(16, 512, 128, 512)], ["trans_fc12"]),
    ("DiffusionTransformer token linear 768x1536", [(1, 512, 768, 1536)],
     ["dit_768x1536", "dit_768x1536_l1"]),
    ("DiffusionTransformer token linear 768x3072", [(1, 512, 768, 3072)],
     ["dit_768x3072", "dit_768x3072_l1"]),
    ("pair Transition fc3", [(16, 512, 512, 128)], ["trans_fc3"]),
    ("OuterProductMean depth contraction", [(1, 16384, 1024, 16384)], ["opm"]),
    ("DiffusionTransformer token linear 1536x768", [(1, 512, 1536, 768)],
     ["dit_1536x768", "dit_1536x768_l1"]),
    ("TriangleMultiplication out-proj", [(512, 512, 128, 128)],
     ["pair_out128_whole", "pair_out128_h64", "pair_out128_flat", "pair_out128_flat_cg"]),
    ("TriangleAttention out-proj", [(1, 262144, 128, 128)],
     ["pair_out128_whole", "pair_out128_h64", "pair_out128_flat", "pair_out128_flat_cg",
      "triatt_out_shipped"]),
    ("PairWeightedAveraging", [(1024, 32, 512, 512)], ["pwa", "pwa_2d", "pwa_flat"]),
    ("atom transformer linear", [(140, 128, 128, 256)], ["atom", "atom_l1", "atom_flat"]),
]

# The fold's own numbers, from perf/roof_budget/ROOF_BUDGET.md. Not re-measured here.
FOLD_TFLOP = 219.49
TRAFFIC_FLOOR_S = 6.934
QB2_CUBE = 104.93


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roofs", type=Path, required=True)
    ap.add_argument("--census", type=Path, default=HERE / "shape_census.json")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    roofs = json.loads(a.roofs.read_text())
    cube = roofs["cube4096_TFLOPs"]
    rate = {r["arm"]: r["TFLOPs"] for r in roofs["rows"]}
    cen = json.loads(a.census.read_text())
    tf = {(r["batch"], r["M"], r["K"], r["N"]): r["TFLOP_per_fold"] for r in cen["rows"]}
    total = cen["total_TFLOP"]

    rows, covered, seconds = [], 0.0, 0.0
    for label, shapes, arms in CLASSES:
        cand = [(rate[n], n) for n in arms if n in rate]
        if not cand:
            print("UNCOVERED %s -- every arm refused" % label)
            continue
        r, arm = max(cand)
        f = sum(tf[s] for s in shapes)
        covered += f
        seconds += f / r
        rows.append({"class": label, "arm": arm, "TFLOPs": r, "pct_of_cube": 100 * r / cube,
                     "TFLOP_per_fold": f, "pct_of_fold": 100 * f / total, "s_per_fold": f / r})

    rows.sort(key=lambda x: -x["TFLOP_per_fold"])
    eff = covered / seconds
    mean_frac = eff / cube
    flat_mean = sum(x["pct_of_cube"] * x["TFLOP_per_fold"] for x in rows) / covered

    w = max(len(x["class"]) for x in rows)
    print("%s  %s  %s" % (roofs["host"], roofs["arch"], roofs["grid"]))
    print("cube %.2f TFLOP/s, covering %.2f of %.2f census TFLOP (%.1f %%)\n"
          % (cube, covered, total, 100 * covered / total))
    print("%-*s  %-22s %9s %9s %9s %8s" % (w, "class", "arm", "TFLOP/s", "% of cube",
                                           "TF/fold", "% fold"))
    for x in rows:
        print("%-*s  %-22s %9.2f %8.1f %% %9.2f %7.2f %%"
              % (w, x["class"], x["arm"], x["TFLOPs"], x["pct_of_cube"],
                 x["TFLOP_per_fold"], x["pct_of_fold"]))
    print("\nFLOP-weighted mean fraction of cube: %.1f %% (harmonic, the one a floor uses)"
          % (100 * mean_frac))
    print("                   arithmetic mean:  %.1f %% -- do not use" % flat_mean)
    print("effective shape-honest rate on this part: %.2f TFLOP/s" % eff)

    out = {"host": roofs["host"], "arch": roofs["arch"], "cube_TFLOPs": cube,
           "covered_TFLOP": covered, "census_TFLOP": total,
           "coverage_pct": 100 * covered / total,
           "effective_TFLOPs": eff, "mean_fraction_pct": 100 * mean_frac,
           "arithmetic_mean_pct": flat_mean, "rows": rows}
    if "BLACK" in roofs["arch"]:
        floor = FOLD_TFLOP / (QB2_CUBE * mean_frac)
        out["qb2_arithmetic_floor_s"] = floor
        out["qb2_cube_TFLOPs"] = QB2_CUBE
        print("\narithmetic floor at the budget's part (qb2 p300c, cube %.2f TFLOP/s):"
              % QB2_CUBE)
        print("  %.2f TFLOP / (%.2f x %.4f) = %.3f s" % (FOLD_TFLOP, QB2_CUBE, mean_frac, floor))
        print("  traffic floor %.3f s -> %s binds"
              % (TRAFFIC_FLOOR_S,
                 "arithmetic" if floor > TRAFFIC_FLOOR_S * 1.1 else
                 "traffic" if floor < TRAFFIC_FLOOR_S / 1.1 else "neither alone"))
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
