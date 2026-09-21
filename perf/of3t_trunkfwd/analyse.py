#!/usr/bin/env python3
"""Turn the three arms into the numbers the reading rests on, and hand them to the campaign's
own recomputation check.

Three things happen here and each one exists because a reading would be unsafe without it:

  * a PER-BLOCK SIDECAR in the campaign's `diff_norm` / `ref_norm` / `section` schema, so
    `perf/of3t_orchestrator/recompute_from_sidecar.py` can recompute the mass-weighted headline
    independently of this row's arithmetic. A result file keeping only extremes cannot be
    re-analysed, and that is a rule this campaign wrote after it lost a figure to it.
  * the ACCUMULATION EXPONENT, fitted. "It compounds over 48 blocks" is a growth ratio, and a
    growth ratio is not amplification without a baseline: k^0.5 is what independent rounding
    does, k^1.0 is a coherent bias. Both exponents are printed beside the fit.
  * the RATIO TABLE against upstream's own bf16, per block AND composed, because a per-block
    figure divided out of a 48-block one would assume the very linearity being measured.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


def sidecar_rows(per_block, arm):
    rows = []
    for x in per_block:
        for curve in ("alone_from_captured_input", "composed_running"):
            for track in ("s_masked", "z_masked"):
                m = x[curve][track]
                rows.append({"tensor": f"{arm}.block{x['block']:02d}.{track}",
                             "section": f"{arm}.{curve}.{track}",
                             "block": x["block"],
                             "diff_norm": m["rel_l2"] * m["ref_norm"],
                             "ref_norm": m["ref_norm"],
                             "rel_l2": m["rel_l2"], "cos": m["cos"],
                             "norm_ratio": m["norm_ratio"]})
    return rows


def fit_exponent(ys):
    """Least squares slope of log(rel) against log(depth). depth = index + 1."""
    xs = [math.log(i + 1) for i in range(len(ys))]
    ls = [math.log(y) for y in ys]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ls) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ls))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fwd", default="perf/of3t_trunkfwd/TRUNK_FORWARD.json")
    ap.add_argument("--floor", default="perf/of3t_trunkfwd/UPSTREAM_FLOOR.json")
    ap.add_argument("--sidecar", default="perf/of3t_trunkfwd/per_block_COMPOSITION.json")
    ap.add_argument("--out", default="perf/of3t_trunkfwd/READING.json")
    ap.add_argument("--recompute",
                    default="perf/of3t_orchestrator/recompute_from_sidecar.py")
    a = ap.parse_args()

    d = json.load(open(a.fwd))
    f = json.load(open(a.floor))
    rows = (sidecar_rows(d["COMPOSITION"]["per_block"], "shipped")
            + sidecar_rows(d["COMPOSITION_transpose_bias_flipped"]["per_block"], "tb_flipped"))
    Path(a.sidecar).write_text(json.dumps(rows, indent=1) + "\n")
    print(f"wrote {a.sidecar} ({len(rows)} rows)")

    rep = {"what": "the reading of of3t-trunkfwd's three arms",
           "inputs": {"forward": a.fwd, "floor": a.floor, "sidecar": a.sidecar}}

    ship_z = d["SHIPPED"]["z_masked"]["rel_l2"]
    ship_s = d["SHIPPED"]["s_masked"]["rel_l2"]
    bf16_z = f["bf16_autocast"]["z_masked"]["rel_l2"]
    bf16_s = f["bf16_autocast"]["s_masked"]["rel_l2"]
    pb_bf16_z = f["per_block_bf16_autocast"]["median_z_masked"]
    pb_bf16_s = f["per_block_bf16_autocast"]["median_s_masked"]
    pb_ours_z = d["COMPOSITION"]["median_alone_z_masked"]
    pb_ours_s = sorted(x["alone_from_captured_input"]["s_masked"]["rel_l2"]
                       for x in d["COMPOSITION"]["per_block"])[24]
    tb_z = d["LEVER_transpose_bias_flipped"]["z_masked"]["rel_l2"]
    tb_pb_z = d["COMPOSITION_transpose_bias_flipped"]["median_alone_z_masked"]

    rep["bars"] = {
        "instrument_floor_masked": {"z": f["float64"]["z_masked"]["rel_l2"],
                                    "s": f["float64"]["s_masked"]["rel_l2"],
                                    "what": "upstream's own blocks re-composed in float64 from "
                                            "the captured input, masked. Everything the "
                                            "instrument itself contributes."},
        "upstream_bf16_composed_masked": {"z": bf16_z, "s": bf16_s},
        "upstream_bf16_per_block_median_masked": {"z": pb_bf16_z, "s": pb_bf16_s},
        "upstream_float64_per_block_median_masked": {
            "z": f["per_block_float64"]["median_z_masked"],
            "s": f["per_block_float64"]["median_s_masked"]},
        "zero_model_A16": d["CONTROL_zero_model"]["z_masked"]["rel_l2"],
        "per_tensor_bar_3d": 0.05}

    rep["ratios_over_upstream_bf16"] = {
        "shipped_composed_z": ship_z / bf16_z, "shipped_composed_s": ship_s / bf16_s,
        "shipped_per_block_z": pb_ours_z / pb_bf16_z,
        "shipped_per_block_s": pb_ours_s / pb_bf16_s,
        "transpose_bias_flipped_composed_z": tb_z / bf16_z,
        "transpose_bias_flipped_per_block_z": tb_pb_z / pb_bf16_z,
        "why": "measured per block AND composed. Dividing one out of the other would assume the "
               "accumulation is linear, which is what the exponents below measure."}

    ours = [x["composed_running"]["z_masked"]["rel_l2"] for x in d["COMPOSITION"]["per_block"]]
    tbs = [x["composed_running"]["z_masked"]["rel_l2"]
           for x in d["COMPOSITION_transpose_bias_flipped"]["per_block"]]
    up = [r["z_masked"]["rel_l2"] for r in f["per_block_bf16_autocast"]["per_block"]]
    rep["accumulation"] = {
        "shipped_exponent_p": fit_exponent(ours),
        "transpose_bias_flipped_exponent_p": fit_exponent(tbs),
        "random_walk_p": 0.5, "coherent_p": 1.0,
        "shipped_growth_ratio_composed_over_per_block_median": ours[-1] / pb_ours_z,
        "upstream_bf16_growth_ratio_composed_over_per_block_median": bf16_z / pb_bf16_z,
        "why": "p is the least-squares slope of log(composed rel) against log(depth) over the 48 "
               "blocks. p = 0.5 is independent rounding, p = 1.0 is a coherent per-block bias. "
               "The growth ratios are stated too because the exponent is a fit and the ratio is "
               "the raw observation.",
        "upstream_per_block_bf16_median_for_scale": sorted(up)[len(up) // 2]}

    rep["tape"] = {"taped_vs_shipped_z": d["TAPED_vs_SHIPPED"]["z_masked"]["rel_l2"],
                   "taped_vs_shipped_s": d["TAPED_vs_SHIPPED"]["s_masked"]["rel_l2"],
                   "bit_identical": d["TAPED_vs_SHIPPED"]["bit_identical"],
                   "share_of_the_disagreement_z":
                       d["TAPED_vs_SHIPPED"]["z_masked"]["rel_l2"] / ship_z}

    rep["reproduction_of_D87"] = {
        "published_by_of3t_pairformer": {"z_masked": 2.796859e-01, "s_masked": 3.185052e-02},
        "this_row_same_configuration_taped": {
            "z_masked": d["PAIRFORMER_ARM_CONFIG_TAPED"]["z_masked"]["rel_l2"],
            "s_masked": d["PAIRFORMER_ARM_CONFIG_TAPED"]["s_masked"]["rel_l2"]},
        "why": "an independent instrument, a different script, the same boundary. If these did "
               "not agree the D87 forward figure would be in question before anything else."}

    rep["composition_is_the_models_not_the_instruments"] = {
        "per_block_loop_vs_module_call_z":
            d["COMPOSITION"]["loop_vs_module_call"]["z_masked"]["rel_l2"],
        "per_block_loop_vs_module_call_s":
            d["COMPOSITION"]["loop_vs_module_call"]["s_masked"]["rel_l2"],
        "why": "arm 3's per-block loop composed to the end is scored against the single "
               "`Pairformer.__call__`. Zero means the loop IS the shipped composition, so a "
               "per-block figure and the composed figure are of one function."}
    rep["D86_transpose_fingerprint"] = d["D86_transpose_fingerprint"]

    Path(a.out).write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rep.items()
                      if k in ("ratios_over_upstream_bf16", "accumulation", "tape",
                               "reproduction_of_D87",
                               "composition_is_the_models_not_the_instruments")}, indent=1))

    # The checker belongs to `of3t-orchestrator`, so it is read out of that branch when it is
    # absent from the tree rather than copied into this one. A second copy of a shared guard is
    # a second thing to keep in step.
    script = a.recompute
    if not Path(script).is_file():
        blob = subprocess.run(["git", "show", f"origin/wk/of3t-orchestrator:{a.recompute}"],
                              capture_output=True)
        if blob.returncode == 0:
            script = "/tmp/recompute_from_sidecar.py"
            Path(script).write_bytes(blob.stdout)
    if Path(script).is_file():
        print("\n--- recompute_from_sidecar.py over the per-block sidecar ---", flush=True)
        r = subprocess.run([sys.executable, script, a.sidecar], capture_output=True,
                           text=True)
        print(r.stdout[-4000:])
        if r.stderr:
            print("stderr:", r.stderr[-2000:])
        print("exit", r.returncode)
    else:
        print(f"{a.recompute} absent from the tree and from origin/wk/of3t-orchestrator -- not run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
