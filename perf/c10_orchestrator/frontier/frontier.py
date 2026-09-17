#!/usr/bin/env python3
"""What 10.0 s would actually require, in the units the campaign has measured.

Everything in this campaign now reduces to one equation. `c10-fixed-cost` measured both terms of
T = F + W/f at a pinned, during-sampled 1350 MHz, so the target is no longer a wall-clock wish: it
is a budget in two currencies, and every lever can be scored against it.

This consolidates that budget, prices every axis the campaign has against it, and names the only
axis with enough headroom to reach the target at all.
"""
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
FLOOR = HERE.parents[1] / "roof_true" / "true_floor_512_qb2c2.json"

F_S, F_SE = 3.9830, 0.1181          # c10-fixed-cost c4e6725bd, 512 aa
W_MCYC, W_SE = 14665.0, 121.1
PIN = 1350.0
TARGET_S = 10.0

# Every Blackhole dense-cube rate the campaign has recorded, with where it came from.
CUBE_CLUSTER = {
    "roof_true/true_floor_512_qb2c2.json (qb2 card 2)": 104.93,
    "c10_orchestrator/floor_mix/floor_mix.json": 108.54279824685312,
    "roof_launch/launch_trace_qb2c1.json (qb2 card 1)": 112.40099491661344,
    "roof_launch/shape_control_qb2c1.json (a)": 113.02145851097598,
    "roof_launch/shape_control_qb2c1.json (b)": 113.72746035078254,
    "roof_launch/launch_trace_qb2c1.json (b)": 114.19605867211541,
}
CUBE_OUTLIER = 67.59      # quoted by shape_rank/ and grid_evidence/ as "the same-session dense cube"
FLOOR_MHZ = 800.0         # the clock floor a throttled p300c sits at

# Every lever the campaign has ever attached a number to, in fold seconds at 1350 MHz.
LEVERS = [
    ("ttnn trace of the diffusion loop", 0.00, "MEASURED on Blackhole at 1350 and 1000 MHz"),
    ("fuse the arithmetic-free elementwise traffic", 0.69, "derived: traffic x a measured 1/3 return"),
    ("TT_BIO_HEAD_PAD_TAIL", 0.21, "Wormhole, no recorded clock"),
    ("TT_BIO_DIT_FUSED_QKV", 0.00, "excluded: jointly 0.713 A with HEAD_PAD_TAIL vs a 0.60 A bar"),
    ("TT_BIO_SDPA_GRID_Q_CHUNK sign", 0.00, "unmeasured, anywhere from -0.46 to +0.21 s"),
    ("per-class grid sizing", 0.00, "withdrawn as a cross-architecture transfer"),
    ("size-independent share of the work term", 0.00,
     "failing its own discriminating test at 768 aa"),
]


def analyse():
    floor = json.loads(FLOOR.read_text())
    w_s = W_MCYC / PIN
    T = F_S + w_s
    need = T - TARGET_S

    out = {
        "scope": "CPU consolidation of measured terms and recorded levers. No device, no new "
                 "measurement, no lever approved.",
        "measured": {"F_s": F_S, "F_se_s": F_SE, "W_Mcyc": W_MCYC, "W_se_Mcyc": W_SE,
                     "W_s_at_pin": w_s, "fold_s": T, "pin_mhz": PIN},
    }

    # --- the budget ---------------------------------------------------------------------------
    out["requirement"] = {
        "target_s": TARGET_S,
        "must_remove_s": need,
        "frontier": "any (dF, dW) with dF + dW/1350 = %.4f s reaches the target" % need,
        "W_alone": {"cut_Mcyc": need * PIN, "cut_pct": 100.0 * need * PIN / W_MCYC},
        "F_alone": {"required_F_s": TARGET_S - w_s,
                    "possible": (TARGET_S - w_s) > 0,
                    "note": "F would have to be negative, so the target cannot be reached by "
                            "removing host and DRAM-bound cost alone no matter how completely"},
    }

    # --- what the campaign actually has ------------------------------------------------------
    priced = sum(v for _, v, _ in LEVERS)
    out["levers"] = [{"item": k, "fold_s": v, "evidence": e} for k, v, e in LEVERS]
    out["levers_total"] = {
        "fold_s": priced,
        "Mcyc": priced * PIN,
        "pct_of_requirement": 100.0 * priced / need,
        "note": "Perturbations stack strongly sub-additively on this fixture, so even this is an "
                "upper bound on what the stack would return, not a forecast.",
    }

    # --- the only axis with enough headroom ---------------------------------------------------
    mm_tflop, mm_s = floor["matmul_TFLOP"], floor["matmul_floor_s"]
    achieved = mm_tflop / mm_s
    cubes = sorted(CUBE_CLUSTER.values())
    lo, hi = cubes[0], cubes[-1]
    out["matmul_rate_axis"] = {
        "matmul_TFLOP_per_fold": mm_tflop,
        "modelled_seconds": mm_s,
        "achieved_TFLOPs": achieved,
        "cube_cluster_TFLOPs": {"min": lo, "median": st.median(cubes), "max": hi,
                                "spread_pct": 100.0 * (hi - lo) / lo, "n": len(cubes)},
        "achieved_pct_of_cube": {"vs_min": 100.0 * achieved / lo, "vs_max": 100.0 * achieved / hi},
        "seconds_if_at_cube": {"vs_min": mm_tflop / lo, "vs_max": mm_tflop / hi},
        "saving_if_at_cube_s": {"vs_min": mm_s - mm_tflop / lo, "vs_max": mm_s - mm_tflop / hi},
        "rate_needed_for_target_TFLOPs": mm_tflop / (mm_s - need),
        "pct_of_the_gap_to_close": 100.0 * need / (mm_s - mm_tflop / st.median(cubes)),
        "reading": "This is the ONLY axis the campaign has found with enough headroom to reach "
                   "10.0 s, and it is also the least trustworthy number in the corpus: the "
                   "per-shape rates behind the modelled seconds were taken on pc's custom "
                   "130-core firmware, not on qb2's 110-core p300c. Not all of the gap is "
                   "recoverable either -- the cube is 4096^3 and the fold's matmuls are not.",
    }

    # --- a fourth clock artifact, found from ratios alone --------------------------------------
    out["fourth_clock_artifact"] = {
        "outlier_TFLOPs": CUBE_OUTLIER,
        "quoted_by": ["shape_rank/", "grid_evidence/"],
        "cluster": CUBE_CLUSTER,
        "implied_clock_MHz": {k: PIN * CUBE_OUTLIER / v for k, v in CUBE_CLUSTER.items()},
        "implied_clock_vs_max_MHz": PIN * CUBE_OUTLIER / hi,
        "clock_floor_MHz": FLOOR_MHZ,
        "reading": "Four independent Blackhole cube measurements cluster within 8.8 %% of each "
                   "other at %.2f to %.2f TFLOP/s. One reads 67.59, and against the cluster "
                   "maximum that implies a chip at %.0f MHz -- the 800 MHz floor. It is the "
                   "campaign's founding failure mode for the fourth time, found here from ratios "
                   "alone with no new measurement." % (lo, hi, PIN * CUBE_OUTLIER / hi),
        "what_survives": "Ratios taken WITHIN that session survive, because both sides were "
                         "throttled together: triangle multiplication at 11.47 TFLOP/s is still "
                         "5.89x off its own session's cube. The absolute rates from it do not, and "
                         "at burst clock they would be about 1.69x higher.",
    }

    out["verdict_material"] = [
        "Removing 4.846 s needs either a 44.6 %% cut in device cycles or a mixture; F alone "
        "cannot do it at any completeness because F is only 3.983 s.",
        "Every lever the campaign has ever numbered sums to %.2f s, %.1f %% of the requirement, "
        "and that sum is an upper bound because perturbations stack sub-additively here."
        % (priced, 100.0 * priced / need),
        "The matmul rate axis is the only one with enough headroom, and the number that says so "
        "rests on shape rates measured on the wrong machine. Re-measuring it on qb2 at a recorded "
        "clock is c10-fold-census's single most valuable deliverable.",
    ]
    out["limits"] = [
        "No lever here is approved, measured as a stack, or scored for accuracy.",
        "matmul_floor_s is modelled, not measured: it is the floor artifact's own number and that "
        "artifact's arithmetic half is contaminated. The achieved rate inherits that.",
        "The frontier is exact arithmetic on two measured terms. Everything scored against it is "
        "not, and the evidence class of each row is carried rather than flattened.",
    ]
    return out


if __name__ == "__main__":
    r = analyse()
    (HERE / "frontier.json").write_text(json.dumps(r, indent=2) + "\n")
    q = r["requirement"]
    print(f"fold {r['measured']['fold_s']:.4f} s at {PIN:.0f} MHz -> must remove "
          f"{q['must_remove_s']:.4f} s")
    print(f"  W alone: {q['W_alone']['cut_Mcyc']:.0f} Mcyc, {q['W_alone']['cut_pct']:.1f} %")
    print(f"  F alone: needs F = {q['F_alone']['required_F_s']:.4f} s -> possible: "
          f"{q['F_alone']['possible']}")
    lt = r["levers_total"]
    print(f"every lever ever numbered: {lt['fold_s']:.2f} s = {lt['pct_of_requirement']:.1f} % of it")
    m = r["matmul_rate_axis"]
    print(f"matmul achieved {m['achieved_TFLOPs']:.2f} TFLOP/s = "
          f"{m['achieved_pct_of_cube']['vs_max']:.1f}-{m['achieved_pct_of_cube']['vs_min']:.1f} % of "
          f"cube; target needs {m['rate_needed_for_target_TFLOPs']:.2f} TFLOP/s "
          f"({m['pct_of_the_gap_to_close']:.0f} % of the gap)")
    a = r["fourth_clock_artifact"]
    print(f"outlier cube {a['outlier_TFLOPs']} TFLOP/s implies "
          f"{a['implied_clock_vs_max_MHz']:.0f} MHz against the cluster max")
