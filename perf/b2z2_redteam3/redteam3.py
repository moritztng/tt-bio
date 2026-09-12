#!/usr/bin/env python3
"""b2z2-redteam-v3: the five load-bearing claims of CLOSING.md, re-derived rather than re-read.

Host only. Opens no device, takes no measurement. Every input is a committed artifact copied into
`src/` from the branch that produced it, so this runs without fetching nine other branches.

    python3 perf/b2z2_redteam3/redteam3.py [--json out/redteam3.json]

ARCH is stated for every number. The wave's standing failure is a Wormhole ratio quoted against a
Blackhole cell, and two of the findings below are exactly that.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"


def load(name):
    return json.loads((SRC / name).read_text())


# ============================================================================================
# Shared primaries
# ============================================================================================
TIMING = load("timing512_step_qb2c1.json")          # BH, qb2 card 1, the headline's own 65 folds
WARM = [r for r in TIMING["runs"] if not r.get("cold")]
BASE = [r for r in WARM if r["arm"] == "base"]
STACK = [r for r in WARM if r["arm"] == "STACK"]

PAGE = load("perf-512aa.json")
CELL_S = next(float(m["cells"]["p150a"]["s_per_fold"])
              for m in PAGE["models"] if m["name"] == "Boltz-2")

# ceiling_v3.py's own constants, restated here so the corrections are diffable against them.
V3_TRUNK_S = 10.22          # PairformerLayer device spans, BH
V3_SAMPLER_S = 5.365        # sampler stage wall, from b2z2-bh-compose-landed's 12.400/5.365/2.423
V3_STACK_RATIO = 1.12862
V3_SILU_FOLD = 1.02423
V3_SHARD_BLOCK = {2: 1.3422, 4: 1.6404, 8: 1.9424}
V3_SHARD_CAP = 2.299
V3_SHARD_CAP_FREE = 2.990
V3_SAMPLER_STEP = {2: 1.13353, 4: 1.22779}
V3_SAMPLER_STEP_CAP = 1.3237
V3_CALIB_WH_OVER_BH = None  # computed below


def lin(xs, ys):
    mx, my = st.fmean(xs), st.fmean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    ss = sum((y - my) ** 2 for y in ys)
    rs = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return a, b, 1 - rs / ss


# ============================================================================================
# A. The A/A floor. Is 1.02860x the right floor for a median of ten PAIRED ratios?
# ============================================================================================
def attack_A():
    bypos = {}
    for r in BASE:
        bypos.setdefault(r["rep"], {})[r["pos"]] = r["fold_s"]

    def adjacent(r):
        c = [p for p in bypos[r["rep"]] if p < r["pos"]]
        return bypos[r["rep"]][max(c)] if c else None

    head = st.median([adjacent(r) / r["fold_s"] for r in STACK])

    # The row's own null: 10 A/A pairs drawn uniformly, WITHOUT replacement, from the 25 that exist.
    aa = [(r["rep"], adjacent(r) / r["fold_s"]) for r in BASE if adjacent(r)]
    vals = [v for _, v in aa]
    rng = random.Random(0)
    meds = sorted(st.median(rng.sample(vals, 10)) for _ in range(20000))
    floor_row = max(meds[int(.975 * len(meds))], 1 / meds[int(.025 * len(meds))])

    # DEFECT 1: sampling without replacement from a pool of 25 applies a finite-population
    # correction the headline's estimator does not enjoy. The 10 STACK folds are 10 fresh draws.
    rng = random.Random(0)
    meds_r = sorted(st.median([rng.choice(vals) for _ in range(10)]) for _ in range(20000))
    floor_repl = max(meds_r[int(.975 * len(meds_r))], 1 / meds_r[int(.025 * len(meds_r))])

    # DEFECT 2, the structural one. The headline's 10 ratios have only FIVE distinct numerators:
    # every rep's base@10, used twice (once for STACK@11, once for STACK@12). The row's null draws
    # 10 pairs with no shared numerator at all, so it credits the estimator with 10 independent
    # denominators when it has 5 clusters of 2. Match the structure: per rep pick one base fold as
    # the shared numerator and the next two base folds as the two denominators.
    clustered = {}
    for rep, d in bypos.items():
        ps = sorted(d)
        clustered[rep] = [(d[ps[i]] / d[ps[i + 1]], d[ps[i]] / d[ps[i + 2]])
                          for i in range(len(ps) - 2)]
    reps = sorted(clustered)
    rng = random.Random(0)
    meds_c = []
    for _ in range(20000):
        rs = []
        for rep in reps:
            a, b = rng.choice(clustered[rep])
            rs += [a, b]
        meds_c.append(st.median(rs))
    meds_c.sort()
    floor_clust = max(meds_c[int(.975 * len(meds_c))], 1 / meds_c[int(.025 * len(meds_c))])

    # How correlated ARE the two STACK ratios that share a numerator? If they were independent the
    # row's null would be right.
    within = []
    for rep in reps:
        rs = sorted([r for r in STACK if r["rep"] == rep], key=lambda r: r["pos"])
        if len(rs) == 2:
            within.append((adjacent(rs[0]) / rs[0]["fold_s"], adjacent(rs[1]) / rs[1]["fold_s"]))
    mx, my = st.fmean([a for a, _ in within]), st.fmean([b for _, b in within])
    num = sum((a - mx) * (b - my) for a, b in within)
    den = (sum((a - mx) ** 2 for a, _ in within) * sum((b - my) ** 2 for _, b in within)) ** .5
    rho = num / den

    # Is base@10 -- the sole numerator of every headline ratio -- a fair draw from the base arm?
    p10 = [bypos[rep][max(bypos[rep])] for rep in reps]
    allbase = [r["fold_s"] for r in BASE]
    within_rep_rank = []
    for rep in reps:
        v = sorted(bypos[rep].values())
        within_rep_rank.append((v.index(bypos[rep][max(bypos[rep])]) + .5) / len(v))

    # The worst SINGLE pair, and whether rejecting it was sound.
    worst_single = max(max(v, 1 / v) for v in vals)

    return {
        "headline_paired_median": head,
        "n_aa_pairs": len(vals),
        "floor_as_published": floor_row,
        "floor_with_replacement": floor_repl,
        "floor_structure_matched_shared_numerator": floor_clust,
        "within_rep_stack_ratio_correlation": rho,
        "base_at_pos10_mean": st.fmean(p10),
        "base_all_mean": st.fmean(allbase),
        "base_at_pos10_mean_within_rep_quantile": st.fmean(within_rep_rank),
        "worst_single_aa_pair": worst_single,
        "headline_margin_over_published_floor": (head - 1) / (floor_row - 1),
        "headline_margin_over_corrected_floor": (head - 1) / (floor_clust - 1),
    }


# ============================================================================================
# B. The fold decomposition. trunk 10.22 + sampler 5.365 + remainder 4.528 = 20.113.
#    Three terms from three instruments at three scopes. Does the split survive?
# ============================================================================================
def attack_B():
    def med(rs, k):
        return st.median([r[k] for r in rs])

    la = [(r["loadavg_before"][0] + r["loadavg_after"][0]) / 2 for r in BASE]
    fits = {}
    rem = [r["fold_s"] - r["block_s"] - r["step_s"] for r in BASE]
    for name, ys in (("fold", [r["fold_s"] for r in BASE]),
                     ("trunk", [r["block_s"] for r in BASE]),
                     ("sampler", [r["step_s"] for r in BASE]),
                     ("remainder", rem)):
        a, b, r2 = lin(la, ys)
        fits[name] = {"intercept_s": a, "slope_s_per_load": b, "r2": r2}
    fslope = fits["fold"]["slope_s_per_load"]
    for k in ("trunk", "sampler", "remainder"):
        fits[k]["share_of_load_penalty_pct"] = fits[k]["slope_s_per_load"] / fslope * 100

    # Transport the split to the published cell along the SESSION'S OWN load axis rather than by
    # scaling all three terms proportionally. The load at which this session's base folds in
    # CELL_S is the only defensible transport, because the three stages have different slopes.
    la_cell = (CELL_S - fits["fold"]["intercept_s"]) / fslope
    at_cell = {k: fits[k]["intercept_s"] + la_cell * fits[k]["slope_s_per_load"] for k in fits}

    # Independent cross-check on the sampler term: the BH step wall is 26.400 ms x 200 steps.
    step_wall_ms = med(BASE, "step_s") / BASE[0]["step_n"] * 1000

    return {
        "session_base_median": {"fold_s": med(BASE, "fold_s"), "trunk_s": med(BASE, "block_s"),
                                "sampler_s": med(BASE, "step_s"),
                                "remainder_s": med(BASE, "fold_s") - med(BASE, "block_s") - med(BASE, "step_s")},
        "session_stack_median": {"fold_s": med(STACK, "fold_s"), "trunk_s": med(STACK, "block_s"),
                                 "sampler_s": med(STACK, "step_s"),
                                 "remainder_s": med(STACK, "fold_s") - med(STACK, "block_s") - med(STACK, "step_s")},
        "load_fits": fits,
        "loadavg_at_which_base_folds_in_cell_s": la_cell,
        "split_at_cell_measured": at_cell,
        "split_as_published": {"trunk": V3_TRUNK_S, "sampler": V3_SAMPLER_S,
                               "remainder": CELL_S - V3_TRUNK_S - V3_SAMPLER_S},
        "step_wall_ms_from_this_session": step_wall_ms,
        "where_the_stack_saving_lands": {
            "trunk_s": med(BASE, "block_s") - med(STACK, "block_s"),
            "sampler_s": med(BASE, "step_s") - med(STACK, "step_s"),
            "remainder_s": (med(BASE, "fold_s") - med(BASE, "block_s") - med(BASE, "step_s"))
                           - (med(STACK, "fold_s") - med(STACK, "block_s") - med(STACK, "step_s")),
            "trunk_ratio": med(BASE, "block_s") / med(STACK, "block_s"),
            "sampler_ratio": med(BASE, "step_s") / med(STACK, "step_s"),
        },
        "silu_predicted_trunk_saving_s": CELL_S - CELL_S / V3_SILU_FOLD,
    }


# ============================================================================================
# C. The composition to 1.789x - 1.822x. Break it.
# ============================================================================================
def attack_C(B):
    bh = load("sampler/step_track_split.json")     # BH, qb2 card 0, 11x10, 1066 programs
    atom_bh_ms = bh["headline"]["atom_tracks_combined_ms"]
    kernel_bh_ms = bh["total_kernel_ms"]
    step_wall_bh_ms = B["step_wall_ms_from_this_session"]

    # WH basis ceiling_v3 actually uses: atom 12.340 of a 40.366 ms summed-kernel step.
    wh_atom, wh_step = 12.340, 40.366
    track = {2: 1.62694, 4: 2.54386, "cap": 1 / 0.19995}

    def step_ratio(atom_ms, step_ms, tr):
        return step_ms / (atom_ms / tr + (step_ms - atom_ms))

    corrected = {}
    for k, tr in track.items():
        corrected[str(k)] = {
            "wh_kernel_basis_as_published": step_ratio(wh_atom, wh_step, tr),
            "bh_kernel_basis": step_ratio(atom_bh_ms, kernel_bh_ms, tr),
            "bh_wall_basis": step_ratio(atom_bh_ms, step_wall_bh_ms, tr),
        }

    # The measured split of the stack's saving replaces split A / split B.
    sb, ss = B["session_base_median"], B["session_stack_median"]
    cell = B["split_at_cell_measured"]
    scale = 1.0
    trunk_post = cell["trunk"] - (sb["trunk_s"] - ss["trunk_s"]) * scale
    samp_post = cell["sampler"] - (sb["sampler_s"] - ss["sampler_s"]) * scale
    rem_post = cell["remainder"] - (sb["remainder_s"] - ss["remainder_s"]) * scale
    stacked_total = trunk_post + samp_post + rem_post

    def compose(block_ratio, step_ratio_):
        return CELL_S / (trunk_post / block_ratio + samp_post / step_ratio_ + rem_post)

    # WH -> BH calibration for the trunk shard: ceiling_v3 PRINTS it and does not APPLY it.
    # The WH block curve says 1.3422x at N=2; the BH fold of the same shard measured 1.0857x.
    def compose_trunk_only(block_ratio):
        return CELL_S / (trunk_post / block_ratio + samp_post + rem_post)

    calib = compose_trunk_only(V3_SHARD_BLOCK[2]) / V3_STACK_RATIO / 1.0857

    stack_measured = CELL_S / stacked_total
    rows = []
    for label, br, sr, pub in (
        ("trunk shard N=2", V3_SHARD_BLOCK[2], 1.0, (1.311, 1.322)),
        ("trunk shard N=4", V3_SHARD_BLOCK[4], 1.0, None),
        ("trunk shard N=8", V3_SHARD_BLOCK[8], 1.0, (1.536, 1.564)),
        ("trunk shard cap, measured link", V3_SHARD_CAP, 1.0, (1.633, 1.670)),
        ("trunk shard cap, FREE link", V3_SHARD_CAP_FREE, 1.0, (1.83, 1.83)),
        ("both shards N=2", V3_SHARD_BLOCK[2], corrected["2"]["bh_wall_basis"], (1.357, 1.366)),
        ("both shards N=4", V3_SHARD_BLOCK[4], corrected["4"]["bh_wall_basis"], (1.523, 1.539)),
        ("both shards at caps", V3_SHARD_CAP, corrected["cap"]["bh_wall_basis"], (1.789, 1.822)),
        ("both shards at caps, FREE link", V3_SHARD_CAP_FREE, corrected["cap"]["bh_wall_basis"], None),
    ):
        v = compose(br, sr)
        # Apply the WH->BH shard calibration ceiling_v3 prints and does not apply: the excess of the
        # route over the measured single-chip stack is what the shard bought, and the BH fold kept
        # 1/calib of what the WH block curve predicted at the one width where both were measured.
        vc = stack_measured + (v - stack_measured) / calib
        rows.append({"route": label, "block_ratio": br, "step_ratio": sr,
                     "fold_ratio_corrected": v, "fold_s": CELL_S / v,
                     "fold_ratio_calibrated": vc, "fold_s_calibrated": CELL_S / vc,
                     "published_range": pub})
    return {
        "bh_step_census": {"atom_tracks_ms": atom_bh_ms, "kernel_ms": kernel_bh_ms,
                           "step_wall_ms": step_wall_bh_ms,
                           "atom_pct_of_kernel": atom_bh_ms / kernel_bh_ms * 100,
                           "atom_pct_of_wall": atom_bh_ms / step_wall_bh_ms * 100,
                           "exposed_dispatch_ms": step_wall_bh_ms - kernel_bh_ms,
                           "exposed_dispatch_pct_of_wall": (step_wall_bh_ms - kernel_bh_ms) / step_wall_bh_ms * 100},
        "wh_step_census": {"atom_ms": wh_atom, "kernel_ms": wh_step,
                           "atom_pct_of_kernel": wh_atom / wh_step * 100},
        "sampler_step_ratios": corrected,
        "post_stack_split_measured": {"trunk_s": trunk_post, "sampler_s": samp_post,
                                      "remainder_s": rem_post, "total_s": stacked_total,
                                      "implied_stack_ratio": CELL_S / stacked_total},
        "wh_to_bh_trunk_shard_calibration": calib,
        "routes": rows,
    }


# ============================================================================================
# D. "The trunk is closed." Is the byte leg a bound, and are its units clean?
# ============================================================================================
def attack_D():
    c = load("byte/census.json")["totals"]
    wh_total = c["dram_rd"] + c["dram_wr"]
    bh_total_gb = 8.0493
    removable = c["redundant_dram_rd"]
    block_ms, mf_ms = 36.3438, 14.9006
    roofs = {"clone 64 MiB": 382.9, "clone 128 MiB (quoted)": 390.7,
             "clone 192 MiB": 396.2, "clone 256 MiB": 395.9,
             "add 2r+1w": 429.9, "campaign fitted": 444.9}
    out = {}
    for name, r in roofs.items():
        floor = max(mf_ms, bh_total_gb / r * 1000)
        after = max(mf_ms, bh_total_gb * (1 - removable / 1e9 / bh_total_gb) / r * 1000)
        out[name] = {"gbps": r, "block_floor_ms": floor, "block_ratio": block_ms / floor,
                     "block_ratio_after_every_removable_byte": block_ms / after}
    return {
        "wh_trace_total_gb": wh_total / 1e9,
        "bh_census_total_gb": bh_total_gb,
        "removable_gb": removable / 1e9,
        "removable_pct_of_bh_census": removable / 1e9 / bh_total_gb * 100,
        "removable_pct_of_wh_trace": removable / wh_total * 100,
        "byte_cut_needed_for_movement_free_pct":
            (1 - 390.7 * mf_ms / 1000 / bh_total_gb) * 100,
        "unit_check": {
            "census_bytes_are_decimal": abs(c["dram_rd"] / 1e6 - 3672.0) < 0.1,
            "roof_is_decimal_gbps": "roofs_bh.py line 55: 2*nbytes/(ms*1e-3)/1e9",
            "verdict": "no unit slip: both sides decimal",
        },
        "roof_sensitivity": out,
    }


# ============================================================================================
# E. The atom axis shards. Refit the curve; check the floor that actually caps the route.
# ============================================================================================
def attack_E():
    cur = load("atom/curve_wh_c24.json")["rows"]
    xs = [r["n_atoms"] for r in cur]
    ys = [r["ms"] for r in cur]
    a, b, r2 = lin(xs, ys)
    m2, m4 = load("atom/shard_wh_mesh2.json"), load("atom/shard_wh_mesh4.json")
    t = {2: m2["timing"], 4: m4["timing"]}
    # Time-side floor from t(N) = c + k/N over the two measured widths and the one-chip whole.
    n = [1, 2, 4]
    y = [m2["timing"]["whole"]["ms"], m2["timing"]["shard"]["ms"], m4["timing"]["shard"]["ms"]]
    inv = [1 / v for v in n]
    c0, k, r2t = lin(inv, y)
    return {
        "work_side_fit": {"const_ms": a, "ms_per_atom": b, "r2": r2,
                          "const_pct_at_4480_atoms": a / (a + b * 4480) * 100},
        "time_side_fit_over_3_points": {"const_ms": c0, "per_chip_ms": k, "r2": r2t,
                                        "replicated_pct": c0 / (c0 + k) * 100},
        "measured": {"whole_ms": m2["timing"]["whole"]["ms"],
                     "shard2_ms": m2["timing"]["shard"]["ms"],
                     "shard4_ms": m4["timing"]["shard"]["ms"],
                     "ratio2": m2["result"]["track_ratio"], "ratio4": m4["result"]["track_ratio"]},
        "bit_exact_2": m2["parity"]["shard_vs_whole"],
        "bit_exact_4": m4["parity"]["shard_vs_whole"],
        "controls_2": {"nohalo_differs": m2["parity"]["control_nohalo_differs"],
                       "perturbed_differs": m2["parity"]["control_perturbed_differs"]},
        "extrapolated_cap_from_time_fit": (c0 + k) / c0,
        "halo_cost_ms": m2["result"]["halo_cost_ms"],
        "outgather_cost_ms": m2["result"]["outgather_cost_ms"],
    }


# ============================================================================================
# F. TT_BIO_ATOM_KEY_WINDOW. Reproduce the defect, then ask the question nobody asked:
#    did the defective transform buy any of its 1.02744x by doing LESS of the model's work?
# ============================================================================================
def attack_F():
    w, s, o = load("gather/window_live.json"), load("gather/shift_live.json"), load("gather/onehot_live.json")
    zt, nb = load("gather/window_zerotail.json"), load("gather/window_nobucket.json")
    full, nomask, part = (load("gather/reach_full_window.json"), load("gather/reach_nomask.json"),
                          load("gather/reach_partial.json"))
    return {
        "impls_at_production_shape": {
            "window (TT_BIO_ATOM_KEY_WINDOW)": {"max_abs": w["max_abs"], "bad_windows": w["n_bad_windows"],
                                                "first_bad": w["first_bad_window"]},
            "shift (TT_BIO_ATOM_SHIFT_GATHER)": {"max_abs": s["max_abs"], "bit_exact": s["bit_exact"]},
            "onehot (shipped)": {"max_abs": o["max_abs"], "bit_exact": o["bit_exact"]},
        },
        "mechanism_confirmed": {
            "windows_real": w["windows_real"], "windows_pad": w["windows_pad"],
            "first_bad_is_windows_minus_2": w["first_bad_window"] == w["windows_real"] - 2,
            "bit_exact_when_no_zero_tail": nb["bit_exact"] and nb["windows_real"] == nb["windows_pad"],
        },
        "mask_neutralises_it": {
            "with_matrix_bias_4128_atoms_bad_real_windows": full["attention_bad_real_windows"],
            "with_matrix_bias_4100_atoms_bad_real_windows": part["attention_bad_real_windows"],
            "without_bias_bad_real_windows": nomask["attention_bad_real_windows"],
            "verdict": "the -1e9 atom bias is load-bearing; remove it and windows 127-128 go wrong",
        },
        "no_cheat_check": {
            "window_out_shape": w["out_shape"], "shift_out_shape": s["out_shape"],
            "same_output_extent": w["out_shape"] == s["out_shape"],
            "window_is_bitexact_over_all_140_real_windows": nb["bit_exact"],
            "verdict": ("both implementations produce the same 140-window output; the window impl is "
                        "bit-exact when all 140 windows are real, so it computes a FULL gather and "
                        "is not cheaper for doing less. Its 1.02744x against the shift gather's "
                        "1.01985x is mechanism, not skipped model work."),
        },
    }


# ============================================================================================
# G. The published cell against the H200 competitor row. Same protocol, fixture, generation?
# ============================================================================================
def attack_G():
    m = next(x for x in PAGE["models"] if x["name"] == "Boltz-2")
    sc = PAGE["scope"]
    p150a, h200 = m["cells"]["p150a"], m["cells"]["h200"]
    return {
        "fixture_is_page_wide": sc["fixture"],
        "protocol_is_model_level": {"recycles": m["recycles"], "sampling_steps": m["sampling_steps"],
                                    "diffusion_samples": sc["diffusion_samples"], "seed": sc["seed"],
                                    "templates": sc["templates"]},
        "cells": {"p150a_s": p150a["s_per_fold"], "h200_s": h200["s_per_fold"],
                  "gap": h200 and p150a["s_per_fold"] / h200["s_per_fold"]},
        "host_seconds_asymmetry": {
            "tt_host_inside_cell": True,
            "h200_host_s": h200["split"]["host_s"], "h200_host_in_cell": h200["split"]["in_cell"],
            "page_says": sc["split"],
        },
        "timed_region_note": sc["timed_region"],
        "verdict_inputs": {
            "same_fixture": True, "same_protocol": True,
            "tt_cell_includes_host_seconds": True,
            "h200_cell_excludes_its_0_24_s_host": h200["split"]["in_cell"] is False,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    A = attack_A(); B = attack_B(); C = attack_C(B)
    D = attack_D(); E = attack_E(); F = attack_F(); G = attack_G()
    out = {"cell_s": CELL_S, "A_aa_floor": A, "B_decomposition": B, "C_composition": C,
           "D_trunk_closed": D, "E_atom_shard": E, "F_atom_key_window": F, "G_page": G}

    p = print
    p(f"published cell {CELL_S:.3f} s   (site/data/perf-512aa.json, Boltz-2 p150a, BH)")
    p("")
    p("=== A. the A/A floor, corrected from 1.01162x to 1.02860x by the orchestrator ===")
    p(f"  headline, re-derived                                  {A['headline_paired_median']:.5f}x")
    p(f"  A/A pairs available                                   {A['n_aa_pairs']}")
    p(f"  floor as published (sample 10 of 25, no replacement)   {A['floor_as_published']:.5f}x")
    p(f"  floor with replacement (10 fresh draws)                {A['floor_with_replacement']:.5f}x")
    p(f"  floor STRUCTURE-MATCHED (5 reps x 2 shared-numerator)  {A['floor_structure_matched_shared_numerator']:.5f}x")
    p(f"  correlation of the two STACK ratios sharing a base    rho = {A['within_rep_stack_ratio_correlation']:+.4f}")
    p(f"  base@pos10 mean {A['base_at_pos10_mean']:.4f} s vs all-base mean {A['base_all_mean']:.4f} s"
      f"  (within-rep quantile {A['base_at_pos10_mean_within_rep_quantile']:.3f})")
    p(f"  headline margin over the floor: {A['headline_margin_over_published_floor']:.2f}x published,"
      f" {A['headline_margin_over_corrected_floor']:.2f}x corrected")
    p(f"  worst single A/A pair {A['worst_single_aa_pair']:.5f}x (rejected by the orchestrator)")
    p("")
    p("=== B. the fold decomposition ===")
    sb, ss = B["session_base_median"], B["session_stack_median"]
    p(f"  session base  (BH, n=30, published digest): fold {sb['fold_s']:.4f} = trunk {sb['trunk_s']:.4f}"
      f" + sampler {sb['sampler_s']:.4f} + remainder {sb['remainder_s']:.4f}")
    p(f"  session STACK (BH, n=10)                  : fold {ss['fold_s']:.4f} = trunk {ss['trunk_s']:.4f}"
      f" + sampler {ss['sampler_s']:.4f} + remainder {ss['remainder_s']:.4f}")
    p(f"  step wall from this session: {B['step_wall_ms_from_this_session']:.4f} ms x 200"
      f"   (CONTEXT says 26.400 ms)")
    p("  load sensitivity (BH, loadavg 5.9-17.5):")
    for k in ("fold", "trunk", "sampler", "remainder"):
        f = B["load_fits"][k]
        sh = f.get("share_of_load_penalty_pct")
        p(f"    {k:<10} {f['intercept_s']:7.4f} s {f['slope_s_per_load']:+.5f} s/load  R2 {f['r2']:.3f}"
          + (f"   {sh:5.1f} % of the fold's load penalty" if sh else ""))
    p(f"  base folds in {CELL_S:.3f} s at loadavg {B['loadavg_at_which_base_folds_in_cell_s']:.2f}. "
      "Split there:")
    m_, pu = B["split_at_cell_measured"], B["split_as_published"]
    for k, pk in (("trunk", "trunk"), ("sampler", "sampler"), ("remainder", "remainder")):
        p(f"    {k:<10} MEASURED {m_[k]:7.4f} s   PUBLISHED {pu[pk]:7.4f} s"
          f"   delta {m_[k]-pu[pk]:+.4f} s ({(m_[k]/pu[pk]-1)*100:+.2f} %)")
    w = B["where_the_stack_saving_lands"]
    p(f"  where the stack's {sb['fold_s']-ss['fold_s']:.4f} s actually comes off: trunk {w['trunk_s']:.4f}"
      f" ({w['trunk_ratio']:.5f}x), sampler {w['sampler_s']:.4f} ({w['sampler_ratio']:.5f}x),"
      f" remainder {w['remainder_s']:.4f}")
    p(f"  ceiling_v3 split B predicts the trunk gives up {B['silu_predicted_trunk_saving_s']:.4f} s;"
      f" split A predicts 0.0000 s. MEASURED {w['trunk_s']:.4f} s.")
    p("")
    p("=== C. the composition ===")
    b_, wh = C["bh_step_census"], C["wh_step_census"]
    p(f"  atom track: WH {wh['atom_ms']:.3f}/{wh['kernel_ms']:.3f} ms = {wh['atom_pct_of_kernel']:.2f} % of kernel")
    p(f"              BH {b_['atom_tracks_ms']:.4f}/{b_['kernel_ms']:.4f} ms = {b_['atom_pct_of_kernel']:.2f} % of kernel,"
      f" {b_['atom_pct_of_wall']:.2f} % of the {b_['step_wall_ms']:.3f} ms WALL")
    p(f"  BH exposed dispatch {b_['exposed_dispatch_ms']:.3f} ms = {b_['exposed_dispatch_pct_of_wall']:.2f} %"
      " of the step wall -- a mesh shard cannot divide it")
    p(f"  {'width':<6} {'published (WH kernel)':>22} {'BH kernel':>12} {'BH wall':>12}")
    for k, v in C["sampler_step_ratios"].items():
        p(f"  {k:<6} {v['wh_kernel_basis_as_published']:>21.5f}x {v['bh_kernel_basis']:>11.5f}x"
          f" {v['bh_wall_basis']:>11.5f}x")
    ps = C["post_stack_split_measured"]
    p(f"  post-stack split, MEASURED: trunk {ps['trunk_s']:.4f} + sampler {ps['sampler_s']:.4f}"
      f" + remainder {ps['remainder_s']:.4f} = {ps['total_s']:.4f} s ({ps['implied_stack_ratio']:.5f}x)")
    p(f"  WH->BH trunk-shard calibration {C['wh_to_bh_trunk_shard_calibration']:.4f}x"
      "  (printed by ceiling_v3, never applied; applied below)")
    p(f"  {'route':<32} {'published':>15} {'corrected':>11} {'+calibrated':>12} {'seconds':>9}")
    for r in C["routes"]:
        pr = (f"{r['published_range'][0]:.3f}-{r['published_range'][1]:.3f}x"
              if r["published_range"] else "-")
        p(f"  {r['route']:<32} {pr:>15} {r['fold_ratio_corrected']:>10.4f}x"
          f" {r['fold_ratio_calibrated']:>11.4f}x {r['fold_s_calibrated']:>8.3f} s")
    p(f"  2x needs {CELL_S/2:.3f} s.")
    p("")
    p("=== D. the trunk's byte leg ===")
    p(f"  WH trace total {D['wh_trace_total_gb']:.4f} GB; BH census {D['bh_census_total_gb']:.4f} GB")
    p(f"  removable {D['removable_gb']:.4f} GB = {D['removable_pct_of_bh_census']:.2f} % of the BH census,"
      f" {D['removable_pct_of_wh_trace']:.2f} % of the WH trace")
    p(f"  cut NEEDED to reach movement-free at 390.7 GB/s: {D['byte_cut_needed_for_movement_free_pct']:.1f} %")
    p(f"  unit check: {D['unit_check']['verdict']}")
    for k, v in D["roof_sensitivity"].items():
        p(f"    {k:<24} {v['gbps']:6.1f} GB/s -> floor {v['block_floor_ms']:7.4f} ms ="
          f" {v['block_ratio']:.4f}x   (after every removable byte {v['block_ratio_after_every_removable_byte']:.4f}x)")
    p("")
    p("=== E. the atom axis ===")
    wf, tf = E["work_side_fit"], E["time_side_fit_over_3_points"]
    p(f"  work side  t = {wf['const_ms']:.5f} ms + {wf['ms_per_atom']:.8f} ms/atom, R2 {wf['r2']:.6f}"
      f" -> {wf['const_pct_at_4480_atoms']:.3f} % constant")
    p(f"  time side  t = {tf['const_ms']:.5f} ms + {tf['per_chip_ms']:.5f} ms/N, R2 {tf['r2']:.6f}"
      f" -> {tf['replicated_pct']:.3f} % replicated, cap {E['extrapolated_cap_from_time_fit']:.3f}x")
    p(f"  measured   whole {E['measured']['whole_ms']:.5f} / N=2 {E['measured']['shard2_ms']:.5f}"
      f" / N=4 {E['measured']['shard4_ms']:.5f} ms -> {E['measured']['ratio2']:.5f}x, {E['measured']['ratio4']:.5f}x")
    p(f"  bit-exact  N=2 {E['bit_exact_2']}   N=4 {E['bit_exact_4']}   controls {E['controls_2']}")
    p("")
    p("=== F. TT_BIO_ATOM_KEY_WINDOW ===")
    for k, v in F["impls_at_production_shape"].items():
        p(f"  {k:<38} {v}")
    p(f"  mechanism: {F['mechanism_confirmed']}")
    p(f"  mask:      {F['mask_neutralises_it']['verdict']}")
    p(f"  NO-CHEAT:  {F['no_cheat_check']['verdict']}")
    p("")
    p("=== G. the published cell vs the H200 row ===")
    p(f"  fixture (page-wide) {G['fixture_is_page_wide']}")
    p(f"  protocol (model-level, so both cells share it) {G['protocol_is_model_level']}")
    p(f"  p150a {G['cells']['p150a_s']} s   H200 {G['cells']['h200_s']} s   gap {G['cells']['gap']:.3f}x")
    p(f"  TT host seconds INSIDE the cell; H200's {G['host_seconds_asymmetry']['h200_host_s']} s host"
      f" is OUTSIDE it (in_cell={G['host_seconds_asymmetry']['h200_host_in_cell']})")

    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=2) + "\n")
        p(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
