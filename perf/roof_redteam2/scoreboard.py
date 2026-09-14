#!/usr/bin/env python3
"""roof-redteam-2: the five attacks, computed. One instrument per number, roofs named inline.

Everything reads the committed artifacts of `origin/wk/roof-budget` (extracted to perf/_rb_tmp,
gitignored) and the committed source of this tree. No device, no new measurement.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RB = ROOT / "perf" / "_rb_tmp"
sys.path.insert(0, str(RB))
sys.path.insert(0, str(ROOT / "perf" / "b2x_difflayer"))
import exec_flops as EF                                                       # noqa: E402
from itemize import itemize                                                   # noqa: E402
from real_traffic import NO_TRAFFIC                                           # noqa: E402

TOP = ["PairformerLayer|1x512x384,1x512x512x128",
       "MSALayer|1x512x512x128,1x1024x512x64",
       "DiffusionModule|"]
CELL = 17.340


def per_op(nodes, widen):
    """(ops, bytes per op). `widen=False` is `real_traffic.py` verbatim; `widen=True` also lets an
    op whose output landed in L1 count as a reader of the DRAM it consumes."""
    ops, rows = itemize({"nodes": nodes})
    dram = [r for r in rows if r["kind"] == "DRAM"]
    ad, aa = defaultdict(int), defaultdict(int)
    for r in rows:
        if r["alloc_op_i"] is not None:
            aa[r["alloc_op_i"]] += r["size"]
            if r["kind"] == "DRAM":
                ad[r["alloc_op_i"]] += r["size"]

    def mv(i):
        n = ops[i]["name"]
        return False if n in NO_TRAFFIC else ((aa if widen else ad)[i] > 0 or n.endswith("_"))

    w, rd = defaultdict(int), defaultdict(int)
    for r in dram:
        readers = [i for i in r["consumers"] if mv(i)]
        if r["alloc_op_i"] is not None:
            w[r["alloc_op_i"]] += r["size"]
        for i in readers:
            rd[i] += r["size"]
            if ops[i]["name"].endswith("_") and ad[i] == 0:
                w[i] += r["size"]
        if not readers:
            rd[r["alloc_op_i"] if r["alloc_op_i"] is not None else -1] += r["size"]
    extra = rd[-1]
    return ops, [w[i] + rd[i] for i in range(len(ops))], extra


def main():
    pub = json.loads((RB / "roof_budget_512_qb2c2.json").read_text())
    base = json.loads((RB / "attrib_512_tip_qb2c2.json").read_text())
    run = json.loads((RB / "attrib2_512_tip_qb2c2.json").read_text())
    ctl = json.loads((RB / "instrument_control.json").read_text())
    st = json.loads((RB / "stream_roof2.json").read_text())
    C = max(r["TFLOPs"] for r in ctl["rows"] if r["label"].startswith("cube")) * 1e12
    S = max(r["GBps"] for r in st["stream"] if r["op"] == "add" and r["N"] == 8192) * 1e9
    row = {r["sig"]: r for r in pub["rows"]}
    tree = run["attrib"]["tree"]
    out = {}

    # ---------------- attack 1: does the ranking survive a non-uniform contention scalar? -------
    T = {s: row[s]["s_per_fold"] for s in TOP}                 # session seconds, median x calls
    R = {s: row[s]["s_at_roof"] for s in TOP}                  # unscaled traffic roof, seconds
    N = {s: row[s]["calls"] * row[s]["ops_per_call"] for s in TOP}
    k0 = pub["summary"]["cell_scale"]
    covered = sum(T.values())                                  # 24.644 s
    cell_top = CELL * covered / base["baseline_summary"]["plain_median_s"]

    def waste(k):
        return {s: T[s] * k[s] - R[s] for s in TOP}

    # model U: one scalar, as published
    wU = waste({s: k0 for s in TOP})
    # model G: host contention costs a fixed amount per dispatched ttnn op, the same everywhere,
    # fitted so the three units still sum to their share of the cell.
    g = (covered - cell_top) / sum(N.values())                 # seconds per op
    kG = {s: 1 - g * N[s] / T[s] for s in TOP}
    wG = waste(kG)
    # the threshold: the per-op gap at which the denoiser and the MSA block change places
    P, M, D = TOP[0], TOP[1], TOP[2]
    gswap = ((T[D] - R[D]) - (T[M] - R[M])) / (N[D] - N[M])
    # the ratio of scalars a reorder needs, holding the third unit at the published scalar
    # holding the other unit at the published scalar, the scalar the challenger needs
    need_D = (T[P] * k0 - R[P] + R[D]) / T[D]                       # denoiser to pass pairformer
    need_M = (T[D] * k0 - R[D] + R[M]) / T[M]                       # MSA block to pass denoiser
    out["attack1_contention"] = {
        "published_scalar": k0,
        "published_scalar_denominator": {
            "plain_reps_s": base["baseline_summary"]["plain_s"],
            "n": base["baseline_summary"]["n"],
            "plain_median_s": base["baseline_summary"]["plain_median_s"],
            "note": "n=2, so the 'median' is the mean of two reps that differ by "
                    "%.2fx; the scalar is 17.340/rep in [%.4f, %.4f]"
                    % (max(base["baseline_summary"]["plain_s"])
                       / min(base["baseline_summary"]["plain_s"]),
                       CELL / max(base["baseline_summary"]["plain_s"]),
                       CELL / min(base["baseline_summary"]["plain_s"]))},
        "times_come_from": "the instrumented fold, %.3f s wall, whose own note says "
                           "'this fold's wall is a diagnostic, not a baseline'"
                           % run["attrib"]["instrumented_fold_s"],
        "ops_per_ms": {s: round(N[s] / (T[s] * 1e3), 2) for s in TOP},
        "published_waste_at_cell": {s: round(wU[s], 3) for s in TOP},
        "per_op_gap_fitted_us": round(g * 1e6, 2),
        "per_op_gap_model_k": {s: round(kG[s], 4) for s in TOP},
        "per_op_gap_model_waste_at_cell": {s: round(wG[s], 3) for s in TOP},
        "per_op_gap_us_that_swaps_denoiser_and_MSA": round(gswap * 1e6, 2),
        "k_denoiser_needed_to_take_rank1": round(need_D, 4),
        "k_denoiser_over_k_pairformer_needed_for_rank1": round(need_D / k0, 3),
        "k_MSA_needed_to_take_rank2": round(need_M, 4),
        "k_MSA_over_k_denoiser_needed_for_rank2": round(need_M / k0, 3),
        "reorders_under_per_op_gap_model": wG[M] > wG[D],
    }

    # ---------------- attack 2: disjoint and exhaustive? ---------------------------------------
    def nodes(sig):
        return EF.nodes_of(str(RB / "captures" / ("cap_" + sig.replace("|", "__") + ".json.gz")))

    kids = {  # sig -> (child sig, calls of child per call of parent)
        TOP[0]: [("TriangleMultiplication|1x512x512x128,1x512x512", 2),
                 ("TriangleAttention|1x512x512x128,1x1x1x512", 2),
                 ("Transition|1x512x512x128", 1),
                 ("AttentionPairBias|1x512x384,1x512x512x128", 1),
                 ("Transition|1x512x384", 1)],
        TOP[1]: [("OuterProductMean|1x1024x512x64,1024x1x1", 1),
                 ("PairWeightedAveraging|1x1024x512x64,1x512x512x128", 1),
                 ("PairformerLayer|1x512x512x128", 1),
                 ("Transition|1x1024x512x64", 1)],
        "DiffusionTransformer|1x512x768,1x512x768":
            [("DiffusionTransformerLayer|1x512x768,1x512x768", 24)],
    }
    add = {}
    for p, cs in kids.items():
        add[p] = {
            "ms_parent": row[p]["ms_per_call"],
            "ms_children": round(sum(row[c]["ms_per_call"] * n for c, n in cs), 3),
            "ops_parent": row[p]["ops_per_call"],
            "ops_children": sum(row[c]["ops_per_call"] * n for c, n in cs),
            "GFLOP_parent": row[p]["GFLOP_per_call"],
            "GFLOP_children": round(sum(row[c]["GFLOP_per_call"] * n for c, n in cs), 1),
            "MB_parent": row[p]["MB_per_call"],
            "MB_children": round(sum(row[c]["MB_per_call"] * n for c, n in cs), 1)}
        add[p]["ms_unattributed"] = round(add[p]["ms_parent"] - add[p]["ms_children"], 3)
        add[p]["ops_unattributed"] = add[p]["ops_parent"] - add[p]["ops_children"]
    out["attack2_tiling"] = {
        "nesting_from_the_instrumented_tree": {
            k: tree[k]["calls"] for k in sorted(tree) if k.count("/") <= 2},
        "top_three_are_disjoint": True,
        "why": "the tree makes PairformerLayer|1x512x512x128 (16 calls) a child of MSALayer and "
               "Diffusion / DiffusionTransformer / DiffusionTransformerLayer children of "
               "DiffusionModule; TOP names none of them",
        "additivity": add,
        "session_s_covered_by_top_three": round(covered, 3),
        "session_plain_fold_s": base["baseline_summary"]["plain_median_s"],
        "coverage_pct": round(100 * covered / base["baseline_summary"]["plain_median_s"], 2),
        "device_s_this_session": base["baseline_summary"]["device_s"],
        "host_s_this_session": base["baseline_summary"]["host_s"],
        "coverage_pct_of_device_s": round(100 * covered / base["baseline_summary"]["device_s"], 2),
    }

    # ---------------- attack 3 + 5: bytes, and where the max is taken -------------------------
    rates = {"dense_cube_qb2c2_104.93": C,
             "shape_honest_pair_transition_transferred_26.38": C * 0.2514}
    floors = {}
    for widen in (False, True):
        agg = 0.0
        per = {n: 0.0 for n in rates}
        for s in TOP:
            nd = nodes(s)
            ops, B, _ = per_op(nd, widen)
            F = [x[2] if x[3] in ("matmul", "eltwise", "noshape") else 0 for x in EF.per_op(nd)]
            n = row[s]["calls"]
            agg += max(sum(F) / C, sum(B) / S) * n
            for label, c in rates.items():
                per[label] += sum(max(f / c, b / S) for f, b in zip(F, B)) * n
        floors["real_traffic_verbatim" if not widen else "L1_output_readers_charged"] = {
            "aggregate_max_s": round(agg, 3),
            **{"per_op_max_at_" + k: round(v, 3) for k, v in per.items()}}
    out["attack3_where_the_max_is_taken"] = {
        "published_floor_s": pub["summary"]["binding_floor_s"],
        "published_rule": "fold_B / stream_roof, summed over the three units; no max in the file",
        "stream_roof_GBps": round(S / 1e9, 1),
        "compute_roof_dense_cube_TFLOPs": round(C / 1e12, 2),
        "shape_honest_rate_source": "wk/roof-pair-transition 6d8d50b00: the pair Transition's own "
                                    "shapes reach 31.40 TFLOP/s against a 124.88 TFLOP/s dense "
                                    "cube in that session, 25.14 %; transferred to this card's "
                                    "104.93 cube as a ratio, 26.38 TFLOP/s",
        "floors": floors,
        "cell_s": CELL,
    }
    out["attack5_byte_instruments"] = {
        "real_traffic_handles_in_place_correctly": True,
        "evidence": "real_traffic.counts charges an in-place op rd += size AND w += size; "
                    "census.py:split_io drops the read entirely (round 1, 8.49 % on a block)",
        "new_defect": "moves_dram() is built from the DRAM rows only, so an op whose output was "
                      "allocated in L1 is not a reader of the DRAM it consumes",
        "fold_TB_real_traffic_verbatim": pub["summary"]["fold_TB"],
    }
    # ---------------- attack 4: is the 35-row MSA a benchmark artifact? ------------------------
    DEPTH = ["OuterProductMean|1x1024x512x64,1024x1x1",
             "PairWeightedAveraging|1x1024x512x64,1x512x512x128",
             "Transition|1x1024x512x64"]
    d_mb = sum(row[s_]["MB_per_call"] for s_ in DEPTH)
    d_s = sum(row[s_]["s_per_fold"] for s_ in DEPTH) * k0
    msa_mb = row[TOP[1]]["MB_per_call"]
    ladder = (64, 128, 256, 512, 1024)

    def pad_to(n):
        for r_ in ladder:
            if n <= r_:
                return r_
        return -(-n // ladder[-1]) * ladder[-1]

    fix, real_cap8192, real_cap16384 = 35, 8192, 8833
    out["attack4_msa_depth"] = {
        "fixture": run["env"]["protocol"]["fixture"],
        "fixture_depth_rows": run["env"]["n_msa"],
        "fixture_depth_is_constant_across_the_ladder":
            "all 30 a3m files in perf/size512/fixtures carry exactly 35 sequences, 128 aa to "
            "1568 aa, so depth is a property of the harness and not of the target",
        "same_target_real_alignment":
            "perf/capacity/cdk2_1hcl_colabfold_deep.a3m.gz, the committed ColabFold search for "
            "the same CDK2 (PDB 1HCL) the fixture tandem-repeats: 8833 rows, 8832 unique",
        "deployed_depth_ceiling": {
            "cli_default_max_msa_seqs": 8192, "worker_default_max_msa_seqs": 16384,
            "subsample_msa_default": False,
            "msa_source_when_unspecified": "tt_bio/main.py:_resolve_msa_default step 5 turns on "
                                           "the online ColabFold server"},
        "pad_rule_today": "tt_bio/token_axis.py MSA_PAD_LADDER = %s (wk/roof-msa-ladder c0802981a)"
                          % (ladder,),
        "depth_axis_MB_per_MSA_block": round(d_mb, 1),
        "depth_axis_share_of_MSA_block": round(d_mb / msa_mb, 4),
        "depth_axis_s_at_cell": round(d_s, 3),
        # The published 1.305 s is the depth axis at a padded depth of 1024. Its cost at any
        # other padded depth scales with that depth, so the ladder's prize is the difference
        # between the old rule's padded depth and the ladder's, both taken at this cell.
        "depth_axis_s_at_cell_by_rule": {
            str(n): {"pad_1024_rule": round(d_s * (-(-n // 1024) * 1024) / 1024, 3),
                     "ladder_rule": round(d_s * pad_to(n) / 1024, 3),
                     "ladder_prize_s": round(d_s * ((-(-n // 1024) * 1024) - pad_to(n)) / 1024, 3)}
            for n in (fix, 1024, real_cap8192, real_cap16384)},
        "cell_s_if_folded_at_the_real_alignment": {
            "depth": real_cap8192,
            "depth_axis_scale": round(real_cap8192 / 1024, 2),
            "projected_s": round(CELL - d_s + d_s * real_cap8192 / 1024, 2)},
    }

    print(json.dumps(out, indent=1))
    (HERE / "scoreboard.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
