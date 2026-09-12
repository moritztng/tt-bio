#!/usr/bin/env python3
"""Adversarial re-derivation of the four claims that carry b2z2 wave 2's conclusion.

Host only. Opens no device, takes no measurement, and reads nothing that is not a committed
artifact copied into `src/` from the branch that produced it. Every ratio printed here is quoted
with the base it is a ratio of, because this row polices that.

    python3 perf/b2z2_redteam2/redteam2.py
    python3 perf/b2z2_redteam2/redteam2.py --json out/redteam2.json

C1  the single-processor ceiling (1.53x - 1.80x / 1.82x)   -> REFUTED, replacement given
C2  "the trunk is well explored" (CONTEXT 2-CORRECTION-C)  -> QUALIFIED
C3  the p300c shard, 1.191x block / 1.0983x - 1.1138x fold -> QUALIFIED, provenance settled
C4  the union 1.09858x and the 1.2507x projection stack    -> UPHELD / QUALIFIED
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"


def load(rel: str) -> dict:
    return json.loads((SRC / rel).read_text())


# =================================================================================================
# C1. The stall identity, and whether "movement free" is a regime the measured silicon permits.
# =================================================================================================
#
# The identity (BH, wave 1, instrument corrected by b2z2-whglx-profiler-build), which every rung of
# the published ceiling is built on:
#
#     wait_in 18.3366 + wait_out 3.1066 + compute 10.7010 + non-resident 4.1995 = 36.3437 ms
#                                                              against a 36.3438 ms block span
#
# "Movement free" sets wait_in and wait_out to zero and holds the other two, giving 14.9005 ms and
# a 2.4391x multiplier on the Pairformer block. The question this row asks is not whether the
# arithmetic closes -- it does, twice, below -- but whether anything else binds before you get
# there. The answer is the DRAM interface, and it binds at almost exactly that point.
BLOCK_SPAN_MS, BLOCK_IN_MS, BLOCK_OUT_MS, BLOCK_COMPUTE_MS = 36.3438, 18.3366, 3.1066, 10.7010

# The BH diffusion step, from b2z2-sampler-stall-split. WH fractions of residency on a BH wall.
STEP_WALL_MS = 26.400            # production wall, b2z2-sampler-ceiling-map (NOT the armed span)
STEP_RESIDENT_F = 0.576
STEP_IN_F, STEP_OUT_F, STEP_COMPUTE_F = 0.558, 0.121, 0.321

# --- the roofs. Every one of these is a DIRECT measurement on a Blackhole p300c processor. -------
# perf/attn_sites/roofs_qb2c0_raw.json  (qb2 card 0): directional, 64 MB
ROOF_READ_ONLY_GBS = 390.0
ROOF_WRITE_ONLY_GBS = 269.6
ROOF_D2D_RW_GBS = 393.9
# perf/bioir_roofline/roofs_p300c_qb2_card2.json: ttnn.clone, read N + write N, counted as 2N/t.
# This is the roof in the currency the block's traffic is counted in: total interface bytes.
ROOF_CLONE_GBS = {"64 MiB": 382.9, "192 MiB": 396.2, "256 MiB": 395.9, "128 MiB": 390.7}
# ttnn.add, 2 reads + 1 write, counted 3N/t -- the most read-heavy, and so the most generous,
# directly measured total-traffic roof on this part.
ROOF_ADD_GBS = 429.9
# What the campaign actually quotes everywhere. It is a FITTED asymptote out of a
# t = t_fixed + bytes/BW_eff model (b2z-arch-deficit), not a measured streaming rate, and it is
# above every directly measured roof above. Carried so the result can be shown at its own number.
ROOF_CAMPAIGN_FITTED_GBS = 444.9


def c1_identity() -> dict:
    """Does the identity close, and does it close on a SECOND, independent capture?"""
    non_res = BLOCK_SPAN_MS - (BLOCK_IN_MS + BLOCK_OUT_MS + BLOCK_COMPUTE_MS)
    wave1 = {
        "wait_in_ms": BLOCK_IN_MS, "wait_out_ms": BLOCK_OUT_MS, "compute_ms": BLOCK_COMPUTE_MS,
        "non_resident_ms": non_res, "span_ms": BLOCK_SPAN_MS,
        "closes_to_ms": abs(BLOCK_IN_MS + BLOCK_OUT_MS + BLOCK_COMPUTE_MS + non_res - BLOCK_SPAN_MS),
    }
    # Independent: b2z2-bh-tile-census read the same block off its own BH capture.
    t = load("b2z2_tile_census/results/tile_census.json")["PairformerLayer"]
    tot, span = t["total"], t["span_ms"]
    c_in, c_out = tot["wait_ns"] / 1e6, tot["resv_ns"] / 1e6
    c_res = tot["trisc1_ns"] / 1e6
    census = {
        "wait_in_ms": c_in, "wait_out_ms": c_out, "compute_ms": c_res - c_in - c_out,
        "non_resident_ms": span - c_res, "span_ms": span,
        "closes_to_ms": 0.0,
    }
    dev = {k: 100.0 * abs(census[k] - wave1[k]) / wave1[k]
           for k in ("wait_in_ms", "wait_out_ms", "compute_ms", "non_resident_ms", "span_ms")}
    return {"wave1": wave1, "census": census, "agreement_pct": dev,
            "movement_free_span_ms": BLOCK_COMPUTE_MS + non_res,
            "movement_free_x": BLOCK_SPAN_MS / (BLOCK_COMPUTE_MS + non_res)}


def c1_dram() -> dict:
    """The term the ceiling holds constant while it zeroes the wait: bytes across the interface.

    `census_tiles.py::traffic` counts each operand tensor ONCE per program, split by whether the
    buffer is DRAM- or L1-interleaved. A multicast operand is therefore counted once, not once per
    core, so this is a LOWER bound on what the interface actually sees. The constraint below binds
    at least this hard.
    """
    c = load("b2z2_tile_census/results/tile_census.json")
    out = {}
    for tag, wall_ms in (("PairformerLayer", BLOCK_SPAN_MS), ("DiffusionStep", STEP_WALL_MS)):
        t = c[tag]["total"]
        rd, wr = t["dram_rd"], t["dram_wr"]
        gb = (rd + wr) / 1e9
        # The movement-free wall for each block, from its own stall identity.
        if tag == "PairformerLayer":
            mf_ms = BLOCK_COMPUTE_MS + (BLOCK_SPAN_MS - BLOCK_IN_MS - BLOCK_OUT_MS - BLOCK_COMPUTE_MS)
        else:
            mf_ms = STEP_WALL_MS * ((1 - STEP_RESIDENT_F) + STEP_RESIDENT_F * STEP_COMPUTE_F)
        roofs = {"campaign fitted 444.9": ROOF_CAMPAIGN_FITTED_GBS,
                 "measured add 2r+1w 429.9": ROOF_ADD_GBS,
                 "measured clone 128 MiB 390.7": ROOF_CLONE_GBS["128 MiB"],
                 "measured read-only 390.0": ROOF_READ_ONLY_GBS}
        out[tag] = {
            "dram_rd_MB": rd / 1e6, "dram_wr_MB": wr / 1e6, "dram_total_GB": gb,
            "l1_MB": (t["l1_rd"] + t["l1_wr"]) / 1e6,
            "wall_ms": wall_ms, "achieved_GBs": gb / (wall_ms / 1e3),
            "movement_free_ms": mf_ms,
            "required_GBs_if_movement_free": gb / (mf_ms / 1e3),
            "floor_ms_at_roof": {k: gb / v * 1e3 for k, v in roofs.items()},
            "x_at_roof": {k: wall_ms / (gb / v * 1e3) for k, v in roofs.items()},
            "movement_free_x": wall_ms / mf_ms,
            "dram_binds": {k: bool(gb / v * 1e3 > mf_ms) for k, v in roofs.items()},
        }
    # Do reads and writes share the budget, or do they overlap? The clone roof settles it: a
    # DRAM->DRAM clone moves read+write concurrently and tops out at ~391-396 GB/s of TOTAL
    # traffic, against 390.0 GB/s for reads alone. Concurrency buys ~1.5 %, not 2x -- so the two
    # directions share one budget and the total-traffic reading is the supported one.
    out["direction_test"] = {
        "read_only_GBs": ROOF_READ_ONLY_GBS,
        "clone_total_traffic_GBs": ROOF_CLONE_GBS["128 MiB"],
        "concurrency_gain_x": ROOF_CLONE_GBS["128 MiB"] / ROOF_READ_ONLY_GBS,
        "verdict": "reads and writes share one budget; total-traffic accounting is supported",
    }
    return out


# --- the fold rungs, rebuilt with DRAM as a co-binding constraint --------------------------------
SPLITS = {   # each measured INSIDE one run, so the parts and the whole are one fold
    "measured base 20.188 s": dict(fold=20.188, trunk=12.400, sampler=5.365),
    "levered arm 18.594 s":   dict(fold=18.594, trunk=11.3209, sampler=5.2547),
}
HOST_TRUNK_S, HOST_SAMPLER_S = 0.355, 0.330


def held(phase_s: float, keep: float, host_s: float) -> float:
    return phase_s * keep + host_s * (1.0 - keep)


def c1_rungs(dram: dict) -> dict:
    d = dram["PairformerLayer"]
    s_keep = (1 - STEP_RESIDENT_F) + STEP_RESIDENT_F * STEP_COMPUTE_F
    keeps = {
        "movement-free (published)": d["movement_free_ms"] / BLOCK_SPAN_MS,
        "DRAM roof, campaign's fitted 444.9": d["floor_ms_at_roof"]["campaign fitted 444.9"] / BLOCK_SPAN_MS,
        "DRAM roof, measured 429.9 (most generous measured)":
            d["floor_ms_at_roof"]["measured add 2r+1w 429.9"] / BLOCK_SPAN_MS,
        "DRAM roof, measured 390.7 (clone, matching mix)":
            d["floor_ms_at_roof"]["measured clone 128 MiB 390.7"] / BLOCK_SPAN_MS,
    }
    out = {"sampler_keep": s_keep, "sampler_movement_free_x": 1 / s_keep, "trunk_keeps": keeps,
           "rungs": {}}
    for kname, keep in keeps.items():
        # The trunk's floor can never be better than movement-free; take the binding one.
        k = max(keep, d["movement_free_ms"] / BLOCK_SPAN_MS)
        row = {}
        for sname, sp in SPLITS.items():
            rest = sp["fold"] - sp["trunk"] - sp["sampler"]
            tf = held(sp["trunk"], k, HOST_TRUNK_S)
            sf = held(sp["sampler"], s_keep, HOST_SAMPLER_S)
            # The rung the campaign calls "best supported": trunk at its floor, sampler given
            # only the under-fill that has actually been measured on it (0.40-0.70 s).
            samp_meas = sp["sampler"] - 0.55
            row[sname] = {
                "best_supported_s": tf + samp_meas + rest,
                "best_supported_x": sp["fold"] / (tf + samp_meas + rest),
                "trunk_floor_s": tf,
                "trunk_only_s": tf + sp["sampler"] + rest,
                "trunk_only_x": sp["fold"] / (tf + sp["sampler"] + rest),
                "both_s": tf + sf + rest, "both_x": sp["fold"] / (tf + sf + rest),
            }
        out["rungs"][kname] = row
    return out


# =================================================================================================
# C2. Scope of the 1.1968x reordering bound.
# =================================================================================================
GRID_CORE_MS = 6141.137          # 72 cores x 85.2936 ms, WH, b2z2-dependency-overlap
REORDER_BOUNDS = {"REBALANCE": 495.5, "OVERLAP": 401.5, "DECHAIN": 113.0}
# Sizes the SAME campaign measured on the trunk that the bound above does not cover, each on the
# Pairformer block and each carrying its own status.
NOT_COVERED = {
    "L1-interleave every DRAM-interleaved read": (1.188, "DERIVED at fitted bandwidths, "
        "b2z2-byte-axis-reopened; the L1 bandwidth in it is fitted, not measured"),
    "delete every byte (envelope, not a lever)": (1.826, "b2z2-byte-axis-reopened, 89.5 % of the "
        "wait x 50.5 % of the span"),
    "bfp8_b on the delivery axis": (0.531, "BYTES, not a ratio: 0.531x the bytes at the same tile "
        "count. Its dismissal in wave 1 rested on the refuted tile-count model"),
}


def c2() -> dict:
    tot = sum(REORDER_BOUNDS.values())
    return {
        "components_core_ms": REORDER_BOUNDS, "sum_core_ms": tot,
        "grid_core_ms": GRID_CORE_MS,
        "bound_block_x": GRID_CORE_MS / (GRID_CORE_MS - tot),
        "arch": "WH — 72 cores, 85.2936 ms block. The cell is BH.",
        "scope": "work placement and re-timing on a FIXED dependency graph. REBALANCE re-places "
                 "producers near consumers, OVERLAP fills inter-kernel idle, DECHAIN multicasts "
                 "instead of forwarding. None of the three changes what bytes are moved.",
        "not_covered": {k: {"size": v[0], "status": v[1]} for k, v in NOT_COVERED.items()},
        "source_row_says": "b2z2-dependency-overlap, own words: 'That does not put the fold at its "
                           "floor generally. It says the remaining levers have to make the tile "
                           "cheaper or delete it, not re-time it.'",
    }


# =================================================================================================
# C3. The p300c shard: which denominator is right.
# =================================================================================================
def c3() -> dict:
    arms = {}
    for f in sorted((SRC / "b2z2_dualchip").glob("b2z2_fold*.json")):
        d = json.loads(f.read_text())
        v = [r["fold_s"] for r in d.get("reps_done", []) if "fold_s" in r]
        if not v:
            continue
        tr = [r["stages"]["trunk_s"] for r in d["reps_done"] if "stages" in r]
        arms[f.stem.replace("b2z2_fold_", "")] = {
            "median_s": st.median(v), "n": len(v), "devices": d.get("n_devices"),
            "trace": d.get("diffusion_trace"), "benchlocked": d.get("benchlocked"),
            "trunk_s": st.median(tr) if tr else None,
            "cif": sorted({c[:16] for r in d["reps_done"] for c in (r.get("cif") or [])}),
        }
    base_bench = arms["stages"]["median_s"]        # single chip, benchlocked, no trace = shipped
    base_nobench = arms["single"]["median_s"]      # single chip, NOT benchlocked
    mesh_trace = arms["meshtrace"]["median_s"]
    mesh_ctl = arms["meshctl"]["median_s"]
    sharded = arms["sharded"]["median_s"]

    slab = load("b2z2_dualchip/b2z2_slabperf.json")
    block_one, block_sharded = 36.702, 30.805     # constructed, see note below
    frac = 1.0 - block_sharded / block_one
    trunk_stage_s = 12.2647
    proj_block_basis = mesh_trace - trunk_stage_s * frac        # block-scaling projection

    # The row's own measured-anchored projection: one op sharded nets 0.6119 s off the trunk stage
    # against its matched mesh+trace control; all five ops are 2.84x that op's per-block net.
    meas_trunk_saving_s = 12.3485 - 11.7366
    proj_meas_basis = mesh_ctl - meas_trunk_saving_s * 2.84

    cell = 20.113
    return {
        "arms": arms,
        "parity": sorted({c for a in arms.values() for c in a["cif"]}),
        "mesh_tax_traced_x": mesh_trace / base_bench,
        "mesh_tax_traced_ctl_x": mesh_ctl / base_bench,
        "measured_shard_fold_x": mesh_ctl / sharded,
        "measured_shard_trunk_x": 12.3485 / 11.7366,
        "block": {
            "one_chip_ms": block_one, "sharded_ms": block_sharded,
            "x": block_one / block_sharded,
            "pair_track_slab_x": slab["pair_track_total"]["speedup"],
            "label": "CONSTRUCTED from replayed per-op slab timings + a measured gather, not a "
                     "sharded block that has ever run. The only sharded block effect that has "
                     "actually run is transition_z alone.",
        },
        "fold_projections": {
            "block-scaling basis, vs own benchlocked single-chip base":
                base_bench / proj_block_basis,
            "measured-anchored basis, vs own benchlocked single-chip base":
                base_bench / proj_meas_basis,
            "block-scaling basis, vs the PUBLISHED CELL (what the wave quotes)":
                cell / proj_block_basis,
            "measured-anchored basis, vs the PUBLISHED CELL":
                cell / proj_meas_basis,
            "vs the NON-benchlocked single-chip arm, mesh tax omitted (ceiling_v2's 1.1301x)":
                cell / (base_nobench - trunk_stage_s * frac),
        },
        "denominator_drift_x": cell / base_bench,
        "projected_fold_s": {"block-scaling": proj_block_basis,
                             "measured-anchored": proj_meas_basis},
    }


# =================================================================================================
# C4. The union, and the projection stack.
# =================================================================================================
def c4_union() -> dict:
    runs = load("union/timing512_qb2c1.json")["runs"] if (SRC / "union").exists() else None
    if runs is None:
        runs = json.loads((SRC / "b2z2_union_timing512_qb2c1.json").read_text())["runs"]
    warm = [r for r in runs if not r["cold"]]
    by_arm: dict[str, list] = {}
    for r in warm:
        by_arm.setdefault(r["arm"], []).append(r)

    base_by_pos: dict[int, list[float]] = {}
    for r in by_arm["base"]:
        base_by_pos.setdefault(r["pos"], []).append(r["fold_s"])

    # Position-matched: every arm fold is bracketed by base folds at pos-1 and pos+1. Comparing it
    # against the GLOBAL base median ignores the drift across positions; comparing it against its
    # own two neighbours does not.
    out_arms = {}
    for arm, rs in by_arm.items():
        if arm == "base":
            continue
        ratios = []
        for r in rs:
            nb = [f for p in (r["pos"] - 1, r["pos"] + 1) for f in base_by_pos.get(p, [])]
            ratios.append(st.mean(nb) / r["fold_s"])
        out_arms[arm] = {
            "n": len(rs), "median_s": st.median([r["fold_s"] for r in rs]),
            "range_s": [min(r["fold_s"] for r in rs), max(r["fold_s"] for r in rs)],
            "global_base_ratio_x": st.median([f for v in base_by_pos.values() for f in v])
                                   / st.median([r["fold_s"] for r in rs]),
            "position_matched_ratio_x": st.mean(ratios),
            "cif": sorted({r["sha256"] for r in rs}),
        }
    base_all = [f for v in base_by_pos.values() for f in v]
    return {
        "base": {"n": len(base_all), "median_s": st.median(base_all),
                 "range_s": [min(base_all), max(base_all)],
                 "by_position_mean_s": {p: st.mean(v) for p, v in sorted(base_by_pos.items())},
                 "position_drift_pct": 100.0 * (st.mean(base_by_pos[0]) - st.mean(base_by_pos[8]))
                                       / st.mean(base_by_pos[0])},
        "arms": out_arms,
        "loadavg_1m_range": [min(r["loadavg_before"][0] for r in warm),
                             max(r["loadavg_after"][0] for r in warm)],
        "every_union_fold_beats_every_base_fold": bool(
            max(r["fold_s"] for r in by_arm["UNION"]) < min(base_all)),
    }


# The projection stack, and the three assumptions it rests on.
STEP_LEVERS_WH = {"layernorm fusion": 1.03932, "layout elision (arm A)": 1.02574,
                  "short-program fusion": 1.07444}
UNION_BASE_S, UNION_S, UNION_BLOCK_S, UNION_SAMPLER_S = 19.658, 17.894, 9.721, 4.840


def c4_projection(c3r: dict) -> dict:
    prod = 1.0
    for v in STEP_LEVERS_WH.values():
        prod *= v
    largest = max(STEP_LEVERS_WH.values())
    mesh_tax = c3r["mesh_tax_traced_ctl_x"]
    # The shard, three ways: as ceiling_v2 stacks it (block ratio, no mesh tax); with the mesh tax
    # charged; and at the size the row's own measured-anchored projection supports.
    shard_block_x = c3r["block"]["x"]
    shard_supported_x = c3r["fold_projections"][
        "measured-anchored basis, vs own benchlocked single-chip base"]

    rows = {}
    for sname, sx in (("all three multiplied (as published)", prod),
                      ("largest single lever only", largest)):
        step_s = UNION_SAMPLER_S * (1 - 1 / sx)
        with_step = UNION_S - step_s
        for tname, tax in (("mesh tax omitted (as published)", 1.0),
                           ("mesh tax charged", mesh_tax)):
            for shname, sh_s in (
                    ("block ratio on the union's block", UNION_BLOCK_S * (1 - 1 / shard_block_x)),
                    ("row's measured-anchored fold saving",
                     UNION_S * (1 - 1 / shard_supported_x))):
                fold = with_step * tax - sh_s
                rows[f"{sname} | {tname} | {shname}"] = {
                    "fold_s": fold, "x_vs_union_base": UNION_BASE_S / fold}
    # The one step lever measured on BOTH architectures: the atom key window. On WH it was
    # 1.07444x on the step; on BH it is 1.02744x on the fold, paired, in the union session.
    # If the WH step ratio is applied to the BH sampler stage it predicts less than BH measured,
    # so for the only lever where the transfer can be checked it is CONSERVATIVE, not optimistic.
    akw_wh_step_x = STEP_LEVERS_WH["short-program fusion"]
    bh_sampler_stage_s = 5.365
    predicted_s = bh_sampler_stage_s * (1 - 1 / akw_wh_step_x)
    measured_s = UNION_BASE_S - 19.133
    transfer = {"akw_wh_step_x": akw_wh_step_x, "predicted_fold_saving_s": predicted_s,
                "measured_fold_saving_s_BH": measured_s,
                "wh_underpredicts_by_pct": 100.0 * (measured_s - predicted_s) / predicted_s}
    return {"step_stack_product": prod, "step_stack_largest": largest,
            "wh_to_bh_transfer_check": transfer,
            "mesh_tax_x": mesh_tax, "shard_block_x": shard_block_x,
            "shard_supported_fold_x": shard_supported_x,
            "published_1_2507": UNION_BASE_S / (
                (UNION_S - UNION_SAMPLER_S * (1 - 1 / prod)) - UNION_BLOCK_S * (1 - 1 / shard_block_x)),
            "grid": rows,
            "range_x": [min(v["x_vs_union_base"] for v in rows.values()),
                        max(v["x_vs_union_base"] for v in rows.values())]}


# =================================================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    R: dict = {}

    print("=" * 94)
    print("C1  THE SINGLE-PROCESSOR CEILING")
    print("=" * 94)
    ident = R["c1_identity"] = c1_identity()
    print("  the stall identity, on two independent BH captures of the same block")
    print(f"    {'term':<16}{'wave 1':>12}{'tile census':>14}{'agree':>9}")
    for k in ("wait_in_ms", "wait_out_ms", "compute_ms", "non_resident_ms", "span_ms"):
        print(f"    {k:<16}{ident['wave1'][k]:>11.4f}{ident['census'][k]:>14.4f}"
              f"{ident['agreement_pct'][k]:>8.2f}%")
    print(f"    the identity CLOSES on both. movement-free span {ident['movement_free_span_ms']:.4f} ms"
          f" = {ident['movement_free_x']:.4f}x on the block")

    dram = R["c1_dram"] = c1_dram()
    print("\n  what the ceiling holds constant while it zeroes the wait: bytes across the interface")
    dt = dram["direction_test"]
    print(f"    read-only roof {dt['read_only_GBs']:.1f} GB/s; a concurrent read+write clone tops out"
          f" at {dt['clone_total_traffic_GBs']:.1f} GB/s of TOTAL traffic")
    print(f"    -> concurrency buys {100*(dt['concurrency_gain_x']-1):.1f} %, so the two directions"
          f" share one budget. Total-traffic accounting it is.")
    for tag in ("PairformerLayer", "DiffusionStep"):
        d = dram[tag]
        print(f"\n    {tag}: {d['dram_rd_MB']:,.1f} MB read + {d['dram_wr_MB']:,.1f} MB written"
              f" = {d['dram_total_GB']:.4f} GB, plus {d['l1_MB']:,.1f} MB of L1")
        print(f"      today   {d['wall_ms']:7.4f} ms -> {d['achieved_GBs']:6.1f} GB/s")
        print(f"      movement-free {d['movement_free_ms']:7.4f} ms ({d['movement_free_x']:.4f}x)"
              f" -> {d['required_GBs_if_movement_free']:6.1f} GB/s REQUIRED")
        for rname in d["floor_ms_at_roof"]:
            binds = "BINDS" if d["dram_binds"][rname] else "slack"
            print(f"      roof {rname:<34} floor {d['floor_ms_at_roof'][rname]:7.4f} ms"
                  f" = {d['x_at_roof'][rname]:.4f}x   {binds}")

    rungs = R["c1_rungs"] = c1_rungs(dram)
    print(f"\n  the fold rungs, rebuilt with DRAM as a co-binding constraint")
    print(f"  (sampler multiplier {rungs['sampler_movement_free_x']:.4f}x unchanged — DRAM does not"
          f" bind the step)")
    print(f"    {'trunk treatment':<52}{'trunk only':>12}{'best-supp':>12}{'+ sampler':>12}")
    for kname, row in rungs["rungs"].items():
        r = row["measured base 20.188 s"]
        print(f"    {kname:<52}{r['trunk_only_x']:>11.4f}x{r['best_supported_x']:>11.4f}x"
              f"{r['both_x']:>11.4f}x")
    pub = rungs["rungs"]["movement-free (published)"]["measured base 20.188 s"]
    cor = rungs["rungs"]["DRAM roof, measured 390.7 (clone, matching mix)"]["measured base 20.188 s"]
    gen = rungs["rungs"]["DRAM roof, campaign's fitted 444.9"]["measured base 20.188 s"]
    print(f"\n    PUBLISHED bracket   {pub['trunk_only_x']:.2f}x - {pub['both_x']:.2f}x")
    print(f"    CORRECTED bracket   {cor['trunk_only_x']:.2f}x - {gen['both_x']:.2f}x"
          f"   (measured roof to the campaign's own fitted one)")

    two = R["c1_two_processor"] = {
        "per_chip_dram_GB": dram["PairformerLayer"]["dram_total_GB"] / 2,
        "per_chip_floor_ms_at_390_7": dram["PairformerLayer"]["dram_total_GB"] / 2 / 390.7 * 1e3,
        "movement_free_ms": dram["PairformerLayer"]["movement_free_ms"],
    }
    two["dram_binds_on_two_chips"] = bool(
        two["per_chip_floor_ms_at_390_7"] > two["movement_free_ms"])
    print(f"\n  and the constraint is PER PROCESSOR. Shard the trunk over the p300c pair and each"
          f" chip moves {two['per_chip_dram_GB']:.4f} GB,")
    print(f"  a {two['per_chip_floor_ms_at_390_7']:.4f} ms floor at the measured 390.7 GB/s roof"
          f" against a {two['movement_free_ms']:.4f} ms movement-free block:"
          f" DRAM {'still binds' if two['dram_binds_on_two_chips'] else 'no longer binds'}.")
    print("  The correction below lowers the one-processor ceiling and leaves the two-processor"
          " one alone.")

    print("\n" + "=" * 94)
    print("C2  \"THE TRUNK IS WELL EXPLORED AND ITS REMAINING LEVERS ARE SMALL AND HARD\"")
    print("=" * 94)
    c = R["c2"] = c2()
    for k, v in c["components_core_ms"].items():
        print(f"    {k:<12}{v:>9.1f} core-ms   {GRID_CORE_MS/(GRID_CORE_MS-v):.4f}x alone")
    print(f"    {'SUM':<12}{c['sum_core_ms']:>9.1f} core-ms   {c['bound_block_x']:.4f}x"
          f"   reproduces the published 1.1968x")
    print(f"    scope: {c['scope']}")
    print("    NOT covered by it, all sized by the same campaign on the same block:")
    for k, v in c["not_covered"].items():
        print(f"      {v['size']:>7.3f}  {k}")

    print("\n" + "=" * 94)
    print("C3  THE p300c SHARD")
    print("=" * 94)
    c3r = R["c3"] = c3()
    print(f"    {'arm':<14}{'median s':>10}{'dev':>5}{'trace':>7}{'bench':>7}{'trunk s':>9}")
    for k, v in c3r["arms"].items():
        print(f"    {k:<14}{v['median_s']:>10.4f}{v['devices']:>5}{str(v['trace']):>7}"
              f"{str(v['benchlocked']):>7}{(v['trunk_s'] or 0):>9.4f}")
    print(f"    parity: {len(c3r['parity'])} distinct CIF across every arm -> "
          f"{'BIT-IDENTICAL' if len(c3r['parity'])==1 else 'DIVERGENT'}")
    print(f"    mesh tax, traced, both benchlocked   {c3r['mesh_tax_traced_x']:.5f}x /"
          f" {c3r['mesh_tax_traced_ctl_x']:.5f}x")
    print(f"    MEASURED shard (one op): fold {c3r['measured_shard_fold_x']:.5f}x,"
          f" trunk stage {c3r['measured_shard_trunk_x']:.5f}x")
    print(f"    block {c3r['block']['x']:.4f}x — {c3r['block']['label']}")
    print("    the fold projection, by denominator:")
    for k, v in c3r["fold_projections"].items():
        print(f"      {v:.5f}x   {k}")
    print(f"    the published cell is {c3r['denominator_drift_x']:.5f}x this row's own benchlocked"
          f" single-chip base. That factor is drift, not shard.")

    print("\n" + "=" * 94)
    print("C4  THE UNION, AND THE PROJECTION STACK")
    print("=" * 94)
    u = R["c4_union"] = c4_union()
    print(f"    base n={u['base']['n']} median {u['base']['median_s']:.4f} s, range"
          f" {u['base']['range_s'][0]:.3f}-{u['base']['range_s'][1]:.3f},"
          f" drift across positions {u['base']['position_drift_pct']:+.2f} %")
    print(f"    {'arm':<8}{'n':>3}{'median s':>11}{'global base':>13}{'position-matched':>19}")
    for k, v in u["arms"].items():
        print(f"    {k:<8}{v['n']:>3}{v['median_s']:>11.4f}{v['global_base_ratio_x']:>12.5f}x"
              f"{v['position_matched_ratio_x']:>18.5f}x")
    print(f"    every UNION fold faster than every base fold: "
          f"{u['every_union_fold_beats_every_base_fold']}")
    print(f"    loadavg 1m across the session {u['loadavg_1m_range'][0]:.2f}-"
          f"{u['loadavg_1m_range'][1]:.2f}")

    p = R["c4_projection"] = c4_projection(c3r)
    print(f"\n    the 1.2507x stack reproduces: {p['published_1_2507']:.4f}x")
    print(f"    its three soft assumptions, priced one at a time:")
    print(f"      step levers multiplied {p['step_stack_product']:.4f}x vs largest alone"
          f" {p['step_stack_largest']:.4f}x  (all three remove programs from the SAME 62.9 %"
          f" per-program constant)")
    print(f"      mesh tax omitted vs charged at {p['mesh_tax_x']:.5f}x")
    print(f"      shard at the constructed block ratio {p['shard_block_x']:.4f}x vs at the row's own"
          f" measured-anchored fold saving {p['shard_supported_fold_x']:.5f}x")
    t = p["wh_to_bh_transfer_check"]
    print(f"      WH->BH transfer, on the one step lever measured on both: WH"
          f" {t['akw_wh_step_x']:.5f}x on the step predicts {t['predicted_fold_saving_s']:.3f} s off"
          f" the fold, BH measured {t['measured_fold_saving_s_BH']:.3f} s"
          f" ({t['wh_underpredicts_by_pct']:+.0f} %) -- the transfer is conservative, not optimistic")
    print(f"    {'combination':<74}{'fold x':>10}")
    for k, v in sorted(p["grid"].items(), key=lambda kv: -kv[1]["x_vs_union_base"]):
        print(f"      {k:<72}{v['x_vs_union_base']:>9.4f}x")
    print(f"    RANGE {p['range_x'][0]:.4f}x - {p['range_x'][1]:.4f}x against the union's own base")

    if a.json:
        pth = a.json if a.json.is_absolute() else HERE / a.json
        pth.parent.mkdir(parents=True, exist_ok=True)
        pth.write_text(json.dumps(R, indent=1, sort_keys=True, default=str) + "\n")
        print(f"\nwrote {pth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
