#!/usr/bin/env python3
"""Re-derive b2z2's closing claim from primaries. Host only, opens no device.

Every input below is either read out of a committed artifact in this repo or carried with the
state doc that measured it. Nothing here is a new measurement and nothing is a re-quote: each rung
is recomputed and cross-checked against a second, independent construction, and the script fails
if the two disagree by more than 1 %.

    python3 perf/b2z2_final/closing.py            # print the tables
    python3 perf/b2z2_final/closing.py --json out/closing.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------------------------
# P1. The published cell, read from the page's own data file rather than from any state doc.
# ---------------------------------------------------------------------------------------------
def published_cell() -> float:
    d = json.loads((ROOT / "site" / "data" / "perf-512aa.json").read_text())
    for m in d["models"]:
        if m["name"] == "Boltz-2":
            return float(m["cells"]["p150a"]["s_per_fold"])
    raise SystemExit("Boltz-2 p150a cell not found")


CELL_S = published_cell()                       # 20.079 s, fc7fed56
CELL_BASE_ARM_S = 22.195                        # the cell's own ref: previous default, n=6, 1.18 %
NEVER_PUBLISHED_S = 23.841                      # b2x-baseline-attrib's internal re-measure

# ---------------------------------------------------------------------------------------------
# P5. Every base measured on the cell's own protocol since the campaign opened. MEASURED, BH.
# ---------------------------------------------------------------------------------------------
BASES = {
    "b2z-orchestrator n=5":           19.865,
    "b2z2-diffusion-loop-attack n=3": 19.937,
    "b2z-levers-default-on n=3":      20.054,
    "b2z2-bh-compose-landed s1 n=10": 20.188,
    "b2z2-bh-compose-landed s2 n=10": 20.788,
}

# ---------------------------------------------------------------------------------------------
# P7. The BH Pairformer stall identity (wave 1, instrument corrected by b2z2-whglx-profiler-build).
# ---------------------------------------------------------------------------------------------
BLOCK_SPAN_MS      = 36.3438
WAIT_IN_MS         = 18.3366
WAIT_OUT_MS        = 3.1066
MOVE_FRACTION      = (WAIT_IN_MS + WAIT_OUT_MS) / BLOCK_SPAN_MS      # 0.58999
MOVE_FREE_MULT     = 1.0 - MOVE_FRACTION                             # 0.41001

# P8. In-bracket host dispatch a movement-free device cannot delete (b2z-host-residual-kill).
HOST_IN_TRUNK_S    = 0.355
HOST_IN_SAMPLER_S  = 0.647
# Superseded on a quiet box by b2z2-diffusion-loop-attack: 0.33 s host-serial, 93.8 % device-bound.
HOST_IN_SAMPLER_QUIET_S = 0.330

# b2z-kernel-cycle-census's "<=32 % of the traced step is in no kernel", which redteam's
# conservative rung holds fixed. Retired by b2z2-diffusion-loop-attack: 2.75 % under trace.
LAUNCH_HELD_STALE  = 0.32
LAUNCH_HELD_MEAS   = 0.0275

# P9. What is actually MEASURED about the sampler's device side: two named under-filled sites.
SAMPLER_MEASURED_HEADROOM_S = (0.40, 0.70)

# P11. Dual chip, all MEASURED unless marked.
LINK_GBPS_PER_DIR  = 20.1
GATHER_MS          = 1.680          # 67.11 MB pair track across the on-card link
GATHERS_PER_BLOCK  = 3              # derived from a PairformerLayer.__call__ walk
BLOCK_MESH_TAX     = 1.0044         # real PairformerLayer, 1x2 replicated vs one chip
SHARDABLE_Q        = 0.7925         # from the measured 3.585x 256->512 scaling
BLOCK_ONE_CHIP_MS  = 36.702         # real block, one chip, own process, median of 7
N_BLOCKS           = 280

# Phase splits, each measured inside ONE run so the parts and the whole are the same fold.
SPLITS = {
    # b2z2-redteam-ceiling section 6, integ base arm, qb2 card 3. Closes to 0.073 s (0.6 %).
    "levered arm 18.594 s":  {"fold": 18.594, "trunk": 11.3209, "sampler": 5.2547, "rest": 2.0184},
    # b2z2-bh-compose-landed session 1, qb2 card 1, n=10, A/A 1.00229x. rest is the residual.
    "measured base 20.188 s": {"fold": 20.188, "trunk": 12.400, "sampler": 5.365, "rest": None},
}


def rungs(split: dict) -> dict:
    """The ceiling rungs for one phase split, each quoted against ITS OWN fold."""
    fold, trunk, sampler = split["fold"], split["trunk"], split["sampler"]
    rest = split["rest"] if split["rest"] is not None else fold - trunk - sampler

    def held(phase_s: float, mult: float, host_s: float) -> float:
        """A movement-free device cannot delete the host dispatch exposed in the same bracket."""
        return phase_s * mult + host_s * (1.0 - mult)

    trunk_floor = held(trunk, MOVE_FREE_MULT, HOST_IN_TRUNK_S)

    # Redteam's two sampler treatments, both of which TRANSFER the trunk's multiplier.
    samp_cons_mult = LAUNCH_HELD_STALE + (1 - LAUNCH_HELD_STALE) * MOVE_FREE_MULT
    samp_cons = held(sampler, samp_cons_mult, HOST_IN_SAMPLER_S)
    samp_opt = held(sampler, MOVE_FREE_MULT, HOST_IN_SAMPLER_S)

    # The no-transfer treatment: only what has been MEASURED about the sampler.
    lo, hi = SAMPLER_MEASURED_HEADROOM_S
    samp_meas = sampler - (lo + hi) / 2.0

    out = {
        "fold": fold, "trunk": trunk, "sampler": sampler, "rest": round(rest, 4),
        "trunk_floor_s": trunk_floor,
        "rung_trunk_only_s": trunk_floor + sampler + rest,
        "rung_trunk_sampler_cons_s": trunk_floor + samp_cons + rest,
        "rung_trunk_sampler_opt_s": trunk_floor + samp_opt + rest,
        "rung_trunk_free_sampler_measured_s": trunk_floor + samp_meas + rest,
    }
    for k in [k for k in out if k.startswith("rung_")]:
        out["x_" + k[len("rung_"):].removesuffix("_s")] = fold / out[k]
    return out


def dual_chip(trunk_pairformer_s: float) -> dict:
    """The shard, priced from this row's own measured inputs. PROJECTED where it says so."""
    w = BLOCK_ONE_CHIP_MS * SHARDABLE_Q               # 29.086 ms, the part a shard halves
    ns = BLOCK_ONE_CHIP_MS - w                        # 7.616 ms single-track, replicated
    g = GATHER_MS * GATHERS_PER_BLOCK
    sharded3 = (w / 2 + ns) * BLOCK_MESH_TAX + g
    sharded2 = (w / 2 + ns) * BLOCK_MESH_TAX + GATHER_MS * 2
    return {
        "block_one_chip_ms": BLOCK_ONE_CHIP_MS, "shardable_ms": w, "single_track_ms": ns,
        "link_ms_per_block": g,
        "block_3gather_ms": sharded3, "block_3gather_x": BLOCK_ONE_CHIP_MS / sharded3,
        "block_2gather_ms": sharded2, "block_2gather_x": BLOCK_ONE_CHIP_MS / sharded2,
        "trunk_pairformer_s": trunk_pairformer_s,
        "saves_3gather_s": trunk_pairformer_s * (1 - sharded3 / BLOCK_ONE_CHIP_MS),
        "saves_2gather_s": trunk_pairformer_s * (1 - sharded2 / BLOCK_ONE_CHIP_MS),
        "extra_host_calls_per_fold": GATHERS_PER_BLOCK * N_BLOCKS,
    }


# ---------------------------------------------------------------------------------------------
# The double-counting check. Every trunk route acts INSIDE the input-tile wait that the
# movement-free envelope already zeroes, so their product must not exceed the envelope. If it
# does, they are not independent and nobody may stack them.
# ---------------------------------------------------------------------------------------------
TRUNK_ROUTES = {
    # name: (block ratio if the route were perfect and free, source)
    "work placement / reordering": (1.1968, "b2z2-dependency-overlap, three disjoint bounds added"),
    "datum rate to zero":          (BLOCK_SPAN_MS / (BLOCK_SPAN_MS - 4.916),
                                    "b2z2-bh-tile-census, 236118 arrivals x 20.82 ns = 4.916 ms"),
    "byte axis, every byte gone":  (1.826, "b2z2-byte-axis-reopened, 89.5 % of wait x 50.5 % of span"),
    "CB ring topology":            (1.0015, "b2z2-cb-depth-prefetch, best arm"),
    "dest capacity":               (1.0000, "b2z2-tile-shape-and-format, C/B = 0.99938"),
    "tile-pass deletion":          (0.9710, "b2z2-pairformer-megakernel-build, -10.2 % passes, SLOWER"),
}


def overlap_check() -> dict:
    envelope = 1.0 / MOVE_FREE_MULT            # movement-free block, the thing they all live inside
    prod = 1.0
    for x, _ in TRUNK_ROUTES.values():
        prod *= max(x, 1.0)                    # a route that loses contributes nothing, not a debit
    return {"envelope_block_x": envelope, "naive_product_block_x": prod,
            "exceeds_envelope_by_pct": 100.0 * (prod - envelope) / envelope,
            "stacking_legitimate": bool(prod <= envelope),
            "routes": {k: v[0] for k, v in TRUNK_ROUTES.items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()

    r = {"cell_s": CELL_S, "cell_base_arm_s": CELL_BASE_ARM_S,
         "cell_levers_x": CELL_BASE_ARM_S / CELL_S,
         "never_published_s": NEVER_PUBLISHED_S,
         "move_fraction": MOVE_FRACTION, "move_free_mult": MOVE_FREE_MULT}

    # --- base spread on the cell, across sessions vs within one session ----------------------
    vals = sorted(BASES.values())
    r["bases"] = BASES
    r["base_median_s"] = st.median(vals)
    r["base_cross_session_spread_pct"] = 100.0 * (vals[-1] - vals[0]) / st.median(vals)
    r["best_within_session_aa_floor_x"] = 1.00229     # b2z2-bh-compose-landed session 1, n=5 pairs

    # --- the ceiling, two independent constructions -------------------------------------------
    r["ceiling"] = {k: rungs(v) for k, v in SPLITS.items()}
    a1, a2 = r["ceiling"]["levered arm 18.594 s"], r["ceiling"]["measured base 20.188 s"]
    r["construction_agreement_pct"] = {
        k.replace("x_", ""): 100.0 * abs(a1[k] - a2[k]) / a1[k]
        for k in a1 if k.startswith("x_")
    }
    worst = max(r["construction_agreement_pct"].values())
    assert worst < 1.0, f"the two constructions disagree by {worst:.2f} %, re-check the inputs"

    # --- the mixed-denominator defect in the published table ----------------------------------
    r["redteam_published"] = {"trunk_only": 1.656, "cons": 1.954, "opt": 2.135}
    r["redteam_reproduced_mixed"] = {
        "trunk_only": CELL_S / a1["rung_trunk_only_s"],
        "cons": CELL_S / a1["rung_trunk_sampler_cons_s"],
        "opt": CELL_S / a1["rung_trunk_sampler_opt_s"],
    }
    r["self_consistent"] = {"trunk_only": a1["x_trunk_only"], "cons": a1["x_trunk_sampler_cons"],
                            "opt": a1["x_trunk_sampler_opt"]}
    r["inflation_x"] = CELL_S / SPLITS["levered arm 18.594 s"]["fold"]
    r["headline_move_pct"] = {
        k: 100.0 * (r["self_consistent"][k] - r["redteam_published"][k]) / r["redteam_published"][k]
        for k in r["redteam_published"]
    }

    # --- what 2x would cost, in seconds, against a base that exists ---------------------------
    base = SPLITS["measured base 20.188 s"]["fold"]
    target = CELL_S / 2.0
    r["two_x"] = {
        "target_s": target,
        "must_remove_from_measured_base_s": base - target,
        "trunk_movement_free_removes_s": a2["trunk"] - a2["trunk_floor_s"],
        "sampler_measured_headroom_s": sum(SAMPLER_MEASURED_HEADROOM_S) / 2,
        "rest_removable_s": 0.0,
    }
    r["two_x"]["available_at_measured_ceilings_s"] = (
        r["two_x"]["trunk_movement_free_removes_s"] + r["two_x"]["sampler_measured_headroom_s"])
    r["two_x"]["deficit_1chip_s"] = (r["two_x"]["must_remove_from_measured_base_s"]
                                     - r["two_x"]["available_at_measured_ceilings_s"])

    # --- two chips ----------------------------------------------------------------------------
    r["dual"] = dual_chip(trunk_pairformer_s=10.22)
    # On an already movement-free trunk there is no movement left to halve, so the shard halves the
    # residual compute and the gathers become pure added cost.
    tf = a2["trunk_floor_s"]
    shard_of_floor = (tf - HOST_IN_TRUNK_S * (1 - MOVE_FREE_MULT)) / 2 * BLOCK_MESH_TAX \
        + GATHER_MS * GATHERS_PER_BLOCK * N_BLOCKS / 1000.0 + HOST_IN_TRUNK_S * (1 - MOVE_FREE_MULT)
    r["dual"]["trunk_floor_1chip_s"] = tf
    r["dual"]["trunk_floor_2chip_s"] = shard_of_floor
    r["dual"]["gather_cost_at_floor_s"] = GATHER_MS * GATHERS_PER_BLOCK * N_BLOCKS / 1000.0
    lo, hi = SAMPLER_MEASURED_HEADROOM_S
    samp_meas = a2["sampler"] - (lo + hi) / 2
    samp_opt = a2["sampler"] * MOVE_FREE_MULT + HOST_IN_SAMPLER_S * (1 - MOVE_FREE_MULT)
    r["dual"]["fold_2chip_sampler_measured_s"] = shard_of_floor + samp_meas + a2["rest"]
    r["dual"]["fold_2chip_sampler_opt_s"] = shard_of_floor + samp_opt + a2["rest"]
    r["dual"]["x_2chip_sampler_measured"] = base / r["dual"]["fold_2chip_sampler_measured_s"]
    r["dual"]["x_2chip_sampler_opt"] = base / r["dual"]["fold_2chip_sampler_opt_s"]
    r["dual"]["deficit_2chip_s"] = (r["two_x"]["must_remove_from_measured_base_s"]
                                    - r["two_x"]["available_at_measured_ceilings_s"]
                                    - (tf - shard_of_floor))
    # The shard as it actually is today: no movement-free anything, just the projected block ratio.
    r["dual"]["x_fold_shard_only"] = (base / (base - r["dual"]["saves_3gather_s"]),
                                      base / (base - r["dual"]["saves_2gather_s"]))
    r["dual"]["throughput_x_two_independent_folds"] = 2.0
    r["overlap"] = overlap_check()

    _print(r)
    if a.json:
        p = Path(a.json)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent / p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(r, indent=1, sort_keys=True) + "\n")
        print(f"\nwrote {p}")
    return 0


def _print(r: dict) -> None:
    print(f"published cell                 {r['cell_s']:.3f} s   (its own base arm "
          f"{r['cell_base_arm_s']:.3f} s = {r['cell_levers_x']:.4f}x)")
    print(f"never published                {r['never_published_s']:.3f} s")
    print(f"movement-free multiplier       {r['move_free_mult']:.5f}   "
          f"(movement {100*r['move_fraction']:.2f} % of the BH block)")
    print(f"\nbase on the cell, {len(r['bases'])} sessions   median {r['base_median_s']:.3f} s, "
          f"cross-session spread {r['base_cross_session_spread_pct']:.2f} %, "
          f"best within-session A/A floor {100*(r['best_within_session_aa_floor_x']-1):.3f} %")
    for k, v in sorted(r["bases"].items(), key=lambda kv: kv[1]):
        print(f"   {v:7.3f} s   {k}")

    print("\nCEILING RUNGS, each quoted against the fold it was derived from")
    print(f"   {'rung':<34}{'levered 18.594 s':>20}{'measured 20.188 s':>20}{'agree':>8}")
    a1, a2 = r["ceiling"]["levered arm 18.594 s"], r["ceiling"]["measured base 20.188 s"]
    for k, lab in [("trunk_only", "trunk movement-free"),
                   ("trunk_sampler_cons", "+ sampler, redteam conservative"),
                   ("trunk_sampler_opt", "+ sampler, redteam optimistic"),
                   ("trunk_free_sampler_measured", "+ sampler MEASURED headroom")]:
        print(f"   {lab:<34}{a1['x_'+k]:>19.4f}x{a2['x_'+k]:>19.4f}x"
              f"{r['construction_agreement_pct'][k]:>7.2f}%")

    print(f"\nMIXED-DENOMINATOR DEFECT   published ceiling divides the {r['cell_s']:.3f} s cell by a "
          f"floor built from the 18.594 s arm")
    print(f"   inflation factor {r['inflation_x']:.4f}x")
    print(f"   {'rung':<20}{'published':>12}{'reproduced':>12}{'self-consistent':>18}{'move':>9}")
    for k in ["trunk_only", "cons", "opt"]:
        print(f"   {k:<20}{r['redteam_published'][k]:>11.3f}x{r['redteam_reproduced_mixed'][k]:>11.3f}x"
              f"{r['self_consistent'][k]:>17.3f}x{r['headline_move_pct'][k]:>8.1f}%")

    t = r["two_x"]
    print(f"\n2x IN SECONDS, against the measured base 20.188 s")
    print(f"   target                              {t['target_s']:.3f} s")
    print(f"   must remove                         {t['must_remove_from_measured_base_s']:.3f} s")
    print(f"   trunk movement-free gives           {t['trunk_movement_free_removes_s']:.3f} s")
    print(f"   sampler measured headroom gives     {t['sampler_measured_headroom_s']:.3f} s")
    print(f"   rest (dispatch ceiling already hit) {t['rest_removable_s']:.3f} s")
    print(f"   DEFICIT, one processor              {t['deficit_1chip_s']:.3f} s")

    d = r["dual"]
    print(f"\nTWO CHIPS   link {LINK_GBPS_PER_DIR} GB/s/dir, gather {GATHER_MS} ms, "
          f"mesh tax {BLOCK_MESH_TAX}x, shardable q={SHARDABLE_Q}")
    print(f"   block {d['block_one_chip_ms']:.3f} -> {d['block_3gather_ms']:.3f} ms "
          f"({d['block_3gather_x']:.3f}x, 3 gathers) / {d['block_2gather_ms']:.3f} ms "
          f"({d['block_2gather_x']:.3f}x, 2 gathers)")
    print(f"   saves off the 10.22 s PairformerLayer track  "
          f"{d['saves_3gather_s']:.2f} - {d['saves_2gather_s']:.2f} s")
    print(f"   fold, shard only, nothing else       "
          f"{d['x_fold_shard_only'][0]:.3f}x - {d['x_fold_shard_only'][1]:.3f}x")
    print(f"   trunk floor 1 chip {d['trunk_floor_1chip_s']:.3f} s -> 2 chips "
          f"{d['trunk_floor_2chip_s']:.3f} s  (gathers alone cost "
          f"{d['gather_cost_at_floor_s']:.3f} s at the floor)")
    print(f"   2-chip ceiling, sampler MEASURED     {d['x_2chip_sampler_measured']:.3f}x")
    print(f"   2-chip ceiling, sampler transferred  {d['x_2chip_sampler_opt']:.3f}x")
    print(f"   DEFICIT, two processors              {d['deficit_2chip_s']:.3f} s")
    print(f"   throughput if the two chips run two independent folds instead  "
          f"{d['throughput_x_two_independent_folds']:.3f}x")

    o = r["overlap"]
    print(f"\nDOUBLE-COUNTING CHECK   do the trunk routes stack?")
    for k, v in sorted(o["routes"].items(), key=lambda kv: -kv[1]):
        print(f"   {k:<32}{v:>8.4f}x on the block")
    print(f"   naive product                   {o['naive_product_block_x']:>8.4f}x")
    print(f"   movement-free envelope          {o['envelope_block_x']:>8.4f}x")
    print(f"   -> the product EXCEEDS the envelope by {o['exceeds_envelope_by_pct']:.1f} %, so the "
          f"routes are NOT independent")
    print(f"   stacking legitimate: {o['stacking_legitimate']}")


if __name__ == "__main__":
    raise SystemExit(main())
