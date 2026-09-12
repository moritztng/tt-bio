#!/usr/bin/env python3
"""Wave 2's ceiling, third derivation -- the two measurements that superseded ceiling_v2.

Host only. Opens no device, takes no new measurement, reads the published cell off the page.
Every input below is MEASURED and names the row that measured it.

What changed since `ceiling_v2.py`, and why that file must not be quoted any more:

  * `b2z2-redteam-v2` REFUTED the 1.53x-1.80x single-processor bracket. The stall identity holds;
    the regime built on it does not. "Movement free" holds the BYTES constant without saying so.
    The Pairformer block moves 8.0493 GB of DRAM traffic per call, so its movement-free time
    implies 540.2 GB/s from a part that measures 390.7 GB/s (ttnn.clone, read+write, same
    currency). The 444.9 GB/s this campaign quoted everywhere is a FITTED asymptote and sits above
    every directly measured roof. Corrected bracket: 1.35x-1.66x, best-supported 1.40x-1.49x.
  * `b2z2-trunk-shard-scale-wh` MEASURED the shard at 2, 4 and 8 chips instead of projecting it.
    It scales and it is bit-exact at every width, but 33.4 % of the block is replicated work that
    no chip count removes, so the block caps at 2.299x with the measured link and 2.990x with a
    free one -- at ANY width.
  * `b2z2-bh-union-step` raised the measured single-chip stack from 1.09858x to 1.12862x on the
    cell, and `b2z2-redteam-v2` re-derived the parent figure independently to within 0.17 %.

The question this file answers: with everything wave 2 has MEASURED, plus an unlimited number of
Blackhole processors, what is the best fold ratio anyone can name? It is not 2x.

    python3 perf/b2z2_orch/ceiling_v3.py
    python3 perf/b2z2_orch/ceiling_v3.py --json perf/b2z2_orch/ceiling_v3.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def published_cell() -> float:
    """The ratio denominator, read from the page rather than from any state doc."""
    d = json.loads((ROOT / "site" / "data" / "perf-512aa.json").read_text())
    for m in d["models"]:
        if m["name"] == "Boltz-2":
            return float(m["cells"]["p150a"]["s_per_fold"])
    raise SystemExit("Boltz-2 p150a cell not found in site/data/perf-512aa.json")


CELL_S = published_cell()

# -- MEASURED inputs -------------------------------------------------------------------------
# Trunk seconds in the fold: PairformerLayer device spans, CONTEXT §2-CORRECTION (BH).
TRUNK_S = 10.22
# The single-chip stack on the cell: b2z2-bh-union-step, BH, paired against its own interleaved
# base, A/A floor 1.01162x, n=10 stack / n=30 base, 65 warm folds in one process.
STACK_RATIO = 1.12862
# Of the stack's five levers, TT_BIO_UNFUSED_SILU is the only one acting inside the Pairformer
# block; b2z2-bh-union-clean measured it at 1.02423x on the fold, alone, on the cell.
SILU_FOLD_RATIO = 1.02423
# Trunk shard, block ratios vs one chip: b2z2-trunk-shard-scale-wh, WH, 512 aa, median of 15 warm
# reps, bit-exact at every width. Fit compute(N) = 28.679 + 58.916/N, R2 0.9964.
SHARD_BLOCK = {1: 1.0000, 2: 1.3422, 4: 1.6404, 8: 1.9424}
SHARD_CAP_MEASURED_LINK = 2.299   # any width, measured link
SHARD_CAP_FREE_LINK = 2.990       # any width, link cost zero
# The same shard, measured on a BH fold at N=2: b2z2-dual-chip-fold, corrected by b2z2-redteam-v2
# C3 against the row's own benchlocked base (the published 1.1142x/1.1301x divided by the cell,
# which is 1.01634x that base, and booked cross-session drift as shard win).
SHARD_FOLD_BH_N2 = 1.0857
# Sampler shard. The TOKEN axis is a measured NO-GO (b2z2-sharded-sampler): 56.3 % of the token DiT
# does not depend on token count, so a token shard tops out at 1.2825x with a free link and measures
# 1.0411x with the real one. The ATOM axis is a measured GO (b2z2-atom-axis-shard, 2026-09-12) and
# that verdict did not transfer: an atom layer's weights are ~0.55 MB against 4.6 MB of activations,
# so a row split divides the LARGE term there and the small one in the token DiT. Fitted over five
# widths, traced: t = 1.23565 ms + 0.00251221 ms/atom, R2 0.998604 -> 9.893 % constant on the work
# side. The TIME floor is worse than the work floor because the collectives grow with the mesh where
# the halo does not: t(N) = 1.24185 ms + 4.96906 ms/N, R2 0.998766 -> 19.995 % replicated.
# Bit-exact (torch.equal, max abs 0.0) at four atom counts and at both mesh widths.
SAMPLER_TOKEN_AXIS_SHARDS = False
SAMPLER_S = 5.365                 # BH sampler stage wall, CONTEXT §1
SAMPLER_STEP = {1: 1.0, 2: 1.13353, 4: 1.22779}   # atom shard composed onto the step, WH
SAMPLER_STEP_CAP = 1.3237         # atom track capped at 5.0x by its 19.995 % time floor, composed
                                  # onto a step in which the atom track is 12.340 of 40.366 ms
# Single-processor bracket after the bandwidth intersection: b2z2-redteam-v2.
ONE_CHIP_BRACKET = (1.35, 1.66)
ONE_CHIP_BEST_SUPPORTED = (1.40, 1.49)


# -- The MSA track: a MEASURED NO-GO on its own row axis (b2z2-msa-axis-shard, 2026-09-12) -----
# The fold's third stage, and the last one nobody had asked the shard question about. It is
# 3.8337 s of a 41.4333 s Wormhole fold (9.25 %, b2z2-msa-layer-census). Fitted over five row
# counts on a real settled `MSALayer.__call__` grabbed out of a 512 aa fold and replayed -- the
# same method both other shard verdicts in this wave were decided with:
#     t = 136.5814 ms + 0.097335 ms/row, R2 0.996770  ->  57.812 % constant at the 1024 bucket.
# The pre-registered falsifier was 45 %. **The MSA track is the TOKEN DiT's machine (56.3 %), not
# the atom track's (9.893 %).** All five widths ran the identical code path (PWA whole/blocked
# 71/0, OPM join_split 0, small-depth 0), so the fit is not a path switch.
#
# The byte-ratio heuristic that decided the atom axis MIS-PREDICTS this block, and that is the
# transferable finding: MSALayer's weights are 1.52 MB against 67.1 MB of activations -- 2.27 %,
# FOUR TIMES more favourable than the atom track's 12 % -- and it still does not shard. The
# heuristic cannot see a sub-unit that carries no row index at all, and MSALayer has one.
MSA_ROW_AXIS_SHARDS = False
MSA_CONSTANT_FRACTION = 0.57812    # measured, five widths, WH, R2 0.996770
MSA_TRACK_S_WH = 3.8337            # WH. The MSA share of the BH cell is NOT measured by any row.
MSA_FOLD_S_WH = 41.4333            # WH
# Where the constant is, per sub-unit, same session, same five widths. `msa_transition` is the
# internal control: the same instrument reads 1.87 % on a sub-unit that IS per row.
MSA_SUBUNIT_CONSTANT = {           # sub-unit -> (constant fraction, ms at 1024 rows)
    "pairformer_layer": (0.9991, 79.613),      # pure pair track, b = 0.000074 ms/row
    "outer_product_mean": (0.7536, 61.587),    # the 536.9 MB post-contraction layout chain
    "pair_weighted_averaging": (0.1428, 72.357),
    "msa_transition": (0.0187, 21.455),
}
# 79.613 + 61.587 * 0.7536 = 125.99 ms of the 238.629 ms call is pair-track work on the
# [tokens, tokens] axes. It has no MSA row index -- but it is the axis the trunk shard already
# divides bit-exactly. The MSA track's constant is not un-shardable; it is on the OTHER axis.
MSA_PAIR_TRACK_MS = 125.99
MSA_LAYER_MS = 238.629             # WH, measured, reproduces the census's 238.626 to 0.001 %


# -- The bandwidth intersection, as a target rather than a verdict -----------------------------
# b2z2-redteam-v2's primaries for the Pairformer block, all MEASURED:
BLOCK_MS_TODAY = 36.3438          # wave 1's capture, reproduced on a second independent BH capture
BLOCK_MS_MOVEMENT_FREE = 14.9006  # the stall identity with the input wait zeroed
BLOCK_GB_PER_CALL = 8.0493        # per program, each operand tensor once -> a LOWER bound
ROOF_GBPS = 390.7                 # ttnn.clone, read+write, the same currency as the byte count
# b2z2-trunk-byte-floor censused the block's bytes buffer-keyed (every operand keyed on its device
# ALLOCATION, not its tensor id) and found how many are redundant rather than how many exist.
REMOVABLE_BYTE_FRACTION = 0.0517  # 416.3 MB of 8.0493 GB. Everything else is read exactly once:
                                  # 58.8 % of the traffic is 63 single-use intermediates with one
                                  # producer and one consumer each, which is program fusion, not
                                  # redundancy -- and the megakernel already lost 2.9 % on that.


def byte_target() -> dict:
    """How far the block's bytes must fall before the bandwidth roof stops binding it.

    The campaign's top rung assumed the block could reach BLOCK_MS_MOVEMENT_FREE. At the measured
    roof that time can only carry ROOF_GBPS * t bytes, so the rung is reachable only if the block
    moves that many bytes or fewer. Everything here is division; the value is that it turns "the
    trunk is bytes-bound" into a number a row can aim at.
    """
    byte_bound_ms = BLOCK_GB_PER_CALL / ROOF_GBPS * 1000.0
    floor_ms = max(BLOCK_MS_MOVEMENT_FREE, byte_bound_ms)
    gb_allowed = ROOF_GBPS * BLOCK_MS_MOVEMENT_FREE / 1000.0
    gb_after = BLOCK_GB_PER_CALL * (1.0 - REMOVABLE_BYTE_FRACTION)
    after_ms = gb_after / ROOF_GBPS * 1000.0
    return {
        "block_ms_today": BLOCK_MS_TODAY,
        "block_gbps_today": BLOCK_GB_PER_CALL / BLOCK_MS_TODAY * 1000.0,
        "byte_bound_ms": byte_bound_ms,
        "movement_free_ms": BLOCK_MS_MOVEMENT_FREE,
        "block_floor_ms": floor_ms,
        "block_ratio_at_floor": BLOCK_MS_TODAY / floor_ms,
        "gb_allowed_at_movement_free": gb_allowed,
        "byte_cut_needed_pct": (1.0 - gb_allowed / BLOCK_GB_PER_CALL) * 100.0,
        "byte_cut_available_pct": REMOVABLE_BYTE_FRACTION * 100.0,
        "gb_after_every_removable_byte": gb_after,
        "byte_bound_ms_after": after_ms,
        "block_floor_ms_after": max(BLOCK_MS_MOVEMENT_FREE, after_ms),
        "block_ratio_at_floor_after": BLOCK_MS_TODAY / max(BLOCK_MS_MOVEMENT_FREE, after_ms),
    }


def split_stack() -> tuple[float, float, float, float]:
    """Where the stack's seconds come off, as a bracket rather than an assumption.

    The composite depends on how much of the stack's saving is already inside the trunk, because
    the shard then has less left to divide. Two ends, both defensible:
      A  none of it is (every lever but silu is a sampler or host lever) -> trunk untouched
      B  all of silu's fold saving is (silu is the one lever acting inside the block)
    """
    stacked_s = CELL_S / STACK_RATIO
    saved = CELL_S - stacked_s
    rest = CELL_S - TRUNK_S
    silu_s = CELL_S - CELL_S / SILU_FOLD_RATIO
    # A: all saving outside the trunk.  B: silu's share comes out of the trunk.
    return (TRUNK_S, rest - saved, TRUNK_S - silu_s, rest - (saved - silu_s))


def composite(trunk_s: float, rest_s: float, block_ratio: float) -> float:
    """Fold ratio for a stack that leaves `rest_s` alone and divides `trunk_s` by the shard."""
    return CELL_S / (rest_s + trunk_s / block_ratio)


def msa_row_axis() -> dict:
    """What the MSA track would have been worth if its row axis HAD divided, and what it is.

    Reported on the WH fold it was measured on, never composed into the BH cell above: no row has
    measured the MSA track's share of a Blackhole fold, and this wave has already retracted one
    cross-architecture comparison. The conclusion does not need it -- the best case is small
    enough on either part to be a rounding error against the trunk and sampler shards.
    """
    c = MSA_CONSTANT_FRACTION
    best = {}
    for n in (2, 4, 8):
        track = 1.0 / (c + (1.0 - c) / n)            # free link, perfect split, no collective
        saved = MSA_TRACK_S_WH * (1.0 - 1.0 / track)
        best[n] = (track, MSA_FOLD_S_WH / (MSA_FOLD_S_WH - saved))
    track_inf = 1.0 / c
    saved_inf = MSA_TRACK_S_WH * (1.0 - 1.0 / track_inf)
    return {"shards": MSA_ROW_AXIS_SHARDS, "constant_fraction": c,
            "ceiling_by_width_wh": best,
            "ceiling_any_width_wh": (track_inf, MSA_FOLD_S_WH / (MSA_FOLD_S_WH - saved_inf)),
            "pair_track_ms_of_layer": MSA_PAIR_TRACK_MS,
            "pair_track_fraction": MSA_PAIR_TRACK_MS / MSA_LAYER_MS}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()

    tA, rA, tB, rB = split_stack()
    rows = []
    for label, ratio in [
        *[(f"shard N={n} (WH block, measured)", r) for n, r in SHARD_BLOCK.items() if n > 1],
        ("shard cap, measured link (any N)", SHARD_CAP_MEASURED_LINK),
        ("shard cap, FREE link (any N)", SHARD_CAP_FREE_LINK),
    ]:
        lo, hi = sorted((composite(tB, rB, ratio), composite(tA, rA, ratio)))
        rows.append({"route": label, "block_ratio": ratio, "fold_lo": lo, "fold_hi": hi})

    # The multi-chip route WITH the sampler's atom-axis shard. One extra assumption, stated: the
    # sampler keeps its proportional share of whatever the stack left in `rest`. The overlap is real
    # and it is why this is not simply the row above times a sampler factor -- three of the stack's
    # five levers live in the sampler stage that this term divides.
    srows = []
    for n in (2, 4):
        for lbl, (tr, rs) in (("A", (tA, rA)), ("B", (tB, rB))):
            frac = SAMPLER_S / (CELL_S - TRUNK_S)
            samp, other = rs * frac, rs * (1.0 - frac)
            srows.append(CELL_S / (other + samp / SAMPLER_STEP[n] + tr / SHARD_BLOCK[n]))
        lo, hi = sorted(srows[-2:])
        rows.append({"route": f"+ BOTH shards, N={n} (WH, PROJECTED)",
                     "block_ratio": SHARD_BLOCK[n], "fold_lo": lo, "fold_hi": hi})
    caps = []
    for tr, rs in ((tA, rA), (tB, rB)):
        frac = SAMPLER_S / (CELL_S - TRUNK_S)
        samp, other = rs * frac, rs * (1.0 - frac)
        caps.append(CELL_S / (other + samp / SAMPLER_STEP_CAP + tr / SHARD_CAP_MEASURED_LINK))
    rows.append({"route": "+ BOTH shards at their caps (any N)", "block_ratio": SHARD_CAP_MEASURED_LINK,
                 "fold_lo": min(caps), "fold_hi": max(caps)})

    # Calibration check: the WH block curve at N=2 against the BH fold measurement of the same
    # shard. If they disagree, every larger-N row inherits the same transfer error.
    n2_lo, n2_hi = rows[0]["fold_lo"], rows[0]["fold_hi"]
    n2_marginal = ((n2_lo + n2_hi) / 2) / STACK_RATIO
    calib = n2_marginal / SHARD_FOLD_BH_N2

    out = {
        "cell_s": CELL_S,
        "cell_source": "site/data/perf-512aa.json, Boltz-2 p150a",
        "measured_single_chip_stack": STACK_RATIO,
        "measured_single_chip_stack_s": CELL_S / STACK_RATIO,
        "one_chip_bracket": ONE_CHIP_BRACKET,
        "one_chip_best_supported": ONE_CHIP_BEST_SUPPORTED,
        "msa_row_axis": msa_row_axis(),
        "sampler_token_axis_shards": SAMPLER_TOKEN_AXIS_SHARDS,
        "sampler_atom_axis_step": SAMPLER_STEP,
        "sampler_atom_axis_step_cap": SAMPLER_STEP_CAP,
        "trunk_s": TRUNK_S,
        "split_A_trunk_rest": [tA, rA],
        "split_B_trunk_rest": [tB, rB],
        "routes": rows,
        "wh_to_bh_shard_calibration": calib,
        "target_2x_s": CELL_S / 2.0,
        "byte_target": byte_target(),
    }

    print(f"published cell                 {CELL_S:.3f} s   (site/data/perf-512aa.json)")
    print(f"2x target                      {CELL_S/2:.3f} s")
    print(f"MEASURED single-chip stack     {STACK_RATIO:.5f}x = {CELL_S/STACK_RATIO:.3f} s"
          "   (b2z2-bh-union-step, BH, A/A floor 1.01162x)")
    print(f"one-processor ceiling          {ONE_CHIP_BRACKET[0]:.2f}x - {ONE_CHIP_BRACKET[1]:.2f}x"
          f"   (best supported {ONE_CHIP_BEST_SUPPORTED[0]:.2f}x - {ONE_CHIP_BEST_SUPPORTED[1]:.2f}x)")
    print(f"trunk / rest of fold           {TRUNK_S:.3f} s / {CELL_S-TRUNK_S:.3f} s")
    print()
    print("stack + shards. The TOKEN axis of the sampler does not shard; the ATOM axis does.")
    print(f"  {'route':<36} {'block':>7}  {'fold ratio':>18}")
    for r in rows:
        print(f"  {r['route']:<36} {r['block_ratio']:>6.3f}x  "
              f"{r['fold_lo']:>7.4f}x - {r['fold_hi']:.4f}x")
    print()
    print(f"WH-block -> BH-fold shard calibration at N=2: {calib:.3f}x")
    print("  (>1 means the WH block curve OVER-predicts the BH fold contribution of the same")
    print("   shard, so every larger-N row above is an upper bound, not an estimate)")
    print()
    bt = byte_target()
    tA, rA, tB, rB = split_stack()
    print("the trunk's own floor, once the bandwidth roof is applied:")
    print(f"  block today                  {bt['block_ms_today']:.4f} ms "
          f"at {bt['block_gbps_today']:.1f} GB/s")
    print(f"  movement-free (stall identity){bt['movement_free_ms']:>8.4f} ms "
          f"-- needs {BLOCK_GB_PER_CALL/BLOCK_MS_MOVEMENT_FREE*1000:.1f} GB/s")
    print(f"  byte-bound at the roof       {bt['byte_bound_ms']:.4f} ms "
          f"at {ROOF_GBPS:.1f} GB/s")
    print(f"  => block floor               {bt['block_floor_ms']:.4f} ms = "
          f"{bt['block_ratio_at_floor']:.4f}x on the block")
    print(f"  the byte cut that would make movement-free reachable: "
          f"**{bt['byte_cut_needed_pct']:.1f} %** "
          f"({BLOCK_GB_PER_CALL:.4f} -> {bt['gb_allowed_at_movement_free']:.4f} GB/call)")
    print(f"  the byte cut that is actually AVAILABLE (measured, b2z2-trunk-byte-floor): "
          f"**{bt['byte_cut_available_pct']:.2f} %** "
          f"-> floor {bt['block_floor_ms_after']:.4f} ms = "
          f"{bt['block_ratio_at_floor_after']:.4f}x on the block")
    print("  => the roof still binds after every redundant byte in the block is deleted.")
    mf_A = composite(tA, rA, BLOCK_MS_TODAY / BLOCK_MS_MOVEMENT_FREE)
    mf_B = composite(tB, rB, BLOCK_MS_TODAY / BLOCK_MS_MOVEMENT_FREE)
    print(f"  and even a FULLY movement-free trunk, on ONE processor, with the measured stack, is "
          f"{min(mf_A, mf_B):.4f}x - {max(mf_A, mf_B):.4f}x")
    print()
    m = msa_row_axis()
    print()
    print("the MSA track, the fold's third stage: it does NOT divide on its row axis.")
    print(f"  constant fraction, measured over five widths (WH)  {100*m['constant_fraction']:.3f} %"
          f"   (falsifier bar was 45 %)")
    print("  ceiling if it HAD sharded -- free link, perfect split, no collective, WH fold:")
    for n, (track, fold) in m["ceiling_by_width_wh"].items():
        print(f"    N={n:<2d}  {track:.4f}x on the track   {fold:.4f}x on the fold")
    t8, f8 = m["ceiling_any_width_wh"]
    print(f"    any N {t8:.4f}x on the track   {f8:.4f}x on the fold")
    print(f"  => the BH composite above is UNCHANGED at {min(r['fold_lo'] for r in rows[-1:]):.4f}x"
          f" - {max(r['fold_hi'] for r in rows[-1:]):.4f}x. The third stage adds nothing.")
    print(f"  {m['pair_track_ms_of_layer']:.2f} ms of the 238.63 ms layer "
          f"({100*m['pair_track_fraction']:.1f} %) is pair-track work with no MSA row index --")
    print("  which is the axis the trunk shard already divides. The constant is on the OTHER axis.")

    best = max(r["fold_hi"] for r in rows)
    print(f"BEST NAMEABLE, unlimited processors, free link: {best:.4f}x = {CELL_S/best:.3f} s")
    print(f"2x needs {CELL_S/2:.3f} s. It is not in this table.")

    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
