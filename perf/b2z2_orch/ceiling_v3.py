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
TRUNK_S = 10.155                  # redteam-v3, -0.64 %: 84.2 % of host contention lands in the
                                  # remainder and only 5.4 % in the trunk
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
# CORRECTED by b2z2-redteam-v3. This file previously composed the sampler shard onto a WORMHOLE step
# census -- "the atom track is 12.340 of a 40.366 ms step" -- and applied it to BLACKHOLE sampler
# seconds. A Blackhole census of the same 1066 programs already existed on disk and was never used
# (b2z2-sharded-sampler's step_track_split.json, qb2 card 0, 11x10). On Blackhole the atom track is
# 6.0538 of 22.0152 ms = 22.93 % of the 26.402 ms wall, because 16.62 % of that wall is EXPOSED
# DISPATCH that a mesh shard cannot divide. Same mistake class as the MSA roof error: a Wormhole
# number standing in for a Blackhole one that already existed.
SAMPLER_S = 5.158                 # BH sampler stage, redteam-v3: 5.365 mixed a stage wall with a
                                  # device-span trunk, a scope mix CONTEXT forbids (-3.85 %)
ATOM_TRACK_FRAC_BH = 0.2293       # atom track as a fraction of the BH step wall
ATOM_SHARD = {1: 1.0, 2: 1.62694, 4: 2.54386}     # MEASURED on the atom track, WH, bit-exact
ATOM_SHARD_CAP = 5.0              # 19.995 % time-side replicated floor
# MSA axis: measured NO-GO. b2z2-msa-axis-shard fits t = 136.5814 ms + 0.097335 ms/row, R2 0.996770
# -> 57.812 % constant against a 45 % pre-registered bar. The MSA track is the token DiT's machine
# (56.3 %), not the atom track's (9.893 %). Free link, perfect split, no collective is 1.2673x on the
# track at two chips = 1.0199x on a WH fold, so it is excluded here rather than modelled.
MSA_AXIS_SHARDS = False
# The measured trunk give-up inside the headline stack. redteam-v3 read the per-fold `block_s` column
# the headline session recorded and nobody had opened: the trunk gives up 0.4542 s, which settles the
# A/B bracket this file used to carry as an assumption and REFUTES its split-A end (0.0000 s).
STACK_TRUNK_GIVEUP_S = 0.4542
# What redteam-v3 published, for this file to check itself against.
REDTEAM3_CAP = 1.7431
REDTEAM3_CAP_CALIBRATED = 1.7021


def sampler_step(n_or_cap) -> float:
    """Fold the atom-track shard into the Blackhole step wall."""
    a = ATOM_SHARD_CAP if n_or_cap == "cap" else ATOM_SHARD[n_or_cap]
    return 1.0 / ((1.0 - ATOM_TRACK_FRAC_BH) + ATOM_TRACK_FRAC_BH / a)
# Single-processor bracket after the bandwidth intersection: b2z2-redteam-v2.
ONE_CHIP_BRACKET = (1.35, 1.66)
ONE_CHIP_BEST_SUPPORTED = (1.40, 1.49)


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


def stack_split() -> tuple[float, float, float]:
    """Where the stack's seconds come off. MEASURED, not bracketed.

    This used to return an A/B bracket over an assumption: does the stack's saving already come out
    of the trunk or not. `b2z2-redteam-v3` settled it by opening the `block_s` column the headline
    session had recorded per fold and nobody had read -- the trunk gives up STACK_TRUNK_GIVEUP_S, so
    the old split-A end (trunk untouched) is REFUTED and the high end of every route this file used
    to print was never an estimate.

    Returns (trunk, sampler, rest) after the stack, in seconds.
    """
    saved = CELL_S - CELL_S / STACK_RATIO
    trunk = TRUNK_S - STACK_TRUNK_GIVEUP_S
    rest_and_sampler = CELL_S - TRUNK_S                      # sampler + everything else, before
    after = rest_and_sampler - (saved - STACK_TRUNK_GIVEUP_S)
    frac = SAMPLER_S / rest_and_sampler                      # the sampler's share of it
    return trunk, after * frac, after * (1.0 - frac)


def composite(trunk_s: float, sampler_s: float, rest_s: float,
              block_ratio: float, step_ratio: float = 1.0) -> float:
    """Fold ratio: divide the trunk by its shard, the sampler by its shard, leave the rest."""
    return CELL_S / (rest_s + sampler_s / step_ratio + trunk_s / block_ratio)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()

    trunk, samp, rest = stack_split()
    rows = []
    for label, br in [*[(f"trunk shard N={n}", r) for n, r in SHARD_BLOCK.items() if n > 1],
                      ("trunk shard cap, measured link", SHARD_CAP_MEASURED_LINK),
                      ("trunk shard cap, FREE link", SHARD_CAP_FREE_LINK)]:
        rows.append({"route": label, "block_ratio": br, "step_ratio": 1.0,
                     "fold": composite(trunk, samp, rest, br)})
    for n in (2, 4):
        sr = sampler_step(n)
        rows.append({"route": f"+ atom-axis sampler shard, N={n}", "block_ratio": SHARD_BLOCK[n],
                     "step_ratio": sr, "fold": composite(trunk, samp, rest, SHARD_BLOCK[n], sr)})
    sr_cap = sampler_step("cap")
    cap_fold = composite(trunk, samp, rest, SHARD_CAP_MEASURED_LINK, sr_cap)
    rows.append({"route": "+ BOTH shards at their caps (any N)",
                 "block_ratio": SHARD_CAP_MEASURED_LINK, "step_ratio": sr_cap, "fold": cap_fold})

    # The WH->BH calibration this file used to print and never apply. redteam-v3's point: the WH
    # block curve over-predicts the BH fold contribution of the same shard, so applying it is not
    # optional once the number is quoted.
    n2_marginal = rows[0]["fold"] / STACK_RATIO
    calib = n2_marginal / SHARD_FOLD_BH_N2
    cap_calibrated = CELL_S / (rest + samp / sr_cap + trunk / (SHARD_CAP_MEASURED_LINK / calib))

    out = {
        "cell_s": CELL_S,
        "cell_source": "site/data/perf-512aa.json, Boltz-2 p150a",
        "measured_single_chip_stack": STACK_RATIO,
        "measured_single_chip_stack_s": CELL_S / STACK_RATIO,
        "one_chip_bracket": ONE_CHIP_BRACKET,
        "one_chip_best_supported": ONE_CHIP_BEST_SUPPORTED,
        "sampler_token_axis_shards": SAMPLER_TOKEN_AXIS_SHARDS,
        "sampler_atom_axis_step": {str(n): sampler_step(n) for n in ATOM_SHARD},
        "sampler_atom_axis_step_cap": sr_cap,
        "msa_axis_shards": MSA_AXIS_SHARDS,
        "trunk_s": TRUNK_S,
        "stack_split_trunk_sampler_rest": [trunk, samp, rest],
        "cap_fold": cap_fold, "cap_fold_calibrated": cap_calibrated,
        "redteam3_cap": REDTEAM3_CAP, "redteam3_cap_calibrated": REDTEAM3_CAP_CALIBRATED,
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
    print(f"trunk / sampler / rest         {trunk:.3f} / {samp:.3f} / {rest:.3f} s  "
          f"(after the stack; trunk give-up {STACK_TRUNK_GIVEUP_S:.4f} s MEASURED)")
    print()
    print("stack + shards. The TOKEN axis of the sampler does not shard; the ATOM axis does.")
    print(f"  {'route':<36} {'block':>7} {'step':>7}  {'fold':>9}  {'s':>7}")
    for r in rows:
        print(f"  {r['route']:<36} {r['block_ratio']:>6.3f}x {r['step_ratio']:>6.3f}x  "
              f"{r['fold']:>8.4f}x  {CELL_S/r['fold']:>6.2f}")
    print()
    print(f"WH-block -> BH-fold shard calibration at N=2: {calib:.3f}x, NOW APPLIED")
    print(f"  cap route {cap_fold:.4f}x uncalibrated -> **{cap_calibrated:.4f}x = "
          f"{CELL_S/cap_calibrated:.2f} s** calibrated")
    print(f"  redteam-v3 published {REDTEAM3_CAP:.4f}x / {REDTEAM3_CAP_CALIBRATED:.4f}x calibrated;"
          f" this file reproduces to {abs(cap_fold/REDTEAM3_CAP-1)*100:.2f} % / "
          f"{abs(cap_calibrated/REDTEAM3_CAP_CALIBRATED-1)*100:.2f} %")
    if max(abs(cap_fold/REDTEAM3_CAP-1), abs(cap_calibrated/REDTEAM3_CAP_CALIBRATED-1)) > 0.02:
        print("  !! DIVERGES from redteam-v3 by more than 2 % -- one of the two is wrong, "
              "and theirs re-derives host-only in perf/b2z2_redteam3/redteam3.py")
    print()
    bt = byte_target()
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
    mf = composite(trunk, samp, rest, BLOCK_MS_TODAY / BLOCK_MS_MOVEMENT_FREE)
    print(f"  and even a FULLY movement-free trunk, on ONE processor, with the measured stack, is "
          f"{mf:.4f}x")
    print()
    print(f"BEST NAMEABLE, unlimited processors: {cap_calibrated:.4f}x = "
          f"{CELL_S/cap_calibrated:.3f} s (calibrated)")
    print(f"2x needs {CELL_S/2:.3f} s. It is not in this table.")
    print("MSA axis excluded: measured NO-GO, 57.812 % constant (b2z2-msa-axis-shard).")

    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
