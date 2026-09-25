"""Fleet trajectory rate for BindCraft 2 on Blackhole, from measured inputs only.

Arithmetic, not a measurement: every input below was measured by another row and is
cited with it. Kept as a script rather than a paragraph so that when an input moves
-- and OCCUPIED_CHIP_S will move when bcx-extramsa lands -- the table moves with it
instead of being re-derived by hand.

Deliberately does NOT compute designs/hour. That needs an acceptance rate, no arm has
produced one, and the lab's ~0.33/trajectory is a three-multimer-model figure that
neither of our arms runs. Multiplying by it would manufacture the per-accepted number
this campaign withdrew from CMP on 2026-09-25.
"""

# bcx-predictor, post_seed100, one Blackhole p300c, AICLK median 1350 over 771 samples
# taken DURING the run, monomer trunk, 125 rounds.
OCCUPIED_CHIP_S = 6591.0

# bcx-seam: card busy 7.86 s of a 39.08 s round -> 20.1 % device, 79.2 % host.
MEASURED_ROUND_S = 39.08
ROUNDS_PER_TRAJECTORY = 125  # BindCraft 2's own settings.py:604 stage plan

# MEASURED 2026-09-25 by bcx-extramsa: the extra-MSA swap, 12 interleaved rounds on qb2
# card 3 at AICLK median 1350 in all 12, seed 100, n=211. OFF 36.369 s median, ON 16.870 s,
# 2.156x, arms fully separated. This replaces the ~18 s projection this file used to carry.
SWAPPED_ROUND_S = 16.870
UNSWAPPED_ROUND_S = 36.369  # that row's OWN control, not bcx-round's 39.08 s at a different n

# qb1 UMD 1 is hardware-dead, qb1 UMD 0 is cardblocked on wedge-at-OPEN history.
USABLE_BLACKHOLE = 6
FULL_HEALTH_BLACKHOLE = 8

BOLTZGEN_CHIP_S_PER_DESIGN = 264.3  # state/mgx-design-scale.md, whglx, AICLK 1000


def table(chip_s, label):
    print(f"\n{label}: {chip_s:.0f} chip-s/trajectory, {3600 / chip_s:.3f} traj/h/chip")
    for n, what in ((USABLE_BLACKHOLE, "usable today"), (FULL_HEALTH_BLACKHOLE, "full health")):
        print(f"  {n} chips ({what:<12}) {n * 3600 / chip_s:6.2f} traj/h  {n * 86400 / chip_s:7.1f} traj/day")


def main():
    table(OCCUPIED_CHIP_S, "MEASURED, pre-swap program")
    # SCALED, not measured: applies the round ratio to a trajectory occupancy measured on the
    # pre-swap program. It ASSUMES the whole trajectory scales like a gradient round, and it does
    # not -- mutate's 15 rounds are forward-only (settings.py gradient_stage_rounds drops mutate),
    # so they do not carry the backward the swap speeds up. Treat as an upper bound on the gain.
    table(OCCUPIED_CHIP_S / (UNSWAPPED_ROUND_S / SWAPPED_ROUND_S), "SCALED post-swap (NOT measured)")
    compute = MEASURED_ROUND_S * 0.201 * ROUNDS_PER_TRAJECTORY
    print(f"\nagainst BoltzGen's {BOLTZGEN_CHIP_S_PER_DESIGN} chip-s/design:")
    print(f"  occupancy {OCCUPIED_CHIP_S / BOLTZGEN_CHIP_S_PER_DESIGN:5.1f}x"
          f"   compute {compute / BOLTZGEN_CHIP_S_PER_DESIGN:5.2f}x")
    print("  (per TRAJECTORY, and a BoltzGen design is a produced design -- not like for like)")
    print("\ndesigns/hour: undefined, acceptance rate unmeasured on every arm run so far")


if __name__ == "__main__":
    main()
