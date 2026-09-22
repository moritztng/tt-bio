# narrow-q: +9.50 s on rf3 at 896 aa, refused by the sampler, and boltz2 cannot see it

## The number

`perf/land_standing/out/narrowq_rf3_896_qb2c1.json`, read by `narrowq_rf3_read.py`. rf3, 896 aa,
qb2 card 1, p300c. Quiet reps only (reps 1 and 2; rep 3 took a co-tenant and is cut on the
benchlock ceiling, which is the honest cut and one the reader can check):

    lever on   96.3, 94.8    median  95.55 s
    shipped   104.6, 105.5   median 105.05 s
    A/A floor +1.570 %   A/B +9.942 %   1.0994x   +9.50 s   effect 6.3x its own floor
    separation: max lever-on 96.3 < min shipped 104.6, no overlap
    every leg peaks at 1350 MHz

## Why it is refused, and it is the instrument not the card

`clock_during.py` requires the longest run at or above 1200 MHz to CONTAIN the timed fold. At a
3 s cadence a sample span understates the true run by up to one interval at each end, so it proves
only 104 s inside a 104.6 s fold. Every shortfall is under one sampling interval. Nothing reads as
a throttle -- but a cell the instrument refuses does not become a shipped second because its
direction is plausible.

**Owed: the same cell at a 1 s cadence, four reps, on a box that stays under the ceiling for the
whole cell.** `narrowq_rf3_retake_1s.sh` does exactly that and refuses rather than waits.

## boltz2 is the wrong model, now measured twice

At 896 aa boltz2 runs 4 heads. The wide candidate 448 fits L1, so the fused path serves without
the fallback and the flag cannot reach anything:

    boltz2  896 aa  OFF  served 560 / declined 560   rejects {('pm_over_l1', (896,4,896,32)): 560}
    boltz2  896 aa  ON   served 560 / declined 560   rejects {('pm_over_l1', (896,4,896,32)): 560}
                         same CIF sha256 c3c6034693fad0d8, same pLDDT 0.803368
    rf3     896 aa  OFF  served   0 / declined 2177  fill_preconditions 1088, pm_over_l1 1087
    rf3     896 aa  ON   served 1088 / declined 2176 pm_over_l1 2174, l1_budget 2

rf3 is the one cell in the whole seven-model six-rung p300c baseline where `fill_preconditions`
declines EVERY call -- the exact pathology the lever was written for, and the same shape the
16.7 s Galaxy Wormhole reading came from.

**So `perf/c14_bfp8/fold_ab.py` cannot price this lever.** It is boltz2-only. Registering narrow-q
in it (done 2026-09-22, and useful for the FLAGS shipped-default repair it forced) does not give
this lever a low-floor harness. What would: the same in-process block design applied to rf3.

## Two corrections to things this row wrote

1. **The 9.707 % A/A floor is the BOX, not the harness.** An earlier pass concluded it was
   intrinsic to `fold_ab_flip.py`'s cold-subprocess-per-leg shape. The same harness on rf3 at the
   same size on quiet reps reads **1.570 %**. The control was already on disk and was not read.
2. **The lever is inert exactly where the production q_chunk divides the padded length**, not at
   "the multiples of 256". 320 and 384 are inert with prod=64. Executed over 28 tile-aligned
   lengths from 256 to 1536: changes the ladder at 20, inert at 8.

## UPDATE 2026-09-22 22:45Z — the sampler objection is CLOSED, and the cell is now banked

`clock_during.py` **passes all eight legs** of the 1 s-cadence retake
(`perf/land_standing/out/narrowq_rf3_retake1s_896.json`). Longest unbroken run at or above
1200 MHz contains the timed fold in every leg, peak 1350 MHz throughout. **The instrument no
longer refuses this cell.** What refused it this time is contention: of3t took the box at
22:11:40Z, six minutes after a pre-flight at loadavg 0.13, and the harness reported an A/A floor
of +32.135 % against a +11.149 % A/B.

**The blocker is a window-size mismatch, not the card and not the sampler.** The cell wants ~19
minutes under the ceiling; this box hands out about 6. So the cell is now assembled from pairs:

- `narrowq_pair.sh off|on` takes ONE interleaved pair, ~5.5 min, and banks it. `NQ_CARD`/`NQ_SIB`
  pick the p300c board, because it is the board PAIR that has to be idle.
- `narrowq_bank.py` assembles the bank, applying per leg: the 2.00 loadavg ceiling over the leg's
  own interval, `clock_during.py`'s contract imported rather than restated, and pair integrity.
  Pooling across windows only ADDS between-window variation to the A/A floor, so the assembled
  cell faces a floor at least as wide as a single-session one.
- The bank **refuses a single-order bank**. Every banked pair is rep 1, so without `--first-arm`
  every pair is off-first, which is the fixed order the 768 aa negative control shows reads as a
  lever (+2.542 % at 3.80x its floor where the policy cannot fire).

**Bank state: one pair.** off-first, lever-on 94.2 s, shipped 103.9 s, **+9.70 s**, both legs at
loadavg 1.07-1.33 and clock-clean. Consistent with the +9.50 s read on two cards in two processes.
One pair carries no floor and is not a result.

**Owed: one on-first pair.** One 5.5-minute window on an idle p300c board pair.
