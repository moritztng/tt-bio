# y-silu-lowering round 4 — predictions, committed and pushed before the device was opened

Round 3 closed the leg's central question (the 2.08x is `DST_ACCUM_MODE` picking an accurate sigmoid,
not a defect in ttnn's SILU lowering). It left two of its own statements as reasoning rather than
measurement, and this round measures both. Anchors from round 3, same card, same wheel:

- fused silu penalty, `[1,30,298,256] x [256,1024]`, `fp32_dest_acc_en=True`: **171.481 us/call**
- the same at `False`: **74.685 us/call**; the gap that prices the lever: **96.796 us/call**
- bare matmul control at `True`: **77.490 us/call**
- 9600 padded output tiles at that shape, so the accurate lowering is **0.01786 us/tile**, the cheap
  one **0.00778**, and the gap **0.010083 us/tile**
- y-silu's shape sweep put the 3-D `transition_s` site at **0.00365 us/tile**, 4.9x below that roof

## Sweep A1 — batch at the 4-D pair shape

`[1, b, 298, 256] x [256, 1024]`, b in {1, 2, 4, 8, 16, 30}, production config.

- **P1.** The fused-silu penalty per padded output tile rises monotonically with b and asymptotes at
  round 3's 0.01786. **b=1 lands under 0.008 us/tile; b=30 lands within 15 % of 0.01786.**
  **WRONG if** b=1's per-tile penalty is within 25 % of b=30's — that kills work-per-core as the
  mechanism, because it would make the per-tile cost batch-independent.
- **P2.** A small-batch 4-D shape is as cheap per tile as the 3-D `c_s=384` site, i.e. **b=1 or b=2
  lands within 2x of 0.00365 us/tile.** That refutes dimensionality as the trigger outright.
  **WRONG if** every 4-D batch, b=1 included, is above 0.010 us/tile — then 4-D really is a different
  program and round 3's mechanism is wrong.

## Sweep A2 — M at the 3-D `transition_s` shape

`[1, M, 384] x [384, 1536]`, M in {298, 596, 1192, 2384, 4768, 9536}. This is the experiment round 3
named as the one that would kill its own mechanism, and did not run.

- **P3.** Per-tile penalty climbs from ~0.004 at M=298 to **within 25 % of 0.01786 by M=9536**.
  **WRONG, and round 3's mechanism is refuted, if** it stays flat within 25 % of its M=298 value
  across the whole sweep.
- **P4.** The climb is concave: most of it happens by M=2384 (≈ 3600 output tiles, ≈ 28 tiles/core),
  because saturation is a threshold in tiles per core rather than a slope. **WRONG if** the sweep is
  linear in M to the end.

## Experiment B — the real 512 aa shape

Read from source, not assumed: `TRANSITION_H_CHUNK_SIZE_BIG = 32` is gated on `W <= 384`
(`tenstorrent.py:2411`), so at W=512 the row chunk is `TRANSITION_H_CHUNK_SIZE = 16`, and
`_ref / (w_eff * c) = 131072 / (512*256) = 1.0` leaves it there. A `transition_z` call at 512 aa
therefore runs **32** fc1 silus at `[1, 16, 512, 256]`, not 10 at `[1, 30, 298, 256]`.

- **P5.** `[1,16,512,256] x [256,1024]` has 8192 padded output tiles and its **gap per tile lands
  within 10 % of 0.010083** — the lever is per-element and both shapes are TRISC1-saturated.
  **WRONG if** it is more than 20 % off, which would mean the 512 aa figure cannot be got by scaling.
- **P6.** The 512 aa lever is worth **1250–1500 ms/fold**, i.e. **below round 3's 1487 extrapolation**,
  because `(512/298)^2 = 2.95` ignores the 298 -> 320 tile padding and the honest padded ratio is
  `(512 x 512) / (298 x 320) = 2.749`. **WRONG if** it lands above 1550 or below 1150.
- **P7.** The chunk-size drop from 32 rows to 16 costs nothing per tile: **`[1,32,512,256]` and
  `[1,16,512,256]` agree on penalty per tile within 10 %.** **WRONG if** the 16-row chunk is more
  than 10 % more expensive per tile, which would make the W<=384 gate on
  `TRANSITION_H_CHUNK_SIZE_BIG` a size cliff worth reporting in its own right.
- **P8.** The bare matmul control at 512 lands within 15 % of 0.00807 us/tile.

## What no outcome here can change

The merge recommendation. Nothing in this round is a production change, and the accurate-vs-21f
sigmoid finding that closed the leg is a source fact about `calculate_silu`, not a timing.
