# `_BATCHED_MATMUL_SATURATION_BLOCKS = 32`: measured, real, bit-exact, and not payable here

The audit's one candidate that is live inside the 298-1024 aa ladder on Boltz-2. Everything below
is **Blackhole p150a, pc card 0, 130-core custom firmware**, ttnn 0.68.0, one process per grid.
A Wormhole row is owed and is not in this file.

Harness `perf/roof_wrong_part/sat_blocks.py`; raw JSON in `out/sat_13x10.json`, `out/sat_11x10.json`,
`out/sat_8x9.json`. Arms are the distinct `per_core_M` the constant can select, fixed order, whole
set once per block, 5-7 blocks, minimum over blocks. **A/A floor: the shipped pick entered twice
under two names, 0.010-0.70 % on every row quoted below** (one row, 320 aa attn@v at 13x10, floated
to 3.5 % and is quoted from the 7-block 11x10 and 8x9 runs instead).

## Bit-exactness: 46 arms, 46 `torch.equal`

`per_core_M` partitions output rows and `in0_block_w` does not move with it, so this is a chunking
change and the parity spend is **0.000 A**. Every arm at every grid is `torch.equal` against the
shipped pick. No arm was excluded, and the check is in the JSON per arm.

## The hypothesis the row was opened on, and its refutation

The constant is an absolute output-block count fitted on qb1's 130-core p150a, read by
`_batched_matmul_search`, which holds `cores = gx * gy` two lines above it. The obvious reading is
"a missing grid term". **It is not.** `TT_BIO_FORCE_GRID` moved the grid 13x10 -> 11x10 -> 8x9 on
the same silicon and the op time at a fixed block count did not move:

| shape | 32 blocks @130c | @110c | @72c |
|---|---|---|---|
| trunk attn@v 1024 aa | 0.1379 ms | 0.1378 ms | 0.1355 ms |
| trunk attn@v 512 aa | 0.0374 | 0.0374 | 0.0368 |
| trunk q@kT 512 aa | 0.0461 | 0.0459 | 0.0451 |

The reason is structural rather than a null result. `MatmulMultiCoreReuseProgramConfig` puts one
output block on one core, so a 32-block config engages 32 cores whether the grid has 72 or 130.
**The grid term is already in this function** -- as the legality ceiling `blocks <= cores`, which
is what removes the 80/96/128-block options at 110 and 72 cores. The saturation target does not
need a second one.

So the defect is not the part. It is that **32 is one number for two op classes whose in1 re-read
cost differs by a factor of Mt**, and it was fitted on the one where 32 is right.

## The measurement

`x` is the shipped pick against the best arm; every arm bit-exact.

| class | Nt | 320 aa | 512 aa | 768 aa | 1024 aa |
|---|---|---|---|---|---|
| trunk APB attn@v, 130c | 1 | 1.130x | 1.081x | 1.192x | **1.301x** |
| trunk APB attn@v, 110c | 1 | 1.127x | 1.078x | 1.200x | 1.182x |
| trunk APB attn@v, 72c | 1 | 1.000x | 1.068x | 1.133x | 1.170x |
| token DiT attn@v, 130c | 2 | 1.000x | 1.003x | 1.165x | 1.221x |
| trunk APB q@kT, 130c | Mt | 1.017x | 1.027x | 1.042x | 1.000x (one legal config) |
| token DiT q@kT, 130c | Mt | 1.000x | 1.011x | 1.004x | 1.000x |

Two readings, and the second is the one that explains the first.

**The narrow-in1 class pays at every size and every grid.** With `Nt` 1 or 2 the in1 re-read the
constant exists to bound is one or two tiles per block, so the trade it is making does not exist
there and more blocks is simply better: at 1024 aa the shipped 32 blocks costs 1.301x against 128.

**On the class it was fitted on, 32 is right and going wider is worse.** Token DiT q@kT at 320 aa,
`Nt = 10`: 32 blocks 0.0232 ms, 80 blocks 0.0257, so the wide arm is 1.11x SLOWER. That is the
reading the original sweep took (`perf/bmm_reconcile/pcm_sweep_c0.json`, 80/32/16 blocks at
W=320), and this run reproduces it. The constant is not wrong; it is single-valued where the
resource is not.

A correct rule has to price the thing that actually differs, which is in1 tiles re-read per output
block, `blocks * Nt * Kt`, against the write and occupancy terms. **I am not proposing a fitted
replacement**: re-fitting one number on one part is the defect this row exists to find, and
refitting it here on a p150a would reproduce it with a different constant.

## Why it is killed anyway: the ranking

On Boltz-2 at 512 aa the only site with more than one legal `per_core_M` is the trunk
`AttentionPairBias`, 264 calls, 0.227 s above roof in `perf/roof_budget/ROOF_BUDGET.md`. The token
DiT site is dark -- `BOLTZ2_TOKEN_DIT_SDPA` has been on by default since 2026-09-11, so those 4800
calls take the fused SDPA and never reach this chooser -- and the atom site is `Mt = 1`, where the
legal set is a single value and the constant cannot move anything.

Measured deltas at the two trunk matmuls, 512 aa, 130-core p150a:

    q@kT     0.0461 -> 0.0450 ms   (-0.0012)
    attn@v   0.0374 -> 0.0346 ms   (-0.0028)
    264 calls                       -1.06 ms per fold

**1.06 ms of a 17.340 s cell, 0.006 %.** At 1024 aa the attn@v arm alone is worth 8.42 ms over 264
calls, still under 0.05 % of a fold at that size. The op ratio clears the 1.05x kill line by 6x and
the fold ratio is invisible, which is the whole reason this campaign ranks by seconds above roof
rather than by op ratio.

**VERDICT for this candidate: NO-GO on the Boltz-2 512 aa budget.** Not because it is wrong, and
not because it is not bit-exact -- because the class it moves is 5 % of a unit that is 1.3 % of the
fold.

## What it is worth elsewhere, flagged not measured

`batched_matmul`'s own docstring records 9.5x on the OpenFold3 trunk triangle attention attn@v and
8-14x on the windowed atom attention from the chooser existing at all. Those are `Nt`-narrow
classes and they are a much larger share of their folds than 264 trunk calls are of Boltz-2's. A
1.2-1.3x on that class is worth a row on OpenFold3 or Protenix-v2 and is not worth one here.

## Transfer

KIND: **placement** for the reading itself (which core holds which output block; no arithmetic and
no eligibility moves, every arm bit-exact and every arm legal on every grid). The campaign's
WH->BH placement constant is k = 1.161 +/- 0.047, and it does not apply in this direction: these
are Blackhole numbers and nothing was screened on Wormhole. A Wormhole row would be predicting
BH -> WH, which is 1/k = 0.861, and I am not applying it, because the one term this measurement
turned out to depend on is DRAM behaviour under a narrow in1 and Wormhole has 12 banks against
Blackhole's 8. **The Wormhole figure is owed to a whglx row and is not in this file.**
