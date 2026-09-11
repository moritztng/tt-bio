# Per-part L1 budgets and the grid a part gives you

`tt_bio/tenstorrent.py` carries a set of L1-edge budgets: how wide the triangle
multiplication's hidden-channel chunk may get, how long a sequence keeps its pair tensor
resident in L1, how many bytes a Transition chunk may hold per core. Every one of them was
fitted by measurement on a 130-core Blackhole p150a.

Two things vary between parts, and only one of them was ever handled.

**Per-core unreserved L1.** `_apply_grid_thresholds` scales the budgets by
`ttnn.get_max_worker_l1_unreserved_size() / _WH_FULL_L1_PER_CORE`, clamped at 1.0, so a part
with less L1 per core tightens and a part at or above the calibration point is unchanged.

**Core count.** The same op spread over fewer cores puts more of itself on each core. This
was not handled: `_apply_grid_thresholds` returns early on any grid of 110 cores or more and
keeps the 130-core values. That is what broke issue #11. A P300 gives 110 cores, ran budgets
fitted for 130, and a 107 aa protein with a ligand died in the trimul in-projection with
`Statically allocated circular buffers ... clash with L1 buffers`. Per-core L1 was never the
variable: a P300 measures 1,532,416 B/core, the same as a p150a.

The trimul now learns the ceiling from the clash instead of predicting it. The clash throws
at program validation, before any kernel runs, so the channel loop catches it, records the
width for that call shape, and re-runs narrower; at the minimum width the shape leaves L1 for
DRAM. Narrowing cannot move a number, because the chunk width only partitions a sum over
independent channels. A shape pays one failed compile per process.

## Measured figures

| part | grid | cores | per-core unreserved L1 | how |
|---|---|---|---|---|
| p150a | 13x10 | 130 | 1,532,416 B | `ttnn.get_max_worker_l1_unreserved_size()`, pc, 2026-08-19 |
| p300c | 11x10 | 110 | 1,532,416 B | same call, tt-quietbox2 device 0, 2026-08-19 |
| Wormhole | 8x8 | 64 | 1,466,080 B | `_WH_MEASURED_L1_PER_CORE`, the L1 the WH re-fit was measured at |
| Wormhole | 8x8 | 64 | 1,572,864 B | `_WH_FULL_L1_PER_CORE`, the WH scaling reference (1.5 MiB) |

Widths measured to clash, so the retry ladder must be able to get below them:

| part | call shape (seq, hidden, batch) | width | evidence |
|---|---|---|---|
| p300c | 140, 256, 1 | 256 | issue #11, and tt-quietbox2's native 11x10 on 2026-08-19: L1 buffer at 1155072, static CB region ends 1159680 — the same addresses on Taylor Singletary's card and on ours |

## The Transition row block reads this budget on every grid

`TRANSITION_L1_CHUNK_BYTES_PER_CORE` is 393,216 B: the most a Transition row chunk
(`x_norm` + `x_1` + `x_2`, bf16, tile-padded) may hold per core. It was fitted on the 8x9
Wormhole Galaxy, where the fc1/fc2 matmuls fit at or below 384 KiB per core and throw a
static-CB clash at or above 400 KiB, 14 points over 6 widths with no exceptions
(`perf/wh-protenix/wh_transition_h.py`).

Until 2026-09-11 only the small grid read it. On a full grid the row-block height came from a
constant fitted at the reference W=1024, c=128, and the ratio term below it only shrinks, so a
part NARROWER than the reference ran a block sized for a part up to four times its width and
never looked at its own L1. Boltz-2's 512 aa pair track held 171,585 B/core out of the
1,532,416 a Blackhole Tensix has, on 64 of 110 cores.

The full grid now raises the height to what the same budget allows at the part's own shape,
**raise only** and **snapped down to a power of two**. Raise-only means nothing that fits today
gets a smaller block. The snap is measured rather than tidy: the cost is set by the chunk
COUNT, so heights that round to the same count cost the same (h=32/33/34 are
6.5699/6.5704/6.5721 ms at 512 aa) and the sawtooth between counts is worth up to 5 %; the raw
budget value is a local worst at both boltz-2 tracks. A power of two also divides the padded
row axis exactly, so no ragged tail block exists.

What that reaches, from `perf/b2x_pairtrack/rowblock_census.py` walking every (channel x size)
the engine can express:

| grid | shapes whose block moves | worst per-core L1 among them | shapes pushed over the budget |
|---|---|---|---|
| 11x10, 110 cores | 107 | 386,067 B | 0 |
| 13x10, 130 cores | 120 | 381,117 B | 0 |

Zero is a property of the expression, not of the shapes that were measured: the snap takes a
power of two at or below the height the budget allows, so a raised block cannot exceed it.

Three shapes sit ABOVE the budget today and raise-only leaves them there. Protenix-v2's pair
Transition at W=298/320/384 ships `TRANSITION_H_CHUNK_SIZE_BIG` = 32 rows, which is 514,755
B/core at 110 cores, and it folds. So 393,216 B/core is 1.31x inside the Blackhole wall on a
part that is not boltz-2 and at a height nobody derived, independent of the boltz-2 sweep that
brackets the same wall between 343,170 B/core (runs) and 686,340 (clashes).

`TT_BIO_TRANSITION_H_CHUNK=<rows>` forces the height, which is how that sweep separates "this
height is bad here" from "this size is the problem". Test-only, unset in production.

## The rule

**A part-specific resource figure entering `tenstorrent.py` gets a row in
`L1_BUDGET_PARTS` in the same commit.** `scripts/release_gate.py --model l1-budget` runs the
budget arithmetic for every row and folds the issue-#11 target across the grid ladder the
running part can express, and it fails if a selectable grid has no row. That is the whole
mechanism: the other gate legs compare numbers, and a part that dies at program creation
produces no numbers to compare.

`TT_BIO_FORCE_GRID=x,y` pins the grid, which is how a 130-core card reproduces a 110-core
one. `TT_BIO_TRIMUL_CHUNK_CAP=<width>` pins the trimul chunk width; the gate uses it to prove
the clash-and-retry path returns the same bytes as a run that never clashed. Both are
test-only and unset in production.
