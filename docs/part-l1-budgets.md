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

The engine picks a smaller grid than the card's in exactly one place: protenix-v2 folds 500 to
507 tokens on 11x10, because 13x10 hangs there (issue #9, `docs/size-generality.md`). Setting
`TT_BIO_FORCE_GRID` overrides that too, so a pin set by hand always wins.
