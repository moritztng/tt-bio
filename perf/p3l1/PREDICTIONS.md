# protenix-trunk--p3-l1-output — Phase 3, does the 298.7 survive a wall?

TASK TYPE: ACCELERATE (Phase 3) | PLAYBOOKS loaded: ACCELERATE | memories read:
`roofline-roof-must-be-measured-not-asserted`, `tt-bio-l1-residency-guard-dead-in-real-folds`,
`perfwar-programconfig-gate-output-not-subtracted`, `ttnn-sync-before-every-timed-region`,
`ttnn-batched-matmul-programconfig-rules`, `tt-bio-matmul-dram-write-serialized-l1-residency-fix`,
`stage-through-l1-fixes-source-not-destination`, `ttnn-scatter-gather-per-element-limited`,
`model-merge-approval-gate`, `tt-device-numbering`

Host qb1, **card 1** (`TT_VISIBLE_DEVICES=1`), ttnn 0.67.4, Blackhole p150a. Branch
`wk/protenix-trunk--p3-l1-output`, branched from `wk/protenix-trunk--p3-narrow-write` at `4d35c9a1`
so X2's landed `_NARROW_PROJ_BW = 1` (31.5 ms/fold bit-exact) is in my baseline and is not re-banked.
Scope: **protenix**-v2, the **trunk**, **298** aa — the pair tensor is `[1, 298, 320, 256]`.

Host state when the predictions were written: load average 1.21, two other folds live on
`/dev/tenstorrent/0` (a `full_parity_gate` worker and a `perfwar-qb1-rebaseline-and-land` arm).
`/dev/tenstorrent/1`, `2` and `3` had no open handle. Every measurement below records the host state
it was taken under, and anything a decision turns on that is inside 5 % gets re-taken quiet
(charter §4.8).

---

## Predictions (before measuring)

Committed as `perf/p3l1/PREDICTIONS.md` and pushed before the device was opened; that file is the
tamper-evident copy and this section is the same text.

**P1 — X2's `projection + add` L1-output pair reproduces on this card, at the production grid.**
X2's probe built its program configs from a module-scope `COMPUTE_GRID_MAIN` = 11x10 while the fold
runs 13x10 (X2's own correction 8), so the reproduction is not a re-run, it is the first measurement
of this pair at the production grid. Reading the grid **after** the device is open, I predict the
bit-exact leg `l1_tuned_bw1_obh5` beats production-today `dram_tuned_bw8_obh5` by **1.08-1.22x**
(X2 at 110 cores: 1.149x) and that the per-call delta lands **60-130 us** (X2: 95.00 us).
**Wrong if** the L1 output is refused at 130 banks, or the ratio is below 1.03x, or `torch.equal`
against the `core_grid`-DRAM leg is False.

**P2 — the two ratios in my brief do not share a denominator, and only one of them prices the
298.7.** 1.53x is `dram_cg` 975.04 us / `l1_tuned_bw1_obh5` 639.20 us — the untuned `core_grid`
baseline. 1.49x is production-today 734.20 us / `l1_tuned_bw8_obh2` 494.16 us. The 298.7 ms/fold and
the 754.7 ms/fold both use production-today as the denominator, so the bit-exact ratio that goes
with 298.7 is **1.149x, not 1.53x**. I predict re-measurement gives **1.40-1.65x** against
`core_grid`-DRAM and **1.08-1.22x** against production today. **Wrong if** the two ratios come out
within 5 % of each other, which would mean the denominators do not in fact differ.

**P3 — the production sequence is not the probe's pair, and the trimul cannot put both of its
projections in L1.** Per Pairformer layer the class runs 4 calls inside `TriangleMultiplication`
(`p_out` and `g_out`, x2 trimuls, `tenstorrent.py:1619/1622`) and 2 inside `TriangleAttention`
(`gate_and_project`, `tenstorrent.py:1817`). The trimul's real chain is
`proj -> proj -> multiply_ -> add_`, not `proj -> add`. One 48.82 MB L1 output is 375.5 kB/bank
across 130 banks and the matmul's own circular buffers at `in0_block_w`=8 / `out_block_h`=5 need
802.8 kB/bank, so a second concurrent L1 output would need **1587.6 kB/bank against 1427.5 kB
available** and I predict it is refused. Production therefore puts `p_out` in L1 and leaves `g_out`
in DRAM. **Wrong if** both fit.

**P4 — the wall, and this is the deliverable.** The instrument is a **region wall**: the
`TriangleMultiplication` / `TriangleAttention` body plus its `ttnn.add_` residual, synchronised on
both sides of each, summed over the fold's 524 c_z=256 `PairformerLayer` executions (charter §4.9's
x524). Converting X2's 95.00 us/call at 3144 calls/fold gives 298.7 ms/fold, so I predict the
measured region wall falls by **150-350 ms/fold** bit-exact, i.e. **50-117 % of the 298.7
projection**, and I expect it **below** the projection rather than above it. **Wrong if** the
region-wall delta is under 50 ms/fold — in which case the 298.7 does not survive contact with a
wall and I report that as the loss, the way X2 reported P6 — or above 400 ms/fold.

**P5 — the fold wall is only an instrument here if the region wall clears ~300 ms.** Base spread on
this harness is 144 ms and X2's fold wall moved **+68 ms against a real 31.5 ms win**, sign unstable
across repeats. If the region wall lands above 300 ms/fold I take a 3-fold A/B and report the median
with the spread beside it; below that I do not claim a fold number at all. **Wrong if** the fold
wall moves more than 400 ms in the direction that contradicts the region wall.

**P6 — the `_transpose_memory_config` interaction, which STATUS.md requires every L1-raising leg to
report.** The ending `TriangleAttention` runs `ttnn.permute(x, (1,0,2), memory_config=
_transpose_memory_config(x))` on the tensor `gate_and_project` just produced. That test is
`2.5 x volume x elem <= per_core x 130`, which at 48.82 MB is 122 MB against roughly 182 MB, so it
does not know `x` is already in L1 and I predict it **still returns L1**. Source plus destination is
then 97.6 MB = 750.9 kB/bank of 1427.5, which fits, so I predict the permute **runs**, and that the
ending variant's region-wall delta lands **within 20 % of the starting variant's**. **Wrong if**
the permute throws, or the ending variant delivers less than half the starting variant's delta.

**P7 — deliverable 2: `tenstorrent.py:2088` is READ-bound, so an L1 output is the wrong lever there
and the source is the right one.** The site reads 48.82 MB of layer-normed `z` and writes
`[1,298,320,16]` = 6.10 MB padded to one tile of width. My baseline (X2's `_NARROW_PROJ_BW=1`
already landed) is 208.4 ms/fold over 484 calls = 430.7 us/call, against 125.7 us for 48.82 MB at
this card's read roof, so the site runs at about **29 % of this card's measured DRAM->L1 read
roof**. An L1 output can only remove the 6.10 MB write, worth 23.0 us/call at the unary write roof
= 11.1 ms/fold at most. The lever with the mass behind it is the **source**: route the immediately
preceding `ttnn.layer_norm` output to L1 so the projection reads its `in0` from L1
(`stage-through-l1-fixes-source-not-destination` says staging fixes a bad source, and this source is
bad). I predict the `layer_norm + projection` region wall falls by **60-160 ms/fold** bit-exact and
that the L1 output alone is worth **under 25 ms/fold**. **Wrong if** the L1 output alone beats
40 ms/fold, or the L1-`in0` arm is *slower* than the DRAM one.

**P8 — roofs on card 1 land within 5 % of X2's card-1 figures, because it is the same card.**
X2 measured read 388.3 GB/s, unary write 265.1, DRAM->DRAM 402.5, square bf16 HiFi4 135.65 TFLOP/s.
I re-measure all four myself and inherit none. **Wrong if** any differs by more than 8 % of X2's
figure, which would say the card's state moved and no figure in X2's doc can be carried into mine.

**P9 — parity.** A memory config decides where the writer puts a tile, not the order the
contraction accumulates, so at fixed `in0_block_w` an L1 output cannot move a bit. I predict
`torch.equal` **True** for the bw=1 L1 arm against the `core_grid`-DRAM reference at the fold's own
`[1,298,320,256]`, and plDDT identical to six decimals against a bw=1 **DRAM** fold. I also predict
the bw=1 L1 arm is **NOT** plDDT-identical to today's main, because main ships `_PAIR_PROJ_BW = 16`
which is a different accumulation order — so this change is simultaneously a speedup and a return
to the bit-exact contraction, and both halves have to be reported. **Wrong if** `torch.equal` is
False, or if the fold's plDDT against main's bw=16 arm is identical to six decimals.

**Priced in advance so the ranking can be wrong too.** I expect the region wall to come in around
200-250 ms/fold bit-exact — real, but below the 298.7 — and the third site to give 60-160 ms from
its source rather than its output. If the L1 output turns out to be refused inside a live block, or
the wall comes in under 50 ms, that is the loss and I report it as one.

---
