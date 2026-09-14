# Phase 0 closed: the roofline is 6.934 s and bandwidth sets it

Delivered by `roof-budget` (branch `wk/roof-budget`, table at `perf/roof_budget/ROOF_BUDGET.md`),
attacked by `roof-redteam` (`wk/roof-redteam`). This file records what the campaign is now ranked
on and the two checks the orchestrator ran on it directly.

## The number

    2.9449 TB  /  424.7 GB/s measured  =  6.934 s
    cell of record                       17.340 s   = 2.50x the floor
    above the floor                      10.406 s

One instrument, one card (qb2 card 2, p300c, 11x10, AICLK 800 MHz), one process, one session, tip
`f072ae02f`. FLOPs and bytes come off the same 26 captures; times come off the same process's
bracketed fold.

The compute roof does not bind. 219.49 TFLOP executed against a measured 104.93 TFLOP/s dense bf16
HiFi4 roof is 2.092 s, and 23 of the 26 units sit between 13.8 and 126.7 FLOP/byte against a
measured machine balance of 247.1 FLOP/byte. Only the three pair/token Transitions are compute-side,
at 303-380 FLOP/byte.

## Why the brief said 10.15 s

The brief's compute term, 873 TFLOP / 86.0 TFLOP/s, priced **a different model**. `roof-redteam`
traced 873 TFLOP to Protenix-v2 at c_z = 256. Boltz-2 at the published configuration executes
219.49 TFLOP. That is also why the campaign's third input — every phase at 3-18 % of the compute
roof — never reconciled: the phase figures were right, and 873 TFLOP was the number that was wrong.

Two attacks on the corrected count both failed, which is what makes it trustworthy: tile padding
inflates it by **1.0142x** (fold-wide tile pad factor 1.00149 in the table's own summary), and the
atom transformer's self-declared "up to 4x" undercount prices out at **1.0102x fold-wide**.

## Two checks the orchestrator ran directly, rather than taking on report

**The three top-level units tile the fold; they do not nest.** The table lists `DiffusionModule`,
`Diffusion` and `DiffusionTransformer` at 317.2 / 317.2 / 290.0 GFLOP/call, which is the shape of
nesting, so the sum was worth checking before ranking anything on it:

    PairformerLayer|1x512x384,1x512x512x128    13.708 s/fold
    DiffusionModule|                            8.226
    MSALayer|1x512x512x128,1x1024x512x64        2.710
    ------------------------------------------------------
    sum                                        24.644
    session fold, measured independently       24.731     0.35 % unaccounted

The fold time is a bracketed wall clock and is not derived from the unit table, so the agreement is
a real check rather than a circular one. `Diffusion|1x4480x3,1` (8.068) and
`DiffusionTransformer|1x512x768,1x512x768` (5.439) are nested children, present in the table and
correctly excluded from the sum.

**The MSA pad is safe to shrink.** `outer_product_mean` already divides by the true row count, not
the padded extent (`tt_bio/tenstorrent.py:9441`, `scale = 1.0 / (n_msa if n_msa is not None else S)`,
with `n_msa` threaded from `:11041`), and the pad rows are mask-zeroed at `:11534`. Shrinking
`MSA_PAD_MULTIPLE` therefore changes the length of a reduction whose extra terms are exactly 0.0.
It is not doing less of the model's own work. The remaining three depth-axis units still have to be
checked the same way and that is `roof-msa-ladder`'s first job, not an assumption it may make.

## The work queue

Seconds above the binding roof, at the 17.340 s cell.

| unit | s above roof | note |
|---|---|---|
| pairformer block | 5.410 | 30.6 % of the 424.7 GB/s stream roof |
| denoiser step | 3.867 | 23.1 % |
| MSA block | 1.067 | 30.7 % |
| — pair Transition | 2.251 | the only compute-side row, 7.6 % of 104.93 TFLOP/s |
| — TriangleMultiplication | 1.458 | |
| — token DiT layer | 1.448 | |
| — TriangleAttention | 1.284 | |
| — atom transformer layer | 0.369 | |
| — OuterProductMean | 0.349 | |

Two caveats travel with the seconds and not with the ranking. The ms/call column was taken at
loadavg 27-28 and scaled by one scalar, 0.7011, while the roof is subtracted unscaled; contention is
not uniform across dispatch-heavy phases. And 6.934 s is a pure traffic sum, so it is optimistic by
however much the three compute-side rows are mispriced. Bytes, FLOPs and the roofs are
contention-immune.

## What Phase 0 also corrected

**The byte number of record was a 1.178x undercount of its own inputs.** `itemize.top_level_spans`
falls back from its STACK rule to its RANGE rule on captures ttnn.graph leaves unbalanced (1413
function_start against 1331 function_end), and STACK collapses 428 top-level ops into 4, charging
one read for a buffer ten ops read. Re-running `roof_table.py` unchanged on its own committed
captures gives 8738.0 MB for the pairformer block where `ROOF_DEFICIT.md` publishes 6650.7. The
balanced DiffusionModule capture is byte-identical under both rules, which is the control. 3.4047 TB
of record is 4.0106 TB; the tip's 2.9449 TB is 0.734x that.

**The two byte instruments still disagree by 8.49 % on in-place ops.** `split_io` drops the
read-modify-write's read, making the block census 8046.8 MB rather than 7416.9 MB, while
`baseline_attrib.py:Census.charge` charges it correctly. The fold's 2.9449 TB comes from the second
and is unaffected; every "% of the block" derived from the first is overstated. Assigned to
`roof-redteam-2`.

## The pass-1 fusion rows, and what they changed about Phase A

Three fusion rows were dispatched off a byte ranking. Two came back NO-GO and the third was refuted
as specified. The mechanism now gates every remaining fusion candidate:

**At the sites measured, a fused program costs roughly twice the value of every byte it deletes.**
`roof-fuse-trimul-out` deleted its ranked 268.4 MB/block in full and bit-exactly, and the block got
slower: one matmul pass through a hand-written `minimal_matmul`-derived descriptor costs ~0.70 ms/
trimul more than the `ttnn.linear` it replaces, against a 134.2 MB round trip worth 0.648 ms at that
card's measured 207.06 GB/s. Deleted bytes are only worth the stream rate when the thing deleting
them is free.

Two narrower lessons, both general:

- **Rank a fusion pair by (operand bytes the consumer must read) - (intermediate bytes deleted),
  never by the intermediate alone.** `roof-fuse-qkv-sdpa`'s ranked Q prize had the wrong sign:
  folding the Q projection into the SDPA reader is +268.4 MB, because the projection contracts 128
  channels into 32 per head and the SDPA runs one head per core, so four heads each re-read all 128
  channels. Fusing across a contraction into a consumer split on the contracted-into axis multiplies
  the operand read by the split factor.
- **Some fusion rows are decidable from the source for free.** `roof-fuse-gate-epilogue` is not
  available at any price: the multiply's other operand is downstream of the in-projection's own
  output, so the dependency is circular rather than a DST-window question.

So `FUSION_PAIRS.md`'s byte prizes are sound as bytes and do not imply seconds, which is what that
table itself said and is now demonstrated rather than warned about. A fusion row is only dispatched
now if its consumer kernel already exists and already pays a fused reader's cost.
