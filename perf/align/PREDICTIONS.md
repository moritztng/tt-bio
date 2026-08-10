# p2-alignment — the unaligned contracted axis: blast radius, mechanism, and what a fix costs

Phase 2, EXPERIMENT. Protenix-v2, trunk only, 298 aa (token axis 298 logical, padded to 320; c_z=256).
qb2 card 0, board 007 chip 0, ttnn 0.68.0. Ratios only, never campaign absolutes (charter §4.8).

**Probes only. No production change: nothing under `tt_bio/` is touched by this leg.** Everything new
is under `perf/align/` on `wk/protenix-trunk--p2-alignment`, plus the block harness taken unmodified
from `wk/protenix-trunk--trimul-rescore` (`perf/ledger_298/pf_block_ops.py`, `--tokens 298` default,
dropped rows recorded as `null`). I did not rebuild it.

## Predictions (before measuring)

Committed and pushed before the device was opened. Instruments named per prediction.

**P1 — the four-arm A/B reproduces on my chip.** At a fixed padded shape `[1, 32, 320, 320]` with the
fold's own `_triangle_mul_program_config(10)` supplied on every arm, I predict 72.8 us for logical
320x320, 72.1 us for 298x320, 112.1 us for 320x298 and 113.5 us for 298x298, each within 5 % of
B2's figure, and a contracted-axis ratio of 1.56x. **Wrong if** the ratio lands outside 1.45x-1.65x,
or if the M-unaligned arm (298x320) is more than 3 % above the aligned arm — that would mean the
"only the contracted axis matters" claim does not hold on this chip.

**P2 — the blast radius is small, and the contraction is essentially all of it.** From the live-block
op capture, the only contracting sites whose contracted axis is logically 298 are the triangle
contraction `matmul` (16 calls/block) and the `attn@v` `matmul` inside `AttentionPairBias`'s
fp32-softmax attention (1 call/block, 0.026 ms/block). Every other matmul and linear in the block
contracts over c=256, 384, 1024 or 1536 and over head_dim=32 — all tile-aligned, so I predict **no
penalty, within 3 %,** on the K=256 pair-track projections. The one site the shape scan cannot see is
`ttnn.transformer.scaled_dot_product_attention`, which contracts over the key axis internally; that
axis is logically 298. I predict SDPA pays **1.00x-1.15x**, not 1.56x, because its flash kernel is
built around explicit per-chunk masks and `ceil(298/64) = ceil(320/64) = 5` k-chunks either way.
**Total blast radius predicted: 360 ms/fold** (345 contraction + ~14 from `attn@v` + ~0 SDPA).
**Wrong if** the measured total exceeds 500 ms/fold or falls under 300 ms/fold, and badly wrong if
SDPA's logical-298 arm is more than 15 % slower than its logical-320 arm at a fixed padded shape.

**P3 — the mechanism is the reader, and the penalty is per output block, not per tile.** Both arms
compile the same reader/writer/compute kernel *names*; what differs is the compiled program (extra
mask defines / runtime args), so an empty-kernel-cache compile of the two arms yields the same kernel
file names and a different program. If the penalty is the NCRISC in0/in1 reader masking and
zero-filling the tail sub-tile of the last K block per output block, its cost is one masked
transaction per output block and therefore roughly **constant in absolute us** as `in0_block_w` goes
10 -> 1 while the arm's total time rises ~2.4x: I predict the penalty stays 35-45 us/call across that
ladder, i.e. it falls from ~56 % of the aligned arm to under 25 % of it. **Wrong if** the penalty
tracks total time as a constant *fraction* across the `in0_block_w` ladder (that puts the cost inside
the compute kernel, which is a different Phase-3 fix), or if the two arms compile different kernel
names (that makes it program selection, not padding-awareness). Instruments: an empty
`TT_METAL_KERNEL_CACHE` compile with the generated kernel tree diffed between arms, the
`in0_block_w` ladder, and a batch ladder that scales output blocks at a fixed K. `TT_METAL_WATCHER`
is not used: T4 rejected it (0 of 600 worker lines caught a live program across 10 dumps, and it
inflates op time 9.9x).

**P4 — the aligned fill is bit-exact.** A zero tail contributes exactly 0.0 to an fp32 accumulator and
the K-tile count is 10 in both arms (`ceil(298/32) = ceil(320/32) = 10`), so the accumulation order is
unchanged. I predict `torch.equal` returns **True** on the 298x298 valid region of the contraction
output, aligned-fill arm against production arm, at the fold's own shape. **Wrong if** it returns
False, in which case I report RMSD and max abs deviation and say which operand's tail carries the
non-zero.

**P5 — the fill is not free, because the tail is not already zero.** `layer_norm` writes its bias into
the padded token rows (a zero row normalises to `beta`, not to 0), so I predict the contraction's
operands carry a **non-zero** padded tail at the point of the contraction and a metadata-only relabel
is therefore not exact. Only one of the two operands needs a zero tail for the product to be inert.
I predict the cheapest exact fill is one masked in-place multiply over one operand, costing
**15-35 us/call** against **41 us/call** saved, so the per-call placement is marginal-to-negative and
the fill has to live once per trimul call at the point the chunk is built, not per matmul call.
**Wrong if** a metadata-only reshape is both bit-exact and under 5 us/call — that would make this the
cheapest large lever in the ledger. I also predict the alignment does **not** survive the block: the
contraction's output is transposed and projected back into `z` at logical 298, so the fill has to be
re-applied at each of the 2 trimul calls per block.

## Roofs, measured on this card

*(this pass — filled after measuring)*

## Experiments and verdicts

*(filled after measuring)*

## ms/fold at stake, after this pass

*(filled after measuring)*

## Parity

*(filled after measuring)*

## Corrections to the inherited record

*(filled after measuring)*
