# protenix-trunk--p2-matmul-ceiling — Phase 2, one mechanism under five sites

TASK TYPE: VERIFY/BENCHMARK (designed experiment) | PLAYBOOKS loaded: ACCELERATE + VERIFY/BENCHMARK |
memories read: `ttnn-batched-matmul-programconfig-rules`, `perfwar-programconfig-gate-output-not-subtracted`,
`tt-bio-l1-residency-guard-dead-in-real-folds`, `roofline-roof-must-be-measured-not-asserted`,
`ttnn-sync-before-every-timed-region`, `tt-device-numbering`, `tt-bio-matmul-dram-write-serialized-l1-residency-fix`

Host qb1, card 0 (`TT_VISIBLE_DEVICES=0`), ttnn 0.67.4, Blackhole p150a, compute grid 13x10,
`CORE_GRID_MAIN` 11x10. Branch `wk/protenix-trunk--p2-matmul-ceiling`. Scope: protenix-v2, the trunk,
298 aa (N=320 padded, so the real tensor is `[1, 298, 320, c]`).

**No production change.** Everything in this pass is a probe under `perf/p2ceiling/`. Phase 3 owns the fix.

---

## Predictions (before measuring)

Committed and pushed before the device was opened. Every arm names the number I expect and what would
count as having been wrong.

**P1 — H1, the in0 reader's circular buffer sets the pace at a narrow output.** At K=256, nt=8, both
operands L1-resident, output DRAM-interleaved, I sweep M through 4096 / 8192 / 16384 / 32768 and fit
`t = c0 + c1*M`. I predict the fit is essentially proportional: `c0` under 10 % of `t(16384)` and the
achieved TFLOP/s flat to within 5 % across the four M. Roof-method measured this cell at 83.42 us and
25.74 TFLOP/s on qb1 card 2; on card 0 I expect 80-90 us at M=16384. **Wrong if** the rate moves by
more than 10 % across the M sweep in either direction: a rate that *rises* with M means a fixed
per-op cost (launch or mcast setup) dominates and the reader does not bind; a rate that *falls* with
M means something size-dependent I have not named. Second falsifier, run in the same arm: in0 in DRAM
instead of L1 at the identical shape. If the per-element cost is the same to within 5 %, the limiter
is not where in0 lives, which is the reader-side reading H1 needs.

**P2 — H2, overlap is set by per-core output block width, not by total bytes.** Both arms write
52.43 MB to DRAM with L1 operands at K=256: nt=8 at M=102400 and nt=64 at M=12800. I predict the
nt=64 arm lands at 250-290 us (near `max(compute, write)`, roof-method's 265.22 us) and the nt=8 arm
at 480-560 us (additive, 6.25x roof-method's 83.42 us cell), so the two differ by 1.7-2.2x. **Wrong
if** the two arms land within 10 % of each other, which would say overlap follows total bytes or op
duration and H2 is dead.

**P3 — H3, `in0_block_w` K-loop depth caps the L1-output rate at low K.** K=256 (Kt=8), L1 in and
L1 out, nt=64, an explicit 1D program config with everything but `in0_block_w` pinned, swept through
1 / 2 / 4 / 8. I predict monotone rise, with `bw=8` between 1.25x and 1.7x `bw=1` (trimul-rescore
measured 1.53x on the DRAM-output nt=8 shape: 587.86 / 468.96 / 424.68 / 385.04 us) and `bw=8`
reaching at least 110 TFLOP/s, i.e. within 25 % of the card's K=4096 rate. **Wrong if** the four
widths land within 5 % of each other, which puts the cap on mcast setup per output block instead.

**P4 — Q12, the write roof is a function of the op's own read:write byte ratio.** Same writer, same
output width nt=8, output always DRAM-interleaved, operands in DRAM, K varied so that
read:write moves through 1:3 / 1:1 / 3:1. I predict achieved write rate **falls monotonically as the
read share rises**: 185-205 GB/s at 1:3, 150-175 GB/s at 1:1, 105-135 GB/s at 3:1. A matched
L1-operand arm at each ratio gives the contention-free rate and should sit at 190-200 GB/s
throughout. **Wrong if** the three DRAM-operand arms spread by less than 10 %, which would mean one
number denominates every row and the org's three figures differ for some other reason. Consequence I
also commit to: on this reading T1's qkv @1428 (read:write 1:3) takes a denominator at the *top* of
the range, so its 96.8 % stands to within a few points and the 156.6 GB/s figure is the wrong
denominator for it. I predict @1428 re-scores in 85-95 % and @1434 (1:1) in 80-92 %, i.e. the two
rows together have 50-250 ms/fold of headroom, not the 3-4x the open question allows for.

**P5 — the two narrow-output sites, and whether a program config fixes them.** Real fold shapes:
`linear`@`tenstorrent.py:2807` is `[1, 298, 320, 256] x [256, 16]` (nt=1), `linear`@`protenix.py:306`
is `[1, 298, 320, 256] x [256, 64]` (nt=2). Baseline is production `core_grid=CORE_GRID_MAIN`, which
is `in0_block_w=1, out_block_h=per_core_M`. Tuned is an explicit 1D config, `in0_block_w=8` (= k_tiles),
`per_core_M=30..32` so `ceil(2980/per_core_M)` lands at 94-100 of 110 cores. T5 measured 458.9 us on
16 of 110 cores at 2807 and 478.5 us on 32 of 110 at 306, on pc. I predict the qb1 card-0 baselines
land within 15 % of those, and that the tuned config gives 1.2-1.6x, i.e. 290-380 us at 2807 and
300-400 us at 306, with cores engaged rising to 94-100 of 110. **Wrong if** the tuned config fails to
beat the baseline by more than 3 % — that is T5's own kill test and it would say the read path binds
and the idle cores are a symptom. Parity: `in0_block_w` is the sole bit-exactness knob on
DRAM-interleaved batched operands (`ttnn-batched-matmul-programconfig-rules`), so I predict the tuned
output is **not** bit-exact against the baseline and I will report `torch.equal` and the RMSD rather
than assert either way.

**Priced in advance, so the ranking can be wrong too.** If P5 lands mid-range the two narrow sites
give (459-335)x240/1000 = 29.8 ms/fold at 2807 and (478-350)x40/1000 = 5.1 ms/fold at 306, about
35 ms/fold combined. That is small next to the 250 ms/fold of ledger accuracy in P4 and next to the
~330 ms/fold of exposed write on the pair-track projections, and I expect the ranking after this pass
to put Q12's re-score first and the narrow sites last.

---

## Roofs, measured on this card

_pending — filled after the run._

## Experiments and verdicts

_pending._

## ms/fold at stake, after this pass

_pending._

## Parity

_pending._

## Corrections to the inherited record

_pending._
