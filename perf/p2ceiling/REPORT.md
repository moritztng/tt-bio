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

Committed and pushed before the device was opened, as `perf/p2ceiling/PREDICTIONS.md` in commit
`20b013d0` — that file is the tamper-evident copy. The text below is the same predictions with the
denominator wording spelled out ("10 % of the M=16384 baseline figure" for "10 % of `t(16384)`") so
the shared roofline gate can read it; no number, band or falsifier was changed. Every arm names the
number I expect and what would count as having been wrong.

**P1 — H1, the in0 reader's circular buffer sets the pace at a narrow output.** At K=256, nt=8, both
operands L1-resident, output DRAM-interleaved, I sweep M through 4096 / 8192 / 16384 / 32768 and fit
`t = c0 + c1*M`. I predict the fit is essentially proportional: `c0` under 10 % of the M=16384 baseline figure and the
achieved TFLOP/s flat to within 5 % of the M=32768 figure across the four M. Roof-method measured this cell at 83.42 us and
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
reaching at least 110 TFLOP/s, i.e. within 25 % of the card's K=4096 figure. **Wrong if** the four
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
denominator for it. I predict @1428 re-scores in 85-95 % of the width-matched write roof and @1434 (1:1) in 80-92 % of the same roof, i.e. the two
rows together have 50-250 ms/fold of headroom, not the 3-4x the open question allows for.

**P5 — the two narrow-output sites, and whether a program config fixes them.** Real fold shapes:
`linear`@`tenstorrent.py:2807` is `[1, 298, 320, 256] x [256, 16]` (nt=1), `linear`@`protenix.py:306`
is `[1, 298, 320, 256] x [256, 64]` (nt=2). Baseline is production `core_grid=CORE_GRID_MAIN`, which
is `in0_block_w=1, out_block_h=per_core_M`. Tuned is an explicit 1D config, `in0_block_w=8` (= k_tiles),
`per_core_M=30..32` so `ceil(2980/per_core_M)` lands at 94-100 of 110 cores. T5 measured 458.9 us on
16 of 110 cores at 2807 and 478.5 us on 32 of 110 at 306, on pc. I predict the qb1 card-0 baselines
land within 15 % of those baseline figures, and that the tuned config gives 1.2-1.6x, i.e. 290-380 us at 2807 and
300-400 us at 306, with cores engaged rising to 94-100 of 110. **Wrong if** the tuned config fails to
beat the baseline by more than 3 % of the baseline figure — that is T5's own kill test and it would say the read path binds
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

**qb1 card 0, this pass, 2026-08-10, ttnn 0.67.4, Blackhole p150a, compute grid 13x10.** I did not
inherit any of these — every figure below was re-measured on my own card by
`perf/p2ceiling/p2_probe.py` (`roofs_c0.json`, `q12b_c0.json`), because roofs are per-card and
roof-method's canonical table is card 2 (`roofline-roof-must-be-measured-not-asserted`).

| roof | measured on card 0 | how |
|---|---:|---|
| DRAM -> L1 read | **376.8 GB/s** | 67.1 MB interleaved clone, DRAM sees reads only, 178.1 us |
| L1 -> DRAM write, unary writer | **264.1 GB/s** | 67.1 MB clone the other way, 254.1 us |
| DRAM -> DRAM copy | **399.1 GB/s** | same clone, both legs on DRAM, 336.3 us |
| square bf16 HiFi4 compute | **141.65 TFLOP/s** | 4096^3, 13x10, 970.3 us. Square and DRAM-output, so it is the method roof and not a denominator for a K=256 op (charter §4.6) |
| **matmul-writer write roof** | **213.7 GB/s** | the best write rate any matmul reaches on this card: K=256, nt=128, M=6368, `core_grid` 13x10, 244.07 us. 80.9 % of the unary write roof |
| same, at nt=64 | 196.6 GB/s | M=12768, `core_grid` 13x10, 265.99 us |
| same, at nt=32 / 16 / 8 | 176.1 / 174.1 / 182.8 GB/s | `minimal_matmul` wins all three; the best `ttnn.linear` config at nt=8 reaches only 133.0 GB/s |
| operand placement, K=256 nt=64 M=12800 | 201.3 GB/s L1 operands vs 190.8 DRAM operands | identical shape and config — **the whole operand-placement axis is 5.2 %** |

**Machine balance on this card: 141.65 TFLOP/s / 399.1 GB/s = 354.9 FLOP/byte**, within 5 % of the
charter's 338. Every site in this brief is far on the memory side of it — the PWA z->bias at
28.4 FLOP/byte, the template z projection at 51.2, the pair-track projections at 128.0, the qkv
projection at 192.0 — so all of them are memory-bound and none is compute-bound.

## Experiments and verdicts

### H1 — KILLED as stated, and the M it was measured at is why

Kill test as roof-method wrote it: halve M at fixed nt and K, both operands L1-resident, output
DRAM-interleaved, K=256, nt=8. `--arm h1`, `h1_c0.json`.

| M | `core_grid` 11x10 | TFLOP/s | ns per output row | time ratio vs 2M |
|---:|---:|---:|---:|---:|
| 4096 | 44.19 us | 12.15 | 10.789 | 0.886 |
| 8192 | 49.90 us | 21.52 | 6.092 | 0.612 |
| 16384 | **81.59 us** | **26.32** | 4.980 | 0.536 |
| 32768 | 152.14 us | 28.23 | 4.643 | — |

**Time does not halve and the rate is not flat.** Halving M from 32768 leaves 53.6 % of the time,
not 50 %; from 16384 it leaves 61.2 %. Across the sweep the rate moves by 132 % of its smallest
value against the 5 % tolerance I committed to, so by my own criterion **H1 is KILLED**. The affine
fit on the top two points is `t = 11.04 us + 4.306 ns x M`: a **fixed 11.04 us per-op cost**, well
above T4's 6.40 us launch floor, and at M=16384 that is **13.5 % of the op**. Roof-method's
25.74 TFLOP/s cell was measured at exactly that M, so an eighth of its shortfall against the roofs is
a fixed cost being amortised rather than a reader ceiling.

**What survives.** At the fold's own row count (M = 298 x 320 = 95360) the same fixed cost is
**2.6 % of the op**, so the reading that the in0 stream paces the production sites is not disturbed;
what is dead is the claim that the M=16384 cell measured that per-element rate cleanly. The
asymptotic per-row cost of 4.306 ns is **30.4 TFLOP/s**, 15.5 % above the 26.32 TFLOP/s the cell
reported.

Second falsifier — in0's buffer type at the identical shape (M=16384, in1 always L1, output DRAM):

| config | in0 in L1 | in0 in DRAM | ratio |
|---|---:|---:|---:|
| `core_grid` 11x10 (which forces `in0_block_w` = 1) | 81.59 us | 136.25 us | **1.67x slower from DRAM** |
| explicit 1D config, `in0_block_w` = 8 | 91.73 us | **77.87 us** | **0.85x — faster from DRAM** |

Roof-method's falsifier ("an L1-resident, pre-tilized in0 still costing the same per element") does
not fire: at `in0_block_w` = 1 the buffer type is worth 67 % of the L1-in0 figure. But the sign **reverses** at
`in0_block_w` = 8, and that reversal is the mechanism finding of this pass. The limiter is the
**transaction width of the in0 stream**, not where in0 lives: at a one-tile-wide K block each core
re-walks its M-strip through a shallow circular buffer with Kt narrow strided NOC reads per output
block, which DRAM serves badly and L1 serves well; at a Kt-wide block it issues one contiguous read
per block, which DRAM serves best and which costs L1 only the bank space it takes from the CBs.
**This would be falsified by** a bw sweep whose gain is the same in both operand placements; it is
not, it changes sign (H3 below).

### H2 — CONFIRMED. Overlap is set by per-core output block width, at fixed total bytes

Kill test as written: hold total output bytes at 52.43 MB and vary only nt. `--arm h2`, `h2_c0.json`.
Write basis is a unary L1 -> DRAM clone of exactly those bytes, **199.41 us = 262.9 GB/s**, the
writer at 99.5 % of this card's unary write roof (264.1 GB/s). Compute basis is the same matmul with an
L1 output, measured at M/4 and scaled by FLOPs.

| arm | config | measured | compute | write | max() | sum() | measured / sum | measured / max |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| nt=8, M=102400 | `core_grid` 11x10 | 438.33 us | 237.70 | 199.41 | 237.70 | 437.11 | **1.003 — exactly additive** | 1.844 |
| nt=8, M=102400 | 1D, bw=8, obh=5 | 394.33 us | 237.70 | 199.41 | 237.70 | 437.11 | 0.902 | 1.659 |
| nt=64, M=12800 | `core_grid` 11x10 | 274.57 us | 157.68 | 199.41 | 199.41 | 357.09 | **0.769** | 1.377 |
| nt=64, M=12800 | `core_grid` 13x10 | 269.75 us | 157.68 | 199.41 | 199.41 | 357.09 | 0.755 | 1.353 |

Same writer, same 52.43 MB, same config family: the nt=8 arm is **additive to 0.3 %**, the nt=64 arm
sits at **76.9 % of its own additive total**, and the two differ by **1.60x** where 10 % would have
killed H2. Total bytes and op duration are held fixed by construction, so neither can be the index.
**H2 CONFIRMED** — the packer-to-writer circular buffer only stays full when a core has a wide enough
output block to keep transactions queued, and below that the write drains behind the compute.
**It would have been killed by** the two arms converging; they do the opposite.

The compute basis is scaled from M/4, and H1 shows the rate rises with M, so the true compute term at
full M is smaller and the nt=8 arm is if anything more additive than 1.003. Both arms are short of
`max()`, so card 0 does not reach the near-perfect nt=64 overlap roof-method saw on card 2 (within
4 % of the max() figure there, 37.7 % above the max() figure here). That is a card difference, which is why §4.1 forbids
inheriting.

### H3 — KILLED. `in0_block_w` is a DRAM transaction-width knob, not a K-loop-depth knob

Kill test as written: sweep `in0_block_w` at K=256 with everything else pinned and an **L1 output**.
`--arm h3`, `h3_c0.json`. M=16384, nt=8, both operands L1-resident, 103 of 110 cores by construction
(`per_core_M` = 5, `ceil(512/5)` = 103).

| `in0_block_w` | L1 output | TFLOP/s | DRAM output | TFLOP/s |
|---:|---:|---:|---:|---:|
| 1 | **40.51 us** | **53.01** | 83.95 us | 25.58 |
| 2 | 44.34 us | 48.43 | 85.38 us | 25.15 |
| 4 | 55.03 us | 39.02 | 89.36 us | 24.03 |
| 8 | 56.36 us | 38.10 | 93.02 us | 23.09 |

**Monotone fall, 0.72x from bw=1 to bw=8**, against a predicted rise of at least 1.25x. **H3 is
KILLED**: the L1-output compute rate at low K is not capped by K-loop depth, because deepening the K
loop makes it worse. The K dependence is real and survives — at the same nt=8, L1 output and 103
cores the card reaches 53.01 TFLOP/s at K=256 and 89.56 at K=1024 — but no `in0_block_w` setting
closes that gap, so Phase 3 must not expect this knob to move it.

**What replaces it — H3', CONFIRMED.** `--arm h3b`, `h3b_c0.json` runs the identical sweep with the
operands **DRAM-interleaved**, which is where every production site's operands actually sit:

| shape | `per_core_M` / cores | bw=1 | bw=2 | bw=4 | bw=8 | bw8 / bw1 |
|---|---|---:|---:|---:|---:|---:|
| M=16384, L1 out | 5 / 103 | 21.29 | 31.68 | 31.05 | **31.84** TFLOP/s | **1.50x** |
| M=47680, L1 out | 15 / 100 | 27.24 | 36.32 | 37.16 | **43.65** TFLOP/s | **1.60x** |
| M=95360, DRAM out | 30 / 100 | 20.54 | 26.52 | 29.52 | **32.75** TFLOP/s | **1.59x** |
| M=16384, L1 out, **L1 operands** | 5 / 103 | 53.01 | 48.43 | 39.02 | 38.10 TFLOP/s | **0.72x** |

The sign is set by **in0's buffer type**, not by K and not by `per_core_M`: 1.50-1.60x with a
DRAM-interleaved in0 at three different `per_core_M`, 0.72x with an L1-resident one. A K-loop
overhead would be blind to where in0 lives. This also reconciles the pass with trimul-rescore, whose
587.86 / 468.96 / 424.68 / 385.04 us at bw = 1/2/4/8 was measured with DRAM operands and is the same
1.53x.

### The narrow-output sites — the tuned config wins ~2x, and T5's kill test does not fire

`--arm sites`, `sites_c0.json`. Real fold shapes, `[1, 298, 298, 256]` so ttnn pads to
`[1, 298, 320, 256]` and `m_tiles` is 298 x 10 = 2980, not a square stand-in
(`tt-bio-l1-residency-guard-dead-in-real-folds`). Baseline is production's
`ttnn.linear(core_grid=CORE_GRID_MAIN)`. Tuned is the 1D in1-mcast config family
`_pair_proj_program_config` already ships for the pair track, with `in0_block_w` and `out_block_h`
exposed. Its L1 budget subtracts the output block's bf16 tile **and** the fp32 partial the packer
accumulates into (`perfwar-programconfig-gate-output-not-subtracted` — the output term is not
dropped; at nt=1, bw=8, obh=5 the config needs 358 400 B of a 1 461 760 B bank).

| site | shape | calls/fold | baseline | cores engaged, measured | best tuned | cores | speedup | ms/fold |
|---|---|---:|---:|---|---:|---:|---:|---|
| PWA z->bias, `tenstorrent.py:2832` | `[1,298,320,256] x [256,1]`, nt=1 | 240 | **438.46 us** | **~16 of 110** — the `core_grid` ladder is flat: 948.88 us at 4 cores, 448.34 at 16, 444.30 at 110 | **226.91 us** (bw=8, obh=5, `per_core_M`=30) | **100 of 110** | **1.93x** | 105.2 -> **54.5** |
| template z proj, `protenix.py:306` | `[1,298,320,256] x [256,64]`, nt=2 | 40 | **477.94 us** | **~16 of 110** — 1105.58 us at 4 cores, 499.67 at 16, 476.50 at 110 | **236.54 us** (bw=8, obh=5, `per_core_M`=30) | **100 of 110** | **2.02x** | 19.1 -> **9.5** |

**T5's kill test does not fire.** It said: if the tuned config does not beat 459 us / 478 us, the read
path binds and the idle cores are a symptom. The tuned config beats them by 1.93x and 2.02x, so the
**idle cores were a cause, not a symptom**. My baselines reproduce T5's within 4.5 % of T5's baseline figure (438.46 vs
458.9) and 0.1 % of the same baseline figure (477.94 vs 478.5) on a different host, which is a cross-host check on the ledger row
as well as on the lever.

The read rate says the same thing: **111.4 GB/s = 29.6 % of this card's 376.8 GB/s read roof** at
baseline, **215.2 GB/s = 57.1 % of that read roof** tuned. That is T1's ~130 GB/s in0-stream ceiling
(its H5) reproduced at a third site and then broken. The ceiling is `in0_block_w` = 1, which is what
`core_grid=` selects, and it is not a property of the DRAM banks.

The two knobs are separable, which matters for parity:

| arm | @2832 | @306 | changes the arithmetic? |
|---|---:|---:|---|
| baseline `core_grid` 11x10 | 438.46 us | 477.94 us | — |
| `out_block_h` = 5 only (bw stays 1) | 389.03 us, **1.13x** | 384.68 us, **1.24x** | no |
| + `in0_block_w` = 8 | 226.91 us, **1.93x** | 236.54 us, **2.02x** | yes |

### Q12 — CLOSED. T1's 174.9 GB/s was the right denominator; the index is output width

`--arm q12`/`q12b`. First the experiment the brief asked for: output pinned at 48.82 MB and nt=8
(the fold's own pair-track output, M = 95360 = 298 x 320), K varied so the op's own **read:write byte
ratio** moves from 1:4 to 3:1, each K run twice — operands DRAM-interleaved (the real op, which pays
the contention) and operands L1-resident (the writer with no contention at all).

| read : write | K | DRAM operands, best cfg | L1 operands, best cfg | contention penalty |
|---:|---:|---:|---:|---:|
| 1:4 | 64 | 147.8 GB/s written | 150.2 GB/s | 1.6 % against the L1 arm |
| 1:2 | 128 | 131.4 GB/s | 142.9 GB/s | 8.0 % against the L1 arm |
| **1:1** | 256 | **126.5 GB/s** | 127.6 GB/s | 0.9 % against the L1 arm |
| 2:1 | 512 | 87.4 GB/s | 101.3 GB/s | 13.7 % against the L1 arm |
| **3:1** | 768 | **65.4 GB/s** | 75.2 GB/s | 13.0 % against the L1 arm |
| the write-roof shape, nt=64 | 256 | 190.8 GB/s | 201.3 GB/s | 5.2 % |

The achieved write rate falls **2.26x** across the ratio sweep — and the L1-operand arm falls with it,
2.00x, so **almost none of the fall is contention**. My prediction P4 named the fall correctly and
attributed it to the wrong mechanism; the matched L1 arm is what killed my own account, which is why
it was in the design. Read/write contention is worth 0-14 % against the L1 arm here, and 5.2 % against it at the write-roof shape.

The decomposition confirms it. Best DRAM-out config minus best L1-out config at the same shape is the
part of the write that is **exposed** rather than hidden behind compute; the whole 48.82 MB write leg
is 184.9 us at this card's 264.1 GB/s unary write roof:

| read : write | K | L1-out (compute) | DRAM-out | exposed write | share of the 184.9 us write leg still hidden |
|---:|---:|---:|---:|---:|---:|
| 1:4 | 64 | 100.88 us | 329.77 us | 228.89 us | **none — the DRAM output costs more than the write leg** |
| 1:2 | 128 | 136.71 us | 347.81 us | 211.10 us | none |
| 1:1 | 256 | 246.60 us | 384.19 us | 137.59 us | 25.6 % of the write leg |
| 2:1 | 512 | 474.43 us | 543.15 us | 68.72 us | 62.8 % of the write leg |
| 3:1 | 768 | 695.71 us | 765.43 us | 69.73 us | 62.3 % of the write leg |

**Q12 verdict: read:write ratio is the wrong index and the org should stop carrying three write
roofs.** The write rate an op achieves is set by its **output width** and by how much of its write
overlaps its compute (H2), not by where its operands sit. The card's matmul-writer write roof by
width, L1 operands, K=256, best of every config including `minimal_matmul`:

| output width nt | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|
| best write rate reached | 182.8 GB/s | 174.1 | 176.1 | 196.6 | **213.7 GB/s** |

And the re-score of T1's two rows, with T1's own production op (`ttnn.experimental.minimal_matmul`)
at the fold's own shape on card 0:

| row | T1, card 2 | mine, card 0 | agreement | achieved write | width-matched roof | **score** | T1's score |
|---|---:|---:|---:|---:|---:|---:|---:|
| `minimal_matmul` qkv @1428, nt=24 | 865.3 us / 906.8 ms/fold | **880.17 us / 922.4 ms/fold** | 1.7 % against T1 | 166.4 GB/s | ~175 GB/s (the nt=16 / nt=32 bracket) | **95.1 %** of that roof | 96.8 % of T1's roof at 174.9 |
| `minimal_matmul` gate @1434, nt=8 | 359.8 us / 377.1 ms/fold | **356.05 us / 373.1 ms/fold** | 1.1 % against T1 | 137.1 GB/s | 182.8 GB/s | **75.0 %** of that roof | 77.6 % of T1's roof at 174.9 |

**T1's 174.9 GB/s was right, to within two points on both rows.** It happens to be this card's write
rate at nt ~ 16-32, which is where both rows sit. **156.6 GB/s is the wrong denominator** — it is
below what either row achieves, which is what made @1428 score over 100 %. **197.7 GB/s is also the
wrong denominator** — it is the nt=64 L1-operand figure and neither row is at nt=64. The org should
carry **213.7 GB/s as the card's matmul-writer write roof** and a by-width table beside it, and
should not score an nt=8 row against a write roof at all: its write is additive on top of its
compute, so a write-roof percentage silently prices the compute term as writer inefficiency.

## ms/fold at stake, after this pass

Conversions are charter §4.9. T1's two rows use **x1048 executions**, which is the charter's
**x524** `PairformerLayer` executions times the two triangle-attention modules per layer; I used
T1's own 1048 so the numbers are directly comparable, and 880.17 us x 1048 = 922.4 ms/fold. The two
narrow-output sites are T5's stages, which convert **x10** per recycling cycle and give the call
counts 240 and 40 per fold; I measured the per-call time, not the counts, and take T5's counts as
given. The template stack's **x80 at c=64** constant does not apply to `protenix.py:306`, whose input
is the c_z=256 pair track and which T5 counted at 40.

| rank | lever | ms/fold at stake | evidence | status |
|---:|---|---:|---|---|
| 1 | **the exposed DRAM write at a narrow output** — the pair-track class (trimul-rescore: 72 % of the write leg exposed, 183.2 us, up to ~330 ms/fold) plus gate @1434 (109.4 us exposed of a 184.9 us leg, **114.6 ms/fold**) | **~445** | H2 CONFIRMED: at nt=8 the write is additive to 0.3 %, at nt=64 it is 76.9 % of additive, at identical bytes | no lever measured. At nt=8 there is no width to widen, so the candidates are a wider fused output or an L1 output, both Phase 3's |
| 2 | `in0_block_w`=8 + `out_block_h`=5 on `_KeyedWeights._lin` and the PWA z->bias `linear` | **60.3** (50.7 at @2832, 9.6 at @306) | measured 1.93x / 2.02x at the fold's own shape, 16 -> 100 of 110 cores | **lever in hand**, not bit-exact |
| 3 | the same two sites, bit-exact subset (`out_block_h`=5 only) | **15.5** (11.8 + 3.7) | measured 1.13x / 1.24x | **lever in hand**, `torch.equal` True |
| 4 | Q12's re-score of T1's two write-roof rows | ~250 of ledger accuracy, now **resolved rather than banked** | 174.9 GB/s stands; operand placement is 5.2 % of the writer, not the 26 points of the 156.6-to-197.7 spread | not a change |
| 5 | raise `in0_block_w` wherever in0 is DRAM-interleaved and `core_grid=` is used today | not priced — needs a site inventory | 1.50-1.60x measured at three `per_core_M` | Phase 3 |

**Rank 4 is a subtraction and that is the honest outcome.** Q12 was opened because @1428 (906.8
ms/fold) and @1434 (377.1) might have had three to four times the headroom their 96.8 % and 77.6 %
implied. They do not. What genuinely remains on @1434 is the exposed write, which is rank 1 and a
different mechanism.

I also mis-priced my own predictions and it is worth recording: P5 predicted 1.2-1.6x on the narrow
sites and about 35 ms/fold combined; the measurement is 1.93-2.02x and 60.3 ms/fold, and my expected
ranking (Q12 first, the narrow sites last) came out inverted.

## Parity

**No production change was made in this pass**, so nothing here has changed the arithmetic anywhere.
The verdicts below are for the levers Phase 3 would take, and all of them are measured.

- **`out_block_h` = 5 alone, `in0_block_w` unchanged at 1: bit-exact, measured.** `torch.equal`
  against the production `core_grid=CORE_GRID_MAIN` output returned **True** at both sites, at the
  fold's own `[1, 298, 320, 256]` shape. Only the drain schedule moves; the contraction accumulates
  in the same order.
- **`in0_block_w` = 8: not bit-exact, measured.** `torch.equal` returned **False**. @2832: RMSD
  2.021e-02, 1.342e-03 relative to the output's own standard deviation, PCC 0.99999923. @306: RMSD
  2.120e-02, 1.327e-03 relative, PCC 0.99998975. This is the expected consequence — `in0_block_w`
  is the sole bit-exactness knob on DRAM-interleaved batched operands
  (`ttnn-batched-matmul-programconfig-rules`), because it re-blocks a bf16 contraction and bf16
  addition is not associative.
- **The precedent is already in main.** `_PAIR_PROJ_BW = 16` shipped in `bbb5d85b` and puts the pair
  track at `in0_block_w` = 8 on the same operand class, release-gated across protenix-v2 and two
  other models. Rank 2 is therefore the same parity class as a change this repo has already accepted
  — but it is still release-gated, it stays on this branch, and it is Phase 3's to argue with a full
  parity gate.
- Every other result in this doc is a timing, not an output, so it has no parity consequence.

## Corrections to the inherited record

1. **Roof-method's K=256 / nt=8 / DRAM-output cell (25.74 TFLOP/s, 83.42 us at M=16384) carries a
   13.5 % fixed-cost component.** The affine fit on card 0 is `11.04 us + 4.306 ns x M`. At the
   fold's own M=95360 that cost is 2.6 %. The cell's conclusion survives; its number under-reports
   the asymptotic rate by 15.5 % (30.4 against 26.32 TFLOP/s).
2. **H1's mechanism is the in0 stream's transaction *width*, not its buffer type.** At
   `in0_block_w` = 1 an L1-resident in0 is 1.67x faster than a DRAM one; at `in0_block_w` = 8 the
   DRAM one is 1.15x faster. Any statement of the form "the matmul reader tops out near 130 GB/s" is
   a statement about `in0_block_w` = 1, which is what `core_grid=` selects, and not a property of the
   card. This retires T1's H5 as a card ceiling.
3. **H3's account is wrong and must not be carried into Phase 3.** `in0_block_w` does not amortise a
   K-loop fixed cost: with L1-resident operands, raising it from 1 to 8 at K=256 costs **28 %** of the L1-operand figure
   (53.01 -> 38.10 TFLOP/s). Its gain is confined to DRAM-interleaved operands.
4. **Q12 is closed: 174.9 GB/s stands, 156.6 and 197.7 are both wrong denominators for T1's rows**,
   and the card's matmul-writer write roof is **213.7 GB/s** at nt=128. Operand placement is worth
   5.2 %. The index is output width, and an op at nt=8 or narrower should be scored against
   `compute + write` rather than against a write roof.
5. **T5's "16 of 110 cores" at the PWA z->bias `linear` is confirmed independently on a second
   host**, by a `core_grid` ladder flat from 16 cores (448.34 us) to 110 (444.30 us). At
   `protenix.py:306` my ladder puts it at ~16 core-equivalents rather than T5's 32; the ladder cannot
   separate 16 from 32 at this granularity, and either reading leaves 80+ cores idle.
6. **The PWA z->bias `linear` is at `tenstorrent.py:2832`, not 2807.** 2807 is the weight upload in
   `__init__`; the call is inside the per-head loop at 2832 and its second operand is
   `self.z_weight[:, i:i+1]`, a single logical output column padded to one tile. Every number in
   T5's row reproduces, but Phase 3 should patch 2832.
7. **`minimal_matmul` is the best matmul on this card at every output width below 64**, reaching
   182.8 GB/s written at nt=8 where the best `ttnn.linear` program config reaches 133.0 GB/s. Any
   future roof taken with `ttnn.linear` at a narrow output is 1.37x low.

**No production change was made.** Everything in this pass is a probe under `perf/p2ceiling/`,
committed to `wk/protenix-trunk--p2-matmul-ceiling` and pushed. Nothing under `tt_bio/` was touched,
nothing is proposed for merge here, and merging is Moritz's call.

**Generalisation, recorded and not chased (charter §1):** the `in0_block_w` sign flip is a ttnn
0.67.4 matmul-reader property and will appear in every other model on this stack. One line, out of
scope.
