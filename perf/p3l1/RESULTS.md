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

## Roofs, measured on this card

**qb1 card 1, this pass, 2026-08-10, ttnn 0.67.4, Blackhole p150a.** Every figure in this section
I **re-measured on this card this pass** — qb1 card 1 — and inherited none of — charter §4.1, and the
point of re-taking them on the card X2 already used is that a same-card disagreement would mean the
card's state moved and nothing in X2's doc could be carried into mine. Produced by
`perf/ledger_298/roofs_card.py` (`perf/p3l1/roofs_c1.json`), host load average 1.21 at the time,
two other folds live on `/dev/tenstorrent/0` and nothing on card 1.

| roof | mine, card 1 | X2's, card 1 | apart, as a % of X2's figure | how |
|---|---:|---:|---:|---|
| DRAM -> L1 read | **388.1 GB/s** | 388.3 | 0.05 % of X2's figure | 64 MB interleaved clone |
| L1 -> DRAM write, unary writer | **264.4 GB/s** | 265.1 | 0.26 % of X2's figure | 64 MB clone the other way |
| DRAM -> DRAM copy | **403.5 GB/s** | 402.5 | 0.25 % of X2's figure | same clone, both legs DRAM |
| square bf16 HiFi4 compute, DRAM out | **135.67 TFLOP/s** | 135.65 | 0.01 % of X2's figure | 6144³, 13x10 |
| machine balance | **349.6 FLOP/byte** | 349.3 | 0.09 % of X2's figure | 135.67 / 388.1 |

**P8 CONFIRMED**, and more tightly than it had to be: four of the five agree to under 0.3 % of X2's figure. The card has not moved, so X2's card-1 numbers and mine are directly comparable and the
reproduction below is a like-for-like one.

**Charter §4.6's second column, at this op's own output width and its own output buffer type.**
The pair-track projection is `[1,298,320,256] @ [256,256]`, `nt`=8, 12.500 GFLOP, output 48.82 MB.
Measured by me on card 1 at the production 13x10 grid (`perf/p3l1/probe_c1.json`):

| config | output buffer | us | write rate | TFLOP/s |
|---|---|---:|---:|---:|
| `ttnn.linear(core_grid=)` | DRAM | 706.81 | 69.1 GB/s | 17.68 |
| tuned `in0_block_w`=1, `out_block_h`=5 | DRAM | 577.91 | 84.5 GB/s | 21.63 |
| tuned `in0_block_w`=8, `out_block_h`=5 — **production today** | DRAM | 375.54 | 130.0 GB/s | 33.29 |
| `ttnn.linear(core_grid=)` | **L1** | **refused** | — | — |
| tuned `in0_block_w`=1, `out_block_h`=5 | **L1** | 389.75 | 125.3 GB/s | 32.07 |
| tuned `in0_block_w`=8, `out_block_h`=5 | **L1** | 265.57 | 183.8 GB/s | 47.07 |
| tuned `in0_block_w`=8, `out_block_h`=2 | **L1** | 247.12 | 197.6 GB/s | 50.58 |

**Which roof binds, and it changes when the output moves.** With a DRAM output the projection is
write-bound: production today writes at 130.0 GB/s, 49.2 % of this card's measured 264.4 GB/s unary
write roof. Take the write away and the binding roof becomes the **in0 DRAM read** — the op still
reads 48.82 MB of activations whatever it does with its result, and the best L1-output arm's
247.12 us is 197.6 GB/s of that read, **50.9 % of this card's measured 388.1 GB/s read roof**. The
square compute roof is not the roof for this row and is not used as one.

**Arithmetic intensity, and which side of the 349.6 FLOP/byte balance.** Every row is on the memory
side, but the L1 output moves one of them most of the way across:

| site | shape | DRAM bytes moved | AI (FLOP/byte) | side of 349.6 |
|---|---|---:|---:|---|
| pair-track projection, DRAM out (today) | `[1,298,320,256] x [256,256]` | 97.65 MB | 128.0 | memory |
| the same projection with an **L1** output | | 48.82 MB | 256.1 | memory, 1.37x under the balance |
| `AttentionPairBias` z->bias, `tenstorrent.py:2088` | `x [256,16]`, nt=1 | 54.93 MB | 28.4 | memory, deep |
| the same with an L1 `layer_norm` source and an L1 output | | ~0 | not DRAM-bound at all | — |

**Core utilisation.** The tuned config puts `per_core_M`=25 on `m_tiles`=2980, so
`ceil(2980/25)` = **120 of the grid's 130 cores** receive work; the remaining 10 are the cost of
`out_block_h`=5 having to divide `per_core_M`. The untuned `ttnn.linear(core_grid=)` baseline is the
one that leaves the grid idle — X2 measured its ladder flat from 16 cores to 110 on this card, which
I did not re-take and am citing as X2's.

**Overlap: compute + comm, not max(compute, comm), and now measured on this row.** Removing the
48.82 MB DRAM write at `in0_block_w`=8 takes the projection from 375.54 to 265.57 us, a saving of
109.97 us where the write on its own is 184.7 us at this card's unary write roof. So **59.5 % of that write roof cost was exposed**
and 40.5 % of that same roof cost was already hidden behind the contraction. The total sits
much nearer compute+comm than max(compute, comm), which is what makes removing the write worth
something at all.

## What changed, and the A/B that measured it

Two changes on `wk/protenix-trunk--p3-l1-output`, both in `tt_bio/tenstorrent.py`, both behind their
own module constant so each carries its own parity decision and its own A/B arm:

| constant | what it does | sites |
|---|---|---|
| `_PAIR_PROJ_L1_OUT` + `_PAIR_PROJ_L1_BW` | the pair-track output projection writes its 48.82 MB result to **L1** instead of DRAM, at main's own `in0_block_w`=8 | `_trimul_out_proj` (`p_out` and `g_out`), `TriangleAttention.gate_and_project`'s `x_out` |
| `_PAIR_BIAS_L1_NORM` | `AttentionPairBias`'s pair `layer_norm` writes to **L1**, so the z->bias projection reads its `in0` from L1, and that projection's own 6.10 MB output goes to L1 too | `tenstorrent.py:2088` and the `layer_norm` above it |

`_pair_proj_program_config` now subtracts the **output** term from the L1 bank budget when the
output lands in L1 (`per_core_M x n_tiles x tile`). The shipped helper omits it because it always
wrote to DRAM; a program-config budget that forgets its output term is exactly how a gate passes a
config the allocator then refuses. Both helpers also fall back to today's DRAM path if the
allocator refuses anyway, and remember the refusal per operand class — the static budget cannot see
what a live block is already holding, so the allocation itself is the only honest test.

### Deliverable 1, step 1 — X2's `projection + add` reproduced, at the production grid

X2 built its probe's program configs from a module-scope `COMPUTE_GRID_MAIN` = 11x10 while the fold
runs 13x10; every figure below reads the grid **after** the device is open and is therefore the
first measurement of this candidate at 130 cores. `perf/p3l1/p3_l1_probe.py --arm pair`,
`[1,298,320,256] @ [256,256]`, host load average 1.2.

| leg | projection alone | **projection + add** | vs production today | `torch.equal` vs `dram_cg` | L1 free while the output is live |
|---|---:|---:|---:|---|---:|
| `dram_cg` — `core_grid=`, DRAM out | 706.81 us | **1049.70 us** | 0.70x | **True** | 1 461 760 B/bank |
| `dram_tuned_bw8_obh5` — **production today** | 375.54 us | **733.23 us** | 1.00x | False, max abs 0.5 | 1 461 760 |
| `dram_tuned_bw1_obh5` | 577.91 us | 927.62 us | 0.79x | **True** | 1 461 760 |
| `l1_cg` — `core_grid=`, L1 out | **REFUSED** | — | — | — | — |
| **`l1_tuned_bw1_obh5` — L1 out, bit-exact** | 389.75 us | **637.30 us** | **1.150x** | **True** | 1 084 928 |
| `l1_tuned_bw8_obh5` — L1 out, release-gated | 265.57 us | 516.24 us | 1.420x | False, max abs 0.5 | 1 084 928 |
| **`l1_tuned_bw8_obh2` — L1 out, release-gated** | 247.12 us | **497.67 us** | **1.473x** | False, max abs 0.5 | 1 084 928 |

**P1 CONFIRMED and X2's measurement reproduces almost exactly.** The bit-exact pair is 1.150x
against production today where X2 got 1.149x, and the per-call delta is **95.93 us** against X2's
95.00 — 1.0 % of X2's figure apart, on a different core count. Production today's pair reproduces to
**0.13 % of X2's figure** and the bit-exact L1 pair to 0.30 % of X2's figure. The candidate is
real and it is not a probe artefact.

**P2 CONFIRMED, and this is the first correction to the record.** Against `core_grid`-DRAM the
bit-exact L1 pair is **1049.70 / 637.30 = 1.647x**; against production today it is **1.150x**. My
brief quotes "1.53x bit-exact, 1.49x release-gated" as if they shared a denominator, and they do
not: 1.53x was X2's ratio over the untuned `core_grid` baseline and 1.49x its ratio over production
today. Only the second kind converts into the 298.7 ms/fold, and the bit-exact ratio that belongs
beside 298.7 is 1.150x. The two differ by 43 % of the smaller figure, so the distinction is not
pedantic.

**One new refusal, and it did not exist at 110 cores.** `l1_cg` — an L1 output with
`ttnn.linear(core_grid=)` — **throws** at the production 13x10 grid ("Statically allocated circular
buffers in program 9 clash with L1 buffers"), where X2 measured it running at 110 cores. The tuned
config's smaller circular buffers are what make an L1 output legal at all here, so "put the output
in L1" is not a knob that can be applied to the untuned call.

### Deliverable 1, step 2 — the sequence production actually runs

The probe's pair is `proj -> add`. Production's trimul is `proj(p_out) -> proj(g_out) ->
multiply_(p,g,sigmoid) -> add_(z, .)`: **four calls of the class per Pairformer layer share two
residual adds**, so pricing 3144 calls x 95.93 us over-counts the add half. Measured as the chain
(`--arm trimul`), all legs producing the same DRAM result:

| leg | chain us | vs production today | `torch.equal` vs the bw=1 DRAM reference | L1 free after `p` | after `p` and `g` |
|---|---:|---:|---|---:|---:|
| `prod_today` — both DRAM, bw=8 | **1427.10** | 1.00x | False, max abs 3.906e-03 | 1 461 760 | 1 461 760 |
| `all_dram_bw1` — the bit-exact DRAM reference | 1835.64 | 0.78x | **True** | 1 461 760 | 1 461 760 |
| `p_l1_bw1` — only `p_out` in L1 | 1571.98 | 0.91x | **True** | 1 084 928 | 1 084 928 |
| **`both_l1_bw1` — both in L1, bit-exact** | **1323.54** | **1.078x** | **True** | 1 084 928 | 708 096 |
| `p_l1_bw8_obh5` | 1194.29 | 1.195x | False | 1 084 928 | 1 084 928 |
| **`both_l1_bw8_obh2` — release-gated** | **1039.64** | **1.373x** | False, max abs 3.906e-03 | 1 084 928 | 708 096 |

**P3 was WRONG, and being wrong here is worth more than the prediction was.** I predicted a second
concurrent 48.82 MB L1 output would be refused: 375.5 kB/bank x 2 plus 802.8 kB/bank of circular
buffers is 1587.6 kB against 1427.5 kB available. It is not refused. Both fit, the allocator reports
**708 096 B of 1 461 760 still free per bank** with two pair tensors resident, and `both_l1_bw1` is
1.078x on the chain against production today while `p_l1_bw1` alone is 0.91x — i.e. **putting only
one of the pair in L1 is a loss and putting both is a win.** My static budget over-counted because
`in0` and `in1` are not double-buffered at their worst case simultaneously on this shape. The
production guard therefore keeps the static budget as a cheap first filter and lets the allocator
have the final word.

### Deliverable 2 — the third site, `tenstorrent.py:2088`, and it is bound by its SOURCE

The site is `AttentionPairBias`'s z->bias projection, `[1,298,320,256] @ [256,16]`, 484 calls/fold,
**225.8 ms/fold** in X2's live fold before `_NARROW_PROJ_BW = 1` and 208.4 ms/fold after it, which
is my baseline. The region is `layer_norm(z) -> linear -> permute(0,3,1,2)`; the projection reads
48.82 MB to write 6.10 MB, so its own write was never the cost. Measured (`--arm site2`):

| leg | region us | projection | permute | vs my baseline | `torch.equal` vs `core_grid` |
|---|---:|---:|---:|---:|---|
| `prod_cg` — before X2's change | 847.16 | 500.06 | 139.99 | 0.95x | **True** |
| **`prod_bw1` — my baseline, X2's landed change** | **801.32** | 450.32 | 141.99 | 1.00x | **True** |
| `bw1_outL1` — L1 **output** only | 750.68 | 440.49 | 104.89 | 1.067x | **True** |
| `normL1_cg` — L1 **source**, untuned projection | 428.49 | 202.20 | 139.21 | 1.870x | **True** |
| **`normL1_bw1` — L1 source** | 371.39 | 136.99 | 114.25 | **2.158x** | **True** |
| **`normL1_bw1_outL1` — L1 source and L1 output** | **306.19** | 109.33 | 76.47 | **2.617x** | **True** |
| `bw8_outDRAM` — `in0_block_w`=8, DRAM source | 615.48 | 266.60 | 138.24 | 1.302x | False, max abs 0.5 |
| `normL1_bw8` — `in0_block_w`=8, L1 source | 444.85 | 215.23 | 115.02 | 1.801x | False, max abs 0.5 |

**P7 was right about the mechanism and badly wrong about the size.** The direction is exactly as
predicted — the L1 **output** alone is worth 50.64 us/call, 24.5 ms/fold, inside the "under
25 ms/fold" I committed to, while the L1 **source** is worth 429.93 us/call — but I predicted 60-160
ms/fold from the source and the probe says 208 ms/fold for the source alone and **239.6 ms/fold for
both**, over 484 calls. `stage-through-l1-fixes-source-not-destination` holds: this op's bad side is
its source, and staging fixed it.

**And the `in0_block_w` sign flip reproduces.** With a DRAM source `in0_block_w`=8 is worth 1.302x;
with an L1 source it is a **loss** against `in0_block_w`=1 (444.85 against 371.39, 0.84x). That is
X2's inventory exclusion, confirmed by direct measurement rather than inherited: the knob that pays
for a DRAM read costs when the read is already free. Everything here stays at `in0_block_w`=1, which
is also the bit-exact side.


### The wall, and the A/B protocol that measured it

Every arm is a **real 298 aa protenix-v2 fold** (10 recycles / 200 sampling steps / 1 sample /
seed 0). Arms run in **separate processes on the same card in one session**, launched back to back
from one script with the unmodified baseline arm restored between them, and in the decisive sweep
the baseline runs **first and last** so process-to-process drift is bracketed rather than assumed
away (`perf/p3l1/run_ab.sh`, `run_post.sh`, `run_post2.sh`). Two instruments:

- **block wall** — `PairformerLayer.__call__` synchronised on both sides, summed over the fold's
  604 executions (524 at c_z=256 plus the template stack's 80 at c=64). Nothing inside the block is
  serialised, so overlap survives.
- **op site wall** — `_pair_proj_linear`, `_narrow_proj_linear` and the layer's residual
  `ttnn.add_`, each synchronised on both sides, keyed by operand class and summed over the fold.
  This removes each op's overlap with its neighbours, so its arms are comparable to each other but
  its parts do not sum to the block.

**The op wall reproduces to under 1 % of the baseline figure, and that is what licenses
the figures below.** Three
independent baseline folds in two sessions:

| class | `base` | `base2` | `base3` | spread |
|---|---:|---:|---:|---:|
| `_narrow_proj_linear` `[1,298,320,256] x [256,16]`, 484 calls | 252.217 ms | 252.107 | 252.082 | **0.05 % of the mean** |
| residual `ttnn.add_`, 3988 calls | 1187.687 ms | 1188.731 | 1192.299 | 0.39 % of the baseline figure |
| `_pair_proj_linear` `[1,298,320,256] x [256,256]`, 2096 calls | 982.238 ms | 985.907 | 985.940 | 0.38 % of the baseline figure |
| `_pair_proj_linear` `[298,320,256] x [256,256]`, 1048 calls | 505.231 ms | 508.181 | 503.422 | 0.94 % of the baseline figure |
| block wall, 604 blocks | 21 143.667 ms | 21 205.382 | 21 210.769 | 0.32 % of the baseline figure |

### Where the time actually went — the op site walls

`base2` and `base3` bracket the two live arms in one session; the control column is their mean.
Conversion is charter §4.9's **x524** for the pair-track projection classes (4 per layer at
`_trimul_out_proj` = 2096 calls, 2 per layer at `gate_and_project` = 1048) and **x484** for the
`AttentionPairBias` site. Call counts are counted in the fold.

| class | calls/fold | control (`base2`/`base3` mean) | **`rg` — L1 out at `in0_block_w`=8 + the 2088 source** | delivered | `both2` — L1 out at `in0_block_w`=1 + the same source | delivered |
|---|---:|---:|---:|---:|---:|---:|
| `_pair_proj_linear` `[1,298,320,256] x [256,256]`, trimul | 2096 | 985.924 ms | **793.581** | **-192.343** | 1072.106 | **+86.182** |
| `_pair_proj_linear` `[298,320,256] x [256,256]`, `gate_and_project` | 1048 | 505.802 ms | **409.410** | **-96.392** | 534.541 | **+28.739** |
| the layer's residual `ttnn.add_` | 3988 | 1190.515 ms | **1057.959** | **-132.556** | 1049.839 | -140.676 |
| `_narrow_proj_linear` `[1,298,320,256] x [256,16]`, **the 2088 site** | 484 | 252.095 ms | **101.434** | **-150.661** | 101.075 | -151.020 |
| `_pair_proj_linear` at the template stack's c=64, both classes | 480 | 92.712 ms | 83.867 | -8.845 | 85.083 | -7.629 |
| `_narrow_proj_linear` `[298,320,256] x [256,1]`, PWA, untouched | 240 | 123.494 ms | 125.641 | +2.147 | 125.941 | +2.447 |
| **sum of the op walls** | | | | **-580.797** | | **-184.404** |
| `body:AttentionPairBias` (the whole `layer_norm -> proj -> permute` region + attention) | 5284 | 3439.420 ms | 3212.978 | **-226.442** | 3256.016 | -183.404 |
| `body:TriangleMultiplication` | 1208 | 7504.885 ms | 7276.200 | **-228.685** | 7566.713 | +61.828 |
| `body:TriangleAttention` | 1208 | 8010.867 ms | 7949.299 | -61.568 | 8096.597 | +85.730 |
| **block wall** | 604 | **21 208.076 ms** | **20 646.229** | **-561.847** | 21 088.307 | **-119.769** |
| fold plDDT | | 0.859489 | **0.859489** | **unchanged** | 0.853952 | moved |

**The block wall and the sum of the op walls agree to 3.4 % of the block figure** on the `rg` arm
(-561.847 against -580.797), which is the cross-check that makes either credible.

**P4 was wrong in the most useful way available: the wall is bigger than the projection, but only
at the `in0_block_w` production already ships, and the arm I predicted from is the bad one.**
I predicted 150-350 ms/fold for the L1 output and specified `in0_block_w`=1 to keep it bit-exact
against the untuned reference. At `in0_block_w`=1 the wall delivers **33-106 ms/fold** — the
projections go 41-115 ms/fold *slower* and eat most of the residual add's saving, so on that arm
**the 298.7 does not survive contact with a wall, it comes in at 11-36 % of the projected figure**.
At main's own `in0_block_w`=8 the same L1 output delivers **430.1 ms/fold** (-192.343 -96.392
-132.556 -8.845), which is **144 % of the 298.7 projection**.

**Why the two arms differ, stated as a mechanism.** The L1 output removes a 48.82 MB DRAM write from
the projection and a 48.82 MB DRAM read from its consumer. The consumer's half is worth 132.6
ms/fold and is the same in both arms (`residual ttnn.add_`, 387-389 -> 320-323 us/call). The
projection's half only materialises if the matmul is left at the blocking it was tuned for:
`in0_block_w`=8 turns the projection from 470.9 to 375.6 us/call, `in0_block_w`=1 turns it into
497.1. The bit-exactness I was chasing at `in0_block_w`=1 costs more than the whole lever is worth.

**P5 CONFIRMED, and the fold wall stays out of the claim.** The four block-instrumented arms'
medians span 132 ms with an ordering that does not match the block wall. The `rg` arm's single
uninstrumented fold is 28.688 s against 29.335-29.722 s for its two bracketing baselines, which is
the right sign and the right rough size, but it is one fold against a 144 ms base spread and I am
not quoting it as the number.

## Delivered ms/fold

Two piles, kept apart. **The bit-exact pile is everything that returns `torch.equal` True against
production today; the release-gated pile is anything that does not, and the two are never added
into one headline number.** As it turns out the whole of this leg's delivery is in the first pile
and the release-gated pile is empty.

**Bit-exact against production today, plDDT identical to six decimals — recommend merging:**

| change | wall | delivered | parity |
|---|---|---:|---|
| **deliverable 1** — L1 output on the pair-track projection class at main's `in0_block_w`=8 | op walls: projections + residual add + template classes | **-430.1 ms/fold** | `torch.equal` True vs production today, max abs 0.0 |
| **deliverable 2** — L1 `layer_norm` source + L1 output at `tenstorrent.py:2088` | the site's own op wall, 484 calls | **-150.7 ms/fold** | `torch.equal` True, plDDT 0.859489 unchanged |
| the same, measured as the region it sits in | `AttentionPairBias` region wall | -226.4 ms/fold | as above |
| **both together** | **block wall, 604 blocks, against a bracketing control** | **-561.8 ms/fold** | plDDT **0.859489**, identical to production today |

**Measured and rejected, not shipped:** the same L1 output at `in0_block_w`=1 delivers 33-106
ms/fold instead of 430 and moves plDDT 0.859489 -> 0.853952. It buys bit-exactness against the
untuned `core_grid` reference, which main does not have either, at a cost of ~330 ms/fold. It stays
in the file as an A/B toggle with that verdict written beside it.

**Release-gated pile: empty.** Nothing proposed here changes the arithmetic. The `in0_block_w`=16
cap that the L1-output path uses is the cap main already ships.

Against the trunk's 21.827 s and a 29.4 s fold, 561.8 ms/fold is **2.6 % of the trunk figure** and
**1.9 % of the fold wall**.

## Parity

Measured at the fold's own `[1, 298, 320, 256]`, through the production helpers rather than a
re-implementation of the config (`perf/p3l1/parity_c1.py` / `parity_c1.json`,
`parity2.py` / `parity2.json`), plus the fold's own plDDT.

| arm | reference | `torch.equal` | max abs | PCC | fold plDDT |
|---|---|---|---:|---:|---:|
| **pair-track projection, L1 out at `in0_block_w`=8 (what is proposed)** | **production today, DRAM out, same config** | **True** | **0.0** | 1.0000000000 | **0.859489** |
| **the trimul chain `proj -> proj -> multiply_ -> add_`, L1 out at `in0_block_w`=8** | **production today** | **True** | **0.0** | 1.0000000000 | **0.859489** |
| **z->bias with an L1 `layer_norm` source and an L1 output** | `ttnn.linear(core_grid=)` | **True** | 0.0 | 1.0000000000 | **0.859489** |
| `layer_norm` to L1 vs to DRAM | itself, to DRAM | **True** | 0.0 | 1.0000000000 | — |
| pair-track projection, L1 out at `in0_block_w`=1 (rejected) | `ttnn.linear(core_grid=)` | True | 0.0 | 1.0000000000 | 0.853952 |
| pair-track projection, production today | `ttnn.linear(core_grid=)` | False | 0.5 | 0.9999991684 | 0.859489 |

**P9 CONFIRMED, and the clause I got wrong is worth more than the ones I got right.** A memory
config cannot move a bit, and it does not: at a fixed `in0_block_w` the L1 output is `torch.equal`
against the DRAM output of the identical config, max abs 0.0. What I missed when writing the
prediction is that this makes the **right** reference production today rather than the untuned
`core_grid` call — and against production today the shipped arm is bit-exact, so there is no parity
trade here at all. I predicted the arm would move plDDT and it does, but only on the
`in0_block_w`=1 variant I have now rejected on performance grounds anyway.

## Merge recommendation

Nothing has been merged and nothing should be without Moritz's explicit OK. Everything sits on
`wk/protenix-trunk--p3-l1-output`, pushed, branched from `wk/protenix-trunk--p3-narrow-write`.

**One proposal, both changes, recommend merging.** `_PAIR_PROJ_L1_OUT = True` with
`_PAIR_PROJ_L1_BW = 16` tracking `_PAIR_PROJ_BW`, plus `_PAIR_BIAS_L1_NORM = True`. **-561.8 ms/fold
on the block wall against a bracketing control**, `torch.equal` True against production today at
the fold's own shape on both the projection and the whole trimul chain, and a live 298 aa fold
returning plDDT 0.859489 — the same to six decimals as production today. **Not release-gated: it
changes no arithmetic**, only where three matmuls and one `layer_norm` put their results. Both
levers sit behind a fit test that returns DRAM when the tensor does not fit, and behind a
try/except that falls back to today's path and remembers the refusal per operand class, so a larger
target keeps today's behaviour exactly.

What a reviewer should look at: the output term added to `_pair_proj_program_config`'s L1 budget
(it was absent because the helper always wrote to DRAM), and the fact that the static budget
over-counts — it says two concurrent 48.82 MB L1 outputs cannot fit and the allocator says they can,
with 708 096 B of 1 461 760 still free per bank. The allocation is the real test and the fallback is
what makes relying on it safe.

**Version sensitivity, which my brief asked me to state explicitly.** qb1 is ttnn **0.67.4** and the
shipped wheel is **0.68.0**, so qb1 is one release behind production. **A 0.68.0 re-check is
required before this merges**, and the reason is not boilerplate: L1-output legality provably moved
*within this pass* — `ttnn.linear(core_grid=)` with an L1 output runs at 110 cores and **throws** at
130, which is a circular-buffer sizing decision inside ttnn's matmul, exactly the kind of thing a
minor release moves. The parity claim needs re-taking there too, since it rests on the L1 and DRAM
writers packing identically.

## Corrections to the inherited record

1. **The 298.7 ms/fold was a projection and it was priced on the wrong arm. At the `in0_block_w`
   production already ships, the L1 output delivers 430.1 ms/fold — 144 % of the projected figure;
   at the bit-exact `in0_block_w`=1 that X2's pricing assumed, it delivers 33-106 ms/fold, 11-36 %
   of it.** The projection was a per-call `proj + add` delta times 3144 projections, and production
   runs **four projections per two residual adds** inside `TriangleMultiplication`, so the add's
   saving — 132.6 ms/fold, and the only part that is the same on both arms — was counted twice. The
   projections' own contribution is worth 288.7 ms/fold at `in0_block_w`=8 and **minus 115 ms/fold**
   at `in0_block_w`=1.
2. **The L1 output is orthogonal to the parity knob and the org had them entangled.** At a fixed
   `in0_block_w` the L1 output is `torch.equal` against the DRAM output of the identical config,
   max abs 0.0, plDDT identical to six decimals in a live fold. There is no bit-exact-versus-gated
   choice to make about the destination; the only parity decision in this class is the one main
   already took when it set `_PAIR_PROJ_BW = 16`.
3. **The brief's "1.53x bit-exact, 1.49x release-gated" are ratios over two different
   denominators.** 1.53x is against the untuned `core_grid`-DRAM baseline (my re-measurement:
   1.647x); 1.49x is against production today (mine: 1.473x). Only the second kind converts into a
   ms/fold against production.
4. **An L1 output is not available to `ttnn.linear(core_grid=)` at the production grid.** X2
   measured `l1_cg` running at 11x10; at the production 13x10 it **throws** ("statically allocated
   circular buffers ... clash with L1 buffers"). The tuned config's smaller circular buffers are
   what make an L1 output legal here at all, so "put the output in L1" is not a standalone knob.
5. **Both of a trimul's 48.82 MB projections fit in L1 at once, and putting only one there is a
   loss.** I predicted the second would be refused on a 1587.6 kB/bank budget against 1427.5
   available; the allocator reports **708 096 B of 1 461 760 still free per bank** with both
   resident. `p_out` alone in L1 is 0.91x against production today on the trimul chain; both is
   1.078x at `in0_block_w`=1 and 1.373x at 8. A static program-config budget over-counts here.
6. **`tenstorrent.py:2088` is bound by its SOURCE, not by its write, and that is where its
   225.8 ms/fold goes.** It reads 48.82 MB to write 6.10 MB. An L1 output alone is worth 24.5
   ms/fold; an L1 `layer_norm` source is worth 208 ms/fold in the probe, and the two together take
   the site's in-fold wall from **252.1 to 101.4 ms/fold** and its region wall down 226.4 ms/fold.
   The 225.8 ms/fold this org carried for the site (208.4 after X2's `_NARROW_PROJ_BW = 1`) is now
   **101.4 ms/fold**.
7. **The `in0_block_w` sign flip is confirmed by direct measurement, not inherited.** At the 2088
   site with a DRAM source `in0_block_w`=8 is 1.302x; with an L1 source it is **0.84x**, a loss
   against `in0_block_w`=1. The knob pays for a DRAM read and costs when the read is already free.
   Note this is the opposite of the pair-track projection's verdict in correction 1, and the reason
   is the same one: the knob is worth exactly what the DRAM read it removes is worth.
8. **The op site wall reproduces to under 1 % across processes and sessions; the block wall does
   not always.** Three independent baseline folds agree to 0.05-0.94 % of the baseline figure on every op
   class and to 0.32 % of the baseline figure on the block wall — but one earlier block-instrumented baseline
   came in 778 ms lower than another. Bracket the arms with a baseline first and last, quote the op
   walls, and treat a block-wall delta under ~800 ms taken across sweeps as unresolved.
9. **`COMPUTE_GRID_MAIN` was read after the device was open in every probe here** — 13x10, recorded
   in the JSON of every run — so none of these figures is X2's 11x10 trap. That is also why the
   `l1_cg` refusal in correction 4 shows up at all.
10. **The `_transpose_memory_config` interaction STATUS.md asks every L1-raising leg to report: it
    still takes the L1 branch and nothing broke.** With `gate_and_project`'s 48.82 MB output now in
    L1, the ending `TriangleAttention`'s `ttnn.permute(x, (1,0,2))` still tests L1 (measured
    `get_max_worker_l1_unreserved_size` = 1 532 448 B, so 2.5 x 48.82 MB = 122 MB against 199 MB
    across 130 cores) and runs. `_L1_OUT_REFUSED` is empty after every one of the eleven folds in
    this pass and no arm threw. **P6 CONFIRMED.**

## Closed, and not reopened here

The **fused wider output is closed and I did not reopen it**: the three output projections share no
input and run sequentially, so it is structurally impossible, and the fusion that does exist is
1.370x bit-exact but nets **0.708x** once the un-fuse's 984.4 us is paid against 388.9 us saved.
Staging through L1 as a general lever is likewise closed — it fixes a bad *source*, which is
precisely why it worked at `tenstorrent.py:2088` (source-bound) and why the pair-track projection's
L1 *output* pays only through its consumer. And I did not chase the fold wall for a sub-150 ms
claim.

## Instruments and artefacts

All on `wk/protenix-trunk--p3-l1-output`, all run on qb1 card 1 with `TT_VISIBLE_DEVICES=1`.

| file | what it produced |
|---|---|
| `perf/p3l1/PREDICTIONS.md` | the predictions, committed in `592e2ffa` before the device was opened |
| `perf/ledger_298/roofs_card.py` (reused) | `roofs_c1.json` — the four card roofs and the machine balance |
| `perf/p3l1/p3_l1_probe.py` | `probe_c1.json` — the `pair`, `trimul` and `site2` arms at the production 13x10 grid |
| `perf/p3l1/fold_ab.py`, `run_ab.sh` | `ab_{base,l1out,bias,both}_r1.json` — four arms, block wall, one interleaved sweep |
| `perf/p3l1/run_post.sh`, `run_post2.sh` | `ops_{base,both,base2,both2,rg,base3}.json` — the per-class op site walls, baseline first and last |
| `perf/p3l1/guard_check.py` | proof the L1 branch fires at the fold's own shapes, and `get_max_worker_l1_unreserved_size` |
| `perf/p3l1/parity_c1.py`, `parity2.py` | `parity_c1.json`, `parity2.json` — `torch.equal` through the production helpers |
| `tt_bio/tenstorrent.py` | the production change: `_PAIR_PROJ_L1_OUT`, `_PAIR_PROJ_L1_BW`, `_PAIR_BIAS_L1_NORM`, the output term in the L1 budget, `_l1_memory_config_if_it_fits` |

**Generalisation, recorded and not chased** (charter §1): two general results, one line each. A
matmul whose result is consumed on device should write it to L1 whenever it fits and its program
config leaves room — the win is in the consumer, not the producer, and it is free of parity because
the destination does not touch the accumulation order. And a narrow-output projection that reads a
whole activation tensor to write one tile of width is bound by its source, so an L1-resident
producer beats anything done to its own output. Two more sites in this repo have the second shape —
`PairWeightedAveraging`'s per-head z->bias (240 calls/fold, 123.5 ms, and its `layer_norm` is
computed **once** for eight consumers) and the template z projection — and neither is touched here.

## What is not done

**The PWA site at `tenstorrent.py:2999` is the obvious next 60-100 ms/fold** and is untouched: it is
the same source-bound shape as 2088, its wall is 123.5 ms/fold over 240 calls, and its `layer_norm`
is already shared across eight head projections so one L1 tensor would serve all of them. **The
0.68.0 re-check has not been done** and the merge recommendation is conditional on it. **The fold
wall was never resolved** and this leg does not claim one.
