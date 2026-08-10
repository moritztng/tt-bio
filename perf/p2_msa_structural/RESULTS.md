# P5 — `protenix-trunk--p2-msa-structural`, Phase 2: C1, C3, C4, C5

Protenix-v2, trunk only, 298 aa. qb2 card 1 (board 007 chip 1), ttnn 0.68.0, so **ratios, not
campaign absolutes**. T5's numbers are pc card 0 — a different card and a different chip — so every
baseline I A/B against is re-measured here this pass and said to be.

**Probes only. No production change**: everything lives under `perf/p2_msa_structural/` on
`wk/protenix-trunk--p2-msa-structural` and nothing under `tt_bio/` is touched.

**Site line numbers moved.** T5 worked a branch where the OPM tail sat at 3034-3037/3043 and the
producing matmul at 3012. On `origin/main` (`bbb5d85b`) the same chain is
`to_layout:3059`, `reshape:3060`, `to_layout:3061`, `permute:3062`, consumer `linear:3068`,
producer `matmul:3037`. Same ops, +25 lines. I use this branch's numbers below and give T5's in
brackets the first time.

---

## Predictions (before measuring)

Committed and pushed before the device was opened
(`perf/p2_msa_structural/PREDICTIONS.md`, commit on this branch). Every one names the number, the
unit, and what would count as having been wrong.

**Roofs, qb2 card 1.** DRAM read at 128 MB **380 ± 40 GB/s**, DRAM write **255 ± 35 GB/s**,
DRAM→DRAM copy **400 ± 50 GB/s**. Wrong if any lands outside its band; a write roof under 200 or
over 300 GB/s would mean I am measuring something other than the writer.

**C1 — the OPM layout chain.**

- **P1.1** The (i,c)x(d,j) → (i,j,cd) reindexing **cannot** be emitted by any matmul that keeps
  a on the M side and b on the N side, because M carries {i,c} and N carries {d,j} while the
  consumer needs {i,j} together. So the *interesting* arm is not "make the matmul emit it" but
  "make the matmul emit a layout the residual relayout is cheap from". I predict the b-side
  reorder arm (build `b_flat` as `(j*D+d, S)` instead of `(d*J+j, S)`) leaves the **matmul's own
  time unchanged within ±5 %** — identical operand shapes, identical K, only the tiny 0.68 MB b
  operand is permuted differently. Wrong if it moves more than 5 %.
- **P1.2** The batched-over-i arm (`a` as `(298, 32, 35)` @ `(35, 9536)`, output `(298, 32, 9536)`)
  is **at least 3x slower** than the flat `(9536, 35) @ (35, 9536)`, because M becomes one tile row
  per batch and ttnn does not distribute the batch dimension. Wrong if it lands within 1.5x.
- **P1.3** The chain is 4 passes over a 182-195 MB intermediate. A **single**-op relayout would move
  182 MB read + 195 MB write = 377 MB, which at a 400 GB/s copy roof is **~940 us/call = 37.6
  ms/fold at x40 OPM calls**. So if any single op can do the reindex, the prize is
  **179.2 - 37.6 ≈ 140 ms/fold**, not 179.2. Wrong if the measured 4-op chain on this card is not
  within 25 % of 4x the single-pass figure.
- **P1.4** Kill test as the brief states it: if the matmul's own time rises by more than **179
  ms/fold (4479 us/call at x40)** under any arm, the relayout is intrinsic and has only moved. I
  predict **no arm trips this** — the b-side reorder is free and the batched arm's penalty, while
  large in ratio, is bounded by the matmul's own 1.3 ms/call, i.e. it cannot cost 4.5 ms/call.

**C3 — the PWA per-head weight slices.**

- **P3.1** us/call is **flat within ±30 %** from a 16 kB slice to a 1.6 MB slice off the same
  tensor. Wrong if it rises roughly linearly (a 100x size step producing more than a 3x time step
  kills the fixed-cost reading and with it the 51.5 ms/fold).
- **P3.2** The fixed cost will be **~40-70 us**, which is **6-11x T4's measured 6.40 us launch
  floor**. So "essentially all of the time is dispatch" does **not** follow, and I name the third
  thing in advance: **sub-tile slicing**. `self.z_weight[:, i:i+1]` takes one logical column out of
  a 32-wide tile face, so the reader must fetch whole tiles and the packer must rebuild a padded
  face. Prediction: a **tile-aligned 32-column** slice of the same output byte count is **at least
  3x faster** than the 1-column slice. Wrong if aligned and unaligned come out within 1.3x — then
  the cost is host-side op overhead, not the tile face.

**C4 — the transition launch floor.**

- **P4.1** `linear@2071` [T5's 2046] and `layer_norm@2063` [2038] are **flat within ±25 % from 1x to
  2x rows** and rise clearly at 4x. Wrong if 2x rows costs 1.8-2.2x the time — that is bandwidth
  and the candidate dies.
- **P4.2** `minimal_matmul@1701` costs **within ±15 % of `@1695`** despite writing a quarter of the
  bytes, on this card as on pc. Wrong if the ratio is near 4x (i.e. byte-driven).

**C5 — the loop-invariant template z projection.**

- **P5.1** Hoisting `self._lin(zn, "…linear_no_bias_z.weight")` out of `for t in range(nt)` removes
  **30 of 40 calls/fold**. On pc that was 14.4 ms/fold. On qb2 card 1 I predict **10-20 ms/fold**,
  and a **`trunk_template` stage wall drop of 1.4-2.8 %** (the stage is ~72 ms/call). Wrong if the
  stage wall does not move by at least 1.0 %, which would mean the call was already overlapped with
  something.
- **P5.2** `torch.equal` on the `_template` stage output, baseline vs hoisted, returns **True**.
  Wrong if it returns False — the operand and weight are both loop-invariant, so any difference is
  a bug in my hoist, not a numerical property.
- **P5.3** C5 and `protenix-trunk--p2-matmul-ceiling`'s program-config fix to the same site are
  **not additive**. If the config fix makes the call k us and the hoist removes 30 of 40 calls,
  the combined figure is `10 x k x 40/1000` ms/fold of remaining cost, not `179 - C5 - C2`.

---

## Roofs, measured on this card

**I did not inherit a single roof.** Every number below was streamed on **qb2 card 1 (board 007 chip
1)** in this pass by `perf/p2_msa_structural/p5_floor_probes.py`, one process, device synchronised on
both sides of every timed region. T5's pc card 0 figures and W1's qb1 figures appear only as the
numbers this leg must not use.

| roof | qb2 card 1, this pass | method |
|---|---:|---|
| DRAM read, ladder to 128 MB | **393.3 GB/s** | DRAM→L1 clone |
| DRAM write | **272.5 GB/s** | L1→DRAM clone |
| DRAM→DRAM copy | **399.2 GB/s** | DRAM→DRAM clone |
| square compute, bf16 HiFi4, `fp32_dest_acc` + `packer_l1_acc`, DRAM out | **102.33 TFLOP/s** (4096³) | 94.52 at 6144³ |
| per-op launch floor | **4.33 us** (1-tile clone), 8.85 us (1-tile relu) | flat to 32 tiles: 5.21 / 9.26 us |
| **machine balance** | **260.2 FLOP/byte** | 102.33e12 / 393.3e9, my own pair |

The read ladder is 313.8 → 365.9 → 377.4 → 392.7 → 393.3 GB/s at 16/32/64/96/128 MB, so it is
saturated by 96 MB, unlike pc's.

**`dev.compute_with_storage_grid_size()` on this chip returns 11x10, identical to `CORE_GRID_MAIN`.**
STATUS's open item "qb2 has never been swept at its full compute grid, so 22.3 % is an upper bound on
the card gap" is answered for chip 1: 11x10 **is** its full grid, so there is no wider sweep to run
and 22.3 % is not inflated for that reason. qb1's 13x10 is a different chip.

The K-corrected rates at my own sites' (K, output width), batched at the fold's own blocking
`298 x (320 x K) @ (K, nt*32)`, DRAM output, `core_grid=11x10`:

| site | K | nt | us/call | TFLOP/s | output write GB/s |
|---|---:|---:|---:|---:|---:|
| OPM consumer `linear:3068` | 1024 | 8 | 1719.5 | 29.08 | 28.4 |
| template pair track | 64 | 2 | 94.4 | 8.27 | 129.3 |
| PWA z→bias `linear:2832` | 256 | 1 | 451.2 | 3.46 | 13.5 |

All three are far below the 102.33 TFLOP/s square roof and none of them should ever be sized against
it — at nt=1 this card reaches 3.4 % of its own square figure.

---

## Experiments and verdicts

Conversions (charter §4.9): both my stages are called **once per recycling cycle**, so a stage figure
converts **x10**. Per-op counts I use, each measured or counted rather than assumed: **x40** for the
OPM chain (4 MSA blocks x 10 cycles), **x240** per PWA slice site (8 heads x 3 PWA-carrying MSA
blocks x 10 cycles), **x40** for the template z projection (4 templates x 10 cycles), and the
charter's **x80** for a template *block*-level op (2 blocks x 4 templates x 10 cycles). x480 / x484 /
x524 are `pf_stack` constants and do not apply to anything in this doc.

### C1 — the OuterProductMean layout chain

Per-op, at the fold's own shapes (I=J=298, C=D=32, S=35, c_z=256), `perf/p2_msa_structural/p5_opm_chain.py`:

| op (this branch / T5's line) | us/call | ms/fold at x40 | achieved | denominator |
|---|---:|---:|---|---|
| `matmul:3037` [3012] | 743.8 | 29.8 | 15.6 TFLOP/s | 15.2 % of the square compute roof |
| **`to_layout:3059` [3034] TILE→RM** | **35627.3** | **1425.1** | **10.2 GB/s** | **2.6 % of the 399.2 GB/s copy roof** |
| `reshape:3060` [3035] | 1672.4 | 66.9 | 217.6 GB/s | 54.5 % of the copy roof |
| `to_layout:3061` [3036] RM→TILE | 1266.0 | 50.6 | 297.8 GB/s | 74.6 % of the copy roof |
| `permute:3062` [3037] | 1028.1 | 41.1 | 380.2 GB/s | 95.2 % of the copy roof — at the roof |
| consumer `linear:3068` [3043] | 1719.5 | 68.8 | 29.08 TFLOP/s | 110 of 110 cores, ladder below |
| **the 4-op chain** | **39593.8** | **1583.8** | | |

**The candidate as T5 framed it survives its own kill test, and then is superseded by what the kill
test turned up.**

**Kill test, as the brief states it: CONFIRMED (the relayout is not intrinsic).** Arm A1 rebuilds
`b_flat` as `(j*D+d, S)` instead of `(d*J+j, S)`, so the matmul's output tile `(i, j)` *is* the
`(c, d)` block. Operand shapes and K are unchanged and **the matmul's own time moves by -0.7 %**
(738.7 vs 743.8 us) — prediction P1.1 confirmed inside its ±5 % margin. Arm A2 (batched over i,
`(298, 32, 35) @ (35, 9536)`) costs 1581.5 us, **2.13x** the flat form, **+33.5 ms/fold**. The brief's
threshold was 179 ms/fold; the worst arm costs 33.5, so **no arm makes the relayout intrinsic**.
Prediction P1.2 said "at least 3x" for A2 and **was wrong** — 2.13x. Recorded as wrong, not reworded.

**A1 is nonetheless KILLED on the residual.** Its remaining relayout to `(298, 298, 1024)` costs
`reshape4d 16255.2 + permute(0,2,1,3) 4005.0 + reshape 16497.1 = 36757.3 us/call`, which is 92.8 % of
the 39593.8 us chain it replaces. The reindex moved; it did not shrink.

**What the pass actually found: 90.0 % of the chain is one op, and its cost is set by the last-dim
width, not by bytes.** Same 182 MB, two shapes, two processes, reproduced:

| tensor | to_layout TILE→RM | DRAM→DRAM clone of the same bytes |
|---|---:|---:|
| `(9536, 9536)` | **35648-35663 us** (three independent runs) | 913-917 us |
| `(298, 1024, 298)` | **1229.5 us** | 974.9 us |
| `(298, 32, 9536)` | **35634.7 us** | — |

**29.0x from the last dimension alone.** Rank is not the variable: reshaping to rank 3 while keeping
the 9536-wide last dim (4.4 us, a free tile-grid regrouping) leaves the untilize at 35634.7 us, so the
follow-up fix I built is **KILLED** — and its output failed `torch.equal` against the production chain
(max abs diff 0.814), so it is not a valid transformation as written either.

**Mechanism hypothesis, falsifiable.** Untilize has to hold one full row of tiles in a circular
buffer before the packer can emit row-major rows. At 298 tile-columns that row is 596 kB, which does
not fit, and the op falls off its multi-core path onto a serialised one: 10.2 GB/s back-solves to
~4 core-equivalents of 110 against the 296 GB/s the identical bytes reach at 10 tile-columns.
**Kill it by** sweeping the last-dim tile count from 8 to 298 at fixed total bytes and looking for a
cliff: a smooth ramp means the CB is not the mechanism and the cost is per-transaction NOC issue on
the strided packer write instead.

**Cross-check against the live stage, and it is the strongest evidence in this doc.** `trunk_msa` on
this card measures **3492.9 ms/fold** (349.3 ms/stage, x10; raw 349.29 / 349.30 / 348.67) against
T5's **1979.3 ms/fold** on pc card 0. The gap is **+1513.6 ms/fold**. The isolated defect predicts
1425.1 ms/fold here against pc's ~38 ms/fold for the same op, i.e. **+1387 ms/fold, 91.7 % of the
measured stage gap**, from one op. A microbenchmark and a live stage wall taken in different
processes agreeing to 8 % is what convicts it.

**Overlap: additive.** The OPM path alone (matmul + chain + consumer, 42057 us/call) x 4 blocks =
168.2 ms of the 349.3 ms stage wall, **48.2 % of the stage**, which is only possible if the device
finishes one op before the next begins. Nothing in this stage hides communication behind a
neighbour's compute, so every ms removed is a ms of wall clock.

**Core engagement.** `linear:3068` core ladder, us/call: 4x4 2682.2, 8x4 3483.0, 11x8 1883.7,
**11x10 1723.4** — it improves to the full grid, so **110 of 110** engaged. The four relayout ops
expose no grid knob; `permute:3062` at 95.2 % of the measured copy roof cannot be core-starved, and
`to_layout:3059` at 2.6 % of it plainly is.

**ms/fold at stake, C1: 1425.1 on this card** if the wide-last-dim untilize can be avoided, against
T5's 179.2 on pc. **This is a qb2 figure and the org ranks on qb1 absolutes — it must be re-measured
on qb1 (0.67.4) before it enters a Phase-3 ranking**, and the pc/qb2 disagreement means one of the
two cards is doing something the other is not.

### C3 — the PairWeightedAveraging per-head weight slices

**Size sweep, aligned starts, from one 4096x4096 tensor: CONFIRMED flat.** 16 kB **7.35 us**, 160 kB
**8.38 us**, 1.6 MB **11.85 us** — a 100x size step buys a 1.61x time step. Prediction P3.1 confirmed
inside ±30 % up to 160 kB; the 1.6 MB point runs at 276.5 GB/s, 69.3 % of the copy roof, which is
where bandwidth finally starts to show. The fixed cost is **~7.2 us against my measured 4.33 us
1-tile clone floor**, so an *aligned* slice really is at this card's launch floor.

**And T4's 6.40 us floor does not explain the production slices, exactly as the brief suspected. The
third thing is the slice's START OFFSET.** At identical output bytes:

| slice | us/call | output |
|---|---:|---|
| `[0:256, 0:32]` aligned start | **7.49** | 16.4 kB |
| `[0:256, 32:64]` aligned start | **7.39** | 16.4 kB |
| `[0:256, 1:33]` **unaligned start** | **202.42** | 16.4 kB |
| `[0:256, 0:1]` aligned start, sub-tile width | **7.22** | 0.5 kB |
| `[0:256, 1:2]` **unaligned start**, sub-tile width | **202.47** | 0.5 kB |

**27.0x, decided entirely by whether the start is a multiple of 32.** Width does not matter and
output size does not matter. **Mechanism, and it is measured rather than argued:** 202.4 us over the
**source** tensor's 33.5 MB is **331.6 GB/s read+write, 83.1 % of the measured 399.2 GB/s copy
roof** — an unaligned slice pays a full read-and-rewrite pass over the *whole source*, at the copy
roof, because the reader cannot start on a tile boundary and the packer has to rebuild every tile
face. **Kill it by** slicing an unaligned start out of sources of 4 / 33 / 130 MB at fixed output
bytes: if us/call tracks the source size the pass is the mechanism, and if it is flat the cost is
host-side dispatch after all.

**On the production shapes**, `tenstorrent.py:2834/2849/2866/2874` [T5's 2809/2824/2841/2849]:
`z_weight[:, i:i+1]` **35.15 us**, `m_weight[:, i*8:(i+1)*8]` **34.64**, `g_weight[...]` **34.71**,
`o_weight[i*8:(i+1)*8, :]` **16.91**. head_dim is 8, so **7 of every 8 heads slice at an unaligned
start**; head 0 is aligned and lands at the 7.4 us floor. Weighted over the 240 calls/fold per site
(210 unaligned + 30 aligned), the four sites cost **26.4 ms/fold** on this card against T5's 51.5 on
pc. **CONFIRMED: the 26.4 ms/fold is recoverable**, because the inputs are constant parameter
tensors and `_KeyedWeights._w_tt` already caches whole weights at construction — and the result is
bit-identical by construction, which C3 can assert because the cached tensor is the same tensor.

### C4 — the sub-60 us launch floor on the transitions

**`minimal_matmul` output-width sweep: CONFIRMED, decisively.** Same input `(320, 320, 64)`, weight
`(64, N)`:

| N | us/call | output MB | write GB/s | verdict |
|---:|---:|---:|---:|---|
| 32 | 287.54 | 6.55 | 22.8 | |
| 64 | 287.42 | 13.11 | 45.6 | |
| 128 | 289.78 | 26.21 | 90.5 | |
| 192 | 289.42 | 39.32 | 135.9 | **6x the output bytes for +0.7 % time** |
| 384 | 448.89 | 78.64 | 175.2 | the writer finally binds, 64.3 % of the write roof |

Prediction P4.2 said `@1726` [T5's 1701] within ±15 % of `@1720` [1695] — measured **0.7 %**.
CONFIRMED. Below ~40 MB of output the time is independent of output size: the input read is 45.3 GB/s
(11.5 % of the read roof) and the write 135.9 GB/s (49.9 % of the write roof), so **neither roof
binds**. **Mechanism hypothesis:** the schedule is set by the input's tile count (6400 tiles at 44.8
ns/tile), and output columns ride along inside the same per-input-tile NOC transaction until the
packer saturates. **Kill it by** doubling the input at fixed N: if us/call doubles the input tile
count binds, and if it is flat the cost is per-launch after all.

**Row-count sweep on the transitions: PARTIALLY KILLED, and my prediction was wrong.** L1-resident
`(35*mult, 320, 128)`, weight `(128, 512)`:

| mult | layer_norm us | linear (plain) us | linear `activation="silu"` us |
|---:|---:|---:|---:|
| 1x (2.87 MB) | 26.38 | 37.29 | 153.22 |
| 2x (5.73 MB) | 38.60 | 60.35 | 262.74 |
| 4x (11.47 MB) | 63.76 | 117.96 | 493.80 |

P4.1 predicted flat within ±25 % from 1x to 2x; measured **1.46x** (layer_norm) and **1.62x**
(linear). **Wrong, and the "these are per-launch-floor ops" reading dies with it.** A straight fit
gives a fixed term of **13.92 us** for `layer_norm` (52.8 % of the 1x call) and **10.40 us** for the
linear (27.9 % of the 1x call), i.e. 2.4-3.2x my measured 4.33 us clone floor — real, but not the
whole cost. **C4's prize is the fixed term only, not the whole ~47 ms/fold**: about half of the
`layer_norm` line and about a quarter of the `linear` line, and only if several calls can be batched
into one.

**Core engagement, measured:** the 1x transition linear ladder is 1x1 1233.65, 4x4 108.51, 8x4 68.57,
**11x8 38.76**, 11x10 38.87 us → **88 of 110 cores**, so this site is not occupancy-starved and the
`nt=1` collapse T5 found elsewhere does not apply to it.

### C5 — hoist the loop-invariant template z projection

`Trunk._template` (`tt_bio/protenix.py:2021-2026`) evaluates
`self._lin(zn, "template_embedder.linear_no_bias_z.weight")` inside `for t in range(nt)` on a
loop-invariant operand and a loop-invariant weight. The probe's hoisted variant is a local function;
production is untouched.

| | ms/stage | ms/fold (x10) |
|---|---:|---:|
| `trunk_template` baseline | 82.008 (raw 82.01/81.98/81.95/82.02/82.04) | 820.1 |
| `trunk_template` hoisted | 80.530 (raw 80.49/80.58/80.53/80.58/80.49) | 805.3 |
| **saved** | **1.479** | **14.8** |

**CONFIRMED.** 1.80 % of the stage wall, removing 30 of 40 calls/fold. Prediction P5.1 said 10-20
ms/fold and 1.4-2.8 % of the stage — confirmed on both. The five raw walls span 0.11 % of the median
on each arm, so the 1.8 % separation is far outside the noise.

**Core engagement, measured on the call itself:** 4x4 482.6, **8x4 448.5**, 11x8 515.1, 11x10 485.6
us/call. Smallest grid within 5 % of the best is 8x4, so **32 of 110 cores**, reproducing T5's pc
measurement of 32 of 110 on a different card. 78 cores receive nothing.

**C5 and `protenix-trunk--p2-matmul-ceiling` shrink each other and I am not adding them.** The site
is `linear@protenix.py:306`, whose program-config half that leg owns and I did not touch. The
combined figure is not `C5 + C2`: after the hoist the site runs **10 calls/fold instead of 40**, so
whatever per-call time k that leg's config reaches, the remaining cost is `10 x k / 1000` ms/fold and
the hoist's own credit falls to `30 x k / 1000`. At today's k = 485.6 us the site is 19.4 ms/fold
total, the hoist takes 14.6 of it (measured 14.8) and the config can only compete for the 4.9 that is
left. **If that leg halves k first, C5 is worth 7.3 ms/fold, not 14.8.** Whichever lands first takes
the larger share.

---

## ms/fold at stake, after this pass

Ranked on this card. **Every figure is qb2 card 1 at ttnn 0.68.0 and is a ratio, not a campaign
absolute** — the org ranks on qb1 and each of these has to be re-measured there before it drives a
Phase-3 decision.

| lever | T5's estimate (pc) | measured here (qb2 card 1) | verdict |
|---|---:|---:|---|
| **C1** — the wide-last-dim untilize inside the OPM chain | 179.2 ms/fold | **1425.1 ms/fold** for the one op, 1583.8 for the whole chain | **CONFIRMED and 8.0x larger than the candidate**; the reindex itself is not intrinsic (matmul -0.7 %) but no arm tried here shrinks the residual |
| **C3** — cache the PWA per-head weight slices | 51.5 ms/fold | **26.4 ms/fold** | **CONFIRMED**, mechanism corrected to unaligned-start slicing |
| **C4** — the transition fixed term | ~47 ms/fold | **fixed term only**: 13.92 us of 26.38 (`layer_norm`), 10.40 of 37.29 (`linear`) | **PARTIALLY KILLED** — not launch-floor bound; `minimal_matmul` half CONFIRMED at 0.7 % over 6x the output |
| **C5** — hoist the template z projection | 14.4 ms/fold | **14.8 ms/fold** | **CONFIRMED**, `torch.equal` True |

C1 is no longer a 179 ms/fold layout-tidying candidate; on this card it is 40.8 % of the whole
`trunk_msa` stage and it is a ttnn op-selection defect rather than a roofline gap.

---

## Parity

- **C5: `torch.equal` on the `trunk_template` stage output, baseline vs hoisted, returned `True`,
  max absolute difference 0.0.** Measured, not argued. The hoist removes redundant evaluations of a
  function of loop-invariant arguments and nothing else.
- **C3** is bit-identical by construction in the strict sense: caching the eight per-head views at
  construction hands the same device tensor to every call instead of re-deriving it, so there is no
  arithmetic to change. Not separately measured, because there is no second value to compare.
- **C1's A1 arm is not parity-neutral as written.** The rank-3-first variant returned
  `torch.equal` **False** against the production chain, max abs diff **0.814**, so it is recorded as
  killed on correctness as well as on cost. Any Phase-3 attempt on this chain owes a `torch.equal`
  before it owes a number.
- **C4** was measured, not changed; nothing here proposes an arithmetic change.
- Protenix-v2 ships in v0.6.1 and parity is sacred (charter §4.7). **No production change was made in
  this pass** — probes only, all under `perf/p2_msa_structural/`, nothing under `tt_bio/` touched.

---

## Corrections to the inherited record

1. **T5's shared defect #3 is wrong.** "No `activation=` kwarg anywhere in either stage" holds for
   `_lin` but not for `Transition.__call__`, which passes `activation="silu"` at
   `tenstorrent.py:2074` and runs in both of my stages (MSA `transition_m`, and `transition_z` inside
   every `PairformerLayer`). Measured on this card at the MSA transition's own shape, the fused form
   costs **153.22 us against 37.29 us for the identical matmul without it — 4.11x**. T3's premium
   applies here too and T5's stages were wrongly excluded from it.
2. **Every site line number in T5's ledger is 25 low against `origin/main` (`bbb5d85b`).** The OPM
   chain is 3059/3060/3061/3062 with the producer at 3037 and the consumer at 3068; the PWA slices
   are 2834/2849/2866/2874; `minimal_matmul` is 1720/1726. T5's `@1695` / `@1701` are also
   **mislabelled**: they are the triangle-attention qkv and gate projections, not "trimul p/g" and
   "trimul out".
3. **`trunk_msa` is 3492.9 ms/fold on qb2 card 1, 1.76x T5's 1979.3 on pc card 0.** That is not a
   22-25 % card gap and it is not a compute gap. 91.7 % of it is one op. `trunk_template` is 820.1
   ms/fold here against T5's 719.0, a 14.1 % spread that *is* card-shaped.
4. **STATUS's open item "qb2 has never been swept at its full compute grid" is closed for chip 1:**
   `compute_with_storage_grid_size()` returns **11x10**, which is `CORE_GRID_MAIN`. There is no
   wider grid on this chip and 22.3 % is not an upper bound for that reason.
5. **qb2 card 1's machine balance is 260.2 FLOP/byte** (102.33 TFLOP/s / 393.3 GB/s, my own pair) at
   the square-default roof, which is the same under-report class STATUS flags for the retracted 253.
   It is not pc's 341.8 and it is not qb1's 337.9.
6. **T5's C3 mechanism — "essentially all of the time is kernel dispatch" — is replaced.** An
   *aligned* slice is at this card's launch floor (7.2-7.5 us against a 4.33 us 1-tile clone). The
   production slices cost 17-35 us because their start offset is not a multiple of 32, and the
   unaligned path is a full read-and-rewrite of the **source** tensor at 83.1 % of the copy roof.
   Same fix, different and much sharper reason, and it generalises to any `ttnn` slice in this
   codebase — recorded, not chased (charter §1).
