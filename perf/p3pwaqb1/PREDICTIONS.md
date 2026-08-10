# p3-pwa-qb1 — predictions, committed before the device was opened

qb1 (`tt-quietbox`), **card 2**, `TT_VISIBLE_DEVICES=2`, ttnn **0.67.4** (`/home/ttuser/tt-bio-dev/env`),
Blackhole p150a, grid expected 13x10. Branch `wk/protenix-trunk--p3-pwa-qb1` off `origin/main` at
`cc39a867d`, which Moritz merged at 08:35 today. Baseline = that commit with `_PWA_L1_NORM` and
`_TEMPLATE_L1_NORM` **off**, which is the DRAM behaviour, so the A/B is a flag toggle.

Co-tenant on this host: `perfwar-qb1-rebaseline-and-land`, three workers, `/dev/tenstorrent/0` held by
pid 930576 at brief time. Load average will be recorded with every measurement.

## What I predict, and what makes each wrong

**P1 — the merged paths RUN on a 13x10 grid.** `COMPUTE_GRID_MAIN` read after device open is
**(13, 10)**, `_trimul_chunk_size(298, 128)` is **64** (32 on qb2's 11x10), `_L1_OUT_REFUSED` is
**empty** after every live 298 aa fold in this pass, and no trimul circular-buffer throw appears with
`_TEMPLATE_L1_NORM` on. Mechanism for the confidence: the merged `_template` takes all nt projections
above the block loop and deallocates `zn` before the first block, so nothing 48.82 MB is resident when
a trimul places its CBs; and an L1-interleaved tensor spreads over **130** banks here against 110, so
its per-bank cost is *lower* on this grid, not higher. **Wrong if** `_L1_OUT_REFUSED` is non-empty after
any fold, or any arm throws.

**P2 — PWA, and I predict the qb1 absolute is LARGER than X10's qb2 ratio.** X7's three qb1 baselines
put the site's own op wall (`_narrow_proj_linear`, `[298,320,256] @ [256,1]`, **240 calls**) at
122.916 / 123.336 / 123.652 ms/fold, mean **123.30**, and `body:PairWeightedAveraging` over **30
regions** at 415.612 / 424.087 / 429.667, mean **423.12**. qb2's control op wall was 102.33 and it lost
71.1 % of it. The saving per call is a removed 48.82 MB DRAM read, and this card's read roof is within
2 % of qb2's, so the absolute saving per call should be the same while qb1's baseline call is slower.
I predict the **region wall** delta is **-80 to -110 ms/fold** and the **op wall** delta **-85 to -95**,
i.e. 5-45 % more than X10's -75.91. **Wrong if** the region delta is outside -80 to -110, or the op
wall's control is outside 120-127 ms/fold.

**P3 — the template z projection, first qb1 figure ever taken at this site.** It is absent from every
one of X7's `ops_*.json` because `protenix.py` imports `_narrow_proj_linear` by name; I patch both
namespaces, so a `w=[256, 64]` row appears at **40 calls** (4 templates x 10 cycles). Scaling qb2's
16.78 ms/fold control by qb1's own 123.30/102.33 = 1.205 on the sister site, I predict a control op wall
of **18-22 ms/fold** and a delta of **-10 to -18 ms/fold**. **Wrong if** the control is outside that
band or the delta is under 8 or over 20 ms/fold.

**P4 — parity re-taken here, not inherited.** `torch.equal` **True**, max abs **0.0**, at the fold's own
`[1, 298, 320, 256] @ [256, 1]` and `@ [256, 64]` against the flags-off DRAM path of the same config,
and every live fold in this pass returns plDDT **0.859489**, identical to six decimals to X7's and
X10's. A memory config decides where a writer puts a tile, not the order a contraction accumulates, so
there is no mechanism for a trade here; the thing actually untested is whether the L1 and DRAM writers
pack identically at 130 cores. **Wrong if** any `torch.equal` is False or any plDDT differs at the sixth
decimal.

**P5 — the roofs, measured on this card this pass, agree with X7's card-1 figures.** Same host, same
board type, same wheel, different card index: I predict DRAM->L1 read within **5 %** of 388.1 GB/s,
L1->DRAM unary write within 5 % of 264.4, DRAM->DRAM copy within 5 % of 403.5, and square bf16 HiFi4
within **8 %** of 135.67 TFLOP/s. I inherit none of them and divide by my own. **Wrong if** any is
outside its band.

**P6 — Q21, the empty cell in the version x core-count matrix.** `ttnn.linear(core_grid=)` with an L1
output at **130 cores**: I predict it **throws** at 0.67.4 (reproducing X7) and **also throws** at
0.68.0 in `/home/ttuser/tt-boltz2/env`, confirming X10's reading that the refusal is a core-count
property and not a version one. It blocks nothing either way — production takes the tuned config, not
`l1_cg`. **Wrong if** it runs at 130 cores on either wheel.

**P7 — hold the card, move the wheel.** qb1's PWA projection wall is 123.30 ms/fold against qb2's
102.33, a 20 % gap that X10 could not attribute because card and wheel moved together. I predict the
0.68.0 spot-check **on this card** puts the same site within **10 %** of its 0.67.4 figure, so the gap
is mostly the **card** (p150a at 130 cores vs P300c at 110), not the wheel. **Wrong if** 0.68.0 on this
card lands within 10 % of qb2's 102.33 instead.

**P8 — the verdict, and the ranking priced in advance.** I expect **cite both**, with both absolutes
larger than X10's ratios. Confidence order: P1 holds (highest), P4 (high), P2's band (medium-high), P3's
band (medium), P7 (medium-low), P6 (medium-low — it is the cell nobody has filled). If PWA comes in
under 60 ms/fold on qb1 I report that as a loss against X10's own figure rather than re-framing the
question.

**One prediction about the instrument, not the change.** X7's qb1 `body:PairWeightedAveraging` is
423.12 ms/fold against qb2's 182.71 — a 2.3x host gap at a region whose projection half is only 20 %
apart. I predict that gap reproduces in my control, i.e. it lives in the region's other ops (softmax,
the `m` projections, the permutes) and not at the site under test. Recorded, not chased (charter §1).

## The instrument, chosen before the numbers exist

A **fold wall cannot resolve either figure and taking one would be a failed deliverable.** X9 measured
this host's fold-wall A/A floor at **224.0 ms** largest apparent delta on identical code over twenty
folds; the targets are 75.91 and 11.49. So: the `body:PairWeightedAveraging` **region wall** over 30
regions and the sites' own **op walls** over 240 and 40 calls, every region synchronised on both sides,
arms in separate processes on one card in one session, with the flags-off baseline restored and run
**first and last** and a third in the middle. The control is the mean of the three with the spread
stated. **PWA is priced on its region wall, never as a per-call delta x 240** — one `layer_norm` feeds
**eight** per-head projections, so multiplying the norm's saving by 240 instead of 30 invents ~+29
ms/fold of phantom win (X10's own words), the same over-count that inflated X2's 298.7.
