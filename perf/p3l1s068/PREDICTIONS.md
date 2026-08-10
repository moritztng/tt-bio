# p3-l1-source-068 — predictions, written before the device was opened

qb2 **chip 2**, ttnn **0.68.0** (`/home/ttuser/tt-bio-dev/env`, `ttnn 0.68.0`), Blackhole **P300c**
(subsystem `0x0046`, board `005` = chips 2+3, chip 3 held empty). Branch
`wk/protenix-trunk--p3-l1-source-068`, reset onto `wk/protenix-trunk--p3-l1-output` at `9ae1ef2d`
so X7's `_PAIR_PROJ_L1_OUT` / `_PAIR_PROJ_L1_BW` / `_PAIR_BIAS_L1_NORM`, the output term in the L1
budget and `_l1_memory_config_if_it_fits` are the baseline and are not re-derived. Scope:
**protenix**-v2, the **trunk**, **298** aa.

Everything here is a **ratio on this card against this card's own baseline** (charter §4.8). The
campaign absolute is X7's, on qb1 at 0.67.4. Host state at the time of writing: load average 0.77,
`fuser` reports no open handle on any of `/dev/tenstorrent/0..3`.

---

**P1 — the L1 output is still legal at 0.68.0, with the tuned config and only with it.**
X7's failure mode is `ttnn.linear(core_grid=)` + an L1 output, which ran at 110 cores on qb1 and
**threw** at 130 ("statically allocated circular buffers clash with L1 buffers"). I predict on this
card at 0.68.0: the tuned 1D program config with an L1 output **runs** at every leg X7 ran it at,
and the `l1_cg` leg **throws**. I read `COMPUTE_GRID_MAIN` after the device is open and predict
**13x10** (the same Blackhole grid qb1 reports), stated explicitly in every JSON.
**Wrong if** the tuned L1-output config throws at any leg, or `l1_cg` runs.

**P2 — parity re-taken, not inherited: `torch.equal` True.** A memory config decides where the
writer puts a tile, not the order the contraction accumulates, and 0.68.0 does not change that. I
predict `torch.equal` **True**, max abs **0.0**, on (a) the pair-track projection L1-out at
`in0_block_w`=8 against production today's DRAM output of the identical config, (b) the whole
`proj -> proj -> multiply_ -> add_` trimul chain, (c) the 2088 z->bias with an L1 `layer_norm`
source and an L1 output, all at the fold's own `[1, 298, 320, 256]`; and a live 298 aa fold whose
plDDT is **identical to six decimals** between the change-on and change-off arms on this card.
**Wrong if** any `torch.equal` is False or the two plDDTs differ at the sixth decimal. That is the
finding that stops the merge.

**P3 — the probe ratios move by less than 10 % of X7's figures.** X7 on qb1/0.67.4 measured, against
production today: `l1_tuned_bw8_obh2` **1.473x** on `proj + add`, `l1_tuned_bw8_obh5` 1.420x, the
bit-exact `l1_tuned_bw1_obh5` 1.150x, and on the real trimul chain `both_l1_bw8_obh2` **1.373x** /
`both_l1_bw1` 1.078x. The absolute microseconds are a different card and are not comparable; the
**ratios** are what carries. I predict every one of those ratios reproduces here **within 10 % of
X7's figure**, and specifically that `both_l1_bw8_obh2` lands **1.24-1.51x**.
**Wrong if** any ratio moves more than 10 % of X7's figure, or changes sign.

**P4 — the in-fold op wall still removes the same fraction of each class.** X7's `rg` arm removed
19.5 % of the trimul projection class's wall (985.9 -> 793.6 ms), 19.1 % of `gate_and_project`'s
(505.8 -> 409.4), 11.1 % of the residual `add_`'s (1190.5 -> 1058.0) and 59.8 % of the 2088 site's
(252.1 -> 101.4). I predict each of those **fractions** reproduces on this card within 20 % of
itself, so the combined op-wall delta on this card is **between 0.8x and 1.2x of this card's own
baseline sum times X7's fraction**. I do NOT predict 561.8 ms: that is a qb1 absolute.
**Wrong if** any class's fraction is outside 20 % of X7's, or any class moves the wrong way.

**P5 — deliverable 2, PWA `tenstorrent.py:3084`: the source is the lever and the per-call pricing
is honest here, unlike at the trimul.** The site is `[298, 320, 256] @ [256, 1]`, **240 calls/fold**,
X7's baseline op wall **123.65 ms/fold** (517.0 us/call, within 1 % of the 2088 site's 522.3). Its
`layer_norm` is at `tenstorrent.py:3074` and runs **once for eight head projections** — 30
`layer_norm` calls/fold against 240 projections. X7's over-pricing correction was about a shared
*consumer* (four trimul projections feeding two residual adds), and this is the mirror case: the
shared object here is the **producer**, and each of the eight projections independently stops
reading 48.82 MB from DRAM, so 240 x the per-call read saving is the right arithmetic. What is
**not** 240x is the `layer_norm`'s own removed write: that is paid 30 times, not 240. Taking X7's
measured 2088 split (the L1 source moved the projection 450.3 -> 137.0 us, and the norm's own write
was worth ~117 us of the region) I predict the site's op wall falls **60-100 ms/fold**, of which
**under 5 ms/fold** is the `layer_norm` half, and that the L1 **output** alone is worth **under
15 ms/fold**. **Wrong if** the delta is under 40 or over 120 ms/fold, or the L1 output alone beats
25 ms/fold, or the L1-source arm is slower than the DRAM one.

**P6 — deliverable 3, the template z projection at `protenix.py:2033`, and it is small.** It is
`self._lin(zn, "template_embedder.linear_no_bias_z.weight")`, `[1, 298, 320, 256] @ [256, 64]`,
inside `for t in range(nt)` with `zn` computed **once above the loop** — the same shared-producer
shape as PWA, with nt=4 consumers instead of 8. It runs **4 templates x 10 cycles = 40 calls/fold**.
At ~520 us/call that is **~21 ms/fold of wall**, so I predict the L1 source is worth **8-18
ms/fold** and that this is **below the bar for a production config change on its own**, shipping
only if it rides along on the same helper the PWA site needs anyway.
I also predict it is **invisible to X7's instrument**: `protenix.py` does
`from .tenstorrent import ... _narrow_proj_linear`, so patching `T._narrow_proj_linear` never sees
it, which is why no `w=[256, 64]` row exists in any of X7's `ops_*.json`. **Wrong if** the site's
measured wall is above 35 ms/fold, or a `w=[256, 64]` row does appear once I patch only `T`.

**P7 — the roofs on this card are NOT qb1's and I inherit none.** qb2 chip 2 is a P300c, qb1 card 1
a p150a. I re-measure DRAM->L1 read, L1->DRAM unary write, DRAM->DRAM copy and square bf16 HiFi4
compute on this chip this pass, and I predict at least one of them differs from X7's qb1 figures
(388.1 / 264.4 / 403.5 GB/s, 135.67 TFLOP/s) by **more than 8 %**, which is exactly why no absolute
crosses between the two docs.

**P8 — the verdict.** I expect to land on "**X7's proposal stands at 0.68.0**": the L1 output legal,
parity `torch.equal` True, the ratios inside 10 %. The single most likely way to be wrong is P1 —
a circular-buffer sizing change inside 0.68.0's matmul that refuses the tuned L1-output config the
way 0.67.4 already refuses `core_grid=`. If that happens I stop deliverable 1 there and report it,
because it blocks the org's largest merge recommendation on the wheel the fleet actually ships.

**Priced in advance so the ranking can be wrong too.** My ordering going in is: deliverable 1 holds
(highest confidence), PWA delivers 60-100 ms/fold (medium), the template site delivers under 20 and
is not worth its own surface area (medium). If the PWA site comes in under 40 ms/fold I report that
as a loss against X7's own 60-100 estimate rather than re-framing it.
