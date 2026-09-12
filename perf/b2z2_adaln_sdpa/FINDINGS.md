# b2z2-step-adaln-sdpa — both sites are short of work units, and the SDPA fix is bit-exact

TASK TYPE: ACCELERATE | PLAYBOOKS loaded: ACCELERATE + ALWAYS-ON | memories read:
`b2z2-radical-2x-wave2`, `no-speedup-by-skipping-the-models-own-work`,
`unified-solution-not-per-model-patches`, `perf-method-floor-screen-predict-then-build`,
`rfd3-isolated-screen-underprices-residency-lever`, `ttnn-split-work-to-cores-grid-height-holes`,
`whglx-unpinned-all-chip-open-breaks-every-cotenant`, `tt-bio-worktree-run-recipe`,
`git-worktree-add-f-steals-branch-ref`, `token-axis-must-bucket-to-multiple-of-32`

STATUS: IN FLIGHT (pass 1). Both sites measured, both of the brief's proposed levers refuted by
  measurement, and a third lever found that neither the brief nor this row's own prediction named.
BRANCH: wk/b2z2-step-adaln-sdpa (pushed, not merged), cut from
  `wk/b2z2-step-fusion-next-sites` @ `9cd6c2ff1` so `TT_BIO_ATOM_L1` is under every number here.
CARD: whglx card 12, one Wormhole_B0 of the 32-chip mesh, **8x9 = 72 cores**, pinned
  `TT_VISIBLE_DEVICES=12`, `TT_BIO_LEASE_CARDS=12`, `TT_BIO_TRACE_REGION_SIZE` 512 MiB, run as
  `tt-admin`. No other chip opened, no reset needed.
ARCH: WH for every number. The published cell is Blackhole; this row holds no BH chip and any BH
  figure is labelled PROJECTED.
BASE: **41.5235 ms/step**, A/A floor **1.00064** over 6 blocks of 10 replays. That reproduces
  `b2z2-step-program-fusion`'s 41.6014 to 0.19 % and `b2z2-step-layout-elision`'s 41.5784 to
  0.13 %, on a third card.

## 1. P0 CONFIRMED — the brief's 2.902 ms/step site is half owned already

`site_rank.py` clusters by op code and cost, and at `[1, 512, 768]` it cannot tell
`AdaLN.s_terms`'s norm from `AdaLN.__call__`'s. The per-call join (`sites_census.py`, 114 recorded
calls, wrapped for exactly one settled step) splits them:

| n | site | shape | affine | owner |
|---|---|---|---|---|
| 48 | `AdaLN.__call__` token | [1, 512, 768] | **none** | free |
| 48 | `AdaLN.s_terms` token | [1, 512, 768] | `weight` | **`b2z2-step-layernorm-fusion`** |
| 12 | `AdaLN.__call__` atom | [1, 140, 32, 128] | **none** | free |
| 6 | swiglu + `Diffusion` singletons | mixed | both | free, 0.39 ms |

**The unowned norm-of-`a` population is 60 programs and 1.566 ms/step, not 97 and 2.902.** That is
3.8 % of the step and it caps the whole site at 1.039x. Predicted 60 / 1.566 before the run.

## 2. P1 CONFIRMED — the fold the brief proposes does not exist to be built

All 60 norms of `a` carry `weight=None, bias=None`, measured, not read off the source: AdaLN's
affine is the *adaptive* one, a per-token `[1, 512, 768]` `s_scale`/`s_bias`, and `ttnn.layer_norm`'s
`weight=`/`bias=` slots take a per-channel 1-D term. There is nothing to fold. The multiply+add
half was already built by `b2z2-step-binaryng-fusion` as one `ttnn.mac`, bit-exact, and measured
**0.99315x**. The brief's step 2 is closed by measurement and this row does not build it.

## 3. P2 CONFIRMED as a MECHANISM, and it is not the one I predicted the fix for

The norm of `a` is core-starved, and the test is direct: give the same op 4x the rows.

    [1,  512, 768]   16 tile-rows   29.40 us
    [1, 2048, 768]   64 tile-rows   55.40 us      4x the work for 1.884x the time

An op that is bandwidth- or arithmetic-bound costs 4x for 4x. This one does not, so at 512 rows it
is waiting on its own core count: ttnn's interleaved LayerNorm hands whole tile-rows to cores, 512
rows is **16 tile-rows, so 16 of 72 cores**, and 1.572 MB in 29.40 us is 53.5 GB/s against a card
whose matmuls reach 23-47 % of a far higher roof. **Predicted 16-24 cores, confirmed.**

**But the fix I pre-registered is not available.** Row count is fixed by the model, and height
sharding cannot split a tile-row, so no row-parallel form reaches more than 16 cores. Reaching 72
means splitting the 768-wide reduction across cores, which changes its association and is therefore
NOT bit-exact — and `b2z2-step-layout-elision` has already measured that this step's sub-ULP levers
**add** rather than cancel (0.37 A + 0.35 A = 0.71 A against a 0.60 A bar). A block-sharded norm is
the one thing left at this site; it is named here and priced in pass 2, not claimed.

## 4. P3 REFUTED — the token SDPA is NOT byte-bound, and the bias dtype proves it

Shapes, measured: q/k/v `[1, 16, 512, 64]` bf16, bias `[1, 16, 512, 512]` bf16, 24 calls a step.
(Head dim is 64, not the 48 I predicted from 768/16 — the projection is wider than the model dim.)

| term | per call | share of bytes |
|---|---|---|
| arithmetic | 1.0737 GFLOP at 125.2 us = **8.58 TF/s, 4.8 % of the 178.6 TF/s roof** | — |
| q + k + v | 3.146 MB | 27 % |
| **bias** | **8.389 MB** | **67 %** |
| out | 1.049 MB | 8 % |
| total | 12.58 MB at 125.2 us = **100.5 GB/s** | |

The arithmetic half of P3 is right and the byte half is wrong. **Dropping the bias to bfp8_b takes
the op to 0.688x of its bytes and buys 1.070x. bfp4_b takes it to 0.521x and buys 1.131x.** A
byte-bound op would have returned ~1.45x and ~1.92x. So bytes explain about a fifth of this op, and
**P4's lever is dead: the bias dtype is not worth an accuracy change.** (For the record it is not
free either: bfp8_b moves the output 0.875 max abs on a 4.365 RMS.)

## 5. The lever neither the brief nor the prediction named: `q_chunk`, and it is BIT-EXACT

The SDPA's work-unit count is `(S / q_chunk) * heads`. Shipped is `q_chunk = k_chunk = 256` from
`_capped_sdpa_chunk_size`, a constant with **no grid term**, which at 512 tokens and 16 heads is
**32 work units on 72 cores**. Sweeping it, every rung checked against the shipped config with
`torch.equal`:

| q_chunk | work units | us | vs shipped | bit-exact |
|---|---|---|---|---|
| 512 | 16 | 182.70 | 0.686x | yes, max abs 0 |
| **256 (shipped)** | 32 | **125.40** | 1.000x | — |
| **128** | **64** | **95.60** | **1.3117x** | **yes, max abs 0** |
| 64 | 128 | 106.00 | 1.183x | yes, max abs 0 |
| 32 | 256 | 140.40 | 0.893x | yes, max abs 0 |

The curve peaks where the work units first fill the grid and fall in one pass (64 units, 72 cores),
and every rung is bit-identical, because `q_chunk` partitions independent query rows — `k_chunk` is
the online-softmax reduction order and that one is NOT free, so it is untouched here.

**24 calls x 29.8 us = 0.715 ms/step = 1.0176x on the step, bit-exact, from a program config.**
That is the site's honest ceiling too: the whole SDPA site is 2.745 ms/step and this takes 26 % of it.

## 6. What pass 2 owes

1. Build the grid-aware `q_chunk` pick — structural, no model name, no sequence length: the
   largest work-unit count that still fits one grid pass. Measure it **on the step**, interleaved,
   with `torch.equal` beside the ratio.
2. Price the block-sharded norm of `a` at the one site that is left, and say plainly whether a
   third sub-ULP lever is worth having on a step that already cannot stack two.
3. Disjointness: neither site is touched by `TT_BIO_ATOM_L1`, `TT_BIO_ATOM_KEY_WINDOW`,
   `TT_BIO_ATOM_KV_PREPROJ` (all atom-branch) or the layernorm row's 48 `s_terms` norms.

## 7. Artifacts

`perf/b2z2_adaln_sdpa/` on `wk/b2z2-step-adaln-sdpa`: `PREDICTED.md` (committed at `c69f319c9`,
before the first device run), `sites_census.py` + `census_wh_c12.json` (the per-call join, both
micro-screens, the step A/A floor), `chunk_screen.py` + `chunks_wh_c12.json` (the q_chunk sweep and
the row-scaling test).

CHEAT-CHECK: clean. 200 sampling steps, 3 recycles, full 35-row MSA depth, seed 0, `cdk2x2_512`,
templates off, one sample — untouched in every measurement here. Nothing measured so far removes a
unit of the model's own work: a `q_chunk` is how the same attention is partitioned across cores and
`torch.equal` at max abs 0 on every rung is the proof that the same arithmetic came out.
TILE-MOVEMENT-DELTA: 0.0 % — a `q_chunk` change moves the same tiles to different cores. Named, not
estimated: the sweep's own bit-exactness is what says no tile pass was added or removed.
