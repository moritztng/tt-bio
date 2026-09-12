# b2z2-step-adaln-sdpa — both of the brief's levers are refuted, and `q_chunk` is a grid parameter nobody set

TASK TYPE: ACCELERATE | PLAYBOOKS loaded: ACCELERATE + ALWAYS-ON | memories read:
`b2z2-radical-2x-wave2`, `no-speedup-by-skipping-the-models-own-work`,
`unified-solution-not-per-model-patches`, `perf-method-floor-screen-predict-then-build`,
`rfd3-isolated-screen-underprices-residency-lever`, `ttnn-split-work-to-cores-grid-height-holes`,
`fused-sdpa-ragged-tile-tail`, `token-axis-must-bucket-to-multiple-of-32`,
`whglx-unpinned-all-chip-open-breaks-every-cotenant`, `tt-bio-worktree-run-recipe`,
`git-worktree-add-f-steals-branch-ref`, `one-size-tuning-is-a-standing-defect-class`

VERDICT: GO on one lever the brief did not name, REFUTED on both it did, and the second site is
  named-not-built exactly as its falsifier requires. `TT_BIO_SDPA_GRID_Q_CHUNK` is **1.01831x on
  the diffusion step and bit-exact** — `torch.equal`, max abs 0.0 — with the arms not overlapping
  at all. It ships behind a flag defaulting OFF and the branch is not merged; the default flip is
  the orchestrator's call and there is nothing accuracy-shaped to weigh, only a size ladder.
BRANCH: wk/b2z2-step-adaln-sdpa (pushed, not merged), cut from
  `wk/b2z2-step-fusion-next-sites` @ `9cd6c2ff1` so `TT_BIO_ATOM_L1` is under every number here.
CARD: whglx card 12, one Wormhole_B0 of the 32-chip mesh, **8x9 = 72 cores**, pinned
  `TT_VISIBLE_DEVICES=12`, `TT_BIO_LEASE_CARDS=12`, `TT_BIO_TRACE_REGION_SIZE` 512 MiB, run as
  `tt-admin`. No other chip opened, no reset needed, no `tt-smi` run.
ARCH: WH for every measured number. The published cell is Blackhole, this row holds no BH chip,
  and every BH figure here is labelled PROJECTED.

PREDICTED: `perf/b2z2_adaln_sdpa/PREDICTED.md`, committed at `c69f319c9` before this row's first
  device run. P0 the brief's 2.902 ms site is a cost-cluster and only 60 programs / 1.566 ms of it
  is unowned. P1 the fold the brief proposes is structurally unavailable. P2 the norm of `a` is
  parallelism-starved on 16-24 of 72 cores, fixable for 1.005x-1.020x. P3 the token SDPA is
  byte-bound with 73 % of its bytes in the bias. P4 so the lever is the bias dtype, 1.015x-1.035x.
MEASURED: **STEP-RATIO 1.01831x (WH), 41.5915 -> 40.8436 ms/step, -0.7479 ms**, 5 blocks of 10
  replays, arms interleaved with the lead alternating, profiler off, one process. **The arms do
  not overlap: the slowest `gridq` block (40.8534) beats the fastest `base` block (41.5674) by
  1.75 %.** Spreads 0.141 % and 0.034 % against a step **A/A floor of 1.00064** — the effect is
  29x the floor. **P0 CONFIRMED exactly** (60 programs, 1.566 ms, from a 114-call per-call join).
  **P1 CONFIRMED** (all 60 norms carry `weight=None, bias=None`). **P2 CONFIRMED as a mechanism
  and its fix REFUTED as unavailable** (§3). **P3 REFUTED on its byte half** (§4) and **P4 dead
  with it**. The lever that pays was named by neither the brief nor the prediction (§5).
PARITY: **BIT-EXACT, `torch.equal`, max abs 0.0**, on the real settled `Diffusion.__call__` at
  512 aa with its shipped shapes, and separately on all five rungs of the op-level `q_chunk` sweep
  against the shipped config. No fold-level scoring is owed: `q_chunk` partitions independent
  query rows, the reduction order lives in `k_chunk`, and that one is untouched. The base arm
  reproduces two other rows' step wall on a third card to 0.19 % and 0.13 %.
DEFICIT-SECONDS: 0.1496 s REMOVED on a WH fold — 0.7479 ms/step x 200 steps, measured on the
  step rather than projected off an op probe. That is 1.00361x on a 41.533 s WH fold. On the
  published Blackhole cell, **PROJECTED only**: the WH ratio on the committed 26.400 ms BH step is
  -0.4747 ms/step, 20.079 -> 19.984 s, **1.00475x**. This row holds no BH chip.
TILE-MOVEMENT-DELTA: +2.321 % — and the step still got 1.83 % faster. Counted, not estimated: a
  narrower `q_chunk` re-reads K and V once per chunk, so 2 chunks per head become 4 and this op's
  tile passes go 7,168 -> 9,216 per call, +2,048 x 24 calls = +49,152 of the step's 2,117,676
  (`b2z2-step-binaryng-fusion`'s census). **This is the mirror of that row's result** — it removed
  4.75 % of the movement and lost 1.6 %; this one adds 2.3 % of it and wins 1.8 %. On the diffusion
  step neither direction of the movement term predicts the sign of the wall.
CHEAT-CHECK: clean. 200 sampling steps, 3 recycles, the full 35-row MSA depth, seed 0,
  `cdk2x2_512`, templates off, one sample — untouched in every measurement here. The lever changes
  how the same attention is partitioned across cores and nothing else; `torch.equal` at max abs 0.0
  on the step output is the strongest available statement that no work was skipped. No step-cut,
  recycle-cut or depth-cut number appears in this pass or its artifacts.

## 1. P0 CONFIRMED — the brief's 2.902 ms/step site is half owned already

`site_rank.py` clusters by op code and cost, and at `[1, 512, 768]` nothing cost-shaped can tell
`AdaLN.s_terms`'s norm (31.94 us) from `AdaLN.__call__`'s (27.88 us). The per-call join
(`sites_census.py`, `ttnn.layer_norm` wrapped for exactly one settled step, 114 calls recorded)
splits them:

| n | site | shape | affine | owner |
|---|---|---|---|---|
| 48 | `AdaLN.__call__` token | [1, 512, 768] | **none** | free |
| 48 | `AdaLN.s_terms` token | [1, 512, 768] | `weight` | **`b2z2-step-layernorm-fusion`** |
| 12 | `AdaLN.__call__` atom | [1, 140, 32, 128] | **none** | free |
| 6 | swiglu + `Diffusion` singletons | mixed | both | free, 0.39 ms total |

**The unowned norm-of-`a` population is 60 programs and 1.566 ms/step, not 97 and 2.902.** That is
3.8 % of the step, and it caps the entire site at 1.039x even if the norm went free.

## 2. P1 CONFIRMED — the fold the brief proposes does not exist to be built

All 60 norms of `a` carry `weight=None, bias=None`, measured rather than read off the source.
AdaLN's affine is the *adaptive* one: a per-token `[1, 512, 768]` `s_scale` and `s_bias`, while
`ttnn.layer_norm`'s `weight=`/`bias=` slots take a per-channel 1-D term. There is nothing to fold.
The multiply+add half is already built — `b2z2-step-binaryng-fusion` replaced it with one
`ttnn.mac`, bit-exact, and measured **0.99315x**. The brief's step 2 is closed by measurement.

Two levers the brief's step 2 implies were screened anyway, on the grabbed step's own operands:

* **the AdaLN chain L1-resident: 0.809x, 24 % SLOWER** (91.8 -> 113.5 us for `layer_norm` +
  `multiply_` + `add_` on `[1, 512, 768]`), bit-exact both ways. `TT_BIO_ATOM_L1` bought 1.08461x
  doing exactly this to the atom branch; the token branch does not repeat it, and the reason is
  that the atom branch's live set was 19.5 MB of DRAM round trips while this one is 2.36 MB.

## 3. P2 CONFIRMED as a mechanism — and the fix I pre-registered is not available

The norm of `a` is core-starved, tested directly by giving the same op 4x the rows:

    [1,  512, 768]   16 tile-rows   29.40 us
    [1, 2048, 768]   64 tile-rows   55.40 us      4x the work for 1.884x the time

An op bound by bytes or by arithmetic costs 4x for 4x. This one does not, so at 512 rows it waits
on its own core count: ttnn's interleaved LayerNorm hands whole tile-rows to cores, 512 rows is
**16 tile-rows, so 16 of 72 cores**, and 1.572 MB in 29.40 us is 53.5 GB/s. Predicted 16-24, and
that is what it is.

**But the row-parallel fix is unavailable at any size.** Height sharding cannot split a tile-row,
so no row-parallel form of this op reaches more than 16 cores at 512 tokens; and the site stops
being starved at all only above **2,304 tokens** (72 tile-rows), where there is nothing left to
win. Reaching the grid at 512 aa means splitting the 768-wide reduction across cores, which
changes its association and is **not bit-exact** — and `b2z2-step-layout-elision` has already
measured that this step's sub-ULP levers **add** rather than cancel (0.36959 + 0.34638 A against a
0.60 A bar). So, per this row's own pre-registered falsifier: **the AdaLN norm of `a` is named and
not built. The size at which it would become a bit-exact lever is above 2,304 tokens, and at that
size the starvation it would fix has gone away on its own.** The block-sharded form is the one
thing left there, it is an accuracy change, and it should be priced by whoever is willing to spend
this step's remaining sub-ULP budget on 1.566 ms.

## 4. P3 REFUTED — the token SDPA is NOT byte-bound, and the bias dtype is what proves it

Measured shapes: q/k/v `[1, 16, 512, 64]` bf16 against a `[1, 16, 512, 512]` bf16 bias, 24 calls a
step. (Head dim is 64, not the 48 I predicted from 768/16 — the projection is wider than the model
dim.)

| term | per call | share of bytes |
|---|---|---|
| arithmetic | 1.0737 GFLOP at 125.2 us = **8.58 TF/s, 4.8 % of the 178.6 TF/s roof** | — |
| q + k + v | 3.146 MB | 27 % |
| **bias** | **8.389 MB** | **67 %** |
| out | 1.049 MB | 8 % |
| **total** | **12.58 MB at 125.2 us = 100.5 GB/s** | |

The arithmetic half of P3 is right, the byte half is wrong, and the falsifier that settles it is
the dtype itself: **bfp8_b on the bias takes the op to 0.688x of its bytes and buys 1.070x;
bfp4_b takes it to 0.521x and buys 1.131x.** A byte-bound op would have returned ~1.45x and
~1.92x. **Bytes explain about a fifth of this op**, so P4's lever is dead — the bias dtype is not
worth an accuracy change, and it is not free either (bfp8_b moves the output 0.875 max abs on a
4.365 RMS).

**This is the answer to the question the brief asked, and it is not the trunk's answer.** Wave 1
closed a Boltz-2-specific SDPA on the TRUNK's, on the grounds that its cost was packer passes. The
SAMPLER's is neither packer passes nor bytes nor arithmetic nor the 9.76 us per-program constant
(7.8 % of 125.2 us). It is the grid.

## 5. The lever: `q_chunk` is a grid parameter and the shipped pick has no grid term

ttnn's SDPA parallelises over `batch * heads * q_chunks`, one chunk per core. The shipped pick is
`_capped_sdpa_chunk_size` = `min(256, padded_len)` — **a constant with no grid term, the same on a
72-core Wormhole and a 110-core Blackhole**. At 512 tokens and 16 heads that is **32 work units on
72 cores**. Sweeping it, every rung checked against the shipped config with `torch.equal`:

| q_chunk | work units | us | vs shipped | bit-exact |
|---|---|---|---|---|
| 512 | 16 | 182.70 | 0.686x | yes, max abs 0 |
| **256 (shipped)** | 32 | **125.40** | 1.000x | — |
| **128** | **64** | **95.60** | **1.3117x** | **yes, max abs 0** |
| 64 | 128 | 106.00 | 1.183x | yes, max abs 0 |
| 32 | 256 | 140.40 | 0.893x | yes, max abs 0 |

The curve peaks where the work units first fill the grid in a single pass (64 units, 72 cores) and
falls off on both sides — idle cores above, a second pass and its tail below. Every rung is
bit-identical because `q_chunk` partitions independent query rows.

`_grid_q_chunk(q_len, work, cap, n_cores)` picks the smallest `q_chunk` whose
`work * padded // q_chunk` still fits one grid pass, clamped to the shipped cap and to a chunk that
divides the padded length. **No sequence length, no head count and no model name is written down**:
both come from the tensor, the grid comes from the device, and all six call sites across
`tenstorrent.py`, `esmc.py`, `esmfold2.py` and `saprot.py` pass `batch * heads` uniformly. The atom
SDPA (560 heads, 32 q) is left exactly where it was, because at 560 work units it was never
starved. `perf/b2z2_adaln_sdpa/test_grid_q_chunk.py` checks seven named shapes and the invariants
over 93 lengths x 6 head counts x 3 grids, off-device.

**On the step: 41.5915 -> 40.8436 ms, 1.01831x, bit-exact.** The op sweep priced it at 0.715 ms
(1.0176x) and the step returned 0.7479 ms, so for once the op screen **under**-priced by 4.6 %
rather than over-pricing — the opposite of this block's usual 38 %.

## 6. Why it is release-gated even though it is bit-exact

It cannot change a value; `torch.equal` at max abs 0.0 settles that at the op and at the step. What
it can do is pick a worse rung on a shape nobody has measured. The curve is **not monotonic** —
q_chunk 32 is 0.893x, worse than shipped — so a rule that lands on the wrong side of the peak costs
real time, and only the 512 aa / 16-head / 72-core point has been measured. `one-size-tuning-is-a-
standing-defect-class` cuts both ways here: the shipped constant is the one-size defect, and a
rule fitted at one point is the next one. `TT_BIO_SDPA_GRID_Q_CHUNK` therefore defaults **False**,
the branch is not merged, and what the flip needs is a size ladder (298 / 512 / 1024 aa) and one
Blackhole run at 11x10, where the rule picks 128 for 64 units of 110 cores and 64 (128 units, two
passes) has never been tried. Not an accuracy check — there is nothing to check.

## 7. Disjointness

Neither site is touched by `TT_BIO_ATOM_L1`, `TT_BIO_ATOM_KEY_WINDOW` or `TT_BIO_ATOM_KV_PREPROJ`
(all three are in the atom branch; this row's SDPA is the token DiT's and its norm site is
`AdaLN.__call__`, not `s_terms`). `b2z2-step-layernorm-fusion` owns the 48 `s_terms` norms and this
row explicitly does not take them. `b2z2-step-layout-elision`'s arm B fuses the token head split
into the qkv matmul, which changes what feeds the SDPA but not how it is chunked. The one lever
that shares a *program* with this one is `b2z2-step-binaryng-fusion`'s `TT_BIO_MAC_FUSE`, which is
NO-GO and off.

## 8. Artifacts

`perf/b2z2_adaln_sdpa/` on `wk/b2z2-step-adaln-sdpa`:
`PREDICTED.md` (committed at `c69f319c9`, before the first device run),
`sites_census.py` + `census_wh_c12.json` (the 114-call per-call join, the AdaLN-chain L1 screen,
the SDPA bias-dtype screen, the step A/A floor),
`chunk_screen.py` + `chunks_wh_c12.json` (the `q_chunk` sweep with per-rung `torch.equal`, and the
4x-rows test that convicts the norm),
`step_ab.py` + `step_ab_wh_c12.json` (the interleaved step A/B),
`test_grid_q_chunk.py` (off-device). Model code: `tt_bio/tenstorrent.py` `_grid_q_chunk` and
`_sdpa_program_config_for_lengths`, plus the `work` argument at all six call sites. Flag off.

**Instrument note, so nobody trusts a blank:** `step_ab.py` records `SDPA_CHUNK_PICKS` and it comes
back **empty for both arms**. That counter is written only by `_tri_att_sdpa_at`, which the
AttentionPairBias path does not go through, so it is blind to the picks this lever makes. The pick
is established by `test_grid_q_chunk.py` and by the sweep, not by that counter.
