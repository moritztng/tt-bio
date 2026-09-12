# b2z2-grid-qchunk-unify — the constant audit, the ladder on three grids, and where the rule stops

TASK TYPE: ACCELERATE | PLAYBOOKS loaded: ACCELERATE + ALWAYS-ON | memories read:
`b2z2-radical-2x-wave2`, `one-size-tuning-is-a-standing-defect-class`,
`unified-solution-not-per-model-patches`, `no-speedup-by-skipping-the-models-own-work`,
`whglx-tt-visible-devices-is-a-umd-logical-id-not-a-device-node`,
`whglx-galaxy-lease-card-number-vs-device-node-mismatch`, `pc-card0-512aa-fold-nondeterminism`,
`qb2-p300c-chip-wedge-fourth-device-open`, `tt-bio-worktree-run-recipe`,
`verification-instrument-drift-is-shared-code-drift`

PREDICTED: `perf/b2z2_gridq/PREDICTED.md`, committed at `b21af2b08` before the first grep and the
first device run.

## What this row settles

**P0 — how many other grid-blind constants: 14 of 25 pickers, and THREE of them are occupancy.**
Predicted 5-9 grid-blind with 1-3 carrying exposure. The count is right and the framing was not:
the interesting number is not how many constants are blind, it is how many of the blind ones spend
a resource that IS the grid.

**P1 — the Blackhole answer: REFUTED, and in the useful direction.** I predicted the win would be
BIGGER on 110 cores than on 72, because the shipped 256 leaves a worse occupancy there (32 of 110
against 32 of 72). Measured on Blackhole silicon at the cell's own 11x10 grid, the diffusion
token SDPA's shape reads **1.1398x**, against Wormhole's **1.3058x**. The occupancy argument gives
the right SIGN everywhere and the wrong SIZE, so occupancy is not the whole cost model.

**P2 — across models: the pick moves on every model on the path, and it never loses on any shape
on any of three grids.** 30 rungs measured (10 shapes x 3 grids), worst case 1.0000x on 72 and on
110 cores and 1.1789x on 130, every rung bit-exact on the part where bit-exactness is meaningful.

**The rule is a floor, not the answer, and Blackhole is where that shows.** It picks the measured
optimum on 8 of 10 shapes at 72 cores and on only 4 of 10 at 110 and 3 of 10 at 130. Every miss is
in the same direction: the true optimum is NARROWER than the first chunk that fills the grid in one
pass. At 512 tokens and 16 heads on 110 cores the rule's 128 reads 1.1398x where 32 reads 1.2522x.

## 1. The audit (`perf/b2z2_gridq/grid_blind_census.py`, `out/census_join.json`)

Asked by experiment, not by reading: every picker is called at the fleet's shipped shapes under
three simulated grids -- 8x9 = 72, 11x10 = 110, 13x10 = 130 -- in three separate processes, and
the returns are diffed. A picker whose answer never moves across a 1.8x change in core count is
grid-blind by measurement.

**11 of 25 move with the grid. 14 do not.** The 14 split by the resource they actually spend, read
against their call sites rather than guessed from their names:

| class | what it means | grid-blind members |
|---|---|---|
| **OCCUPANCY** | sets how many cores run the op | **`_capped_sdpa_chunk_size`**, `_sdpa_chunks_shipped`'s (64,64) band, `_tri_att_q_chunks` |
| inner-loop | sets a loop inside the kernel | `_dividing_sdpa_chunk_size` (k_chunk), `_trimul_in0_block_w` |
| bytes | bounds an L1 or DRAM peak | `row_block`, `pwa_depth_block`, `l1_resident_budget_bytes`, `atom_pair_budget_bytes`, ESMC's pair-FFN row block, ESMFold2's OPM fallback rows, Boltz-2's host row block |
| host | blocks work on the CPU | RFD3's `_ATTN_ROW_BLOCK` (measured at 8 and 16 CPU threads) |
| guard | avoids a hang, not a tuning knob | Protenix's `PAIRCOND_MM_NARROW_MAX_TILES` |

**Of the three occupancy members, two are not exposed.** The band and the tri-att ladder both sit
on the triangle-attention path, where `work = seq * heads` is 4096 units at 512 aa: that path is
oversubscribed by 57x on a 72-core grid and cannot be starved by a chunk size. `_tri_att_q_chunks`
is additionally L1-discovered rather than grid-declared, and its own docstring says so.

**So the audit's answer is the opposite of the brief's expectation, and it is the more useful one:
the occupancy-deciding surface in `tt_bio` is small, and nearly all of it already derives from the
device.** `_triangle_mul_program_config` takes `per_core_M/N` from `gx, gy`; `_fp32_softmax_core_budget`
clamps to the grid; `sdpa_generic.plan`, `rfd3_bias` and `reblock_permute._split_plan` all read
`num_cores` off the device. `_capped_sdpa_chunk_size` was the one site on that surface that did
not, and it was starved. It is not one of many.

**The byte budgets are grid-blind on purpose and correctly.** `l1_resident_budget_bytes` and
`atom_pair_budget_bytes` read the part's own L1 and DRAM, which is the right resource; they show
up as blind here only because L1 and DRAM do not move when the core count does.

### The second finding the audit turned up: `q_chunk` has TWO pickers with opposite policies

`_grid_q_chunk` takes the NARROWEST chunk that still fills the grid in one pass. `_tri_att_q_chunks`
offers the WIDEST chunk first and narrows only on an L1 refusal. They set the same kernel parameter
on the same op family and neither knows the other exists. Both are right where they are -- the
diffusion SDPA is starved and the tri-att SDPA is oversubscribed -- but "widest" and "narrowest"
are the two ends of one curve, and nothing in the engine writes down which end a call is on. That
is the unification this row can name and has not built: one picker, `units = work * padded / qc`
against the core count, with the tri-att ladder's L1 discovery as its ceiling.

## 2. The ladder, re-run on this build on three grids

Inherited nothing: `rule_ladder.py` is the parent's instrument, vendored verbatim so the two are
comparable, and every number below was measured on this branch. Synthetic operands of the shipped
shapes and dtypes, `torch.equal` against the shipped config on every rung.

### WH 8x9 = 72
| S | heads | shipped | rule | measured best | rule/shipped | best/shipped | bit-exact |
|---|---|---|---|---|---|---|---|
| 320 | 8 | 256 (365.90 us) | **64** (217.8) | 160 (153.50) | **1.68x** | 2.3837x | True |
| 320 | 16 | 256 (370.20 us) | **160** (163.2) | 160 (163.20) | **2.2684x** | 2.2684x | True |
| 512 | 8 | 256 (113.10 us) | **64** (58.5) | 64 (58.50) | **1.9333x** | 1.9333x | True |
| 512 | 16 | 256 (126.40 us) | **128** (96.8) | 128 (96.80) | **1.3058x** | 1.3058x | True |
| 768 | 8 | 256 (218.40 us) | **96** (113.3) | 96 (113.30) | **1.9276x** | 1.9276x | True |
| 768 | 16 | 256 (255.50 us) | **192** (196.9) | 192 (196.90) | **1.2976x** | 1.2976x | True |
| 1024 | 8 | 256 (179.80 us) | **128** (139.5) | 128 (139.50) | **1.2889x** | 1.2889x | True |
| 1024 | 16 | 256 (257.60 us) | **256** (257.6) | 256 (257.60) | **1.0x** | 1.0x | True |
| 1536 | 8 | 256 (462.20 us) | **192** (326.0) | 192 (326.00) | **1.4178x** | 1.4178x | True |
| 1536 | 16 | 256 (665.70 us) | **256** (665.7) | 192 (617.10) | **1.0x** | 1.0788x | True |
worst rule/shipped: 1.0  optimum hit: 8 / 10

### BH 11x10 = 110
| S | heads | shipped | rule | measured best | rule/shipped | best/shipped | bit-exact |
|---|---|---|---|---|---|---|---|
| 320 | 8 | 256 (306.20 us) | **32** (104.2) | 32 (104.20) | **2.9386x** | 2.9386x | True |
| 320 | 16 | 256 (305.90 us) | **64** (137.2) | 32 (111.60) | **2.2296x** | 2.741x | True |
| 512 | 8 | 256 (98.90 us) | **64** (68.7) | 32 (63.00) | **1.4396x** | 1.5698x | True |
| 512 | 16 | 256 (100.30 us) | **128** (88.0) | 32 (80.10) | **1.1398x** | 1.2522x | True |
| 768 | 8 | 256 (173.80 us) | **64** (102.2) | 32 (100.40) | **1.7006x** | 1.7311x | True |
| 768 | 16 | 256 (178.40 us) | **128** (132.4) | 128 (132.40) | **1.3474x** | 1.3474x | True |
| 1024 | 8 | 256 (236.40 us) | **128** (167.8) | 32 (147.40) | **1.4088x** | 1.6038x | True |
| 1024 | 16 | 256 (245.50 us) | **256** (245.5) | 64 (222.80) | **1.0x** | 1.1019x | True |
| 1536 | 8 | 256 (357.60 us) | **128** (256.7) | 128 (256.70) | **1.3931x** | 1.3931x | True |
| 1536 | 16 | 256 (385.90 us) | **256** (385.9) | 256 (385.90) | **1.0x** | 1.0x | True |
worst rule/shipped: 1.0  optimum hit: 4 / 10

### BH 13x10 = 130
| S | heads | shipped | rule | measured best | rule/shipped | best/shipped | bit-exact |
|---|---|---|---|---|---|---|---|
| 320 | 8 | 256 (326.60 us) | **32** (113.6) | 32 (113.60) | **2.875x** | 2.875x | True |
| 320 | 16 | 256 (328.30 us) | **64** (151.9) | 32 (126.20) | **2.1613x** | 2.6014x | True |
| 512 | 8 | 256 (144.50 us) | **32** (97.6) | 32 (97.60) | **1.4805x** | 1.4805x | True |
| 512 | 16 | 256 (149.50 us) | **64** (104.9) | 32 (100.00) | **1.4252x** | 1.495x | True |
| 768 | 8 | 256 (275.40 us) | **64** (153.6) | 32 (136.20) | **1.793x** | 2.022x | True |
| 768 | 16 | 256 (283.10 us) | **96** (176.8) | 96 (176.80) | **1.6012x** | 1.6012x | True |
| 1024 | 8 | 256 (384.60 us) | **64** (203.7) | 32 (184.60) | **1.8881x** | 2.0834x | True |
| 1024 | 16 | 256 (387.90 us) | **128** (268.7) | 64 (264.60) | **1.4436x** | 1.466x | True |
| 1536 | 8 | 256 (598.00 us) | **96** (354.1) | 32 (331.10) | **1.6888x** | 1.8061x | True |
| 1536 | 16 | 256 (608.10 us) | **192** (515.8) | 96 (496.20) | **1.1789x** | 1.2255x | True |
worst rule/shipped: 1.1789  optimum hit: 3 / 10

**Wormhole reproduces the parent on a different chip.** whglx card 24 against the parent's card 12:
512/16 reads 126.40 -> 96.80 us here and 127.50 -> 97.60 there, 0.9 % apart, same pick, same verdict
(8/10, worst 1.0000x).

**Blackhole is `pc` card 0, a p150a, at its own 13x10 and forced to 11x10 with `TT_BIO_FORCE_GRID`.**
That is not the published cell's part -- the cell is a p300c chip on qb2, and all four qb2 chips
were held when this ran (cards 1, 2, 3 leased, card 0 running a full parity gate). What it is: real
Blackhole silicon at the cell's own core count. `pc` card 0 is the known-faulty matmul card
(`pc-card0-512aa-fold-nondeterminism`), so **its bit-exact column is not evidence** and the parity
claim in this row rests on Wormhole. The fault is a wrong value, not a slow one; timing stands.

## 3. What the arch split says about the mechanism

| | 72 cores (WH) | 110 cores (BH) | 130 cores (BH) |
|---|---|---|---|
| rule's pick at 512/16 | 128 | 128 | 64 |
| rule / shipped at 512/16 | **1.3058x** | **1.1398x** | **1.4252x** |
| rule picks the optimum | 8/10 | 4/10 | 3/10 |
| worst rule / shipped | 1.0000x | 1.0000x | 1.1789x |
| direction of every miss | narrower would win | narrower would win | narrower would win |

The one-pass rule is a **floor on both architectures and a tight one only on Wormhole.** On
Blackhole the curve keeps falling past the point where the grid is full: at 320 tokens and 8 heads,
`q_chunk` 32 is 104.2 us against 256's 306.2, and 32 puts 80 units on 110 cores -- under one pass,
so the rule agrees there -- but at 512/16 the optimum 32 is 256 units, **two and a bit passes**, and
it still beats the one-pass 128 by 1.10x. A pass count is therefore not the cost, and the reason the
rule survives anyway is that it never goes to the wrong side of the peak.

The mechanism that would close it is the one the parent named and did not fit: a narrower chunk
re-reads K and V once per chunk, so the cost is `passes(qc) * qc * a + units(qc) * kv_bytes * b`,
and `b` is smaller on the part with more DRAM bandwidth. Blackhole tolerating narrower chunks than
Wormhole is exactly what that predicts. **This row does not fit it either** -- 30 rungs on three
grids is enough to see the sign and not enough to fit two coefficients across two architectures --
and says plainly what it leaves on the table: **1.10x-1.25x at 110 cores, 1.02x-1.21x at 130.**

## 4. Across models: the pick moves on every model on the path

`model_pick_census.py` wraps the picker and records, for every SDPA call a real run makes, the
`(q_len, k_len, work)` it arrives with. Run on whglx card 24, 8x9 = 72 cores, flag OFF so the
census sees the shipped run:

| model | run | calls | q_len | work | shipped | rule | occupancy |
|---|---|---|---|---|---|---|---|
| ESMC-300M | `embed examples/prot.fasta` | 30 | 128 | 15 | 128 | **32** | 0.208 -> **0.833** |
| SaProt-650M | `saprot examples/prot.fasta` | 33 | 128 | 20 | 128 | **64** | 0.278 -> **0.556** |

**Every call of both models gets a different chunk**, and both sit at a quarter of the grid under
the shipped constant. Neither is Boltz-2: this is the first evidence the lever is not a Boltz-2
patch.

### And the shapes they ask at are below what the ladder can measure

ESMC and SaProt call at 128 tokens, where the op is 13-16 us. The inherited ladder settings (20
reps) read **0.9322x** there -- the first loss on 35 rungs, on a production shape. **It is not a
loss.** Three repeats at 60 reps of the same configuration on the same card:

    128 tokens, 15 heads, 110 cores      shipped arm        rule arm      ratio
      repeat 1                            15.80 us          11.9 us       1.3277x
      repeat 2                            15.90 us          27.5 us       0.5782x
      repeat 3                            15.90 us          12.2 us       1.3033x

The shipped arm is stable to 0.6 % across the three and the rule's arm swings 2.3x. A 13 us op is
below what this harness resolves; the parent's ladder never ran under 320 tokens, so nothing had
tested that. **At 256 tokens the same instrument is stable and the win is large:**

| shape (110 cores) | repeat 1 | repeat 2 | repeat 3 |
|---|---|---|---|
| 256 tokens, 15 heads | 1.4326x | 1.4062x | 1.4397x |
| 256 tokens, 20 heads | 1.3906x | 1.3709x | 1.3953x |

**So the PLM exposure is real and its size is not yet measured.** What is established: the pick
moves on 63 of 63 calls across two models that are not Boltz-2, the occupancy it leaves rises from
~0.25 to 0.56-0.83, and at the nearest resolvable size the win is 1.37x-1.44x. What is owed is an
instrument that can resolve a 13 us op.

## 5. What is owed

1. **The Blackhole STEP on the published cell's part.** All four qb2 chips were held (three
   leased, and card 0 running `b2z2-union-gate-ship`'s parity gate), so the Blackhole leg here is
   an op ladder on a p150a, not a step on the p300c. **Not projected into any headline.**
2. **Boltz-2's own pick census, and a caveat the merge decision needs.** A 512 aa fold through
   `tt_bio.main` records **zero** calls, and the engine's own branch counter agrees:
   `B2_TOKEN_DIT_SDPA_STATS` reads **[0, 0]** -- served zero, declined zero -- so
   `AttentionPairBias.__call__` never ran in the process the hook is in. The fold runs in spawned
   device workers and a `sitecustomize` on PYTHONPATH reached only the parent. **This is census
   plumbing, not a model finding**, but it means something real is still unshown: every Boltz-2
   number for this lever, the parent's 1.01831x included, was taken on a step GRABBED out of a
   truncated precursor fold, and the branch it lives on is gated on `seq_mask is None`. Nobody has
   yet demonstrated that a production fold takes it. That is one counter read away and it should be
   read before the flag is flipped.
3. **A resolvable instrument under ~30 us**, without which the PLM shapes cannot be priced.
4. **The two-term rule** that would take the remaining 1.10x-1.25x on Blackhole.
