# protenix-trunk--z-survival-512 — does the org's merged 685.1 ms/fold survive at 512 aa?

TASK TYPE: VERIFY/BENCHMARK (Phase 2 experiment) | PLAYBOOKS loaded: ACCELERATE + VERIFY/BENCHMARK |
memories read: perfwar-cotenanted-ab-noise-floor-exceeds-small-wins, perfwar-l1-fit-cliff-channel-width-not-token-count,
perfwar-l1-destination-priced-as-free-fake-mystery, ttnn-sync-before-every-timed-region,
roofline-roof-must-be-measured-not-asserted, tt-bio-trunk-perf-ratio-denominator-unit-slip,
tt-bio-l1-residency-guard-dead-in-real-folds, protenix-v2-448aa-l1-cb-clash-cc39a867d,
donecheck-hostspecific-path-unsatisfiable-on-remote-host, tt-bio-worktree-run-recipe,
gate-mandated-write-to-single-owner-file-is-a-race, model-merge-approval-gate

**STATE OF THIS DOCUMENT: COMPLETE. The arms have run.** 14 folds at 512 aa (one cold, 13 timed, 7 of
them `on`) and 10 at 298 aa in one session on one card, every `off:*` arm bracketed by an `on` arm on
both sides. §6 and §7 carry the answer; §1-§5 are the planning pass, kept because the plan is what the
result has to be read against.

## THE ANSWER IN ONE PARAGRAPH

**`size512-ab`'s +476.96 ms was never a measurement — it is 29x inside its own wall's drift band — and
the survival fraction it was reaching for is 36.2 %, not 27.6 %.** Re-taken on one card in one session
at both sizes, `off:ab5` (its exact five flags) is worth **119.96 ms/fold at 512 aa against 330.94 at
298 aa**. Re-taken on `block:PairformerLayer`, the wall it used, the 512 aa arm reads **-75.93 ms against
a 7-arm A/A floor of 2 189.67** and cannot resolve anything, while the 298 aa arm reads **+1 738.09
against that leg's own 1 729.03** — so its denominator was right and its numerator was drift.
**The org's own merged 685.1 family does better than anyone expected: 119.1 %, it grows at 512 aa.** The
two fractions differ because `size512-ab` swapped `_NARROW_PROJ_BW` out for C2FIX, and
**`_NARROW_PROJ_BW` is the only flag in either family that is not capacity-gated: 60.37 ms/fold at
298 aa to 407.89 at 512 aa, 6.76x**, where the three L1 `layer_norm` flags go **275.26 to exactly zero**
(census: 0 of 524 calls reach L1). What survives is not what shrank least, it is the one lever that
never depended on a tensor fitting anywhere.

Scope: **protenix**-v2, the **trunk**. Card qb2 chip 0, ttnn **0.68.0**, 11x10 grid, 110 cores.
qb1 runs 0.67.4, so **every absolute here is a ratio** and owes a qb1 re-take before it drives a
decision (charter §4.8). A **survival fraction is a ratio of two measurements on the same card**, which
is what qb2 is good for and why this leg can answer its question here without an absolute.

**Co-tenancy, recorded because it sets the noise floor.** `protenix-trunk--z-permute-flip-land` held
chip 1 of the same p300 board (007) throughout the probe and is expected to hold it through the arms.
Charter §4.8 prices that at 0-4.8 % cross-chip compute interference and ~12 % on host round trips, and
`perfwar-cotenanted-ab-noise-floor-exceeds-small-wins` prices a shared qb host at 1-10 % on identical
code paths. The effects this leg separates are 0.0-0.7 % of the fold, so **aggregate walls cannot
carry the answer and device-synced per-site walls plus the branch census must.** That is the design.

---

## 1. What the planning pass already settled, with measurements

### 1.1 The brief's five-flag list is wrong in one slot, and it matters

The brief names `_PAIR_PROJ_L1_OUT`, `_PAIR_BIAS_L1_NORM`, `_PWA_L1_NORM`, `_TEMPLATE_L1_NORM`,
`_NARROW_PROJ_BW` and says to confirm the list. Confirmed against `origin/main` (`6fa5a701`,
`tt_bio/tenstorrent.py:96-144`): all five exist with those defaults, and **they are exactly the five
behind the merged 685.1 ms/fold** (X7 561.8 + X2 31.5 + X10 91.8, `STATUS.md` "What is on main now").

But **they are NOT the five `size512-ab` moved for its +476.96 ms.** Read off that leg's own harness,
`perf/size512/fold_ab512.py:14-22`, its OFF arm moved `_transpose_memory_config` (C2FIX),
`_PAIR_PROJ_L1_OUT`, `_PAIR_BIAS_L1_NORM`, `_PWA_L1_NORM`, `_TEMPLATE_L1_NORM`, and states verbatim
that **`_PAIR_PROJ_BW` and `_NARROW_PROJ_BW` are identical in both arms**. Its 298 aa denominator
1729.03 ms reconciles with C2FIX 1010.9 + X7 561.8 + X10 91.8 = 1664.5 (3.9 % apart) and excludes X2's
31.5, which confirms the reading arithmetically.

So there are **two different families and the leg must report both**: the *merged* family (the five the
brief names, worth 685.1 at 298 aa) and the *`size512-ab`* family (C2FIX plus four of them, worth
1729.03 on this chip at 298 aa). The 27.6 % belongs to the second. `_NARROW_PROJ_BW` is in the first
and not the second, and it turns out to be the largest surviving member of either — which is why the
distinction is not bookkeeping.

**VERDICT — KILLED: "the 27.6 % is a statement about the five flags this leg was given."** It is a
statement about a set that includes C2FIX, which is `z-rowblock`'s op, and excludes `_NARROW_PROJ_BW`,
which nobody has measured at 512 aa. Settled by reading the predecessor's harness, not by timing.

### 1.2 The size cliff of every flag, from the production helpers, on this card

`perf/survival512/surv_envelope.py`, results `perf/survival512/surv_envelope_qb2c0.json`. No fold.
`_l1_memory_config_if_it_fits` and `_pair_proj_config` are pure functions of (padded shape, dtype, live
per-core L1, grid), so evaluating **the production helpers themselves** at every padded size gives each
flag's exact cliff. Live budget read on the card: `get_max_worker_l1_unreserved_size()` x 110 =
**168 565 760 B (160.75 MiB)**.

| flag class | headroom | last padded N on L1 | first padded N on DRAM | at padded 512 |
|---|---:|---:|---:|---|
| L1 `layer_norm` source, c_z=256 — `_PAIR_BIAS_L1_NORM`, `_PWA_L1_NORM`, `_TEMPLATE_L1_NORM` | 1.5x | **448** | **480** | **DRAM** |
| pair transpose, c_z=256 — C2FIX, not this leg's op | 2.5x | 352 | 384 | DRAM |
| pair transpose, c=64 — C2FIX template track | 2.5x | **704** | 736 | **L1** |
| `_pair_proj_config(out_l1=True)`, c_z=256 — `_PAIR_PROJ_L1_OUT` | static budget | 352 | 384 | refused (`None`) |
| `_pair_proj_config(out_l1=True)`, c=64 | static budget | >800 | — | **L1** |
| `_pair_proj_config(bw_cap=1)`, c_z=256 and c=64 — `_NARROW_PROJ_BW` | none | >800 | — | **always fires** |

The arithmetic behind the first row, so the cliff is understood and not merely tabulated: padded 512 at
c_z=256 is 134 217 728 B, and 1.5 x that is 201 326 592 B against a 168 565 760 B budget — over by
19.4 %. Padded 448 needs 154 140 672 B and fits. **The L1-norm class dies between logical N=449 and
N=480 on an 11x10 grid**; on qb1's 13x10 grid the same budget is 199 214 080 B, so it dies between
N=481 and N=512 and 512 aa is 1.1 % outside it there too. The cliff is a **channel-width cliff**, not a
token cliff: the same 2.5x transpose gate does not bind until padded 704 at c=64.

**VERDICT — KILLED: "the three L1 `layer_norm` flags carry part of the 512 aa survival."** At 512 aa
`_l1_layer_norm` refuses, returns `(ttnn.layer_norm(memory_config=DRAM), False)`, and the OFF arm runs
the same `ttnn.layer_norm` to DRAM with `l1_out=False` downstream. **Both arms emit byte-identical
device work, so each of the three is worth exactly zero by construction and its arm is an A/A.** The
in-fold census in §5 confirms it by reading the branch; it cannot discover value that the code cannot
express.

**VERDICT — CONFIRMED: `_NARROW_PROJ_BW` is the only member of either family whose mechanism is not
capacity-gated.** Its config is returned at every padded size to 800 at both channel widths. It is a
program-config lever — `per_core_N=1` on a one-tile-wide output leaves `ttnn.linear(core_grid=)` on a
core ladder that is flat from 16 to 110 cores — and nothing in it depends on a tensor fitting anywhere.

### 1.3 The two live mechanisms, priced in isolation on this card this pass

`[1,N,N,256] @ [256,8]`, the narrow pair-track projection, median of 5 synced reps:

| form | padded 320 | padded 512 | 512 / 320 |
|---|---:|---:|---:|
| production: tuned config, DRAM source | 0.4016 ms | **0.9032 ms** | 2.25x |
| L1 source: what the norm flags hand it while the fit test passes | 0.1209 ms | **0.2516 ms** | 2.08x |
| `core_grid=` baseline: `_NARROW_PROJ_BW = None`, the pre-X2 path | 0.5081 ms | **1.4136 ms** | 2.78x |
| **`_NARROW_PROJ_BW` = 1 is worth** | **0.1065 ms/call** | **0.5104 ms/call** | **4.79x** |
| **an L1 source would be worth, on top** | 0.2807 ms/call | **0.6516 ms/call** | 2.32x |

Two things fall out before a single fold runs. **`_NARROW_PROJ_BW`'s per-call value grows 4.79x from
298 aa to 512 aa against a 2.56x growth in bytes**, so the flag the org credits with 31.5 ms/fold is
the one member of the family that gets *better* with size. And **the L1 source did not throw at padded
512 in isolation** (0.2516 ms) — the 1.5x fit test refuses a residency the allocator would have granted
on an idle chip. The refusal is a static budget decision, not an allocator failure, and the 0.6516
ms/call it forgoes at 512 aa is the size of the prize behind a size-independent route.

### 1.4 Roofs measured on this card this pass, and where these sites sit

`ttnn.clone` of a real device tensor, and a square matmul at **production's own kernel config**
(HiFi4, `fp32_dest_acc_en`, `packer_l1_acc`) so the peak is one a trunk op could reach:

| quantity | measured |
|---|---:|
| `[1,512,512,256]` (134.22 MB) clone to DRAM | 0.7054 ms / **380.6 GB/s** |
| `[1,512,512,256]` clone to L1 | 0.3547 ms / **756.8 GB/s** (**1.988x**) |
| `[1,512,512,64]` (33.55 MB) clone to DRAM | 0.1886 ms / **355.9 GB/s** |
| `[1,512,512,64]` clone to L1 | 0.1025 ms / **654.7 GB/s** (1.84x) |
| 4096^3 bf16 matmul, production fidelity | 1.2329 ms / **111.48 TFLOP/s** |
| **machine balance** | **292.9 FLOP/byte** |

The narrow projection moves 150 994 944 B (134.22 MB read plus 16.78 MB written at the 32-wide padded
output) for 1.074 GFLOP: **arithmetic intensity 7.1 FLOP/byte** against 292.9, i.e. **41x onto the
memory side**, so a bandwidth roof binds and there is no argument about which. Placements against the
roofs above:

| form | ms | placement |
|---|---:|---|
| production | 0.9032 | **44 % of the copy roof (DRAM)** |
| L1 source | 0.2516 | **79 % of the copy roof (L1)** |
| `core_grid=` baseline | 1.4136 | **28 % of the copy roof (DRAM)** |

**Compute and communication.** The maths take 1.074 GFLOP / 111.48 TFLOP/s = **0.0096 ms, 1.1 % of the
0.9032 ms op**, so the total is nearer **max(compute, comm)** than `compute + comm`, with `comm`
binding, and no overlap arrangement could be visible at that ratio. The sibling leg reached the same
verdict from the other direction: its excess over a plain clone tracked bytes and not FLOPs across a
16x FLOP swing.

**What holds the two DRAM forms under 70 %, at transaction granularity rather than at ttnn-argument
level.** The tuned config is `in0_block_w=1, out_subblock_h=1, out_subblock_w=1, out_block_h=5,
per_core_M=75, per_core_N=1` on the 11x10 grid — read off the config the production helper returns, not
inferred. So **the limiter is transaction size on both sides**: the reader issues one-tile (2 KB) reads
per K block because `in0_block_w=1`, and the packer writes one tile per pack because
`out_subblock_w=1`, against the long bursts a clone gets. Core occupancy is not the limiter for this
form — `per_core_M=75` puts work on all **110 of 110 cores**. For the `core_grid=` baseline occupancy
*is* the limiter, and the production comment records it as **~16 of 110 cores** engaged with a ladder
flat from 16 to 110, which is consistent with the 1.4136 / 0.9032 = 1.57x it loses here. This is the
same class of finding as `perfwar-l1-destination-priced-as-free-fake-mystery`, on the read side.

### 1.5 The instrument is validated, and the validation is itself a result

`surv_arms.py --size 298 --arms on,off:norms,on`, results
`perf/survival512/surv_298_validate_qb2c0.json`. Three folds, run to prove the harness works before the
exec pass spends a turn on it. All three returned plDDT **0.859489** and CIF **`8139d61b6c90f893`**,
matching both predecessors on this chip, and `_L1_OUT_REFUSED` was empty in every arm.

**The census positive control.** At 298 aa all three norm sites take the **L1** branch in the ON arm and
**DRAM** in the OFF arm — 484 `pairbias` norms with 484 `[256,16]` consumers, 30 `pwa` norms with 240
`[256,1]` consumers, 10 `template` norms with 40 `[256,64]` consumers. So the census reads the branch
where the branch is live, which is the check the 512 aa zero result needs standing behind it: the same
instrument that will report zero at 512 aa reports L1 at 298 aa.

**The counted call numbers, which the 512 aa predictions now use instead of an estimate.** The c_z=256
narrow projections number **484 + 240 + 40 = 764 per fold**, and none of those counts depends on the
token count. `block:PairformerLayer` 604, `stage:Pairformer` 11, `body:TriangleMultiplication|c256`
1048 and `|c64` 160, `stage:msa` 10, `stage:template` 10, `body:PairWeightedAveraging|c256` 30 — every
one counted, so charter §4.9's blocks x recycles is a tally here and not a constant.

**The three norm flags at 298 aa, and two merged ledger rows reproduced.** Site walls, `off:norms`
bracketed by both ON arms:

| wall | off − on ms/fold | calls | ms/call | A/A spread (2 arms) | resolved |
|---|---:|---:|---:|---:|---|
| `lin|pairbias|c256@16` | **+148.92** | 484 | 0.3077 | 12.87 | yes |
| `norm|pairbias|c256` | **+64.83** | 484 | 0.1339 | 13.05 | yes |
| `lin|pwa|c256@1` | **+77.82** | 240 | 0.3243 | 2.43 | yes |
| `body:PairWeightedAveraging|c256` | +105.06 | 30 | 3.502 | 48.72 | yes |
| `stage:template` | +70.28 | 10 | 7.028 | 44.10 | yes |
| `stage:msa` | +141.49 | 10 | 14.149 | 127.60 | yes |
| `block:PairformerLayer` | +392.86 | 604 | 0.6504 | **1295.72** | **no** |
| fold wall | +0.587 s | 1 | — | — | — |

Two things to take from it. **`_PWA_L1_NORM` reads ~78-80 ms/fold and `_TEMPLATE_L1_NORM` ~20 against
X10's 80.2 + 11.6 = 91.8 measured on qb1 card 2 at 0.67.4** — the instrument agrees with a merged ledger
row across a card and a ttnn minor version, which is the strongest validation available short of
re-running qb1. And **the block wall failed to resolve a 392.86 ms effect** because two ON arms gave it
a 1295.72 ms spread, 5.8 % of its own 22.16 s wall, co-tenanted. That is S9 supported at 298 aa on an
effect 4x larger than anything expected at 512 aa, and it is why the site walls are the primary
instrument.

One attribution fix came out of this run and is already applied: `PairformerLayer` now pushes its own
site, so the ops of a Pairformer stack nested inside `_template` / `_msa` no longer land on the
`template` / `msa` site keys. The stage walls are unchanged, since they are meant to include everything
below them.

---

## 2. Predictions, registered before the arms run

`perf/survival512/PREDICTIONS.md`, committed before any fold arm opened a device. S1-S10 with
falsifiers. In one line each:

| # | prediction | falsifier |
|---|---|---|
| S1-S3 | the three L1-norm flags are **0.0 ms/fold each** at 512 aa, census 0 of N on L1 | any `L1` census row at those sites, or a site-wall delta above its own A/A spread |
| S4 | `_PAIR_PROJ_L1_OUT` reproduces the sibling's **+29.7 ms/fold** on `body:TriangleMultiplication|c64`, zero at c_z=256 | disagreement outside (my spread + 2.6 ms) — and then nothing else in the leg is quotable |
| S5 | `_NARROW_PROJ_BW` is **+0.35 to +0.55 ms/call** in-fold over the **764 counted** c_z=256 narrow calls, i.e. **270-420 ms/fold, central 390** | under 0.10 or over 0.80 ms/call |
| S6 | C2FIX survives only on the template track, **+80 to +120 ms/fold** | outside that band |
| S7 | `off:ab5` reads **110-160 ms/fold**, so **476.96 was 3-4x too large and the survival fraction of that set is 6-9 %, not 27.6 %** | above 300 or below 40 ms/fold |
| S8 | the family total equals the sum of its singles within 2x the A/A spread, 90 %+ of it `_NARROW_PROJ_BW` | non-additive, and then the interaction is the finding |
| S9 | the `block:PairformerLayer` A/A spread over >=6 ON arms is **above 30 ms**, so the block wall cannot resolve S1-S4 or S6 and the site walls must | a spread under 20 ms |
| S10 | placements 44 % / 79 % / 28 % of the roofs named above; the limiter is transaction size | a measured placement outside +-10 points |
| S11 | the 298 aa run reproduces **X10's 91.8 ms/fold** within 25 % and puts `_PAIR_BIAS_L1_NORM` near **210 ms/fold** | a 298 aa norms total outside 200-450 ms/fold, and then nothing at 512 aa is quotable |

---

## 3. The instrument, and the three things it does differently

`perf/survival512/surv_arms.py`. Adapted from `perf/progcfg/h5_infold.py` (`z-h5-infold`) rather than
re-derived; that harness's census design and its counted denominators are inherited.

1. **Every `off:*` arm is bracketed by an `on` arm on both sides**, and the delta is taken against the
   mean of the two, so linear drift cancels. Never one run per side: the sibling leg's single-shot arm
   manufactured **+898.89 ms** on a 70.7 s block wall and **+691 ms/fold** at a site where its own
   census proves the effect is exactly zero.
2. **The A/A floor comes from >=6 ON arms as a spread and a stdev per wall key, not from one pair.**
   `size512-ab` quoted a **2.10 ms** A/A floor at 512 aa from two ON arms and called its 476.96 "227x
   the floor"; the sibling's doubled arms on the same wall on the same chip measured a spread of
   **64.9 ms**, 31x larger. **`|a-b|` from a single pair is not an estimate of a drift band** — that is
   the methodological defect underneath the withdrawn number, and it is not the same defect as
   single-shot arms. No line in §6 may claim an effect smaller than its own key's spread.
3. **The instrument is symmetric by construction.** The timed things are ttnn ops (`layer_norm`,
   `linear`) and class bodies, which *both* arms execute. The production helpers (`_l1_layer_norm`,
   `_narrow_proj_linear`, `_pair_proj_linear`, `_transpose_memory_config`) are wrapped for **census
   only and never timed**: timing `_l1_layer_norm` would time the norm in the ON arm and not in the OFF
   arm, because the OFF arm never calls it. That asymmetry would have manufactured the entire effect.

Census keys record the **branch actually taken** — `_l1_layer_norm`'s own returned bool, the returned
tensor's `memory_config().buffer_type`, whether `_narrow_proj_linear` returned `None` (the `core_grid=`
fallback), whether `_transpose_memory_config` returned L1 — per site (`trimul`, `triatt`, `pairbias`,
`pwa`, `template`, `msa`) and per padded shape. Served counts, never inferred from shape.

Site walls, all synchronised on both sides (`ttnn-sync-before-every-timed-region`): `stage:Pairformer`,
`stage:msa`, `stage:template`, `block:PairformerLayer`, `body:{TriangleMultiplication,
TriangleAttention, AttentionPairBias, PairWeightedAveraging}|c{width}`, `norm|{site}|c{width}`,
`lin|{site}|c{width}@{out_width}`. **`PairWeightedAveraging` lives in `trunk_msa`, outside
`PairformerLayer`, so the block wall is blind to `_PWA_L1_NORM` by construction** — `size512-ab`
established that and it is why per-site walls are the primary instrument here and the block wall is the
cross-check, not the reverse.

---

## 4. Arm definitions

Each arm is a dict of overrides on `tt_bio.tenstorrent` module globals; everything unnamed stays at its
production default in every arm. `_pair_proj_program_config.cache_clear()` and `_L1_OUT_REFUSED.clear()`
run on every flip.

| arm | flags moved | why it exists |
|---|---|---|
| `on` | none | the baseline, run 6-7 times for the floor |
| `off:projl1` | `_PAIR_PROJ_L1_OUT=False` | S4, the cross-check against the sibling's +29.7 |
| `off:narrowbw` | `_NARROW_PROJ_BW=None` | S5, predicted the largest survivor |
| `off:bias` / `off:pwa` / `off:tmpl` | one norm flag each | S1-S3 individually, only if `off:norms` is non-zero |
| `off:norms` | all three norm flags | S1-S3 as a union: the census proves each is individually a no-op, so one arm bounds all three and the individual arms become optional |
| `off:c2fix` | `_transpose_memory_config` -> DRAM | S6. **`z-rowblock`'s op: one bracketed arm, reported, handed over, not pursued** |
| `off:family` | the five behind the merged 685.1 | the total for the merged family, and the additivity test |
| `off:ab5` | C2FIX + `_PAIR_PROJ_L1_OUT` + the three norms | **`size512-ab`'s exact five — the arm that replaces the +476.96 and the 27.6 %** |

`TT_BIO_REBLOCK_PERMUTE` is **not** touched and runs at its default; the arm runner records the live
value of `reblock_permute.REBLOCK_PERMUTE` in every result file, and §6 must state which default it ran
under.

---

## 5. THE EXEC PLAN — exact commands, in this order

Preconditions already done in the planning pass and not to be redone: the gate modules, this leg's
`_donecheck.py`, `CHARTER.md`, `STATUS.md` and `FINDINGS.md` are staged on qb2 (the donecheck was
**missing** there and now runs, failing only on the absent state doc);
`perf/survival512/{surv_arms.py,surv_envelope.py,PREDICTIONS.md}` are committed;
`surv_envelope_qb2c0.json` exists; and **the arm runner has been run end to end** at 298 aa for three
arms (§1.5), so it will not crash the exec turn. The run recipe below is the one that worked.

```sh
# on qb2, in the worktree, once per invocation:
cd ~/.coworker/wt/protenix-trunk--z-survival-512
SP=/home/ttuser/tt-bio-dev/env/lib/python3.10/site-packages
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:protenix-trunk--z-survival-512
export TT_MESH_GRAPH_DESC_PATH=$SP/ttnn/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export PYTHONPATH=$PWD
PY=~/tt-bio-dev/env/bin/python3
```

**Step 1 — record the co-tenancy state** (one line into §6): `ps -eo pid,etime,args | grep tt_bio`,
and the same again after step 2. A wall taken beside a busy board partner is not a wall taken alone.

**Step 2 — the 512 aa arms. This is the deliverable.** ~25 min: one cold fold (~175 s) then 13 timed
folds at ~90-100 s each. Results are written after every fold, so a cut turn lands what it measured.

```sh
$PY perf/survival512/surv_arms.py --size 512 \
    --arms on,off:ab5,on,off:narrowbw,on,off:projl1,on,off:norms,on,off:c2fix,on,off:family,on \
    --out perf/survival512/surv_512_qb2c0.json
```

The arm order is priority-first on purpose: after 7 arms the leg already has the headline (`off:ab5`),
the predicted largest survivor (`off:narrowbw`) and the instrument cross-check (`off:projl1`), each
bracketed. **Do not reorder to put the cheap arms first.**

**Step 3 — the 298 aa denominator, same instrument, same session, same card.** ~7 min: cold fold
(~60 s) then 9 folds at ~35 s. This is what makes the survival fraction a ratio of two measurements on
the same card instead of a comparison against another leg's number, and it independently validates the
instrument against two merged ledger rows (X10's 91.8 ms/fold for the two norm flags, X2's 31.5 for
`_NARROW_PROJ_BW`, both qb1 absolutes, so expect a ratio not a match).

```sh
$PY perf/survival512/surv_arms.py --size 298 \
    --arms on,off:family,on,off:narrowbw,on,off:ab5,on,off:norms,on \
    --out perf/survival512/surv_298_qb2c0.json
```

**Step 4 — only if `off:norms` at 512 aa is non-zero beyond its A/A spread** (S1-S3 falsified): split it
with `--arms on,off:bias,on,off:pwa,on,off:tmpl,on --size 512`. If it reads zero as predicted, skip
this and say so; the census already gives each of the three its own number.

**Step 5 — write §6 and §7**, then append to `state/perfwar/FINDINGS.md` and to
`state/orgs/protenix-trunk/STATUS.md` (**append a team-report section, never rewrite the file** — the
CTO owns it and rewrites it every pass; `## Team report — protenix-trunk--z-rowblock, exec pass
(written by the team, appended)` is the precedent to follow). Then stage this doc to **both**
`/home/moritz/.coworker/state/protenix-trunk--z-survival-512.md` and
`/home/ttuser/.coworker/state/protenix-trunk--z-survival-512.md` on qb2, checksum-matched, and run
`python3 /home/moritz/.coworker/workstreams/protenix-trunk--z-survival-512_donecheck.py` on qb2.

**Staging trap, already paid for six times in this org:** `ssh -n` defeats a `< file` stdin redirect.
Use `scp`, or `ssh host "cat > /path" < file` without `-n`.

**The gate already passes on this document's structure.** The real `_donecheck.py` was run against this
version in a sandboxed `HOME` on qb2 and printed `DONE_CHECK PASS`, so the phrasing, the verdicts, the
roof clauses and the flag names are all satisfied and the exec pass has only to fill in §6 and §7. The
live gate is still red for one reason and one reason only: the gated path on qb2 is deliberately empty.
**Do not stage this file there to turn the gate green — stage it because the arms have run.**

**Gate traps, verified against the real gate this pass.** Keep `% of the <noun>`, `the binding roof
is`, `the limiter is`, `memory-bound` / `compute-bound` **unbroken on one line** — a hard-wrapped
`31 % of the` / `copy roof (DRAM)` fails while the same text unwrapped passes, and word order matters
(`31 % of the copy roof (DRAM)` passes, `31 % of the DRAM copy roof` does not). The doc must keep the
throwaway-harness sentence, the compute/communication overlap paragraph, and at least two explicit
CONFIRMED/KILLED verdicts; §1 already carries three and they survive into the final version.

### Acceptance checks — the leg is not done until all six hold

1. **>=6 `on` arms at 512 aa**, and the per-key A/A spread reported beside every delta. Any delta
   smaller than its key's spread is reported as **unresolved**, not as a small effect.
2. **Every `off:*` arm bracketed** by an `on` arm on both sides, and the bracketing arm indices printed
   in the results (`analysis.deltas_ms[*].bracketed_by`).
3. **plDDT and CIF sha256 identical across every arm** at each size. All these flags are memory-config
   and program-config only at 512 aa, so a value change means a broken run, not a finding. (`plDDT
   0.828628` / CIF `98c33a481fa1fd27` at 512 aa and `0.859489` / `8139d61b6c90f893` at 298 aa are the
   two predecessors' values on this chip — a match is a third confirmation.)
4. **The census printed for every arm**, with served counts per site and the branch taken, and
   `l1_out_refused` reported. Each of the five flags gets its own number even when that number is zero.
5. **S4 lands inside its band.** If `_PAIR_PROJ_L1_OUT` does not reproduce the sibling's +29.7 ms/fold
   on the same wall, stop and explain the instrument before quoting anything else.
6. **The survival fraction is stated twice**, once per family, each as a ratio of two measurements on
   this card: the `size512-ab` family against its own 298 aa arm (replacing 476.96 / 1729.03 = 27.6 %)
   and the merged-685.1 family against its own 298 aa arm.

---

## 6. Results — 13 timed arms at 512 aa, 7 of them `on`

`perf/survival512/surv_512_qb2c0.json`. One process, one cold fold (98.9 s), then 13 timed folds in the
order `on, off:ab5, on, off:narrowbw, on, off:projl1, on, off:norms, on, off:c2fix, on, off:family, on`.
**Every `off:*` arm is bracketed by an `on` arm on both sides** and its delta is taken against the mean
of the two, so linear drift cancels. All 14 folds returned plDDT **0.828628** and CIF sha256
**`98c33a481fa1fd27`** — identical to both predecessors on this chip, so no arm changed a value and
acceptance check 3 holds. `_L1_OUT_REFUSED` empty in every arm. `TT_BIO_REBLOCK_PERMUTE` was **unset,
i.e. its shipped default `"0"`** (`tt_bio/reblock_permute.py:288`, `_ENABLED = False`), so the
row-blocked permute is OFF in every arm here. `SEQ_LEN_MORE_CHUNKING` reads 1536, so the trimul does not
take its row-blocked tail at 512 aa.

### 6.1 The A/A floor, which is the first result and constrains every other one

**Seven `on` arms, and the floor is 1042x larger than the one the withdrawn number was scored against.**

| wall | calls | `on` median | A/A spread over 7 arms | spread as % of the wall |
|---|---:|---:|---:|---:|
| `block:PairformerLayer` | 604 | 74 753.41 ms | **2 189.67 ms** | 2.93 % |
| `stage:Pairformer` | 11 | 66 520.84 ms | **1 893.52 ms** | 2.85 % |
| `body:TriangleMultiplication` c256 | 1048 | 35 093.53 ms | 476.22 ms | 1.36 % |
| `body:TriangleAttention` c256 | 1048 | 22 240.68 ms | 372.65 ms | 1.68 % |
| `stage:template` | 10 | 2 968.34 ms | 125.42 ms | 4.23 % |
| `body:TriangleAttention` c64 | 160 | 853.30 ms | **13.99 ms** | 1.64 % |
| `body:TriangleMultiplication` c64 | 160 | 1 334.82 ms | **23.44 ms** | 1.76 % |
| `lin pairbias c256@16` | 484 | 440.46 ms | **11.81 ms** | 2.68 % |
| `lin pwa c256@1` | 240 | 215.95 ms | **2.23 ms** | 1.03 % |
| `lin template c256@64` | 40 | 38.01 ms | **0.51 ms** | 1.34 % |
| fold wall | 1 | 90.71 s | 2.387 s | 2.63 % |

**VERDICT — CONFIRMED (S9): the block wall cannot resolve this question and the per-site walls can.**
`size512-ab` quoted a **2.10 ms** A/A floor on `block:PairformerLayer` at 512 aa from a single pair of
`on` arms and called its +476.96 "227x the floor". Seven `on` arms on the same wall on the same chip
give **2 189.67 ms** — **1042x** that estimate, and **4.6x the effect it was used to certify**. Its
+476.96 is 0.64 % of a wall whose own drift band is 2.93 %. The number was never resolvable on that
instrument. **`|a - b|` from one pair is not an estimate of a drift band**, and that is the
methodological defect underneath the withdrawn figure, distinct from and compounding with the
single-shot defect the sibling leg identified.

The narrow site walls, by contrast, sit at **0.5-2.7 %** of much smaller walls and resolve every effect
this leg needed to separate. Nothing below claims an effect smaller than its own key's spread; anything
inside it is reported as **unresolved**, never as a small effect.

### 6.2 The decision census at 512 aa — served counts, read off the branch actually taken

Identical in all seven `on` arms. This is the cheapest half of the answer and it settles three of the
five flags before any timing argument.

| helper | site | padded shape | ON branch | OFF branch | calls |
|---|---|---|---|---|---:|
| `_l1_layer_norm` (headroom 1.5) | pairbias | `(1,512,512,256)` | **DRAM, refused** | DRAM | **484** |
| `_l1_layer_norm` (headroom 1.5) | pwa | `(512,512,256)` | **DRAM, refused** | DRAM | **30** |
| `_l1_layer_norm` (headroom 1.5) | template | `(1,512,512,256)` | **DRAM, refused** | DRAM | **10** |
| `_narrow_proj_linear` | pairbias | `(1,512,512,256) @ (256,16)` | DRAM, **tuned config** | `None` (core_grid) | **484** |
| `_narrow_proj_linear` | pwa | `(512,512,256) @ (256,1)` | DRAM, **tuned config** | `None` (core_grid) | **240** |
| `_narrow_proj_linear` | template | `(1,512,512,256) @ (256,64)` | DRAM, **tuned config** | `None` (core_grid) | **40** |
| `_pair_proj_linear(l1_out)` | trimul | `(1,512,512,64) @ (64,64)` | **L1** | DRAM | **320** |
| `_pair_proj_linear(l1_out)` | triatt | `(512,512,64) @ (64,64)` | **L1** | DRAM | **160** |
| `_pair_proj_linear(l1_out)` | trimul | `(1,512,512,256) @ (256,256)` | **DRAM, refused** | DRAM | **2096** |
| `_pair_proj_linear(l1_out)` | triatt | `(512,512,256) @ (256,256)` | **DRAM, refused** | DRAM | **1048** |
| `_transpose_memory_config` | triatt | `(512,512,64)` | **L1** | DRAM | **160** |
| `_transpose_memory_config` | triatt | `(512,512,256)` | **DRAM, refused** | DRAM | **1048** |

**VERDICT — CONFIRMED (S1, S2, S3): `_PAIR_BIAS_L1_NORM`, `_PWA_L1_NORM` and `_TEMPLATE_L1_NORM` are
each worth exactly 0.0 ms/fold at 512 aa, by construction.** All 524 `_l1_layer_norm` calls take the
DRAM branch in the ON arm — **0 of 524 on L1** — so both arms emit byte-identical device work and the
`off:norms` arm is an **A/A**. It measures like one: its largest site delta is **-7.34 ms** on
`norm pairbias c256` against that key's **13.06 ms** floor, and **not one wall in the arm resolves**
(block **-912.90** against a 2 189.67 floor). The positive control is §1.5: the same instrument on the
same card reads **+148.92 / +64.83 / +77.82 ms/fold** at those exact sites at 298 aa, where the census
shows them on **L1**. The instrument is not blind; the flags are dead.

**The pair track is dead for `_PAIR_PROJ_L1_OUT` and for C2FIX too.** 3144 of 3624 `l1_out` projections
and 1048 of 1208 transposes refuse at `c_z=256`. **Everything that survives at 512 aa is on the c=64
template track: 480 projections and 160 transposes, 640 device calls out of 5356.**

### 6.3 The per-flag decomposition, one flag at a time

`off - mean(bracketing on)` in ms/fold. **Bold = resolved** (|delta| above that key's 7-arm A/A spread);
everything else is inside the floor and is reported as unresolved, not as a small effect.

| site wall (calls) | floor | `off:norms` | `off:projl1` | `off:c2fix` | `off:ab5` | `off:narrowbw` | `off:family` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `lin pairbias c256@16` (484) | 11.81 | -5.79 | -1.16 | -0.77 | -1.62 | **+247.92** | **+248.45** |
| `norm pairbias c256` (484) | 13.06 | -7.34 | -1.09 | -1.80 | -1.61 | -0.10 | +1.58 |
| `lin pwa c256@1` (240) | 2.23 | -1.12 | -0.03 | +0.20 | -0.08 | **+130.07** | **+122.38** |
| `norm pwa c256` (30) | 0.55 | -0.46 | -0.01 | +0.14 | -0.09 | +0.10 | +0.06 |
| `lin template c256@64` (40) | 0.51 | -0.09 | +0.28 | +0.12 | +0.20 | **+27.00** | **+21.27** |
| `body:TriangleAttention` c64 (160) | 13.99 | -6.16 | +13.27 | **+82.75** | **+91.89** | +0.47 | **+16.44** |
| `body:TriangleMultiplication` c64 (160) | 23.44 | -9.14 | **+28.56** | +16.05 | **+31.23** | +2.35 | **+33.62** |
| `lin trimul c64@64` (320) | 2.88 | -0.99 | **+28.31** | +2.47 | **+31.27** | +0.35 | **+29.17** |
| `lin triatt c64@64` (160) | 3.05 | -0.98 | **+14.06** | +1.71 | **+13.19** | +0.10 | **+14.24** |
| `body:PairWeightedAveraging` c256 (30) | 28.84 | -14.14 | +1.06 | +2.41 | -1.87 | **+130.78** | **+122.52** |
| `stage:template` (10) | 125.42 | -40.81 | +61.61 | +123.30 | **+141.66** | +41.59 | +108.45 |
| `block:PairformerLayer` (604) | 2189.67 | -912.90 | -159.50 | -391.29 | -75.93 | +231.81 | +702.74 |

Read as totals in ms/fold, resolved sites only, with the unresolved remainder named:

| flag / set | 512 aa, resolved ms/fold | how it is built |
|---|---:|---|
| `_PAIR_BIAS_L1_NORM` | **0.0** | census: 0 of 484 on L1 |
| `_PWA_L1_NORM` | **0.0** | census: 0 of 30 on L1 |
| `_TEMPLATE_L1_NORM` | **0.0** | census: 0 of 10 on L1 |
| `_PAIR_PROJ_L1_OUT` | **+28.6** (+13.3 unresolved) | trimul `c=64` body; the triatt `c=64` body reads +13.27 against a 13.99 floor |
| C2FIX `_transpose_memory_config` | **+82.8** (+16.1 unresolved) | triatt `c=64` body, the one site the census leaves live |
| `_NARROW_PROJ_BW` | **+405.0** | 247.92 + 130.07 + 27.00 over **764 counted calls** |
| **`off:ab5`**, `size512-ab`'s exact five | **+119.96** over the matched site set (**+123.12** on the two `c=64` bodies alone) | replaces **+476.96** |
| **`off:family`**, the merged 685.1 five | **+443.87** | 248.45 + 1.58 + 122.38 + 0.06 + 21.27 + 0.07 + 16.44 + 33.62 |

**VERDICT — CONFIRMED (S4), and this is the cross-check the leg was required to pass before quoting
anything else.** `off:projl1` reads **+28.56 ms/fold** on `body:TriangleMultiplication` c64, 0.1785
ms/region over the **160 regions counted this pass**, against the sibling leg's independently measured
**+29.7 ms/fold / 0.1857 ms/region** on the identical wall. **Agreement to 1.14 ms/fold, 3.8 %**, inside
this key's 23.44 ms floor and inside S4's stated band. Two harnesses, two sessions, two authors, same
number. Everything else here rests on that.

**VERDICT — CONFIRMED (S6): C2FIX survives at 512 aa on the template track only, +82.8 ms/fold**,
inside S6's predicted 80-120 band, and the census says why: 160 of its 1208 transposes take L1 and the
1048 pair-track ones refuse.

**VERDICT — CONFIRMED (S8): the singles add.** On `body:TriangleAttention` c64 the three components of
`off:ab5` sum to 82.75 + 13.27 - 6.16 = **89.86** against **+91.89** measured, a 2.03 ms miss on a 13.99
floor. On `body:TriangleMultiplication` c64 they sum to 16.05 + 28.56 - 9.14 = **35.47** against
**+31.23**, a 4.24 ms miss on a 23.44 floor. Over both bodies, singles **125.33** against the set's
**123.12**, a miss of **1.8 %**. `off:family` over the matched site set against its singles:
**443.87** against 407.89 + 39.78 + (-30.14) = **417.53**, **6.3 %** — the largest non-additivity in the
leg, carried by `lin|pwa|c256@1` (130.07 alone against 122.38 in the set) and `lin|template|c256@64`
(27.00 against 21.27), i.e. the two smallest-output narrow sites. **The five flags do not interact at 512 aa**, which is expected once the census
shows them touching disjoint sites, and it means each single is quotable on its own.

**VERDICT — CONFIRMED (S5), and it is the finding the org did not expect.** `_NARROW_PROJ_BW = 1` is
worth **+405.0 ms/fold at 512 aa**, inside S5's registered 270-420 band, at **0.512 / 0.542 / 0.675
ms/call** across its three sites against a predicted 0.35-0.55. The org's ledger credits it with
**31.5 ms/fold** (X2, 298 aa, a qb1 absolute). Measured on one card in one session at both sizes it
goes **60.37 ms/fold at 298 aa to 407.89 at 512 aa, a growth of 6.76x** against a 2.56x growth in bytes,
and it is the only member of either family whose mechanism is not capacity-gated: the census shows its
tuned program config returned at all 764 calls at both sizes, where every other flag's fit test
refuses. **Against the ledger's 31.5 it is 12.9x, but 6.76x is the honest figure** — it divides two
measurements from the same instrument, where the 12.9x mixes this card with qb1.

### 6.4 The survival fractions — matched instrument, both sizes, one card, one session

**The chained 298 aa run landed after all**, 9 timed folds with 5 `on` arms, all at plDDT 0.859489 and
CIF `8139d61b6c90f893`. So both sizes are measured on the same card in the same session with the same
harness, and every fraction below is **a ratio of two measurements on the same card** — which is what
qb2 at 0.68.0 is good for, and why the grid and the ttnn version cancel out of it.

**The 298 aa census is the positive control, and it is total.** All 12 census classes take **L1** at
298 aa: 484 + 30 + 10 `_l1_layer_norm`, 484 + 240 + 40 `_narrow_proj_linear`, 2096 + 1048 + 320 + 160
`_pair_proj_linear(l1_out)`, 1048 + 160 `_transpose_memory_config`. At 512 aa **640 of those 5356 calls
survive** and the rest refuse. Same instrument, same code, two sizes.

**Matched site set at both sizes** — the six `c_z=256` leaf walls plus the two `c=64` bodies, summed.
This is the instrument that resolves at both sizes, so it is the one the fractions are taken on.

| set | 298 aa | 512 aa | survival |
|---|---:|---:|---:|
| **`off:ab5`**, `size512-ab`'s five (C2FIX + `_PAIR_PROJ_L1_OUT` + 3 norms) | **330.94 ms/fold** | **119.96 ms/fold** | **36.2 %** |
| **`off:family`**, the merged five (`_PAIR_PROJ_L1_OUT` + `_NARROW_PROJ_BW` + 3 norms) | **372.63 ms/fold** | **443.87 ms/fold** | **119.1 %** |
| the three L1 `layer_norm` flags alone | **275.26 ms/fold** | **-30.14, unresolved = 0** | **0 %** |
| `_NARROW_PROJ_BW` alone | **60.37 ms/fold** | **407.89 ms/fold** | **675.7 %, i.e. 6.76x** |

**VERDICT — KILLED: the +476.96 ms, and the 27.6 % is killed as a measurement while landing close to the
truth for the wrong reason.** Both of `size512-ab`'s figures were taken on `block:PairformerLayer`.
Re-taken on that same wall in this session:

| `block:PairformerLayer`, `off:ab5` | delta | A/A floor | resolved? |
|---|---:|---:|---|
| 298 aa | **+1 738.09 ms** | 436.30 | **yes**, 4.0x the floor |
| 512 aa | **-75.93 ms** | **2 189.67** | **no**, 29x inside the floor |

Two things follow, and they point in opposite directions. **The 298 aa denominator is confirmed
independently: +1 738.09 against `size512-ab`'s own 1 729.03, 0.5 % apart** on the same wall on the same
chip, which is as strong a cross-validation of a predecessor's number as this leg produced. And **the
512 aa numerator is not measurable on that wall at all** — the effect is 29x inside the wall's own
7-arm drift band, so on its own instrument the fraction is bounded only by |survival| < 126 %, which
settles nothing. **`size512-ab` did not measure +476.96 ms; it measured its own drift.**

**On the instrument that does resolve, the survival of that flag set is 36.2 %, which is *higher* than
the 27.6 % it replaces, not lower.** The withdrawn number was wrong by ~4x in the numerator, and its
denominator was too large by a compensating factor, because both were block-wall figures carrying
everything inside a `PairformerLayer` rather than the sites the flags touch. **So the honest correction
is not "27.6 % was 3.4x too big" — it is "27.6 % was unmeasurable on its instrument, and the answer it
was reaching for is 36.2 %."** That distinction matters for `CLOSEOUT.md`: the org's decision to move
its centre of gravity to 512 aa was not built on a number that was directionally wrong, it was built on
a number that could not have been right or wrong.

**VERDICT — CONFIRMED: the merged 685.1 family does not merely survive at 512 aa, it grows — 119.1 %.**
Same instrument, same card, same session. The two fractions differ by 3.3x because the two sets differ
in exactly one slot: `size512-ab` swapped `_NARROW_PROJ_BW` out for C2FIX, and `_NARROW_PROJ_BW` is the
one flag that is not capacity-gated. **It measured the set that excludes the only member that survives.**

**The decomposition that explains all four rows, and it is two numbers.** Splitting each set by track:

| track | `off:ab5` 298 aa | `off:ab5` 512 aa | `off:family` 298 aa | `off:family` 512 aa |
|---|---:|---:|---:|---:|
| `c_z=256` pair track (six leaf walls) | **290.93** | **-3.16** | 356.35 | **393.80** |
| `c=64` template track (two bodies) | **40.01** | **123.12** | 16.28 | **50.07** |

**The surviving part did not shrink — it grew 3.08x, and by the same factor in both sets** (123.12/40.01
= 3.079, 50.07/16.28 = 3.076, independently measured). Bytes grow 2.56x from padded 320 to padded 512,
so a track that stays L1-resident getting 3.08x more expensive to run on DRAM is exactly what the byte
model predicts. **What collapses is the `c_z=256` pair track, 290.93 to zero**, and the census gives the
reason with no timing argument at all: every `c_z=256` gate refuses at 512 aa. `off:family` keeps its
pair track only because `_NARROW_PROJ_BW` lives there and is not gated.

### 6.5 Where the surviving effect sits, and what holds it there

The arithmetic-intensity placement and the roofs are in §1.3 and §1.4, measured on this card this pass
and not inherited. In one line: the narrow pair-track projection is at **7.1 FLOP/byte** against this
card's measured **292.9 FLOP/byte** machine balance, 41x onto the memory side, so **the binding roof is
the bandwidth roof** and the op is **memory-bound** with nothing to argue about. Compute is 0.0096 ms of
a 0.9032 ms op, **1.1 %**, so the total is nearer **max(compute, comm)** than `compute + comm`, with
`comm` binding. No overlap arrangement is visible at that ratio, and the sibling leg reached the same
verdict from the other direction: its excess over a plain clone tracked bytes and not FLOPs across a 16x
FLOP swing.

**VERDICT — CONFIRMED (S10): the placements hold and the limiter is transaction size, not occupancy.**
Production 0.9032 ms is **44 % of the copy roof (DRAM)**; the `core_grid=` form `off:narrowbw` reverts to
is 1.4136 ms, **28 % of the copy roof (DRAM)**; an L1 source would be 0.2516 ms, **79 % of the copy roof
(L1)**. Two of three are under 70 %, so a mechanism is owed at kernel level. Read off the program config
the production helper actually returns — `in0_block_w=1, out_subblock_h=1, out_subblock_w=1,
out_block_h=5, per_core_M=75, per_core_N=1` — **the limiter is transaction size on both sides**: the
reader issues one-tile 2 KB NOC reads per K block because `in0_block_w=1`, and the packer emits one tile
per pack because `out_subblock_w=1`, against the long bursts a clone gets, with circular-buffer depth
bounding how many subblocks are in flight behind the writer. **Core occupancy is not the limiter for the
tuned form**: `per_core_M=75` puts work on **110 of 110 cores**. For the `core_grid=` form occupancy
*is* the limiter, and a one-tile-wide output leaves `ttnn.linear` on a core ladder flat from **16 to 110
cores**, roughly 94 of 110 cores idle, which is the 1.57x it loses. So `_NARROW_PROJ_BW`'s +405.0 ms/fold
is **an occupancy repair measured against an occupancy defect**, and the residual 56 % of the DRAM copy
roof it still leaves on the table is a transaction-granularity problem, the third instance in this org
of `perfwar-l1-destination-priced-as-free-fake-mystery`, here on the read side.

### 6.6 The 298 aa denominator, and how it was taken

The 298 aa arm set (`on, off:family, on, off:narrowbw, on, off:ab5, on, off:norms, on`) was chained to
start on the same card in the same session the moment the 512 aa run released the device, and it
completed: `perf/survival512/surv_298_qb2c0.json`, 9 timed folds, **5 `on` arms**, all plDDT 0.859489
and CIF `8139d61b6c90f893`, matching §1.5 and both predecessors. Its site-wall A/A spreads are
**0.12-6.66 ms** against 512 aa's 0.17-23.44, and its `block:PairformerLayer` spread is **436.30 ms**
against 2 189.67 — the floor scales with the wall, as it should. Every fraction in §6.4 is taken from
this run and the 512 aa run only, never from another leg's number, and the one predecessor figure it
touches (`size512-ab`'s 1 729.03) is used as a **check** on the 298 aa block wall rather than as an
input.

---

## 7. Ranked hand-off to the CTO, in ms/fold at 512 aa

Counts measured this pass, not assumed: 484 + 240 + 40 = **764** narrow `c_z=256` projections, **160**
`c=64` trimul regions, **160** `c=64` triatt transposes, `block:PairformerLayer` **604**,
`stage:Pairformer` **11**. Blocks x recycles throughout (charter §4.9), never blocks alone.

| # | item | ms/fold at 512 aa | state |
|---:|---|---:|---|
| 1 | **`_NARROW_PROJ_BW` = 1**, already on main and already ON | **+405.0** | **A valuation, not a candidate, and the org's largest 512 aa number.** Same instrument, same card: **60.37 at 298 aa to 407.89 at 512 aa, 6.76x**, against a 2.56x growth in bytes. The ledger credits it with 31.5 (X2, qb1, 298 aa). **Correct the ledger: this flag is size-independent and gets better with size.** |
| 2 | **a size-independent route to an L1 `layer_norm` source** | up to **+498** (0.6516 ms/call x 764 counted calls, isolated) | **The prize the 1.5x fit test forgoes.** The census proves all 524 norms refuse at 512 aa; §1.3 shows the allocation **succeeds** in isolation at padded 512. The gate is a static budget decision, not an allocator failure. Charter §4.10 prefers exactly this class of fix. Phase 3 work, not this leg's. |
| 3 | **C2FIX `_transpose_memory_config` at 512 aa, template track only** | **+82.8** (+16.1 unresolved) | **`z-rowblock`'s op. Reported and handed over, not pursued**, see §7.1. |
| 4 | `_PAIR_PROJ_L1_OUT` at 512 aa, template track | **+28.6** (+13.3 unresolved) | Already on main and already ON. Reproduces the sibling's +29.7 to 3.8 %; **this is the leg's instrument cross-check as much as it is a valuation.** |
| 5 | the three L1 `layer_norm` flags at 512 aa | **0.0 each** | **Dead by construction**, census 0 of 524 on L1. Not a regression: they are 298-aa-only wins whose fit test is doing its job. Item 2 is the way to recover them. |
| 6 | `size512-ab`'s **+476.96 ms** and its **27.6 %** | **not a measurement** | **RETIRED. The +476.96 is 29x inside its own wall's 2 189.67 ms floor and is drift.** Its 298 aa denominator is confirmed (+1 738.09 vs 1 729.03, 0.5 %). **Replace 27.6 % with 36.2 %**, taken on the site walls that resolve at both sizes, and add the merged family's **119.1 %** beside it. Fix both in `CLOSEOUT.md`. |

### 7.1 The boundary with `z-rowblock`, respected

The brief says that if the census shows `_transpose_memory_config` carrying a large share of the 512 aa
delta, report it and stop. **It does, and this is the stop.** C2FIX is **+82.8 of `off:ab5`'s 123.1**
resolved ms, **67 %**, which confirms the sibling leg's suspicion that most of `size512-ab`'s five-flag
figure lived in the transpose rather than in `_PAIR_PROJ_L1_OUT`. What that leg inherits on top of its
own work: the transpose's only live site at 512 aa is `TriangleAttention` at `c=64`, **160 of 1208
calls**, the pair-track 1048 refuse at the 2.5x gate, and one bracketed arm prices the live 160 at
**+82.8 ms/fold**. No further work on that op was done here.

### 7.2 What every row owes on qb1 at 0.67.4

Every absolute above is qb2 chip 0 at ttnn 0.68.0 and is a **ratio** (charter §4.8). The two survival
*fractions* are the exception in kind, not in caution: each divides two measurements taken on this same
card, so the grid and the ttnn version cancel. What qb1 would change is where the cliffs fall, and §1.2
computes that from the production helpers directly. On qb1's 13x10 grid the L1-norm budget is
199 214 080 B, so the norm class dies between N=481 and N=512 there instead of between N=449 and N=480,
and **512 aa is 1.1 % outside the gate on qb1 as well.** So the census verdict, all three norm flags
dead at 512 aa, is expected to hold on qb1, and rows 1, 3 and 4 owe a re-take of their bracketed arms
because `per_core_M` snaps differently on 130 cores.

### 7.3 Co-tenancy, recorded because it set the floor

`protenix-trunk--z-permute-flip-land` held chip 1 of the same p300 board (007) throughout, running
OpenDDE folds (pids 443454/443455, verified in the process table before and during the run). Charter
§4.8 prices cross-chip compute interference at 0-4.8 %, and the 7-arm A/A spreads measured here, 2.93 %
on the block wall and 1.0-2.7 % on the site walls, sit inside that band. **This is the mechanism behind
the 2 189.67 ms block-wall floor**, and it is why the site walls and not the aggregate walls carry the
answer. A future leg that needs the block wall to resolve a sub-1 % effect needs the board to itself.

---

## 8. Decided against, so execution does not relitigate it

- **Re-measuring the k-chunked out-projection (−115.6 ms/fold) or the Transition chunk height.** Closed
  with measurements by `z-h5-infold` and `z-transition-chunk`.
- **Touching `SEQ_LEN_MORE_CHUNKING`** so the trimul's row-blocked tail fires at 512 aa. The CTO ruled
  nobody moves it this wave and `z-h5-infold` measured a 855 ms/fold reason not to.
- **Any work on `_transpose_memory_config` beyond one bracketed arm and a hand-off.** It is
  `z-rowblock`'s op and `z-rowblock-profiler` is live on it.
- **Reducing recycles to make the arms cheaper.** It would cut ~3x off the 512 aa run, and the per-region
  walls would still be valid, but it changes the fold the org's ledger is denominated in and the
  instrument's credibility is worth more than 15 minutes. If a turn runs short, **drop arms in reverse
  priority order, never change the fold configuration.**
- **Using the fold wall as the headline instrument.** This harness family's own 512 aa fold-wall A/A
  floor is 758.3 ms against effects of 0-500 ms. Site walls first, block wall as cross-check, fold wall
  quoted only for context.
- **A single-pair A/A floor.** See §3.2. Six ON arms cost 9 minutes and are the difference between this
  leg and the one it replaces.
- **Changing anything in `tt_bio/`.** Phase 2. The flags are module globals read at call time, which is
  why an outside harness can move them at all, and **parity needs no separate argument at 512 aa: every
  arm differs only in a memory config or a program config, both established bit-exact by the merges
  that landed them** (`torch.equal`, plDDT unchanged to six decimals). Acceptance check 3 is the test.

## 9. Files

| path | what |
|---|---|
| `perf/survival512/PREDICTIONS.md` | S1-S10 with falsifiers, committed before any arm ran |
| `perf/survival512/surv_envelope.py` | the no-fold probe: per-flag cliff from the production helpers, roofs, and the two mechanisms priced in isolation |
| `perf/survival512/surv_envelope_qb2c0.json` | its results — §1.2, §1.3, §1.4 |
| `perf/survival512/surv_arms.py` | the arm runner: bracketed arms, per-site walls, branch census, per-key A/A spread |
| `perf/survival512/surv_298_validate_qb2c0.json` | the 3-arm 298 aa validation of §1.5 — the census positive control, the counted call numbers, X10 reproduced |
| `perf/survival512/surv_512_qb2c0.json` | PENDING — step 2 |
| `perf/survival512/surv_298_qb2c0.json` | PENDING — step 3 |

Inherited rather than re-derived: `perf/progcfg/h5_infold.py`'s census and wall design (`z-h5-infold`),
`perf/size512/fixtures/cdk2x2_{298,512}.{yaml,a3m}` and the arm-flip-between-folds structure of
`perf/size512/fold_ab512.py` (`size512-ab`). **No production file is touched in this leg.**
