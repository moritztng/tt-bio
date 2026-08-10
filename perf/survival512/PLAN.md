# protenix-trunk--z-survival-512 — does the org's merged 685.1 ms/fold survive at 512 aa?

TASK TYPE: VERIFY/BENCHMARK (Phase 2 experiment) | PLAYBOOKS loaded: ACCELERATE + VERIFY/BENCHMARK |
memories read: perfwar-cotenanted-ab-noise-floor-exceeds-small-wins, perfwar-l1-fit-cliff-channel-width-not-token-count,
perfwar-l1-destination-priced-as-free-fake-mystery, ttnn-sync-before-every-timed-region,
roofline-roof-must-be-measured-not-asserted, tt-bio-trunk-perf-ratio-denominator-unit-slip,
tt-bio-l1-residency-guard-dead-in-real-folds, protenix-v2-448aa-l1-cb-clash-cc39a867d,
donecheck-hostspecific-path-unsatisfiable-on-remote-host, tt-bio-worktree-run-recipe,
gate-mandated-write-to-single-owner-file-is-a-race, model-merge-approval-gate

**STATE OF THIS DOCUMENT: PLANNING PASS. The no-fold probe has run and its numbers are real. THE
IN-FOLD ARMS HAVE NOT RUN. Do not conclude this leg, do not quote a survival fraction from this
version, and do not stage this file to qb2's `~/.coworker/state/` until the arms in §5 have produced
the tables in §6.** The DONE_CHECK is deliberately still red for exactly that reason: the gated path
`/home/ttuser/.coworker/state/protenix-trunk--z-survival-512.md` is intentionally absent.

**This leg is a throwaway experiment harness under `perf/survival512/`, not production. Nothing in
`tt_bio/` is touched, nothing is proposed for merge, and every flag is flipped from outside on module
globals that production reads at call time.** The word "merged" appears throughout because the
question is whether the org's *merged* 685.1 ms/fold survives; the code that answers it is a probe.

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

---

## 2. Predictions, registered before the arms run

`perf/survival512/PREDICTIONS.md`, committed before any fold arm opened a device. S1-S10 with
falsifiers. In one line each:

| # | prediction | falsifier |
|---|---|---|
| S1-S3 | the three L1-norm flags are **0.0 ms/fold each** at 512 aa, census 0 of N on L1 | any `L1` census row at those sites, or a site-wall delta above its own A/A spread |
| S4 | `_PAIR_PROJ_L1_OUT` reproduces the sibling's **+29.7 ms/fold** on `body:TriangleMultiplication|c64`, zero at c_z=256 | disagreement outside (my spread + 2.6 ms) — and then nothing else in the leg is quotable |
| S5 | `_NARROW_PROJ_BW` is **+0.35 to +0.55 ms/call** in-fold, i.e. **370-580 ms/fold** if the counted call number is near 1048 | under 0.10 or over 0.80 ms/call |
| S6 | C2FIX survives only on the template track, **+80 to +120 ms/fold** | outside that band |
| S7 | `off:ab5` reads **110-160 ms/fold**, so **476.96 was 3-4x too large and the survival fraction of that set is 6-9 %, not 27.6 %** | above 300 or below 40 ms/fold |
| S8 | the family total equals the sum of its singles within 2x the A/A spread, 90 %+ of it `_NARROW_PROJ_BW` | non-additive, and then the interaction is the finding |
| S9 | the `block:PairformerLayer` A/A spread over >=6 ON arms is **above 30 ms**, so the block wall cannot resolve S1-S4 or S6 and the site walls must | a spread under 20 ms |
| S10 | placements 44 % / 79 % / 28 % of the roofs named above; the limiter is transaction size | a measured placement outside +-10 points |

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
`surv_envelope_qb2c0.json` exists.

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

## 6. Results — PENDING, the arms have not run

| flag | 298 aa, this session | 512 aa, this session | A/A spread | survives? | census at 512 aa |
|---|---:|---:|---:|---|---|
| `_PAIR_PROJ_L1_OUT` | PENDING | PENDING | PENDING | PENDING | PENDING |
| `_PAIR_BIAS_L1_NORM` | PENDING | PENDING | PENDING | PENDING | PENDING |
| `_PWA_L1_NORM` | PENDING | PENDING | PENDING | PENDING | PENDING |
| `_TEMPLATE_L1_NORM` | PENDING | PENDING | PENDING | PENDING | PENDING |
| `_NARROW_PROJ_BW` | PENDING | PENDING | PENDING | PENDING | PENDING |
| C2FIX `_transpose_memory_config` (hand-off) | PENDING | PENDING | PENDING | PENDING | PENDING |
| **`off:ab5` total** — replaces **+476.96 ms / 27.6 %** | PENDING | PENDING | PENDING | | |
| **`off:family` total** — the merged 685.1 family | PENDING | PENDING | PENDING | | |

## 7. Ranked hand-off — PENDING, with the two rows already visible

| # | item | ms/fold at 512 aa | state |
|---:|---|---:|---|
| 1 | `_NARROW_PROJ_BW` = 1, already on main and already ON | PENDING (predicted 370-580) | a valuation, not a candidate — but if it lands near the prediction it is the org's largest 512 aa number and it was credited with 31.5 |
| 2 | a size-independent route to an L1 `layer_norm` source | PENDING (0.6516 ms/call x counted calls) | the prize the 1.5x fit test forgoes; the isolated allocation **succeeds** at padded 512 on an idle chip |
| 3 | C2FIX at 512 aa, template track only | PENDING (predicted 80-120) | **`z-rowblock`'s op.** Reported and handed over |

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
| `perf/survival512/surv_512_qb2c0.json` | PENDING — step 2 |
| `perf/survival512/surv_298_qb2c0.json` | PENDING — step 3 |

Inherited rather than re-derived: `perf/progcfg/h5_infold.py`'s census and wall design (`z-h5-infold`),
`perf/size512/fixtures/cdk2x2_{298,512}.{yaml,a3m}` and the arm-flip-between-folds structure of
`perf/size512/fold_ab512.py` (`size512-ab`). **No production file is touched in this leg.**
