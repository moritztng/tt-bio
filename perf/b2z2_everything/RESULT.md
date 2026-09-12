# b2z2-everything-union-wh — eleven levers are nine, and together they are 1.10789x on a WH fold

TASK TYPE: ACCELERATE (compose + VERIFY/BENCHMARK) | PLAYBOOKS loaded: ACCELERATE +
VERIFY/BENCHMARK + ALWAYS-ON | memories read: `git-merge-no-conflict-markers-can-still-be-
semantically-broken`, `unified-solution-not-per-model-patches`, `no-speedup-by-skipping-the-
models-own-work`, `merged-lever-defaults-off-is-not-a-landed-win`, `model-merge-approval-gate`,
`perf-page-cell-is-historical-not-live-baseline`, `rfd3-isolated-screen-underprices-residency-
lever`, `whglx-cross-account-artifacts-strand-off-worktree`, `whglx-unpinned-all-chip-open-breaks-
every-cotenant`, `whglx-galaxy-lease-card-number-vs-device-node-mismatch`,
`negative-control-must-break-what-check-reads`, `cdk2x2-chimeric-fixture-cannot-score-non-bit-
exact-parity`, `perf-baseline-cell-keyed-to-card-type-not-dispatch-grant`,
`ssh-remote-background-launch-stdin-hang`, `git-worktree-add-f-steals-branch-ref`

VERDICT: GO — **the whole union is 1.10789x paired on a 512 aa Wormhole fold, n=6, against an
A/A floor of 1.00136x, and it passes parity at both fixture sizes.** All six paired ratios are
positive and span 1.09535x to 1.12464x. The six bit-exact levers inside it write the base CIF
byte for byte. The brief's eleven levers are **nine**: `wk/b2z2-step-program-fusion` contributes
only the defective key window and is an ancestor of four of the other branches, and
`TT_BIO_MAC_FUSE` is its own row's NO-GO.
ARCH: WH — every number in this doc is Wormhole. The published cell is Blackhole and this row
holds no BH chip; no BH claim is made, projected or otherwise.
BRANCH: `wk/b2z2-everything-union-wh`, cut from `origin/main` `84da2a49b`, pushed. Not merged.
Every flag defaults OFF.
CARD: whglx (j10glx02) card 8 for both fold sessions, card 9 for the parity session — a sibling
that had been free 49 min, taken with the grant widened on that one command
(`TT_BIO_LEASE_CARDS=8,9`). Both pinned, `TT_BIO_TRACE_REGION_SIZE` 512 MiB,
`TT_METAL_DEVICE_PROFILER` absent. Run as `tt-admin` from `/home/tt-admin/wt-everything-union`
because the lease dir is tt-admin-owned; **every artifact was copied back into the checkout this
doc commits from** and is committed under `perf/b2z2_everything/`.

PREDICTED: `state/b2z2-everything-union-wh.PREDICTION.md`, also committed at
`perf/b2z2_everything/PREDICTED.md` in `e36c800f0`, before the first device process of this row.
Scored in §7, and it is wrong in an informative direction.
MEASURED: 53 folds in three sessions on two chips. 21 folds base/BX/ALL
(`fold_everything_wh_c8.json`), 35 folds base/SAMP/TRUNK/MSA/HOSTONLY
(`singles_everything_wh_c8.json`), 14 folds at two sizes and three seeds for parity
(`parity_seeds_wh_c9.json`, scored to `parity_score_wh_c9.json`). Drivers
`perf/b2z2_everything/fold_ab.py`, `parity_seeds.py`, `discount.py`, `lever_overlap.py`,
all results committed beside them.
UNION-RATIO: **1.10789x** paired on a 512 aa WH fold, n=6, A/A floor **1.00136x** — 79x clear.
BITEXACT-SUBSET: **1.04112x**, CIF `da476491dbb2a847…` identical to base at plDDT 0.847187.
UNION-DISCOUNT: **-0.144 %** — the union measures 1.10789x against a 1.10630x product of the
four in-session stage singles. **The pre-registered falsifier needed 5 % BELOW and the union came
in above. It does not fire.** Second construction, crossing no session boundary on the numerator:
the host trio's marginal on top of the six bit-exact levers is **1.06413x** against **1.0683x**
standing alone, a **0.39 %** discount. `perf/b2z2_everything/discount.json`.
PARITY: **PASS at both sizes.** 512 aa worst per-pseudo-domain all-atom **0.40355 Å** against a
0.60 Å kill bar and a **1.86363 Å** seed floor; 298 aa **0.29521 Å** against a 1.24155 Å seed
floor. The A/A repeat is byte-identical at both sizes, so the parity leg's own floor is exactly
zero, and the `on` arm differs from `off` at every seed, so the comparison demonstrably breaks.
Against the experimental structure 1HCL the union is not worse than base at either size:
CA RMSD 1.234/1.239 Å vs 1.270/1.280 Å at 512 aa, 0.790 Å vs 0.822 Å at 298 aa.
DEFICIT-SECONDS: 3.991 s REMOVED on a WH fold — **40.561 s to 36.570 s**, the medians of 6 and 6
timed folds interleaved in one process against one device open. Measured on the fold, not
projected from a step. No Blackhole projection is made.
TILE-MOVEMENT-DELTA: 0.0 % — **named, not measured**. No armed profiler capture was taken in this
pass; all three sessions went to paired folds and the parity ladder, and a watcher-armed run is
not a ratio. What is arithmetic from the shapes rather than measured: `qkvg_heads` deletes one
67.1 MB re-read of the normed pair tensor per triangle attention (134.2 MB per block), the gather
elision deletes about 110 MB per sampler step of read+write that computes nothing, and
`TT_BIO_DEVICE_CONDITIONING` deletes a 201 MB download and the upload after it once per fold.
Per `b2z2-trunk-byte-round2` those byte figures are a **lower bound** on what the deletions are
worth, because a deleted read is priced by its reader and the coefficient runs 1.3 - 2.9.
CHEAT-CHECK: clean — no lever removes a unit of the model's own work; both drivers assert
`steps == 200` and `recycles == 3` before the device opens and read `step_n == 200` back off the
sampler on every fold, six of the nine levers are `torch.equal` and the subset containing them
writes the base CIF byte for byte, and the three that are not bit-exact move work from the host
to the device rather than deleting it.

## 1. The population, and the ancestry answer beside each row

`perf/b2z2_everything/lever_overlap.py`, output committed at `lever_overlap_out.txt`. Ten refs,
45 pairs, **23 not independent**. The two verdicts it separates are used as it intends: a
CONFIRMED same site is overlapping changed lines, a collision risk is only a shared class or def,
and **I read the diffs for every risk that survived into the union** — said explicitly below.

| lever | flag | stage | branch @ tip | ancestry / overlap | in the union? |
|---|---|---|---|---|---|
| atom shift gather | `TT_BIO_ATOM_SHIFT_GATHER` | sampler | `wk/b2z2-layout-op-elision` @ `9ca6c3308` | same site as the key window; same site as `ATOM_L1` | **yes** |
| atom L1 residency | `TT_BIO_ATOM_L1` | sampler | `wk/b2z2-step-fusion-next-sites` @ `9cd6c2ff1` | descendant of `step-program-fusion` | **yes** |
| K/V pre-projection | `TT_BIO_ATOM_KV_PREPROJ` | sampler | `wk/b2z2-sampler-union-wh` @ `9b4783cbd` (matrix-proved form) | same site as both gathers by construction | **yes** |
| SDPA grid q-chunk | `TT_BIO_SDPA_GRID_Q_CHUNK` | sampler | `wk/b2z2-step-adaln-sdpa` @ `39efa31a4` | **branch is a DESCENDANT of `step-fusion-next-sites`**; taken as its 52-line marginal diff | **yes** |
| fused qkv+gate | `TT_BIO_TRIATT_FUSED_QKVG` | trunk | `wk/b2z2-trunk-byte-floor` @ `ccad967cc` | disjoint from every other member | **yes** |
| PWA residency | `TT_BIO_PWA_RESIDENCY` | MSA | `wk/b2z2-msa-movement-attack` @ `d39798d11` | disjoint from every other member | **yes** |
| device conditioning | `TT_BIO_DEVICE_CONDITIONING` | host | `wk/b2z2-host-device-compose` @ `a5f66aec` | 1 shared changed line with the sampler union | **yes** |
| device z_init | `TT_BIO_DEVICE_ZINIT` | host | same | same | **yes** |
| device conf pair assembly | `TT_BIO_DEVICE_CONFIDENCE` | host | same | same | **yes** |
| ~~atom key window~~ | `TT_BIO_ATOM_KEY_WINDOW` | sampler | `wk/b2z2-step-program-fusion` @ `7ea55dd92` | **same transform as the shift gather, at the same lines** | **no — computes the wrong gather, max abs 4.21875** |
| ~~BinaryNg / MAC fuse~~ | `TT_BIO_MAC_FUSE` | sampler | `wk/b2z2-step-binaryng-fusion` @ `ae0812ed6` | descendant of `step-program-fusion` | **no — its own row's NO-GO, 1.6 % slower** |

`wk/b2z2-step-matmul-group` appears in the brief's ancestry row and is not a lever at all:
`git diff origin/wk/b2z2-step-program-fusion origin/wk/b2z2-step-matmul-group -- tt_bio/` is
empty. **So the brief's eleven rows are nine levers.**

### The two collision risks that survived into the union, and what reading them said

* **`layout-op-elision` vs `step-adaln-sdpa`, both inside `AttentionPairBias`.** Read: the elision
  replaces the one-hot gather that builds `s_kv`; the SDPA lever only changes the
  `program_config=` argument of four `scaled_dot_product_attention` calls and the helper that
  builds it. Different statements in the same class. Composed.
* **`sampler-union-wh` vs `host-device-compose`, one changed line in common at
  `tt_bio/tenstorrent.py:9750`, inside `DiffusionModule`.** Read: the sampler union wraps the
  cached `keys_indexing` in an `_AtomShiftGather` sentinel; the host branch rewrites where
  `bias_token` comes from, about thirty lines away. **Line-number coincidence on the branch side,
  not a shared site.** Composed.

The one pair the tool could not have caught and the sampler row did: `TT_BIO_SDPA_GRID_Q_CHUNK`'s
branch already CONTAINS `TT_BIO_ATOM_L1` and the key window, so its published 1.01831x is a
marginal on top of them. Taken here as the marginal diff `step-fusion-next-sites...step-adaln-sdpa`
— 52 lines in `tenstorrent.py` plus the `work` argument at the shared helper's call sites in
`esmc.py`, `esmfold2.py` and `saprot.py`. The lever is unified across the four models that share
that helper, not a boltz2 patch.

## 2. The branch

Five composition commits off `origin/main` `84da2a49b`. The sampler subset arrives by
fast-forwarding onto `wk/b2z2-sampler-union-wh`, itself a linear composition onto the same
`origin/main`, so its provenance and its two harnesses come with it. The other four levers are
applied as diffs against `origin/main` restricted to `tt_bio/`, not merged as trees — all the
sampler branches edit the same twenty lines and a three-way merge there is a guess.

**Every flag defaults OFF.** One change of substance was needed: the three PairWeightedAveraging
knobs default ON on their own branch, so here they take their default from one
`TT_BIO_PWA_RESIDENCY`, default OFF, and `TT_BIO_PWA_RESIDENCY=1` reproduces exactly the
configuration that branch measured.

## 3. The fold — `fold_everything_wh_c8.json`, commit `e36c800f`

18 timed folds and 3 cold, one process, one device open, `cdk2x2_512` with its fixed 35-row a3m,
**200 sampling steps, 3 recycles, full MSA depth**, 1 sample, seed 0, arm order reversed on
alternate reps.

| arm | levers | median fold | paired ratio | every rep | CIF sha256 | plDDT |
|---|---|---|---|---|---|---|
| base | — | **40.561 s** | 1.00000x | — | `da476491dbb2a847…` | 0.847187 |
| BX | the six bit-exact | 38.918 s | **1.04112x** | 1.0135 – 1.0475 | **`da476491dbb2a847…`** | **0.847187** |
| **ALL** | all nine | **36.570 s** | **1.10789x** | 1.0954 – 1.1246 | `bce2170b5bf6c9cf…` | 0.84647 |

**A/A floor 1.00136x** on the same estimator — the median of consecutive base-fold pairs, six base
folds spanning 40.389 to 40.612 s (spread 1.00552x) on a box whose loadavg ran 7.15 to 14.88.

**Both hash bars pass, and they are the check that matters.** BX writes the base CIF byte for byte
at the same plDDT, one sha256 across all six reps — six levers compose without moving one atom.
ALL does not, and that is the point: three of its members recompute in bf16 on the device, and an
identical hash on both arms of a non-bit-exact A/B would have meant the toggle never reached the
code. Engagement is counted rather than assumed: the shift gather served 1200 calls and fell back
0, `ATOM_L1` 1200 L1 / 0 DRAM, `qkvg_heads` 560 served / 0 fallbacks on every armed fold and zero
on base.

### Per-stage walls, so the composition can be attributed

| stage | base | BX | ALL | BX ratio | ALL ratio |
|---|---|---|---|---|---|
| trunk (all 4 recycles, MSA inside) | 27.1398 s | 26.5979 s | 25.9888 s | 1.02037x | **1.04429x** |
| diffusion conditioning | 0.8171 s | 0.8527 s | 0.2747 s | 0.95825x | **2.97452x** |
| sampler (200 steps) | 9.9922 s | 8.9184 s | 8.4098 s | 1.12040x | **1.18816x** |
| confidence | 2.1300 s | 2.1148 s | 1.3472 s | 1.00719x | **1.58106x** |
| sampler, ms/step | 50.2116 | 44.8852 | 42.1826 | 1.11867x | 1.19034x |

Two things fall straight out of that table. **The trunk is 66.9 % of this fold** — much the largest
stage, and the two levers aimed at it move it by 4.4 % between them, which is why a union of nine
lands at 1.108x and not higher. And **the union's sampler is faster than the bit-exact subset's**
(8.41 s against 8.92 s) although no sampler lever separates them: `TT_BIO_DEVICE_CONDITIONING`
leaves the token bias on the device, so the sampler stops paying for an upload it used to make.
A host lever with a sampler-stage effect is exactly the cross-stage term a product of per-stage
singles cannot contain, and it is why the union beats that product.

## 4. The singles, and the additivity discount — `singles_everything_wh_c8.json`, `discount.json`

30 timed folds and 5 cold, same card, same discipline, one base interleaved with four stage arms.
The box was much busier during this session — loadavg 6.56 to 46.75, partly this row's own parity
session on card 9 — and the A/A floor says so: **1.00708x**, five times the fold session's.

| arm | flags | paired ratio | multiple of the floor | readable alone? | bit-exact |
|---|---|---|---|---|---|
| SAMP | the four sampler levers | **1.02170x** | 3.06 | **yes** | yes, base CIF |
| TRUNK | `TRIATT_FUSED_QKVG` | 1.00693x | 0.98 | **no** | yes, base CIF |
| MSA | `PWA_RESIDENCY` | 1.00660x | 0.93 | **no** | yes, base CIF |
| HOSTONLY | the three host levers | **1.06830x** | 9.65 | **yes** | no, `bce2170b…` |

    product of the four in-session singles      1.10630x
    measured union (fold session)               1.10789x
    DISCOUNT                                    -0.144 %      the union is ABOVE its product

    host trio's marginal on top of BX, one session   1.06413x
    host trio alone, same box                        1.06830x
    DISCOUNT                                          0.39 %

**The falsifier needed the union more than 5 % BELOW the product and it landed above.** Two
constructions, one crossing a session boundary and one not, agree: composing nine levers across
four stages of this fold costs between nothing and 0.4 %. Composing is not where this campaign
loses accuracy in its stacked projections; **double-counting one site as two levers is, and that
was worth 7.10 % when `b2z2-sampler-union-wh` caught it.**

**`TRUNK` and `MSA` are inside their session's own floor and are therefore not individually
readable as fold levers.** Both are bit-exact and both are in the union on that basis, not on a
fold ratio. Their published numbers are block and layer ratios — 1.01459x and 1.02603x on
instruments a quarter to a tenth the size of a fold — and what a fold can say about them is that
they do not cost anything. `b2z2-trunk-byte-round2`'s finding that a deleted read is priced by its
reader rather than its bytes, with a coefficient of 1.3 to 2.9, argues the trunk lever could be
worth more than its DRAM ledger says; the fold measured 1.00693x either way, and a fold
measurement outranks a ledger in both directions.

**HOSTONLY reproduces its own row independently**: 1.06830x here against `b2z2-host-device-
compose`'s 1.06495x, 0.31 % apart, different session, different base.

## 5. Parity — `parity_seeds_wh_c9.json`, scored to `parity_score_wh_c9.json`

14 folds, card 9, two sizes, three seeds per arm plus an A/A repeat, `base` and the full union as
`off` and `on`, `perf/b2z2_fusebias/score.py` unchanged. `cdk2x2_512` cannot be read whole — it is
CDK2 fused to its own first 214 residues with no inter-domain interface, so the hinge saturates
whole-molecule RMSD — which is why every reading below is per pseudo-domain and priced against the
same arm's own seed spread.

| reading | 512 aa union | 512 aa seed floor | 298 aa union | 298 aa seed floor |
|---|---|---|---|---|
| worst per-domain all-atom | **0.40355 Å** | 1.86363 Å | **0.29521 Å** | 1.24155 Å |
| worst per-domain CA | 0.25642 Å | 1.40434 Å | 0.14380 Å | 0.85702 Å |
| CA-lDDT, worst | 0.99567 | 0.91869 | 0.99968 | 0.98915 |
| hinge | 6.93° | 160.87° | — | — |
| A/A repeat | **byte-identical** | — | **byte-identical** | — |

Against the 0.60 Å kill bar `CLOSING.md` records, **0.40355 Å passes with room**, and it is 4.6x
inside the sampler's own seed spread. The 298 aa reading is **0.29521 Å — the host union's
0.295206 Å to five decimals**, which is the sharpest confirmation this row has that the six
bit-exact levers change no structure at all: the union's parity IS the host trio's parity, not
near it.

**On the amendment's warning, explicitly: none of this row's parity evidence is a block output
against `tt_bio.reference`.** Every reading is a whole fold against a whole fold with the real
checkpoint. The negative control is not asserted, it is exercised: the A/A repeat comes back
byte-identical at both sizes while the union differs from base at every seed, so the comparison
is demonstrably able to fail.

## 6. The instrument traps, each one checked rather than recited

* **One cold fold PER ARM, not one per session.** An arm that adds device programs compiles them
  on its first fold, and a single session-level cold fold would put ALL's compile inside the first
  timed base. The cold folds are in the JSON with `warmup: true` and they show it: cold base
  48.382 s against a 40.561 s median.
* **`TT_METAL_DEVICE_PROFILER` is absent, not 0.** Both drivers assert it is not in the
  environment at all.
* **The grabbed-step over-pricing is not in these numbers.** Nothing here is a replayed step;
  every ratio is a whole fold or a stage of one. For scale, the sampler row's step harness read
  1.13920x where the in-fold sampler read 1.08702x, a 4.80 % over-price.
* **The arms cannot collapse into each other.** All thirteen flag names are asserted absent from
  the environment before the device opens; the arms are module globals and env vars set in
  process, and each fold records the flags it read back off the code, not off the setter.
* **Six `TT_FATAL Out of Memory` lines are in the fold log and none belong to this row.** All six
  are in the first cold fold, before any lever arm ran: a 67108864 B L1 allocation refused across
  72 banks, which is `main`'s own probe-and-fall-back for the L1 layer norm. Zero appear in any
  armed fold.
* **The sibling card was checked free before it was taken.** Card 9's lease had been released 49
  minutes earlier by a concluded row; the grant was widened to `8,9` on that one command and both
  opens were pinned, because a process that can see every chip brings up every chip.

## 7. The prediction, scored

Written before the first device process, `perf/b2z2_everything/PREDICTED.md` @ `e36c800f0`, not
edited since.

* **P1 CONFIRMED, at the top of the band.** Predicted 1.085x within 1.06x - 1.11x; measured
  **1.10789x**. The point estimate was low by 2.1 % and the reason is P6.
* **P2 WRONG IN THE CONSERVATIVE DIRECTION, and the falsifier did not fire.** Predicted a 1.5 %
  discount inside a 0 - 4 % band; measured **-0.144 %**, slightly super-additive, outside my band
  on the low side. I predicted the falsifier would not fire and it did not, by a factor of 35.
* **P3 CONFIRMED on both named levers.** I predicted QKVG and PWA would not be readable on the
  fold; they read 0.98 and 0.93 of their session's A/A floor. I also predicted no lever loses,
  and none does. SDPAQ was not separately measured — the SAMP arm bundles all four sampler levers
  — so that clause is **untested**, not confirmed.
* **P4 CONFIRMED, all three clauses, including the one I most expected to be wrong about.** BX
  writes the base CIF byte for byte; ALL does not; and **ALL and HOSTONLY both write
  `bce2170b5bf6c9cf`**, so six bit-exact levers compose onto the host path changing nothing.
* **P5 CONFIRMED exactly.** 0.29521 Å at 298 aa against the host union's published 0.295206 Å.
* **P6 WRONG ON EVERY STAGE, all four in the same direction.** Predicted sampler 1.10 - 1.12x,
  trunk 1.01 - 1.03x, confidence ≥1.15x, conditioning ≥1.3x; measured **1.18816x, 1.04429x,
  1.58106x, 2.97452x**. Every stage beat its band, and the fold ratio still landed inside P1's
  band because I had the trunk at 50 % of the fold when it is **66.9 %**. Two errors in opposite
  directions cancelled, which is worth saying rather than hiding.

## 8. What is NOT done

1. **No Blackhole number.** This row held whglx cards 8 and 9 and nothing else. `CLOSING.md`'s
   cell is one Blackhole processor of a p300c, and a WH fold ratio is not a BH fold ratio — this
   wave has seen one transfer within 2 % and another under-predict by 41 %.
2. **No single session timed every arm.** The discount is two constructions, one of which crosses
   a session boundary on the denominator. A seven-arm session at six reps is 48 folds, about 33
   minutes, and would collapse both into one number. The two constructions agree to 0.5 %, which
   is why it was not worth holding the row open for.
3. **SDPAQ has no isolated fold leg.** It is inside SAMP and inside the union; its own ratio on
   the fold is unmeasured.
4. **No armed capture**, so TILE-MOVEMENT-DELTA is named rather than measured.
5. **Nothing is merged and no flag defaults on.** `model-merge-approval-gate` stands, and the
   three host levers are not bit-exact, so they need Moritz. The six bit-exact ones need an
   OOM/size check rather than an accuracy argument.

## 9. Running it

    cd /home/tt-admin/wt-everything-union            # tt-admin: the lease dir is tt-admin-owned
    TT_VISIBLE_DEVICES=8 TT_BIO_LEASE_CARDS=8 \
    TT_BIO_LEASE_HOLDER=worker:b2z2-everything-union-wh \
    TT_BIO_TRACE_REGION_SIZE=$((512*1024*1024)) PYTHONPATH=$PWD \
    /home/mthuening/work/tt-bio/env/bin/python3 perf/b2z2_everything/fold_ab.py \
      --out perf/b2z2_everything/fold_everything_wh_c8.json \
      --cifs perf/b2z2_everything/cif --arms base,BX,ALL --reps 6

    ... perf/b2z2_everything/parity_seeds.py --out <json> --cifdir <dir> --seeds 0,1,2
    ... perf/b2z2_fusebias/score.py <cifdir> --runs <json> --split 298 --out <json>
    python3 perf/b2z2_everything/discount.py        # host only, reads the committed folds

Arms: `base`, `BX` (the six bit-exact), `ALL` (all nine), `SAMP`, `TRUNK`, `MSA`, `HOSTONLY`.
The union is a set of env flags you turn on: `TT_BIO_ATOM_SHIFT_GATHER`, `TT_BIO_ATOM_KV_PREPROJ`,
`TT_BIO_ATOM_L1`, `TT_BIO_SDPA_GRID_Q_CHUNK`, `TT_BIO_TRIATT_FUSED_QKVG`, `TT_BIO_PWA_RESIDENCY`,
`TT_BIO_DEVICE_CONDITIONING`, `TT_BIO_DEVICE_ZINIT`, `TT_BIO_DEVICE_CONFIDENCE`.
