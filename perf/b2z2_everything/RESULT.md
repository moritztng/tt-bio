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

VERDICT: GO — **the whole union is 1.10789x paired on a 512 aa Wormhole fold, n=6, against a
bootstrapped A/A floor of [0.99650, 1.00351], and it passes parity at both fixture sizes.** All six paired ratios are
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
MEASURED: 80 folds in four sessions on two chips. 21 folds base/BX/ALL
(`fold_everything_wh_c8.json`), 35 folds base/SAMP/TRUNK/MSA/HOSTONLY
(`singles_everything_wh_c8.json`), 27 folds base/TRUNK/MSA at n=8
(`readers_everything_wh_c8.json`), 14 folds at two sizes and three seeds for parity
(`parity_seeds_wh_c9.json`, scored to `parity_score_wh_c9.json`). Drivers
`perf/b2z2_everything/fold_ab.py`, `parity_seeds.py`, `floor.py`, `discount.py`,
`lever_overlap.py`, all results committed beside them.
UNION-RATIO: **1.10789x** paired on a 512 aa WH fold, n=6, against a floor **bootstrapped at the
same n** — [0.99650, 1.00351], 20000 resamples of the median of six base-vs-base ratios. **30.7x
clear.** The folded-median floor this row first quoted (1.00136x) is superseded: it is a median of
n-1 one-sided values, not a two-sided band on a median of n.
BITEXACT-SUBSET: **1.04112x**, CIF `da476491dbb2a847…` identical to base at plDDT 0.847187.
UNION-DISCOUNT: **-0.144 %** — the union measures 1.10789x against a 1.10630x product of the four
in-session stage singles, all of them **paired FOLD ratios; no block or step ratio is multiplied
anywhere in this row's arithmetic**. **The pre-registered falsifier needed 5 % BELOW and the union
came in above. It does not fire.** Second construction, crossing no session boundary on the
numerator and the only one whose every term clears its own bootstrapped floor: the host trio's
marginal on top of the six bit-exact levers is **1.06413x** against **1.0683x** standing alone, a
**0.39 %** discount. `perf/b2z2_everything/discount.json`.
READ-DELETERS: five of the nine levers delete a reader of a tensor somebody else still reads
(`ATOM_SHIFT_GATHER`, `ATOM_KV_PREPROJ`, `ATOM_L1`, `TRIATT_FUSED_QKVG`, `PWA_RESIDENCY`), three
delete the LAST reader of what they remove (the host trio) and one deletes no reader at all
(`SDPA_GRID_Q_CHUNK`). **The split predicts which numbers transfer** and §4-A shows it did.
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

**A/A floor [0.99650, 1.00351]**, bootstrapped at the quoted n: 20000 resamples of the median of
six base-vs-base ratios drawn from this session's own six base folds, which span 40.389 to
40.612 s on a box whose loadavg ran 7.15 to 14.88. The union clears the upper bound by **30.7x**
and the bit-exact subset by **11.7x**.

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

## 4. The singles, the discount, and which levers transfer

30 timed folds and 5 cold on the same card, one base interleaved with four stage arms, plus a
later 27-fold session at n=8 for the two levers that turned out to need it. **Both sessions ran on
a contended box** — the first to loadavg 46.7 — and the bootstrapped fold floor says so:
[0.97455, 1.02786] and [0.97567, 1.02540]. At that width a lever worth 1 % of a fold is not
measurable on the fold at all, and three of the four stage arms are inside it.

| arm | paired FOLD ratio | inside its session's fold floor? | bit-exact |
|---|---|---|---|
| SAMP, the four sampler levers | 1.02170x | **yes, not readable** | yes, base CIF |
| TRUNK, `TRIATT_FUSED_QKVG` | 1.00693x (n=6), 1.01705x (n=8) | **yes, not readable** | yes, base CIF |
| MSA, `PWA_RESIDENCY` | 1.00660x (n=6), 1.00966x (n=8) | **yes, not readable** | yes, base CIF |
| HOSTONLY, the three host levers | **1.06830x** | no — **readable**, 2.45x the floor | no |

**This corrects what this row said one pass ago.** With the folded-median floor the same session
made SAMP look readable at "3.06x the floor"; the bootstrapped null puts it inside. The correction
runs the opposite way to `b2z2-trunk-fold-ab-bh`'s — a folded median of adjacent pairs is too
NARROW on a noisy box, where a split-half floor is too wide — but the lesson is the same one:
**quote the floor of the estimator you actually quote, at the n you quote it at.**

### 4-A. The instrument, not the lever: two levers are readable on the stage they act in

A trunk lever's honest instrument is the trunk stage of a real fold, and a stage wall is an order
of magnitude quieter than the fold that contains it — floor [0.99270, 1.00740] against the fold's
[0.97567, 1.02540] in the same session. Same folds, same paired estimator, same bootstrap.

| arm | trunk-stage ratio | floor | verdict |
|---|---|---|---|
| TRUNK `qkvg`, n=8 | **1.01166x** | [0.99270, 1.00740] | **READABLE** |
| TRUNK `qkvg`, n=6, other session | **1.01255x** | [0.99405, 1.00599] | **READABLE** |
| MSA `PWA`, n=8 | 1.00904x | [0.99270, 1.00740] | readable |
| MSA `PWA`, n=6, other session | 1.00464x | [0.99405, 1.00599] | **not** readable |
| BX, fold session | 1.02028x | [0.99738, 1.00263] | **READABLE** |
| ALL, fold session | 1.04437x | [0.99738, 1.00263] | **READABLE** |
| HOSTONLY (`z_init`), n=6 | 1.02366x | [0.99405, 1.00599] | **READABLE** |

**`qkvg` is a real in-fold lever and this row's earlier "not readable" was an instrument verdict,
not a lever verdict.** Two independent sessions put it at 1.01166x and 1.01255x on the trunk stage,
**0.09 pp apart**, against a published 1.01459x on the block. Solving for the share that reconciles
them, the Pairformer blocks are **80.2 %** of the trunk stage — so the block ratio reaches the
stage at a transfer coefficient of essentially **1.0**, which is `b2z2-trunk-fold-ab-bh`'s good
half, reproduced on Wormhole. Amdahl on the 65.6 % trunk share puts it at **~1.008x of the fold**,
and a contended WH fold cannot resolve 0.8 %. The lever is fine; the fold is the wrong ruler.

**`PWA` stays unresolved and this row will not claim it.** 1.00904x and 1.00464x in two sessions is
not 0.09 pp agreement, and both are above the 1.0034x its 1.02603x layer ratio predicts on a 13 %
MSA share of the trunk stage. What can be said is that it costs nothing and is bit-exact.

**And the trunk stage adds almost exactly**: BX 1.02028x x HOSTONLY 1.02366x = 1.04423x against
ALL's measured **1.04437x**, 0.014 pp apart, on the noisiest stage's quietest instrument.

### 4-B. The discount

    product of the four in-session stage FOLD singles      1.10630x
    measured union (fold session)                          1.10789x
    DISCOUNT                                              -0.144 %     the union is ABOVE it

    host trio's marginal on top of BX, one session          1.06413x
    host trio alone, same box                               1.06830x
    DISCOUNT                                                 0.39 %

**The falsifier needed the union more than 5 % BELOW the product and it landed above.** The second
construction is the one to trust: both its terms clear their own bootstrapped floors, where three
of the first construction's four factors do not. Two constructions, one crossing a session boundary
and one not, agree: composing nine levers across four stages of this fold costs between nothing and
0.4 %. **Composing is not where this campaign loses accuracy in its stacked projections.
Double-counting one site as two levers is, and that was worth 7.10 % when `b2z2-sampler-union-wh`
caught it.**

**No block or step ratio is multiplied anywhere in this row's arithmetic.** Every factor above is a
paired fold ratio or a paired stage wall of a real fold. The prediction this row pre-registered
DID build its product out of block and step ratios, which is exactly the arithmetic
`b2z2-trunk-fold-ab-bh` has since measured wrong; §7 scores it on that basis.

### 4-C. Which levers delete the LAST reader, and which only delete a share

`discount.json` carries the classification. It predicts what transferred here.

| lever | deletes | transferred? |
|---|---|---|
| `TT_BIO_DEVICE_CONDITIONING` | **the last** reader — a 201 MB host download, nothing else reads it | **yes**: the trio reads 1.06830x here against 1.06495x published, 0.31 % apart on a different session |
| `TT_BIO_DEVICE_ZINIT` | **the last** — z_init never crosses PCIe | yes, and it is visible as 1.02366x on the trunk stage |
| `TT_BIO_DEVICE_CONFIDENCE` | **the last** — the head reads the pair tensor the trunk left on device | yes, 1.64930x on the confidence stage |
| `TT_BIO_TRIATT_FUSED_QKVG` | **a share** — the normed pair tensor goes from two readers to one, and a **third** reader remains | **partly**: full transfer to the trunk stage, invisible at the fold |
| `TT_BIO_PWA_RESIDENCY` | **a share** — the normed MSA rows stop being re-read `2*n_heads` times | **unresolved at the fold** |
| `TT_BIO_ATOM_SHIFT_GATHER`, `ATOM_KV_PREPROJ`, `ATOM_L1` | **a share** — a gather matmul, three of four projections, DRAM readers of a resident tensor | as a group, 1.09242x on the sampler stage, readable; individually not measured here |
| `TT_BIO_SDPA_GRID_Q_CHUNK` | **nothing** — it is occupancy, not movement | inside SAMP, no isolated leg |

**The three levers that transfer at full value are exactly the three that delete the last reader of
what they remove, and the two that do not reach the fold are the two that delete a share of the
readers of a tensor that keeps its other consumers.** That is `b2z2-trunk-fold-ab-bh`'s rule,
measured independently here on Wormhole, on a different lever set, at a different size.

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
checkpoint, so the 23 zero-initialised PairformerLayer weights that let `b2z2-trunk-fold-ab-bh`'s
first run pass with a control that could not fail are not in this instrument at all.

**And the negative control is not a synthetic perturbation, so it cannot be the one that rounds
away.** A bf16 control needs a perturbation above 2^-8 or it lands back on the same tile; this row
ran none, because it did not need one. The control here is the union arm itself: it differs from
base at every seed at both sizes, by a different sha256 and a different plDDT, while the A/A
repeat comes back byte-identical at both sizes. The comparison is shown able to fail on the same
folds that produce the result, not on a probe beside them.

## 6. The instrument traps, each one checked rather than recited

* **The A/A floor is bootstrapped at the n the ratio is quoted at**, 20000 resamples of the
  median of n base-vs-base ratios (`floor.py`). Both the split-half floor and the folded median of
  adjacent pairs answer a different question, and the folded median this row first quoted was too
  NARROW on a contended box — it made a 1.02170x arm look like 3x its floor when the proper null
  contains it. The band is reported two-sided and for both null populations, all pairs and
  adjacent pairs only.
* **A stage wall is a legitimate in-fold instrument and a much quieter one**, and using it is not
  the same as grabbing a block: nothing is replayed, the fold runs whole, the stage is timed with a
  device sync at each boundary. That is how `qkvg` became readable.
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

* **P1 CONFIRMED, at the top of the band — and its arithmetic is now known to be the wrong kind.**
  Predicted 1.085x within 1.06x - 1.11x; measured **1.10789x**. But the 1.1015x product that band
  was built on multiplied BLOCK and STEP ratios, which `b2z2-trunk-fold-ab-bh` has since measured
  wrong for read-deleting levers. The prediction landed for reasons partly unrelated to its method,
  and this row's reported discount does not use that method.
* **P2 WRONG IN THE CONSERVATIVE DIRECTION, and the falsifier did not fire.** Predicted a 1.5 %
  discount inside a 0 - 4 % band; measured **-0.144 %** and **+0.39 %** by the two constructions,
  at or below my band's floor. I predicted the falsifier would not fire and it did not, by a
  factor of at least 12.
* **P3 HALF RIGHT, and the half I got wrong is this pass's finding.** I predicted QKVG and PWA
  would not be readable on the fold and they are not — but I attributed that to the levers being
  too small, and **QKVG is readable on the trunk stage in two independent sessions, 0.09 pp
  apart**. What the fold cannot resolve is the instrument's limit, not the lever's size. PWA
  stays unresolved. "No lever loses" holds: no arm reads below its floor. SDPAQ has no isolated
  leg, so that clause is **untested**, not confirmed.
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
4. **`PWA_RESIDENCY` is unresolved in-fold.** Two sessions put it at 1.00904x and 1.00464x on the
   trunk stage, readable in one and not the other, and both above what its layer ratio predicts.
   It is in the union because it is bit-exact and costs nothing, not because a fold measured it.
5. **Every fold number here was taken on a contended box.** The singles session ran to loadavg
   46.7 and the n=8 session sat at 15-18 throughout. The union's own ratio is 30.7x its floor and
   does not care; the 1 %-class levers do, and that is why they needed the stage instrument. A
   benchlocked quiet box would resolve them on the fold directly.
6. **No armed capture**, so TILE-MOVEMENT-DELTA is named rather than measured.
7. **Nothing is merged and no flag defaults on.** `model-merge-approval-gate` stands, and the
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
    python3 perf/b2z2_everything/floor.py           # host only, bootstraps every session's floor
    python3 perf/b2z2_everything/discount.py        # host only, reads the committed folds

Arms: `base`, `BX` (the six bit-exact), `ALL` (all nine), `SAMP`, `TRUNK`, `MSA`, `HOSTONLY`.
The union is a set of env flags you turn on: `TT_BIO_ATOM_SHIFT_GATHER`, `TT_BIO_ATOM_KV_PREPROJ`,
`TT_BIO_ATOM_L1`, `TT_BIO_SDPA_GRID_Q_CHUNK`, `TT_BIO_TRIATT_FUSED_QKVG`, `TT_BIO_PWA_RESIDENCY`,
`TT_BIO_DEVICE_CONDITIONING`, `TT_BIO_DEVICE_ZINIT`, `TT_BIO_DEVICE_CONFIDENCE`.
