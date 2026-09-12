# b2z2-everything-union-wh — eleven levers are nine, and together they are 1.10789x on a WH fold

TASK TYPE: ACCELERATE (compose + VERIFY/BENCHMARK) | PLAYBOOKS loaded: ACCELERATE +
VERIFY/BENCHMARK + ALWAYS-ON | memories read: `git-merge-no-conflict-markers-can-still-be-
semantically-broken`, `unified-solution-not-per-model-patches`, `no-speedup-by-skipping-the-
models-own-work`, `merged-lever-defaults-off-is-not-a-landed-win`, `model-merge-approval-gate`,
`perf-page-cell-is-historical-not-live-baseline`, `rfd3-isolated-screen-underprices-residency-
lever`, `whglx-cross-account-artifacts-strand-off-worktree`, `whglx-unpinned-all-chip-open-breaks-
every-cotenant`, `ssh-remote-background-launch-stdin-hang`, `git-worktree-add-f-steals-branch-ref`

VERDICT (pass 1): **the whole union is 1.10789x paired on a 512 aa Wormhole fold, n=6, against
an A/A floor of 1.00136x, and the six bit-exact levers inside it write the base CIF byte for
byte.** All six paired ratios are positive and span 1.09535x to 1.12464x. The brief's eleven
levers are **nine**: `wk/b2z2-step-program-fusion` contributes only the defective key window and
is an ancestor of four of the other branches, and `TT_BIO_MAC_FUSE` is its own row's NO-GO.
BRANCH: `wk/b2z2-everything-union-wh` @ `7b28cd3a5`, cut from `origin/main` `84da2a49b`, pushed.
Not merged. Every flag defaults OFF.
CARD: whglx (j10glx02) card 8, one Wormhole_B0 of the 32-chip mesh, 8x9 grid, pinned
`TT_VISIBLE_DEVICES=8` `TT_BIO_LEASE_CARDS=8`
`TT_BIO_LEASE_HOLDER=worker:b2z2-everything-union-wh`, `TT_BIO_TRACE_REGION_SIZE` 512 MiB,
`TT_METAL_DEVICE_PROFILER` absent. Run as `tt-admin` from `/home/tt-admin/wt-everything-union`
(the lease dir is tt-admin-owned); **every artifact was copied back into the checkout this doc
commits from** and is committed under `perf/b2z2_everything/`.
ARCH: Wormhole for every number here. The published cell is Blackhole and this row makes no BH
claim, projected or otherwise.

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

* **`layout-op-elision` vs `step-adaln-sdpa`, both inside `AttentionPairBias`.** Read: the
  elision replaces the one-hot gather that builds `s_kv`; the SDPA lever only changes the
  `program_config=` argument of four `scaled_dot_product_attention` calls and the helper that
  builds it. Different statements in the same class. Composed.
* **`sampler-union-wh` vs `host-device-compose`, one changed line in common at
  `tt_bio/tenstorrent.py:9750`, inside `DiffusionModule`.** Read: the sampler union wraps the
  cached `keys_indexing` in an `_AtomShiftGather` sentinel; the host branch rewrites where
  `bias_token` comes from, about thirty lines away. **Line-number coincidence on the branch side,
  not a shared site.** Composed.

The one pair the tool could not have caught and the sampler row did: `TT_BIO_SDPA_GRID_Q_CHUNK`'s
branch already CONTAINS `TT_BIO_ATOM_L1` and the key window, so its published 1.01831x is a
marginal on top of them. It is taken here as the marginal diff
`step-fusion-next-sites...step-adaln-sdpa`, which is 52 lines in `tenstorrent.py` plus the `work`
argument at the shared helper's call sites in `esmc.py`, `esmfold2.py` and `saprot.py` — the
lever is unified across the four models that share that helper, not a boltz2 patch.

## 2. The branch

Five commits off `origin/main` `84da2a49b`. The sampler subset is taken by fast-forwarding onto
`wk/b2z2-sampler-union-wh`, which is itself a linear composition onto the same `origin/main`, so
its provenance and its two harnesses come with it. The other four levers are applied as diffs
against `origin/main` restricted to `tt_bio/`, not merged as trees — all the sampler branches
edit the same twenty lines and a three-way merge there is a guess.

**Every flag defaults OFF.** One change of substance was needed: the three PairWeightedAveraging
knobs default ON on their own branch, so here they take their default from one
`TT_BIO_PWA_RESIDENCY`, default OFF, and `TT_BIO_PWA_RESIDENCY=1` reproduces exactly the
configuration that branch measured.

## 3. The fold — `perf/b2z2_everything/fold_everything_wh_c8.json`, commit `e36c800f`

18 timed folds and 3 cold, one process, one device open, `cdk2x2_512` with its fixed 35-row a3m,
**200 sampling steps, 3 recycles, full MSA depth**, 1 sample, seed 0, arm order reversed on
alternate reps, `step_n == 200` read back off the sampler on all 21 folds.

| arm | levers | median fold | paired ratio | every rep | CIF sha256 | plDDT |
|---|---|---|---|---|---|---|
| base | — | **40.561 s** | 1.00000x | — | `da476491dbb2a847…` | 0.847187 |
| BX | the six bit-exact | 38.918 s | **1.04112x** | 1.0135 – 1.0475 | **`da476491dbb2a847…`** | **0.847187** |
| **ALL** | all nine | **36.570 s** | **1.10789x** | 1.0954 – 1.1246 | `bce2170b5bf6c9cf…` | 0.84647 |

**A/A floor 1.00136x** on the same estimator — the median of consecutive base-fold pairs, six
base folds spanning 40.389 to 40.612 s (spread 1.00552x) on a box whose loadavg ran 6.15 to
18.12. The union clears that floor by **79x**; the bit-exact subset by 30x.

**Both hash bars pass, and they are the check that matters.** BX writes the base CIF byte for
byte at the same plDDT, one sha256 across all six reps — so six levers compose without moving one
atom. ALL does not, and that is the point: three of its members are host levers that recompute
in bf16 on the device, and an identical hash on both arms of a non-bit-exact A/B would have meant
the toggle never reached the code. Engagement is also counted rather than assumed: the shift
gather served 1200 calls and fell back 0, `ATOM_L1` 1200 L1 / 0 DRAM, `qkvg_heads` 560 served /
0 fallbacks, on every armed fold and zero on base.

### Per-stage walls, so the composition can be attributed

| stage | base | BX | ALL | BX ratio | ALL ratio |
|---|---|---|---|---|---|
| trunk (all 4 recycles, MSA inside) | 27.1398 s | 26.5979 s | 25.9888 s | 1.02037x | **1.04429x** |
| diffusion conditioning | 0.8171 s | 0.8527 s | 0.2747 s | 0.95825x | **2.97452x** |
| sampler (200 steps) | 9.9922 s | 8.9184 s | 8.4098 s | 1.12040x | **1.18816x** |
| confidence | 2.1300 s | 2.1148 s | 1.3472 s | 1.00719x | **1.58106x** |
| sampler, ms/step | 50.2116 | 44.8852 | 42.1826 | 1.11867x | 1.19034x |

Two things fall straight out of that table. **The trunk is 66.9 % of this fold** — much the
largest stage, and the two levers aimed at it move it by 4.4 % between them, which is why a union
dominated by sampler levers lands at 1.108x and not higher. And **the union's sampler is faster
than the bit-exact subset's sampler** (8.41 s against 8.92 s) although no sampler lever separates
them: `TT_BIO_DEVICE_CONDITIONING` leaves the token bias on the device, so the sampler stops
paying for an upload it used to make. A host lever with a sampler-stage effect is exactly the
kind of cross-stage term a product of per-stage singles cannot contain.

## 4. The additivity discount — there is none, the union is slightly ABOVE the product

Product of the surviving singles, each converted to a fold contribution by Amdahl on **this
session's measured base shares** (sampler 24.635 %, trunk 66.910 %, MSA track 8.82 % from
`CLOSING.md`, confidence 5.251 %):

    sampler  1.08702x in-fold (SHG+KVP+L1) x 1.01831x (SDPAQ step)  = 1.10693x  -> 1.02437x fold
    trunk    1.01459x on the block, block <= the whole trunk        -> <= 1.00971x fold
    MSA      1.02603x on MSALayer, track 8.82 %                     -> 1.00224x fold
    host     1.06495x, measured directly on a WH fold               -> 1.06495x fold
    PRODUCT                                                            1.10396x
    MEASURED UNION                                                     1.10789x
    DISCOUNT                                                          -0.36 %   (super-additive)

**The pre-registered falsifier does not fire, and it does not come close: it needed the union to
land more than 5 % BELOW the product and the union landed 0.36 % above it.** The sampler stage
says the same thing on its own instrument: the bit-exact sampler levers predict 1.10693x on the
sampler wall and measure **1.12040x**, 1.22 % better than their product.

**The honest caveat, and it is why a second session is running.** Every single in that product
comes from another row's session on another base, so the comparison mixes sessions and the trunk
term is an upper bound (the block is not the whole trunk). A four-arm singles session — SAMP,
TRUNK, MSA, HOSTONLY against the same interleaved base, six reps each — was launched on the same
card and the same device-open discipline at 22:42 UTC and writes
`perf/b2z2_everything/singles_everything_wh_c8.json`. **The in-session discount is that file's
answer, not this one's**, and until it lands the -0.36 % above is a cross-session figure and
labelled as such.

## 5. The instrument traps, each one checked rather than recited

* **One cold fold PER ARM, not one per session.** The harness folds every arm once and discards
  it before the first timed rep, because an arm that adds device programs compiles them on its
  first fold and a single session-level cold fold would put ALL's compile inside the first
  timed base. The cold folds are in the JSON with `warmup: true` and they show it: cold base
  48.382 s against a 40.561 s median.
* **`TT_METAL_DEVICE_PROFILER` is absent, not 0.** The harness asserts it is not in the
  environment at all.
* **The grabbed-step over-pricing is not in this number.** Nothing here is a replayed step;
  every ratio is a whole fold or a stage of one. For the record the sampler row's step harness
  read 1.13920x where the in-fold sampler read 1.08702x, a 4.80 % over-price, and this row's
  in-fold BX sampler ratio of 1.12040x includes SDPAQ on top of that 1.08702x.
* **The arms cannot collapse into each other.** Every one of the thirteen flag names is asserted
  absent from the environment before the device opens; the arms are module globals and env vars
  set in process, and each fold records the flags it read back off the code, not off the setter.
* **Six `TT_FATAL Out of Memory` lines are in the log and none of them belong to this row.** All
  six are in the first cold fold, before any lever arm ran: a 67108864 B L1 allocation refused
  across 72 banks, which is `main`'s own probe-and-fall-back for the L1 layer norm. Zero appear
  in any armed fold.

## 6. CHEAT-CHECK

Clean, and the driver asserts it rather than the author: `--steps == 200` and `--recycles == 3`
are asserted before the device opens, and `sp.n_steps == 200` is read back off the sampler on
every one of the 21 folds. Full MSA depth, the fixture's own 35-row a3m, one sample, seed 0,
templates off. No lever removes a unit of the model's own work: six of the nine are `torch.equal`
and the union containing them writes the base CIF byte for byte, and the three that are not
bit-exact move work from the host to the device rather than deleting it.

## 7. What is NOT done

1. **Parity.** The host levers are not bit-exact and this pass has not scored them. The 512 aa
   CIFs are committed (`perf/b2z2_everything/cif/`) and `perf/b2z2_fusebias/score.py` is on the
   branch; what is owed is the 298 aa fold pair and both scorings. `CLOSING.md`'s bar is 0.60 Å
   worst per-pseudo-domain all-atom at 512 aa.
2. **The in-session singles**, running now — the discount in §4 is cross-session until it lands.
3. **P4's third clause is untested**: I predicted ALL writes the same CIF as HOSTONLY byte for
   byte, because bit-exact levers change no structure. The singles session produces the HOSTONLY
   CIFs that settle it. If they differ, bit-exactness does not compose through the host path and
   that is a larger finding than the ratio.
4. **No Blackhole number.** This row holds whglx card 8 and nothing else.
5. **Nothing is merged**, and no flag defaults on. `model-merge-approval-gate` stands.

## 8. Running it

    cd /home/tt-admin/wt-everything-union            # tt-admin: the lease dir is tt-admin-owned
    TT_VISIBLE_DEVICES=8 TT_BIO_LEASE_CARDS=8 \
    TT_BIO_LEASE_HOLDER=worker:b2z2-everything-union-wh \
    TT_BIO_TRACE_REGION_SIZE=$((512*1024*1024)) PYTHONPATH=$PWD \
    /home/mthuening/work/tt-bio/env/bin/python3 perf/b2z2_everything/fold_ab.py \
      --out perf/b2z2_everything/fold_everything_wh_c8.json \
      --cifs perf/b2z2_everything/cif --arms base,BX,ALL --reps 6

Arms: `base`, `BX` (the six bit-exact), `ALL` (all nine), `SAMP`, `TRUNK`, `MSA`, `HOSTONLY`.
