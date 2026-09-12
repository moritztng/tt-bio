# b2z2-atoml1-size-curve — TT_BIO_ATOM_L1 across the sizes users fold, both arms, both parts

TASK TYPE: VERIFY/BENCHMARK (+ ACCELERATE, one lever lifted onto main) | PLAYBOOKS loaded:
VERIFY/BENCHMARK + ACCELERATE + ALWAYS-ON | memories read:
`tt-bio-tuned-at-512-l1-gates-go-dark-above-640aa`,
`whglx-tt-visible-devices-is-a-umd-logical-id-not-a-device-node`,
`whglx-galaxy-lease-card-number-vs-device-node-mismatch`,
`whglx-cross-account-artifacts-strand-off-worktree`,
`perf-baseline-cell-keyed-to-card-type-not-dispatch-grant`,
`rfd3-isolated-screen-underprices-residency-lever`, `one-size-tuning-is-a-standing-defect-class`,
`no-speedup-by-skipping-the-models-own-work`, `model-merge-approval-gate`,
`benchlock-one-shot-check-blind-to-mid-run-contention`,
`sibling-perf-campaigns-need-namespaced-output-paths`,
`qb2-tt-smi-reset-resets-board-pair-not-chip`

ARCH: WH+BH — Wormhole is whglx `j10glx02`, 8x9 grid, cards 24/25/26/28/0, five sizes.
Blackhole is qb2 `tt-quietbox2`, 11x10 grid, one processor of a p300c, cards 2 and 3, two sizes.
ttnn 0.68.0 on both. `perf/size512/fixtures/cdk2x2_{512,640,768,896,1024}.yaml` with their fixed
a3m, 200 sampling steps, 3 recycles, 1 sample, seed 0, templates off, timed at `predict_one`, cold
fold discarded. **Every table below names its own part.**

VERDICT: NO-GO on the size hypothesis, and the 768 aa lead is RETIRED. `TT_BIO_ATOM_L1` does not
get more valuable as the target grows. It is a small, real, bit-exact lever worth **~1.08x on the
200-step sampler wall on Blackhole and ~1.05x on Wormhole, at every size from 512 to 1024 aa**, and
that is the same thing the 512 aa cell already said.

The pre-registered falsifier FIRED. On Wormhole, five sizes, n=5 per size, the two arms scale at
**N^2.0028 +/- 0.0701 (base)** and
**N^1.9530 +/- 0.0186 (`TT_BIO_ATOM_L1`)** —
a separation of **0.0498 against a combined stderr of
0.0725, 0.69 sigma**. The curves do not part anywhere. So
`b2z2-bh-stack-atom`'s **85.504 -> 45.243 s at 768 aa was load noise on a contended box**, exactly
as the honest reading of one unpaired fold per arm with loadavg rising said it was. Paired at the
same size on Wormhole the fold reads **0.94060x**, inside a
1.02156x A/A floor. The lead is closed and the campaign should stop chasing it.

What the lever DOES do, at every one of the seven size-by-part points measured:

* **The gate never declines.** `ATOM_L1_STATS` reads `{'l1': 1200, 'dram': 0}` at 512 / 640 / 768 /
  896 / 1024 aa on Wormhole and at 512 / 1024 aa on Blackhole. 1200 = 200 steps x 6 atom layers. It
  is not a lever that silently fell back; it fired on every single call.
* **It is bit-exact everywhere.** Base and `TT_BIO_ATOM_L1` write the same CIF sha256 at every
  size on both parts, including **`a91aa44441f0d9c5` at 512 aa on Blackhole, the digest the perf
  page publishes**.
* **It earns ~1.05x (WH) / ~1.08x (BH) on its own scope, flat in N.** Sampler-wall ratio on
  Wormhole 1.045 -> 1.062 across the
  whole range; on Blackhole 1.085 -> 1.082.
  **Blackhole is worth about 3 points more than Wormhole at every size**, which is the one
  architecture effect in the data and it is a level shift, not a slope.

BRANCH: `wk/b2z2-atoml1-size-curve` — the lever, both harnesses, the fitter, the prediction and
every artifact, pushed to origin. **Not merged to main**, and `model-merge-approval-gate` stands.

CARD: whglx **card 24** (this row's grant) plus idle siblings **25, 26, 28, 0** under the ALWAYS-ON
card-fanout rule, each checked free against the fleet lease files in `~/.coworker/state/leases/` —
the only namespace that means anything on that box, because `TT_VISIBLE_DEVICES`, the lease card
number and `/dev/tenstorrent/N` are three different namespaces there. Blackhole on qb2 **cards 2
and 3**, chosen off the board `000004613193410d` that card 0's live parity gate is NOT on, and no
reset was taken. Every process runs `TT_BIO_LEASE_CARDS=<cards> TT_VISIBLE_DEVICES=<card>
TT_BIO_LEASE_HOLDER=worker:b2z2-atoml1-size-curve TT_BIO_TRACE_REGION_SIZE=536870912` under
`env -u TT_METAL_DEVICE_PROFILER`, so the profiler variable is **absent, not zero**.

DEFICIT-SECONDS: 0.0 s — this row REMOVED a claim rather than adding one. `TT_BIO_ATOM_L1` does not
clear its own session's A/A floor on the Wormhole fold at four of five sizes, and the one seconds
figure the campaign might have banked from it — the **40.3 s/fold at 768 aa** the unpaired screen
appeared to offer — is not there. That is a deletion from the ledger. The lever's real contribution
is on the sampler wall and it is already inside `b2z2-bh-stack-atom`'s Blackhole stack number.

TILE-MOVEMENT-DELTA: 0.0 % — the reason is scope, and this row now has the control on both parts.
`TT_BIO_ATOM_L1` acts inside the atom branch of `Diffusion.__call__`; the 18.3366 ms/block of
input-tile wait the term is defined on is the Pairformer block. The Pairformer wall is recorded on
every one of this row's folds and it does not move. Median Pairformer wall per fold, 280 blocks,
base vs L1: Wormhole 23.989 / 23.994 s at 512 aa, 40.683 / 40.695 at 640, 60.297 / 60.289 at 768,
94.706 / 94.721 at 896, 127.707 / 127.697 at 1024; Blackhole 10.292 / 10.255 at 512 and
47.907 / 47.904 at 1024. **Worst deviation across all seven points: 0.360 %, and on Wormhole it is
under 0.03 % everywhere.** That matches `b2z2-bh-stack-atom`'s direct Blackhole control
(1.00003x on the Pairformer wall against a 1.00779x block floor). The lever does not touch the term.

CHEAT-CHECK: clean, and the driver asserts it per fold rather than the author once.
`size_curve.py` raises unless `step_n == 200` on **every** fold before that fold is recorded, and
`--steps`/`--recycles` are asserted equal to 200 and 3 before the device opens; `block_n` (280) is
read back off the model and stored per fold. **All 109 folds in this pass carry `step_n = 200` and
`block_n = 280` in the artifact.** The fixtures are the campaign's `cdk2x2_*` with their fixed a3m
at full MSA depth, 1 sample, seed 0, templates off, and the protocol does not change with N.
`TT_BIO_ATOM_L1` sets a `ttnn.MemoryConfig` and nothing else: it adds no op, removes no op and
changes no operand, so it cannot do less of the model's own work — and the identical CIF sha256 at
every size on both parts proves it did not.

PARITY: bit-exact is the parity claim and it is the right one for this lever. Base and
`TT_BIO_ATOM_L1` write byte-identical structures at all five Wormhole sizes and both Blackhole
sizes, so there is no accuracy argument to make and no Angstrom number to defend. Checked against
BASE ON THE SAME BOX AND SIZE and **never against `tt_bio.reference`**, which zero-initialises 23 of
a PairformerLayer's weights including both trimuls' `p_out` and would pass for a correct arm, a
wrong arm, and an arm that computed nothing.

MEASURED: **109 folds** in seven interleaved processes — five on Wormhole (one per size) and two on
Blackhole — one device open each, arms ordered `base, L1, base` inside every rep, 5 reps plus a
discarded cold fold, n=5 at every point except Wormhole 1024 aa (n=4 at the time of writing, its
leg still deepening). Driver `perf/b2z2_sizecurve/size_curve.py`, fitter
`perf/b2z2_sizecurve/fit_curve.py`, copier `perf/b2z2_sizecurve/harvest.sh`; artifacts
`curve_<size>_<host>_c<card>.json`, `fit_wh.json`, `fit_bh.json`, per-fold CIFs and logs, all
committed to the branch.

PREDICTED: `state/b2z2-atoml1-size-curve.PREDICTION.md`, committed to the branch as
`perf/b2z2_sizecurve/PREDICTED.md` in `d04f16505` **before this row opened a device** and not
edited since. Eight numbered predictions, scored in full in section 5.

## 1. The gate, done on paper first and then measured on both parts

`_atom_branch_memory_config` prices the branch on its own bytes: `live = B*K*D*(W + 4*ATOM_DIM)*elem`,
which at B=1, W=32, ATOM_DIM=128, D=128, bf16 is **`live = K * 139264 B`**, K = bucketed_atoms/32,
compared against `0.5 * _l1_bank_bytes() * gx * gy`.

| part | grid | budget | declines above | i.e. roughly |
|---|---|---|---|---|
| Wormhole | 8x9 = 72 | **52.62 MB** | K = 378 windows | ~1530 aa |
| Blackhole | 11x10 = 110 | **80.40 MB** | K = 577 windows | ~2340 aa |

**Wormhole, measured:**

| size | K | live | share of the 52.62 MB budget | `ATOM_L1_STATS` | CIF sha256, both arms |
|---|---|---|---|---|---|
| 512 aa | 140 | 19.50 MB | 37 % | `l1 1200, dram 0` | `da476491dbb2a847` |
| 640 aa | 182 | 25.35 MB | 48 % | `l1 1200, dram 0` | `6ab60bdefca5f483` |
| 768 aa | 210 | 29.25 MB | 56 % | `l1 1200, dram 0` | `7860ea474a8139b1` |
| 896 aa | 238 | 33.14 MB | 63 % | `l1 1200, dram 0` | `2108872e687af9ac` |
| 1024 aa | 266 | 37.04 MB | 70 % | `l1 1200, dram 0` | `17bd0c67e2bb4128` |

**Blackhole, measured:**

| size | K | live | share of the 80.40 MB budget | `ATOM_L1_STATS` | CIF sha256, both arms |
|---|---|---|---|---|---|
| 512 aa | 140 | 19.50 MB | 24 % | `l1 1200, dram 0` | `a91aa44441f0d9c5` |
| 1024 aa | 266 | 37.04 MB | 46 % | `l1 1200, dram 0` | `d6ece9fab83c5ca3` |

The lever's own gate has no knee anywhere in the range a user folds, on either part, and the
counters agree at all seven points. **Blackhole's larger grid makes this gate looser, not tighter** —
the live set has no grid term in it and the budget does. The brief that preceded this row had that
backwards; it is the reusable part of the arithmetic.

## 2. The fit, which is the deliverable

`perf/b2z2_sizecurve/fit_wh.json`. Log-log least squares of median fold seconds against N, five
sizes per arm, n=5 per size (1024 aa n=4).

| arm | exponent | stderr | R^2 | points |
|---|---|---|---|---|
| base | **2.0028** | 0.0701 | 0.99634 | 5 |
| `TT_BIO_ATOM_L1` | **1.9530** | 0.0186 | 0.99973 | 5 |

**Separation base - L1 = 0.0498 against a combined stderr of
0.0725: 0.69 sigma. The arms scale the same.** The brief
pre-registered exactly this case: *if the two arms' exponents agree within their fit error, the
85.504 -> 45.243 s screen was load noise on a contended box.* They do, and it was. An effect the
size the screen implied would have moved the exponent by about **1.0**, not 0.05, and it would be
unmissable on five points.

Blackhole has two sizes, so it gives a slope and no error bar and is reported as such: base
N^1.7544, `TT_BIO_ATOM_L1` N^1.7851 across
512 -> 1024 aa. Same conclusion, no fit quality to quote.

**Fold-level, Wormhole. Each size's ratio beside the A/A floor of its own session, both medians
over the same interleaved reps:**

| size | reps | base median s | L1 median s | ratio | A/A floor | resolved |
|---|---|---|---|---|---|---|
| 512 aa | 5 | 43.032 | 44.293 | 0.97153 | 1.01760 | 1.000x |
| 640 aa | 5 | 69.665 | 69.601 | 1.00092 | 1.05126 | 1.000x |
| 768 aa | 5 | 91.947 | 97.754 | 0.94060 | 1.02156 | 1.000x |
| 896 aa | 5 | 134.363 | 134.147 | 1.00161 | 1.01281 | 1.000x |
| 1024 aa | 4 | 174.440 | 171.335 | 1.01812 | 1.00391 | **1.01812x** |

**Fold-level, Blackhole:**

| size | reps | base median s | L1 median s | ratio | A/A floor | resolved |
|---|---|---|---|---|---|---|
| 512 aa | 5 | 22.321 | 21.619 | 1.03247 | 1.01560 | **1.03247x** |
| 1024 aa | 5 | 75.308 | 74.507 | 1.01075 | 1.01024 | **1.01075x** |

**768 aa paired reads 0.941x on Wormhole, not 1.89x.** The screen's base
fold was 85.504 s where this row's paired base median is 91.947 s,
and its L1 fold was 45.243 s where the paired L1 median is 97.754 s.
**The quiet box landed on the treatment arm**, which is the opposite of what the rising loadavg
made it look like — and it is why arms have to be reversed inside a rep rather than run in order.

## 3. Where the value actually is: the sampler wall, and it is flat in N

The fold is trunk-dominated. The 200-step sampler wall is the scope `TT_BIO_ATOM_L1` acts in, and
it is recorded per fold.

**Wormhole:**

| size | base step s | L1 step s | ratio (>1 = lever wins) |
|---|---|---|---|
| 512 aa | 9.532 | 9.119 | **1.04529** |
| 640 aa | 14.141 | 13.637 | **1.03696** |
| 768 aa | 14.175 | 13.658 | **1.03785** |
| 896 aa | 17.837 | 16.820 | **1.06046** |
| 1024 aa | 18.269 | 17.196 | **1.06240** |

**Blackhole:**

| size | base step s | L1 step s | ratio (>1 = lever wins) |
|---|---|---|---|
| 512 aa | 5.344 | 4.925 | **1.08508** |
| 1024 aa | 9.702 | 8.969 | **1.08173** |

**This is the answer to the brief's question and it is a flat line.** Wormhole moves from
1.045 at 512 aa to 1.062 at 1024 aa — a
1.6-point swing across a doubling of N, against per-size medians whose own fold-level floors run
0.4 to 5.1 %. Blackhole is 1.085 and 1.082,
flat to within 0.3 points. **The lever is worth what it is worth at 512 aa, at every size.** The
Blackhole number independently reproduces `b2z2-bh-stack-atom`'s 1.07175x on the same scope, on a
different chip, on a branch carrying this lever and nothing else.

The one architecture effect: **Blackhole pays about 3 points more than Wormhole at every size**
(~1.08x vs ~1.05x). Consistent with the budget arithmetic in section 1 — the same live set is
24-46 % of Blackhole's banks and 37-70 % of Wormhole's — but with two Blackhole sizes this is a
level shift the data supports, not a mechanism the data proves.

### A correction to this row's own earlier pass

At n=1-2 per size, mid-run, this row's Wormhole step ratios read
**1.054 / 0.900 / 0.891 / 0.838 / 0.833** and were written up as "the lever costs on Wormhole and
costs more with N", with a crowding mechanism proposed for it. **That was wrong, and deepening to
n=5 removed it**: the same five points now read
1.045 / 1.037 / 1.038 / 1.060 / 1.062.
One or two reps on a box at loadavg 25-61 produced a clean-looking monotone trend with a plausible
mechanism attached, pointing the opposite way from the truth. It is the same error as the 768 aa
screen this row was sent to retire, committed by this row, and caught only by adding reps.

## 4. The knee

Local exponents, the form a knee actually takes. A global fit averages a cliff away.

**Wormhole:**

| interval | base | `TT_BIO_ATOM_L1` |
|---|---|---|
| 512->640 aa | 2.159 | 2.025 |
| 640->768 aa | 1.522 | 1.863 |
| 768->896 aa | 2.461 | 2.053 |
| 896->1024 aa | 1.955 | 1.832 |

**The knee is not where `tt-bio-tuned-at-512-l1-gates-go-dark-above-640aa` puts it, on this tree.**
That memory recorded N^2.03 from 256 -> 512 and **N^3.62 from 512 -> 768**; here 512 -> 640 is
2.16 and 640 -> 768 is
1.52. Nothing in this range reaches N^2.5 except
768 -> 896 in the base arm, at 2.46. Either the three
gates the memory names have been fixed since 2026-08-13, or this fixture's shape does not trip
them; this row measured the fold, not the gates, and cannot say which. **P6 is REFUTED as stated:
the cliff is not between 512 and 640 aa.**

**P7 stands: `TT_BIO_ATOM_L1` leaves the knee alone.** The two arms' local exponents track each
other, and the small difference runs the safe way — the L1 arm's are visibly smoother
(2.03 / 1.86 /
2.05 / 1.83
against base's 2.16 /
1.52 / 2.46 /
1.95), which is what a constant few-per-cent saving on
a sub-term looks like. The gate arithmetic in section 1 said it could not be otherwise: the gates
the memory names are in the trunk and this lever is in the diffusion atom branch.

The base arm's alternation (high, low, high, low) is the signature of a box whose load moved
between sizes, not of four regime changes; the five legs ran concurrently on one contended Galaxy
and each size sat on a different card.

## 5. The eight predictions, scored

| # | prediction | outcome |
|---|---|---|
| P0 | `{l1: 1200, dram: 0}` at all five sizes | **CONFIRMED**, and at both Blackhole sizes too |
| P1 | base exponent 2.35 +/- 0.20 | **MISS.** 2.0028 +/- 0.0701, below the band |
| P2 | L1 exponent 2.22 +/- 0.20, 0.10-0.20 below base | **MISS on the separation.** 1.9530 +/- 0.0186, i.e. below base by 0.0498, not 0.13 |
| P3 | ratios 1.03 / 1.05 / 1.07 / 1.09 / 1.12 | **REFUTED.** No rise with N on either part |
| P4 | curves separate visibly only above 768 aa | **REFUTED.** They do not separate anywhere |
| P5 | 768 aa pairs at 1.05x-1.12x, **not** 1.89x | **HALF HIT.** 1.89x is dead (0.941x paired); the 1.05-1.12x half was too generous |
| P6 | knee real, between 512 and 640 aa | **REFUTED as stated.** 512->640 is N^2.16 |
| P7 | `ATOM_L1` leaves the knee alone | **CONFIRMED** |
| P8 | identical CIF sha256 at every size | **CONFIRMED** at all five WH sizes and both BH sizes |

Two of eight hit, one half, five missed — and the five misses all point the same way: **I predicted
a small real size effect and there is none at all.** I was closer to the truth than the screen was
and still on the wrong side of it. The prediction did call the outcome that mattered: it said the
exponents might fail to separate at n=5 and named that as the honest reading.

## 6. What it means for the product

**The perf page publishes 512 aa and it is not hiding anything.** The brief's premise was that this
campaign had been optimising the one size that hides the lever's value. It has not: the lever is
worth ~1.08x on the Blackhole sampler wall at 512 aa and ~1.08x at 1024 aa. **A user folding an
800 aa complex gets the same thing a user folding a 512 aa one gets.** Nothing goes in front of
Moritz as a separate size-dependent claim, because there is no size-dependent claim to make.

**Recommendation: `TT_BIO_ATOM_L1` is safe but it does not earn a default flip on its own.** It is
bit-exact at every size on both parts, its gate never declines below ~1530 aa on the tighter part,
and it costs nothing. But on the fold it clears its own A/A floor at exactly two of seven points
(Blackhole 512 aa 1.03247x against a 1.01560x floor, and
Blackhole 1024 aa 1.01075x against 1.01024x, which is
marginal). Its honest home is inside `b2z2-bh-stack-atom`'s Blackhole stack, where it was measured,
and not as a standalone default. **Nothing merged.**

## 7. Reproducing this

```
ssh whglx-admin  'cd /home/mthuening/work/wt/b2z2-atoml1-size-curve && ...'   # Wormhole, 8x9
ssh tt-quietbox2 'cd /home/ttuser/.coworker/wt/b2z2-atoml1-size-curve && ...' # Blackhole, 11x10
env -u TT_METAL_DEVICE_PROFILER TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C \
    TT_BIO_LEASE_HOLDER=worker:b2z2-atoml1-size-curve TT_BIO_TRACE_REGION_SIZE=536870912 \
    python3 perf/b2z2_sizecurve/size_curve.py --out <json> --cifdir <dir> --size $S --reps 5

perf/b2z2_sizecurve/harvest.sh                                  # copies both hosts onto pc
python3 perf/b2z2_sizecurve/fit_curve.py perf/b2z2_sizecurve/curve_*whglx*.json --out fit_wh.json
python3 perf/b2z2_sizecurve/fit_curve.py perf/b2z2_sizecurve/curve_*qb2*.json   --out fit_bh.json
```

`fit_curve.py` refuses to put two architectures on one regression line and tolerates a partial run,
reporting `reps_complete` per size. `harvest.sh` stages every copy and parses it before it replaces
a committed artifact: the driver rewrites its JSON after every fold, so a plain `scp` of a live run
can land a short file, and the qb2 checkout additionally carries the *committed* Wormhole JSONs,
which a plain copy would have written backwards over the live ones.
