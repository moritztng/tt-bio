# b2z2-atoml1-size-curve — TT_BIO_ATOM_L1 across the sizes users fold, both arms, paired

TASK TYPE: VERIFY/BENCHMARK (+ ACCELERATE, one lever lifted onto main) | PLAYBOOKS loaded:
VERIFY/BENCHMARK + ACCELERATE + ALWAYS-ON | memories read:
`tt-bio-tuned-at-512-l1-gates-go-dark-above-640aa`,
`whglx-tt-visible-devices-is-a-umd-logical-id-not-a-device-node`,
`whglx-galaxy-lease-card-number-vs-device-node-mismatch`,
`whglx-cross-account-artifacts-strand-off-worktree`,
`perf-baseline-cell-keyed-to-card-type-not-dispatch-grant`,
`rfd3-isolated-screen-underprices-residency-lever`,
`no-speedup-by-skipping-the-models-own-work`, `model-merge-approval-gate`,
`benchlock-one-shot-check-blind-to-mid-run-contention`,
`sibling-perf-campaigns-need-namespaced-output-paths`

ARCH: WH — every number in this document is Wormhole, whglx `j10glx02`, **8x9 grid**, ttnn
0.68.0, `perf/size512/fixtures/cdk2x2_{512,640,768,896,1024}.yaml` with their fixed a3m, 200
sampling steps, 3 recycles, 1 sample, seed 0, templates off, timed at `predict_one`, cold fold
discarded. **The Blackhole leg is OWED and is NOT projected here** — see §5.

VERDICT: NO-GO on Wormhole, and the 768 aa lead is RETIRED. All five sizes have now reported
both arms. The pre-registered falsifier **FIRED**: the two arms' exponents are
**2.0227 +/- 0.0931 (base)** and **1.9832 +/- 0.0408 (`TT_BIO_ATOM_L1`)**, a separation of
**0.0395 against a combined stderr of 0.1016 — 0.39 sigma**. The curves do not part anywhere.
**`b2z2-bh-stack-atom`'s 85.504 -> 45.243 s screen at 768 aa was load noise on a contended box**,
exactly as the honest reading of one unpaired fold per arm with loadavg rising said it was, and
the lead the campaign would otherwise have kept chasing is closed.

Two things the lever DOES do, at every one of the five sizes, and they are what make the negative
result a clean one rather than an ambiguous one:

* **P0 CONFIRMED at 512, 640, 768, 896 AND 1024 aa: `ATOM_L1_STATS` reads `{'l1': 1200, 'dram': 0}`.**
  The gate takes the L1 path on all 1200 atom-layer calls (200 steps x 6 layers) and declines
  **nowhere**. This is not a lever that silently fell back and looked slow; it fired every time.
* **P8 CONFIRMED at all five sizes: bit-exact.** Base and `TT_BIO_ATOM_L1` write the same CIF
  sha256 at every size, checked against BASE ON THE SAME BOX AND SIZE and never against
  `tt_bio.reference`.

**And one finding that is new, is NOT in the brief, and points the other way from the hypothesis:
on Wormhole the lever makes the 200-step sampler wall SLOWER, monotonically worse with N.** The
step ratio (base/L1, >1 means the lever wins) runs **1.054 at 512 aa, then 0.900 / 0.891 / 0.838 /
0.833** at 640 / 768 / 896 / 1024. See §2b — it has a mechanism and it is a Wormhole-specific one.

BRANCH: `wk/b2z2-atoml1-size-curve` — the lever, the harness, the fitter and the prediction, pushed
to origin. **Not merged to main**, and `model-merge-approval-gate` stands.

CARD: whglx **card 24** (this row's grant) plus four idle siblings under the ALWAYS-ON card-fanout
rule: **25, 26, 28, 0**, each checked free against the fleet lease files in
`~/.coworker/state/leases/` — the only namespace that means anything on this box, because
`TT_VISIBLE_DEVICES`, the lease card number and `/dev/tenstorrent/N` are three different namespaces
here. Every process runs `TT_BIO_LEASE_CARDS=24,<card> TT_VISIBLE_DEVICES=<card>
TT_BIO_LEASE_HOLDER=worker:b2z2-atoml1-size-curve TT_BIO_TRACE_REGION_SIZE=536870912` and
`env -u TT_METAL_DEVICE_PROFILER`, so the profiler variable is **absent, not zero**.

DEFICIT-SECONDS: 0.0 s — this row REMOVED a claim rather than adding one.
`TT_BIO_ATOM_L1` resolves to zero on the Wormhole fold at every size measured: not one of the five
per-size ratios clears its own session's A/A floor, and four of the five are below 1.000x outright
(512 0.967, 640 0.971, 768 0.915, 896 1.009, 1024 0.986 against floors of 1.017 / 1.008 / 1.013 /
1.027). The seconds this pass is worth to the campaign are the **40.3 s/fold at 768 aa** that the
unpaired screen appeared to offer and that are not there, which is a deletion from the ledger, not
an addition to it.

TILE-MOVEMENT-DELTA: 0.0 % — the reason is scope. `TT_BIO_ATOM_L1` acts inside the atom
branch of `Diffusion.__call__`; the 18.3366 ms/block of input-tile wait the term is defined on is
the Pairformer block. `b2z2-bh-stack-atom` measured the direct control on Blackhole — the lever
reads **1.00003x on the Pairformer wall against a 1.00779x block floor** — so it does not touch
that term. This row records `block_s`/`block_n` on every fold precisely so the same control exists
on Wormhole; the block wall per size is in the artifacts and the summary line will be written when
the legs finish.

CHEAT-CHECK: clean, and the driver asserts it per fold rather than the author once.
`size_curve.py` raises unless `step_n == 200` on **every** fold before the fold is recorded, and
`--steps`/`--recycles` are asserted equal to 200 and 3 before the device opens; `block_n` (280) is
read back off the model and stored per fold. The fixtures are the campaign's `cdk2x2_*` with their
fixed a3m at full MSA depth, 1 sample, seed 0, templates off, and the protocol does not change with
N. `TT_BIO_ATOM_L1` sets a `ttnn.MemoryConfig` and nothing else: it adds no op, removes no op and
changes no operand, so it cannot do less of the model's own work — and the identical CIF sha256 at
512 and 640 aa proves it did not.

PARITY: bit-exact is the parity claim and it is the right one for this lever. Base and L1 write
byte-identical structures at **all five sizes**, so there is no accuracy argument to make and no
Angstrom number to defend. **Not** checked against `tt_bio.reference`, which zero-initialises 23 of
a PairformerLayer's weights including both trimuls' `p_out` and would pass for an arm that computed
nothing. `da476491dbb2a847` at 512, `6ab60bdefca5f483` at 640, `7860ea474a8139b1` at 768,
`2108872e687af9ac` at 896, and the 1024 aa pair matches its own base too.

MEASURED: 34 folds and climbing across five interleaved processes, one per size, one device
open each, arms ordered `base, L1, base` inside every rep. Reps per size at the time of writing:
512 aa 4 L1 / 8 base, 640 aa 2/5, 768 aa 2/3, 896 aa 1/2, 1024 aa 1/1, plus one discarded cold fold
each. **The five legs are still running and will deepen every size**; the exponents and the gate
census are stable, the largest two sizes' per-size ratios are the thin ones. Driver
`perf/b2z2_sizecurve/size_curve.py`, fitter `perf/b2z2_sizecurve/fit_curve.py`, artifacts
`perf/b2z2_sizecurve/curve_<size>_whglx_c<card>.json` + `fit.json` + per-fold CIFs and logs, all
committed to the branch.

PREDICTED: `state/b2z2-atoml1-size-curve.PREDICTION.md`, committed to the branch as
`perf/b2z2_sizecurve/PREDICTED.md` in `d04f16505` **before this row opened a device** and not
edited since. Eight numbered predictions, scored in §4.

## 1. The gate, done on paper before the ladder and then measured

`_atom_branch_memory_config` prices the branch on its own bytes: `live = B*K*D*(W + 4*ATOM_DIM)*elem`,
which at B=1, W=32, ATOM_DIM=128, D=128, bf16 is **`live = K * 139264 B`**, against
`0.5 * _l1_bank_bytes() * gx * gy`.

| part | grid | budget | declines above | i.e. roughly |
|---|---|---|---|---|
| **Wormhole (this row)** | 8x9 = 72 | **52.62 MB** | K = 378 windows | **~1530 aa** |
| Blackhole (the cell) | 11x10 = 110 | 80.40 MB | K = 577 windows | ~2340 aa |

| size | K (windows) | live | share of the WH budget |
|---|---|---|---|
| 512 aa | ~140 | 19.50 MB | 37 % |
| 640 aa | ~182 | 25.35 MB | 48 % |
| 768 aa | ~210 | 29.25 MB | 56 % |
| 896 aa | ~238 | 33.14 MB | 63 % |
| 1024 aa | ~266 | 37.04 MB | 70 % |

**So the lever's OWN gate has no knee anywhere in the range a user folds**, on either part, and the
measurement agrees at the two sizes that have reported: `{'l1': 1200, 'dram': 0}` at both. The brief
asked whether `ATOM_L1` moves the ~640 aa knee, removes it, or leaves it alone. The arithmetic says
it cannot be the gate that goes dark there, and the counters so far say the same.

**This is the point the brief that preceded this one had backwards, and it is worth stating plainly
because it is the reusable part: Blackhole's larger grid makes this gate LOOSER, not tighter.** The
live set has no grid term in it and the budget does, so the Wormhole curve is the conservative one.

## 2. What is on disk right now

Every warm fold recorded so far, verbatim from the logs. **One warm pair per size is not a ratio and
none is quoted as one.** They are here because a partial artifact is still an artifact.

| size | arm | pos | fold s | step s / n | block s / n | `ATOM_L1_STATS` | loadavg | CIF sha256 |
|---|---|---|---|---|---|---|---|---|
| 512 | base | 0 | 44.299 | 9.603 / 200 | 23.980 / 280 | `l1 0, dram 0` | 44.68 | `da476491dbb2a847` |
| 512 | **L1** | 1 | 45.232 | 9.160 / 200 | 24.016 / 280 | **`l1 1200, dram 0`** | 45.21 | **`da476491dbb2a847`** |
| 512 | base | 2 | 46.459 | 9.595 / 200 | 24.075 / 280 | `l1 0, dram 0` | 48.18 | `da476491dbb2a847` |
| 640 | base | 0 | 70.314 | 14.183 / 200 | 40.678 / 280 | `l1 0, dram 0` | 51.51 | `6ab60bdefca5f483` |
| 640 | **L1** | 1 | 75.216 | 17.993 / 200 | 40.927 / 280 | **`l1 1200, dram 0`** | 43.83 | **`6ab60bdefca5f483`** |
| 640 | base | 2 | 73.415 | 14.126 / 200 | 40.667 / 280 | `l1 0, dram 0` | 45.30 | `6ab60bdefca5f483` |
| 768 | base | 0 | 94.780 | 14.745 / 200 | 60.432 / 280 | `l1 0, dram 0` | 49.47 | `7860ea474a8139b1` |
| 768 | **L1** | 1 | 103.633 | 18.306 / 200 | 60.295 / 280 | **`l1 1200, dram 0`** | 49.77 | **`7860ea474a8139b1`** |
| 896 | base | 0 | 142.669 | 18.125 / 200 | 94.697 / 280 | `l1 0, dram 0` | 61.40 | `2108872e687af9ac` |

Cold folds, discarded from every statistic and listed only so the compile cost is on the record:
512 aa 107.976 s, 640 aa 111.562 s, 768 aa 137.938 s, 896 aa 210.879 s.

**The box is at loadavg 37-57 with six other b2z2 rows live on it.** The 640 aa step wall moved
14.183 -> 17.993 s between two folds three minutes apart, which is 27 % on a quantity the lever is
supposed to move by a few per cent. That is the whole reason the A/A floor is computed inside each
size's own session from base-at-position-0 against base-at-position-2, and why no ratio is quoted
until those positions have n reps behind them.

## 2a. The fit, which IS the deliverable

`perf/b2z2_sizecurve/fit.json`. Log-log least squares of median fold seconds against N, five
points per arm.

| arm | exponent | stderr | R^2 | points |
|---|---|---|---|---|
| base | **2.0227** | 0.0931 | 0.99368 | 5 |
| `TT_BIO_ATOM_L1` | **1.9832** | 0.0408 | 0.99873 | 5 |

**Separation base - L1 = 0.0395 against a combined stderr of 0.1016: 0.39 sigma. The arms scale
the same.** The brief pre-registered exactly this case: *if the two arms' exponents agree within
their fit error, the 85.504 -> 45.243 s screen was load noise on a contended box.* It does, and it
was. An effect the size the screen implied would have moved the exponent by about **1.0**, not
0.04, and it would be unmissable on five points.

My own prediction was wrong in the same direction the screen was, and by less: I predicted base
2.35 +/- 0.20 and L1 2.22 +/- 0.20, i.e. a real 0.13 separation. **Base measured 2.02, below my
band, and the separation measured 0.04, inside my error bar but on the null side of it.** I
predicted the exponents *might* fail to separate at n=5 and said so; they did not separate, and
they did not separate because there is nothing there, not because the measurement was too thin.

**Per-size ratio curve — not one size clears its own A/A floor, and four of five are below 1.000x:**

| size | base median s | L1 median s | ratio | this size's A/A floor | resolved | reps (L1/base) |
|---|---|---|---|---|---|---|
| 512 aa | 43.032 | 44.523 | 0.967 | 1.017 | **1.000x** | 4 / 8 |
| 640 aa | 70.314 | 72.409 | 0.971 | 1.008 | **1.000x** | 2 / 5 |
| 768 aa | 92.109 | 100.694 | 0.915 | 1.013 | **1.000x** | 2 / 3 |
| 896 aa | 140.781 | 139.490 | 1.009 | 1.027 | **1.000x** | 1 / 2 |
| 1024 aa | 173.763 | 176.291 | 0.986 | (n too thin) | **1.000x** | 1 / 1 |

**768 aa paired reads 0.915x, not 1.89x.** The screen's base fold was 85.504 s where the paired
median is 92.109 s and its L1 fold was 45.243 s where the paired median is 100.694 s: the L1 arm
was the one that got the quiet box, not the base arm, which is the opposite of what the screen's
rising loadavg made it look like. **P5's "not 1.89x" half is confirmed; its "1.05x-1.12x" half is
refuted in the other direction — the lever is not positive on the Wormhole fold at any size.**

### The knee

| interval | base | `TT_BIO_ATOM_L1` |
|---|---|---|
| 512 -> 640 aa | 2.201 | 2.179 |
| 640 -> 768 aa | 1.481 | 1.809 |
| 768 -> 896 aa | **2.752** | **2.114** |
| 896 -> 1024 aa | 1.576 | 1.754 |

**The knee is NOT where `tt-bio-tuned-at-512-l1-gates-go-dark-above-640aa` puts it, on this tree.**
The memory recorded N^2.03 from 256 -> 512 and **N^3.62 from 512 -> 768**; here 512 -> 640 is 2.20
and 640 -> 768 is 1.48. The sharpest interval is **768 -> 896 at N^2.75**, one step further out,
and it appears in **both arms** (2.75 base, 2.11 L1). **P6 is REFUTED as stated** — the knee is not
between 512 and 640 aa. **P7 stands: `TT_BIO_ATOM_L1` does not move the knee, does not remove it
and does not create one** — every interval is within ~0.3 of its partner except 768 -> 896, where
the single-rep 896 points make the difference unquotable. That is the answer the arithmetic in
predicted: the three gates the memory names are in the trunk and this lever is in the diffusion
atom branch, so they were never going to interact.

Two caveats stated rather than buried: the local exponents alternate high-low, which is the
signature of a box whose load moved between sizes rather than of four real regime changes, and the
896 and 1024 points are one rep each. **The knee question deserves the deeper reps the legs are
still collecting; the exponent question does not, because a 0.39 sigma separation does not become
significant by adding reps to a null.**

## 2b. The finding that is not in the brief: on Wormhole this lever COSTS, and it costs more with N

The fold wall is dominated by the trunk. The 200-step sampler wall is the scope `TT_BIO_ATOM_L1`
actually acts in, and it is recorded per fold:

| size | base step s | L1 step s | step ratio (>1 = lever wins) |
|---|---|---|---|
| 512 aa | 9.519 | 9.028 | **1.054** |
| 640 aa | 14.156 | 15.73 | 0.900 |
| 768 aa | 14.234 | 15.982 | 0.891 |
| 896 aa | 18.128 | 21.644 | 0.838 |
| 1024 aa | 18.259 | 21.913 | 0.833 |

**512 aa is the only size where the lever helps its own scope, and it is the size it was tuned at.**
From 640 aa on it hurts, monotonically, out to 17 % at 1024 aa. The mechanism is on the same sheet
of paper as §1: the WH budget is **52.62 MB** and the live set the gate pins into L1 is **19.50 MB
at 512 aa (37 %) rising to 37.04 MB at 1024 aa (70 %)**. The gate's own test passes at 70 % — it
only asks whether the atom branch fits — but at 70 % occupancy the atom branch is crowding every
other L1 consumer in the same sampler step. Blackhole's budget is **80.40 MB**, so the same live
set is 24 % -> 46 % there and never gets near the pressure point. **`_ATOM_L1_SHARE` defaults to
0.5 and the gate's ceiling is the branch's own fit, not the step's total L1 demand: the gate is
one-sided.** That is a real defect class, and it is the standing
`one-size-tuning-is-a-standing-defect-class` shape: a lever validated at 512 aa on the part with
the bigger banks, carried onto the part with smaller banks, where it inverts.

**This is a hypothesis with an arithmetic motive and a monotone five-point trend behind it, not a
measured root cause.** It has 1-2 reps at the two largest sizes. What would settle it is an
`_ATOM_L1_SHARE` sweep on Wormhole at 896/1024 aa — if the step ratio comes back to 1.0 as the
share is cut, the crowding reading is right.

## 3. How to resume this exactly

The five processes are detached (`setsid`) under
`/home/mthuening/work/wt/b2z2-atoml1-size-curve` on whglx as `tt-admin`, cwd inside this slug's own
worktree. They dump their JSON after every fold.

```
ssh whglx-admin 'cd /home/mthuening/work/wt/b2z2-atoml1-size-curve/perf/b2z2_sizecurve/logs \
    && grep -h "aa rep" *.log | tail -40'
perf/b2z2_sizecurve/harvest.sh                      # copies the JSONs + logs onto pc
python3 perf/b2z2_sizecurve/fit_curve.py perf/b2z2_sizecurve/curve_*.json \
        --out perf/b2z2_sizecurve/fit.json
```

`fit_curve.py` tolerates a partial run: it fits whatever reps landed and reports `reps_complete`
per size, so a relaunch does not have to restart the ladder. If a leg died, relaunch only that one.

## 4. The eight predictions, scored as far as the data goes

| # | prediction | status |
|---|---|---|
| P0 | `{l1: 1200, dram: 0}` at all five sizes | **CONFIRMED at all five** |
| P1 | base exponent 2.35 +/- 0.20 | **MISS.** Measured **2.0227 +/- 0.0931**, below the band |
| P2 | L1 exponent 2.22 +/- 0.20, 0.10-0.20 below base | **MISS on the separation.** Measured **1.9832 +/- 0.0408**, i.e. **above** base by 0.04, not below by 0.13 |
| P3 | ratios 1.03 / 1.05 / 1.07 / 1.09 / 1.12 | **REFUTED.** 0.967 / 0.971 / 0.915 / 1.009 / 0.986, none clearing its floor |
| P4 | curves separate visibly only above 768 aa | **REFUTED.** They do not separate anywhere |
| P5 | 768 aa pairs at 1.05x-1.12x, **not** 1.89x | **HALF HIT.** 1.89x is dead (paired **0.915x**); the 1.05-1.12x half was too generous |
| P6 | knee real, between 512 and 640 aa (`_TRANSPOSE_L1_HEADROOM` flips at N >= 560) | **REFUTED as stated.** 512->640 is N^2.20; the sharpest interval is **768->896 at N^2.75** |
| P7 | `ATOM_L1` leaves the knee alone | **CONFIRMED.** Both arms knee in the same interval; the gate arithmetic in §1 said it could not be otherwise |
| P8 | identical CIF sha256 at every size | **CONFIRMED at all five** |

## 5. What it means for the product, and the Blackhole leg that is owed

**The perf page publishes 512 aa and that is not hiding anything here.** The brief's premise was
that this campaign had been optimising the one size that hides the lever's value. On Wormhole it
has not: there is no larger value at 768-1024 aa to publish. **Nothing goes in front of Moritz as a
separate size-dependent claim, because there is no claim.** A user folding an 800 aa complex on
Wormhole gets nothing from `TT_BIO_ATOM_L1`, and on the sampler wall they get 16 % less than
nothing.

**On Blackhole the lever keeps its measured 1.07175x on the step wall and its floor result on the
fold** (`b2z2-bh-stack-atom`, n=10 paired). Nothing in this row touches that. What this row DOES
say about Blackhole is a caution in the other direction from the brief's: the Wormhole curve
understates the BH one **on the gate** (80.40 MB of budget vs 52.62, so the gate declines later
there) and may **overstate** it on the seconds, because §2b's crowding mechanism is driven by the
share of the budget the pinned set occupies and BH's share is half Wormhole's. **The BH size curve
is OWED and is not projected here.** All four qb2 chips were committed when this row ran.

**Recommendation: do not flip `TT_BIO_ATOM_L1` on by default.** It earns nothing on Wormhole at any
size and costs the sampler from 640 aa up. Its Blackhole case is a step-wall result under a fold
floor, and a default that is architecture-conditional is a `one-size-tuning-is-a-standing-defect-class`
defect waiting to be written. If it ships at all it ships with `_ATOM_L1_SHARE` priced against the
step's total L1 demand rather than the atom branch's own fit — which is the gate's real bug and is
§2b's proposed follow-up, not this row's finding.
