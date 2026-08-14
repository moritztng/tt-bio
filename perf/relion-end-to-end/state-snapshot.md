# RELION end to end on Tenstorrent: a full refinement runs through our code and lands on RELION's own published resolution to the digit, the port sits 15x closer to RELION's reference than RELION's own CUDA backend does, and on this dataset we lose to a single H200 on wall-clock and 15-36x on energy because the job cannot fill either box

Task `relion-end-to-end` | worktree `/home/moritz/.coworker/wt/relion-end-to-end` on pc, branch
`wk/relion-end-to-end`. Continues `relion-acc-backend`. **Not merged. Proposed with evidence.**

TASK TYPE: ACCELERATE (whole-pipeline, inside a third-party codebase) | PLAYBOOKS loaded: ALWAYS-ON +
§ACCELERATE + §VERIFY/BENCHMARK | memories read: `perf-method-floor-screen-predict-then-build`,
`roofline-roof-must-be-measured-not-asserted`, `relion-backprojection-kernel-result`,
`relion-precision-fsc-result`, `ttnn-fft-blackhole-kernel-result`,
`ttnn-scatter-gather-per-element-limited`, `tt-bio-isolated-op-timing-oversync-inflates-cost`,
`pc-disk-space-critical`, `galaxy-shared-customer`, `perf-gate-single-shot-legs-recurring-false-alarm`,
`land-task-stale-premise-and-grep-overcount`, `cdk2x2-chimeric-fixture-cannot-score-non-bit-exact-parity`.

**THE MILESTONE IS MET.** Two complete RELION 3D auto-refinements ran to convergence on the RELION-5
tutorial dataset, on one binary with one variable, both `rc=0`, both writing RELION's own output tree.
Both print **`Auto-refine: + Final resolution (without masking) is: 3.79378`**, byte-identical, and
identical to the value RELION's own precalculated `Refine3D/job019` prints for this job. The arm with
the coarse squared-difference kernel running in `tt_bio/cryoem/relion.py` converged on the same
iteration, through the same per-iteration accuracy estimates, to the same answer.

**E6 RAN.** A 2x H200 box was rented, three arms measured (1 GPU, 2 GPU, and the same box's CPU),
and the instance destroyed: **$2.77 of the $40 cap, post-teardown balance $46.5885, 0 instances
running.** Six independent runs across three architectures now converge to `3.79378 Å`. §7.

---

## 1. Re-scope, because three predecessors land here and one of them was wrong in the favourable direction

| inherited from | claim | status after this pass |
|---|---|---|
| `relion-acc-backend` §4.5 | RELION runs one iteration with its coarse compare in our code, 4452/4452 orientations bit-identical | **holds, and §3 now extends it to a whole trajectory.** |
| `relion-acc-backend` §3.3 | "the whole-program ceiling from accelerating the E-step alone is ~41x" | **SUPERSEDED.** It substituted a coarse-only device figure for the entire E-step particle loop. Measured, coarse-only on one card is **3.20x on a whole refinement**. §5. |
| `relion-acc-backend` §4.6 | exact-trilinear-on-device is dead, citing a scatter/gather figure from another workload | **REFUTED by measurement.** §10: the gather clears its pre-registered bar at 1.60x. |
| `relion-acc-backend` §4.12 | the shear interpolant misses RELION's reference by 6.0% median at `--pad 1` | **holds, and it no longer blocks anything** — §10 removed the need for the shear. |
| `relion-projection-optimize` P8 | lever B is 1.25x at 51.20% of the measured floor | **holds, and it is now off the critical path** for the same reason. |
| `relion-intercard-scaling` §15.3 | 32 chips give 12.38x on this job, 38.7% of linear, crossover ~7,060 particles | **holds, and it is the row that decides §7.** This job has 4,452 particles, below the crossover. |
| `relion-intercard-scaling` §8 | per dollar cannot be sourced defensibly | **RETIRED.** Both sides have public list prices. §7. |
| this task, previous pass | the whole-refinement ceiling is ~3.19x, from a projected `expectation_6` | **CONFIRMED at 3.20x** by the finished reference arm, and the projection it rested on was 0.3% off. |

**The bottleneck did not move to I/O or to the M-step.** The M-step is 5.617 s of a 922.19 s
refinement and it is the control that proves the bridge touches nothing else (§4). It stayed an
E-step task. What *did* move is which route the E-step should take: the previous pass believed a
device gather was impossible and that the only buildable kernel changed RELION's answer. §10 measured
the gather and it is alive, so the accuracy fork that dominated this lineage's risk is gone.

**The comparison arm went from unfundable to measured mid-pass, and both versions are kept.** E6 was
blocked on a $0 vast.ai balance; Moritz funded it and a 2x H200 box ran three arms for $2.77 (§7).
Independently, `Tutorial5.0/Refine3D/job019` turned out to be RELION's own 4-GPU run of this exact
job with its file mtimes intact — a same-span, same-MPI-configuration, RELION-authored GPU reference
that was on disk the whole time. **Both are in §7 and they agree**: one H200 does this job in 129.89 s
and RELION's own four 2023-era GPUs did it in 119 s, which says the job saturates at roughly 100-130 s
whatever GPU you point at it.

---

## 2. The milestone: two complete refinements, one binary, one variable

`perf/relion-end-to-end/e2e_campaign.sh` on qb1, both arms under `benchlock.sh`, both continuing from
RELION's own `Refine3D/job019/run_it012_optimiser.star`, both `mpirun -n 5 --j 6 --pool 6 --cpu
--dont_combine_weights_via_disc --preread_images`. One binary, `relion/build-e2e` (`ALTCPU=ON TT=ON`
plus `-DTIMING`). The only variable is `TT_RELION_BACKEND`: `ttnn` makes the bridge decline every
coarse call so RELION runs its own `CpuKernels`, `torch` makes the bridge handle it.

| arm | wall, s | CPU-s user+sys | maxrss | rc | iterations | RELION's own final resolution |
|---|---|---|---|---|---|---|
| **ref**, RELION's own kernels | **922.19** | 20771.53 + 32.91 | 2.84 GB | 0 | 13→17, converged | **3.79378 Å** |
| **tt**, coarse compare in our code | **3988.48** | 80855.73 + 3300.38 | 5.54 GB | 0 | 13→17, converged | **3.79378 Å** |

Both logs are line-for-line identical in structure: the same five `Expectation iteration` banners, the
same `Auto-refine: Estimated accuracy angles=` at every iteration (0.546, 0.53, 0.523, 0.52, 0.52),
the same `Refinement has converged, entering last iteration where two halves will be combined`, the
same `Refinement has converged, stopping now`. **The bridge does not perturb the trajectory's shape at
all — it reproduces RELION's control flow decision for decision.**

**It takes five iterations where job019 needed four from the same point.** Auto-refine's convergence
counter reset at iteration 15 because the angular change went *up* (0.115668° at it14 → 0.17011° at
it15) before settling. Not a `--continue` artifact and not a defect: it costs one iteration of wall
and lands on the same answer. Both arms pay it, so the A/B is unaffected, but a wall quoted "per
refinement" has to say five iterations.

**The bridge arm is 4.325x SLOWER than RELION's own CPU kernel, and that is the expected and correct
result for what it is.** `tt_bio/cryoem/relion.py`'s torch backend is a host-side transcription whose
purpose is to prove the plumbing and the numerics, not to be fast — its docstring says so. It is the
arm that makes "RELION's answer computed by our code" a measurable fact. **It is not the port's
performance claim and it must never be quoted as one.** The performance claim is §5's device floor.

### 2.1. The A/A, so the deltas below are interpretable

There are two, at two scales, and **they do not give the same answer — which is the point**.

- **Per iteration: 1.1%.** The same iteration ran twice under `benchlock.sh` at **142.50 s** and
  **140.98 s**, both acquiring the lock at loadavg 0.17-0.52 — against the **2.01x** that
  `relion-acc-backend` §4.8 measured for the same arm *unlocked* on this host.
- **Per whole refinement: 11.2%.** `ref` and `ref2` are the same binary, same backend, same input,
  both under the lock: **922.19 s** and **1025.91 s** (21184.71 + 34.81 CPU-s, `rc=0`, same
  `3.79378 Å`). `ref2` acquired at loadavg 2.65 against the threshold of 3.0, so this is co-tenancy
  the lock permits rather than measurement error.

**Quoting the 1.1% as this task's noise floor would understate it by 10x.** A whole-refinement A/B has
to clear 11.2%, not 1%. The 4.325x in §2 clears it by 39x, so it stands, but no refinement-scale
result under ~12% should be read as a signal on this host.

The `-DTIMING` + CTIC/CTOC instrument is separately nearly free: this run's `expectation_6` is
**131.826 s** against the un-instrumented iteration 13 at **132.756 s**, **0.7% apart**.

---

## 3. Is it the same answer? Yes, and the port is closer to RELION's reference than RELION's own GPU backend is

`perf/relion-end-to-end/e2e_compare.py` and `e2e_disagree.py`, over the whole trajectory.

### 3.1. Resolution and the maps

| arm | source | gold-standard FSC **0.143**, unmasked | `relion_postprocess` FINAL RESOLUTION | half1 sha256/16 | half2 sha256/16 |
|---|---|---|---|---|---|
| ref, RELION's own kernels | converged | **3.9942 Å** | **4.033896 Å** | `8b742aba54ab9233` | `df8605b9b55caaf8` |
| tt, coarse in our code | converged | **3.9941 Å** | **4.033896 Å** | `26454fbd8788d05c` | `6f5f54f5583feff7` |
| RELION's own 4-GPU job019 | converged | **3.9804 Å** | — | `712d2c0e57656222` | `989f70473b79a896` |

- **Resolution delta at FSC 0.143: −0.0001 Å**, against the program's standing bar of 0.1 Å. RELION's
  own GPU backend sits **+0.0139 Å** from our CPU reference on the same data, 139x further away.
- **`relion_postprocess` prints 4.033896 Å for both arms**, to all six digits it emits.
- **Cross-FSC, ref half-*k* against tt half-*k*: min over all 128 shells = +0.999898 and +0.999987**,
  relative L2 1.008e-02 and 3.567e-03. The two arms reconstructed the same volume to the Nyquist edge.
- **The sha256 per half map differs between the arms and is reported, NOT used as an identity check.**
  `relion-acc-backend` §4.8 ran the *reference* arm twice and the half-map sha256 differed while all
  4,452 assignments and the FSC crossing were identical: RELION's ALTCPU reconstruction is not
  bit-reproducible run to run. A doc that reads a sha difference here as a bridge effect reads noise.

### 3.2. The assignments, and the control that makes the number mean something

Exact equality is the right test for two runs of one binary on one box differing in one variable. It
is the **wrong** test the moment the builds differ, and having both here is what makes the verdict
defensible. `e2e_disagree.py`, angles compared modulo 360 so a hair either side of the wrap does not
score as 359.9° of disagreement:

| pair | column | bit-identical | median \|Δ\| | p99 \|Δ\| | max \|Δ\| |
|---|---|---|---|---|---|
| **tt vs ref** (our bridge vs RELION's CPU) | `_rlnAngleRot` | **4427/4452** | **0.000000°** | **0.000000°** | 0.402428° |
| | `_rlnAngleTilt` | **4427/4452** | **0.000000°** | **0.000000°** | 0.199784° |
| | `_rlnAnglePsi` | **4423/4452** | **0.000000°** | **0.000000°** | 0.469614° |
| | `_rlnOriginXAngst` | **4449/4452** | **0.000000 Å** | **0.000000 Å** | 0.396072 Å |
| | `_rlnOriginYAngst` | **4452/4452** | **0.000000 Å** | **0.000000 Å** | 0.000000 Å |
| **job019 vs ref** (RELION's CUDA vs RELION's CPU) | `_rlnAngleRot` | 0/4452 | 0.086043° | 0.785993° | 179.913288° |
| | `_rlnAngleTilt` | 0/4452 | 0.039283° | 0.369273° | 0.960041° |
| | `_rlnAnglePsi` | 1/4452 | 0.046830° | 0.769189° | 10.988781° |
| | `_rlnOriginXAngst` | 0/4452 | 0.053072 Å | 0.232550 Å | 0.618711 Å |
| | `_rlnOriginYAngst` | 0/4452 | 0.053072 Å | 0.232550 Å | 0.513498 Å |

**The parity verdict, and it is the strongest statement this lineage has been able to make.** Our
bridge leaves RELION's answer **bit-identical on 99.4% of particles, with a median and a 99th
percentile disagreement of exactly zero, and a worst case under half a degree**. RELION's *own* CUDA
backend, on the same job, agrees with RELION's own CPU backend on **zero** particles out of 4,452, with
a median disagreement of 0.086° and a worst case of 11°. **The port is one to two orders of magnitude
closer to RELION's reference than the accelerator RELION itself ships.** That is the yardstick a
customer actually cares about, and it came free from an artifact already on disk.

The 179.9° max on `_rlnAngleRot` is the Euler degeneracy, not a disagreement: at `Tilt` near zero the
Rot/Psi decomposition is not unique, so Rot can differ by ~180° with Psi compensating. It appears in
the RELION-against-RELION row and not in ours.

### 3.3. Does the disagreement compound? The count grows, the magnitude does not, and the answer does not move

| iteration | it012 | it013 | it014 | it015 | it016 |
|---|---|---|---|---|---|
| **tt vs ref** — the bridge against RELION's kernels | 0/4452 | 0/4452 | 6/4452 | 13/4452 | **18/4452 = 0.404%** |
| **ref2 vs ref** — RELION's kernels against *themselves*, rerun | 0/4452 | 0/4452 | 1/4452 | 7/4452 | **13/4452 = 0.292%** |

Read on its own the first row is "the bridge changes RELION's answer and it compounds", and it would
be the wrong reading. **The second row is the control that settles it, and it is the pre-registered
outcome: RELION's own reference arm, rerun byte-for-byte identically, drifts from its own first run
in the same shape and the same order.** `ref2` is `ref` in every respect including
`TT_RELION_BACKEND=ttnn`; the only difference is that it is a second run.

| pair | Rot bit-identical | Rot max \|Δ\| | Psi max \|Δ\| | FSC 0.143 | vs ref | cross-FSC min | rel L2 |
|---|---|---|---|---|---|---|---|
| **ref2 vs ref** (RELION against itself) | **4428/4452** | 0.402428° | 0.469614° | 3.9943 Å | **+0.0001 Å** | 0.999924 / 0.999957 | 8.7e-3 / 6.5e-3 |
| **tt vs ref** (the bridge) | **4427/4452** | 0.402428° | 0.469614° | 3.9941 Å | **−0.0001 Å** | 0.999898 / 0.999987 | 1.0e-2 / 3.6e-3 |

**One particle apart on the count, identical to six decimal places on the worst case, and on opposite
sides of the reference by the same 0.0001 Å.** The bridge's disagreement with RELION is
indistinguishable from RELION's disagreement with itself. The identical `0.402428°` and `0.469614°`
maxima in both rows are not a coincidence: they are the same particles landing on the same
sampling-step boundary, which is what the mechanism below predicts.

**What grows is the count of particles whose values are no longer bit-identical. What does not grow is
the size of any disagreement**: at the end of the trajectory the median is still exactly zero, the p99
is still exactly zero, and the largest single move is 0.47° — smaller than the 0.79° p99 that RELION's
own two *backends* already differ by at every particle. `relion_postprocess` prints the same six
digits for both arms, and `ref2` reaches RELION's own `3.79378 Å` as well.

**PRE-REGISTERED before `ref2` ran** (`perf/relion-end-to-end/e2e_ctrl.sh`, committed before launch):
"if the drift is RELION's own, ref2-vs-ref shows a reassignment trajectory of the same shape and order
as tt-vs-ref, and the bridge is exonerated. If ref2-vs-ref is 0/4452 at every iteration, the 18
particles ARE the bridge and the parity claim has to say so." **It landed on the first branch.**

**The mechanism, named:** the bridge reduces the whole image at once where RELION accumulates per
256-pixel block, which changes the last bits of the score and nothing else (`tt_bio/cryoem/relion.py`
docstring). Almost always the argmin over orientations is unaffected. Occasionally two orientations
score within the last bits of each other and the tie breaks the other way; that particle's assignment
then differs by one sampling step, and because a refinement feeds its output back in, the *set* of
such particles widens slowly. It is a widening tie-break set, not a growing error.

### 3.4. The full parity ladder, once the rented arms are added

E6 (§7) built stock RELION at the same commit on rented hardware, which turns the parity question into
a ladder with four independent controls rather than one comparison. Every row is graded against the
same `ref` arm, `_rlnAngleRot`, angles modulo 360:

| what it is | bit-identical | median \|Δ\| | p99 \|Δ\| | max \|Δ\| |
|---|---|---|---|---|
| `ref2` — RELION's CPU path against **itself**, rerun | 4428/4452 | 0.000000° | **0.000000°** | 0.402° |
| **`tt` — OUR BRIDGE** | **4427/4452** | **0.000000°** | **0.000000°** | **0.402°** |
| `xeoncpu` — stock RELION CPU path, **different box, different build** | 4439/4452 | 0.000000° | **0.000000°** | 0.497° |
| `h200x1` — RELION's **own CUDA path** on an H200, stock build | 4064/4452 | 0.000000° | 0.370508° | 179.2° |
| `h200x2` — the same on 2 GPUs | 4044/4452 | 0.000000° | 0.374485° | 2.33° |
| `job019` — RELION's own CUDA path, 2023 silicon | 0/4452 | 0.086043° | 0.785993° | 179.9° |

**The ladder separates cleanly into two bands, and our bridge is in the tight one.** Every CPU-path
RELION — ours, RELION's rerun against itself, and a stock build on entirely different silicon with a
different compiler — sits at **p99 exactly zero** and ~4430/4452 bit-identical. Every CUDA-path RELION
sits at p99 0.37-0.79° and 0-4064/4452. **Our bridge disagrees with RELION's reference on 25 particles
of 4,452; RELION's own GPU backend disagrees on 388 on modern silicon and on all 4,452 on 2023
silicon.** The bridge is 15x tighter than the accelerator RELION itself ships, and it is
indistinguishable from RELION's CPU path running on another machine.

**So there are four independent controls and they all agree.** `ref2` bounds RELION's own run-to-run
nondeterminism and puts the bridge one particle away from it; `xeoncpu` bounds a different
box-and-build of the same code path and puts the bridge 12 particles inside it; `job019` and the two
H200 arms bound RELION's own cross-backend disagreement and put the bridge 15-178x inside it.
**Nothing here leaves room for a bridge effect on RELION's answer.**

---

## 4. Where the wall-clock actually goes, by stage, across a whole refinement

RELION's accelerated E-step had no observable stage breakdown before this lineage. It has one now:
`src/acc/cpu/cpu_benchmark_utils.h`'s CTIC/CTOC macros were defined empty with the real body sitting
inside a `/* */` comment. Giving them a thread-safe body instruments 51 regions at a measured 0.7%.

**TTPROF, cumulative CPU-seconds summed over threads, four follower ranks, whole trajectory.** Only
the % column is comparable between arms; the ranks agree to 0.14 points, which is the instrument's
own reproducibility.

| region | ref arm, % of `oneParticle` | tt arm, % of `oneParticle` |
|---|---|---|
| `oneParticle`, the whole per-particle E-step | 100% | 100% |
| **`getAllSquaredDifferencesCoarse`** | **78.17 / 78.25 / 78.23 / 78.31** | **88.85 / 88.74 / 88.43 / 88.64** |
| `storeWeightedSums` (wavg + backproject) | 9.28 / 9.21 / 9.20 / 9.19 | 6.12 / 6.14 / 6.42 / 6.24 |
| `getAllSquaredDifferencesFine` | 7.16 / 7.05 / 7.10 / 7.02 | 3.52 / 3.61 / 3.66 / 3.63 |
| `getFourierTransformsAndCtfs` | 5.37 / 5.46 / 5.45 / 5.46 | 1.49 / 1.49 / 1.47 / 1.48 |
| `backproject` (inside `storeWeightedSums`) | 4.86 / 4.83 / 4.81 / 4.84 | 3.80 / 3.76 / 4.01 / 3.82 |
| `weightPass`, the wrapper over both difference passes | 85.35 / 85.33 / 85.35 / 85.35 | 92.38 / 92.36 / 92.11 / 92.28 |

**The instrument's nesting checks out arithmetically**, which matters because a tic/toc tree that
double-counted would inflate exactly the share this task turns on: `weightPass` = 85.35% against
coarse + fine = 78.17 + 7.16 = **85.33%**. The wrapper equals the sum of its children to 0.02 points,
so the regions nest rather than overlap.

**Iteration 13 alone put coarse at 90.40%. Over the whole trajectory it is 78.2%, and the 12 points
are one iteration.** The pre-registered prediction, written before the reference arm ran, was "below
90.4%, landing in 80-90%, and if it lands below 70% the ceiling table is wrong". **Measured 78.2%:
right about the direction, outside the band, above the kill line.** What moved the mix is iteration 17,
the final combined-halves pass, which RELION itself warns "will use data to Nyquist frequency" and
which took **~274 s of the 922 s wall on its own** and is not coarse-dominated.

**RELION's own Timer, wall seconds per iteration**, and the M-step is the control:

| | it13 | it14 | it15 | it16 | it17 | ref total (13-16) | tt total (13-16) | tt/ref |
|---|---|---|---|---|---|---|---|---|
| `expectation`, ref | 137.209 | 164.671 | 160.483 | 162.893 | ~274 | 625.256 | | |
| `expectation`, tt | 636.334 | 667.753 | 735.263 | 687.537 | ~1250 | | 2726.887 | **4.361x** |
| `expectation_6`, the particle loop, ref | 133.870 | 156.769 | 156.792 | 157.780 | ~264 | 605.211 | | |
| `expectation_6`, tt | 632.123 | 664.668 | 715.588 | 675.097 | | | 2687.476 | **4.441x** |
| **`maximization` = the M-step, ref** | 1.390 | 1.408 | 1.409 | 1.410 | | **5.617** | | |
| **`maximization`, tt** | 1.403 | 1.531 | 1.503 | 1.557 | | | **5.994** | **1.067x** |

**The M-step is the control and it holds at 1.067x.** The bridge replaces one kernel inside the E-step
and touches nothing else; if it had perturbed anything global, the M-step would have moved with it. The
6.7% it does move is the arm's larger resident set (5.54 GB against 2.84 GB) pressuring the same box.

**Residual accounting, and there is no material gap.** `expectation_6` over the whole trajectory is
605.211 measured for it13-16 plus ~264 for it17 scaled from its own `expectation` share = **871.2 s,
94.5% of the 922.19 s wall**. The remaining **51.0 s** is `expectation_1/2`, the M-step, the solvent
flatten, `writeOutput`, process startup and the optimiser read — all named, none unattributed.

---

## 5. The pipeline floor, and what each optimization bought

The pipeline floor is not what you get by adding up the primitive floors. The primitives add to a
fraction of a second; the pipeline cannot go below **51.0 s per refinement** because that is RELION's
own host code, and `src/backprojector.cpp` contains **zero** `_CUDA_ENABLED`/`do_gpu` references, so
every GPU RELION in the world pays the M-step on the host too.

**Applying §4's measured shares to the measured 871.2 s particle loop:**

| term | s per refinement | binding limit | seam? |
|---|---|---|---|
| `getAllSquaredDifferencesCoarse` | **681.3** | compute-bound on CPU; the trilinear **gather** inside the projection is the mechanism, not the compare (`relion-acc-backend` §4.6: projection is 95.6% of the call) | **yes, already hooked** |
| `storeWeightedSums` | 80.1 | wavg + trilinear **scatter** into the destination volume | yes, `runWavgKernel` + `runBackProjectKernel` |
| `getAllSquaredDifferencesFine` | 61.9 | same kernel shape, significant subset only | yes, `runDiff2KernelFine` |
| `getFourierTransformsAndCtfs` | 47.0 | host FFTW per particle | no seam in any RELION backend |
| host residue outside the particle loop | **51.0** | setup, the M-step, solvent, `writeOutput`, startup | **no** |
| **refinement** | **922.19** | | |

**The device coarse pass is MEASURED, not assumed: 9.37 s per cs=196 iteration on one p150** (§10),
so five iterations is **46.8 s**. That figure is the gather *and* the arithmetic, because §10's overlap
sweep showed the blend and the compare hide under the gather with 14.6x of headroom.

| what is on the device | host work left, s | refinement wall, s | speedup | status |
|---|---|---|---|---|
| nothing — RELION's own CPU kernels | — | **922.19** | **1.00x** | **MEASURED** |
| the coarse pass, one p150 — *the kernel already hooked* | 240.9 | **287.8** | **3.20x** | device term MEASURED, composition EXTRAPOLATED |
| the coarse pass, 32-chip Galaxy at the measured 12.38x fanout | 240.9 | **244.7** | **3.77x** | EXTRAPOLATED |
| coarse + fine + `storeWeightedSums`, one p150 | 98.0 | **154.7** | **5.96x** | EXTRAPOLATED |
| coarse + fine + `storeWeightedSums`, Galaxy | 98.0 | **102.6** | **8.99x** | EXTRAPOLATED |
| + a device per-particle FFT, Galaxy | 51.0 | **56.6** | **16.30x** | EXTRAPOLATED |
| **the floor: RELION host code that has no seam in any backend** | **51.0** | **51.0** | **18.08x** | MEASURED by difference |

The three EXTRAPOLATED middle rows carry one named assumption: that the fine pass and
`storeWeightedSums` reach the same device-to-CPU ratio the coarse pass measured (0.0688). For the fine
pass that is defensible, it is the same kernel on a subset. **For `storeWeightedSums` it is
optimistic** — it is a scatter, and `relion-backprojection-kernel-result` measured our backprojection
kernel at 16-17% of its own floor. Those rows are an upper bound on the speedup, and they are labelled.

**Attack the largest named cost — and note where that puts the residual.** The largest cost is the
coarse pass at 681.3 s, and putting it on the device at its measured floor removes 634.5 s of the
922.19 s wall. **What is left is 287.8 s of which 240.9 s is host code, so after the single biggest
optimization this pipeline is 84% host-bound.** That is the residual and its mechanism is named: the
per-particle FFT and the M-step and the setup, none of which any RELION backend accelerates. The
scaling row makes the same point more sharply — spreading the device term across 32 chips at the
measured 12.38x fanout moves the refinement from 3.20x to 3.77x. **A 32x increase in silicon buys 18%,
because the thing being scaled is already only 16% of the wall.**

**Bytes/time against a measured roof, in both directions.** Forward: the coarse call moves at least
490 MB (231 MB of model gather for 186 × 19,404 × 8 corners at 8 B, 260 MB of difference temporaries,
1.4 MB of shift stack) and the host bridge does it in 151.2 ms = **3.24 GB/s**, two orders below this
EPYC's socket DRAM roof and physically consistent with the ~108%-of-600% single-core utilisation
measured at the time — that is a sanity check that says the bridge is latency-bound on its own Python
and torch dispatch, not on memory. Backward: the device side is checked against the **measured**
404.9 GB/s DRAM roof and 42.48 GB/s composed ring bandwidth from `relion-intercard-scaling` §13.1/§15.2,
not against a datasheet. **This lineage published 668 GB/s on a ~400 GB/s card once**; every roof
quoted here was measured on the silicon it is quoted for.

### 5.1. The one lever tried on the bridge itself, and its pre-registered gate

The bridge arm's per-particle cost got *worse* with more MPI ranks: `relion-acc-backend` §4.5 measured
325 s per follower for 2,226 particles on `-n 3` (0.146 s/particle) against this campaign's 636.334 s
for 1,113 on `-n 5` (0.572 s/particle), same box, same binary. **Candidate mechanism, stated as a
candidate: torch sizes its intra-op pool per process from the core count, so four ranks ask the
32-core box for twice the threads two ranks did, on top of RELION's own `--j 6` per rank.**
`TT_RELION_TORCH_THREADS` is the one-line arm; the screen is one iteration against the same binary's
measured it013 of 636.334 s. **Pre-registered before the run: below 424 s (1.5x) confirms
oversubscription; above 1.1x refutes it, and a refutation is a result.** Queued behind the same
benchlock hold as `ref2` at the end of this pass. **It cannot change any headline** — the bridge's
wall is not the port's performance claim, §2.

---

## 6. Per card and per server

| unit | what it does per refinement | source |
|---|---|---|
| **1 × p150 (Blackhole)** | the coarse pass in **46.8 s**, 130 cores, model 31.7 MB L1-resident at 238 kB/core, 135 W p50 measured sustained | §10, MEASURED |
| **1 × qb1 (32-core EPYC, 5 MPI × 6 threads)** | the whole refinement in **922.19 s** on RELION's own kernels | §2, MEASURED |
| **1 × p150 + host** | whole refinement **287.8 s**, 3.20x, of which 240.9 s is host | §5, EXTRAPOLATED composition |
| **32-chip Galaxy (6U) + host** | whole refinement **244.7 s**, 3.77x today's kernel; **102.6 s**, 8.99x with three | §5, EXTRAPOLATED |
| **RELION's own 4-GPU node** | whole job **237 s**; the it012→convergence span our arms cover in **119 s** | §7, MEASURED from RELION's own artifact |

**The per-server number is not 32x the per-card number and this is the point.** `relion-intercard-scaling`
§15.3 measured the fanout on this job at **12.38x on 32 chips, 38.7% of linear**, with the crossover
into good scaling at **~7,060 particles**. This dataset has **4,452**. The job is below the size where
a Galaxy is the right shape of machine, and no kernel work changes that.

---

## 7. Galaxy against an H200: a MEASURED pair on both sides, wall-clock, dollars and watts

**E6 ran.** vast.ai was funded mid-pass ($49.36), a 2x H200 box was rented, three arms ran on it, and
it was destroyed. **Total spend $2.77 of the $40 cap; post-teardown balance $46.5885 and
`vastai show instances` returns 0 instances.** The `vastai` CLI is not packaged on pc or qb1 — it
installs into a venv with `pip install vastai`, which is what this pass did rather than reporting the
missing binary as a blocker a second time.

### 7.1. The rented arms

Stock RELION at **`e4a4aad06f079dda646bff870713ae97f3c829a6`**, the same commit qb1 runs, cloned from
github and built on the rented box — not our patched tree, because a GPU reference a customer could
reproduce must not carry our bridge or our instrument. Same span (`--continue` from RELION's own
`run_it012_optimiser.star` to convergence), same `-n 5 --j 6 --pool 6 --dont_combine_weights_via_disc
--preread_images`. Box: 2x NVIDIA H200, Intel Xeon Platinum 8480+, cgroup quota 53.76 cores, so the
30-thread configuration is not oversubscribed. `perf/relion-end-to-end/e6_gpu.sh`.

| arm | hardware | wall, s | CPU-s user+sys | rc | RELION's own final resolution |
|---|---|---|---|---|---|
| **g1** | **1x H200** | **129.89** | 2134.57 + 45.53 | 0 | **3.79378 Å** |
| **g2** | **2x H200** | **98.34** | 1554.30 + 43.14 | 0 | **3.79378 Å** |
| **c** | **Xeon 8480+, 5x6, ALTCPU** | **527.12** | 11628.73 + 52.76 | 0 | **3.79378 Å** |
| ref (qb1) | EPYC 32-core, 5x6, ALTCPU | 922.19 | 20771.53 + 32.91 | 0 | **3.79378 Å** |
| tt (qb1) | the bridge | 3988.48 | 80855.73 + 3300.38 | 0 | **3.79378 Å** |
| job019 | RELION's own 4 GPUs, Oct 2023 | 119 | — | — | **3.79378 Å** |

**Six independent runs, three architectures, two CPU vendors, three RELION build configurations, and
every one converges to `3.79378 Å`.** That is a stronger statement about RELION than about any of the
hardware, and it is the backdrop the rest of this section is read against.

**The rental bought exactly the two unknowns §7.1 previously could not close.**

- **The host correction is now measured: `922.19 / 527.12` = qb1's EPYC is 1.749x slower than the
  Xeon on this job.** The old job019 comparison divided qb1's wall by their GPU wall and got 7.75x;
  **corrected onto one CPU that ratio is `527.12 / 119` = 4.43x.** The unquantified host unknown was
  worth 1.75x, and it was inflating our own reported GPU gap.
- **The GPU is now a known part.** A single H200 does this job in **129.89 s** against RELION's own
  four 2023-era GPUs at **119 s**. The two independent estimates agree that this job saturates at
  roughly 100-130 s *regardless of how many GPUs or which generation* — which is the host bound
  asserting itself, not a GPU result.

**The GPU scaling slope is measured, not assumed: 1 -> 2 H200 is 1.321x, 66% of linear.** Carried one
more doubling at the same efficiency, 4 GPUs land at ~74.5 s and 8 at ~56.4 s. Those two are
EXTRAPOLATED from a measured slope, which is a large improvement on the previous pass where the whole
GPU column was extrapolated from someone else's rounded prose.

### 7.2. Prices and power, and the datasheet TDP was wrong by 5x for this workload

| | Tenstorrent Galaxy Blackhole | H200 |
|---|---|---|
| **list price** | **$110,000** per 6U node, 32 chips | **$320k-420k** for an 8-GPU DGX/HGX, ~$370k typical OEM |
| source | Tenstorrent launch pricing, April 2026: [The Register, 2026-04-28](https://www.theregister.com/2026/04/28/tenstorrent_galaxy_blackhole_ai_servers_ga/); [Dealroom](https://app.dealroom.co/news/feed/tenstorrent-launches-110k-galaxy-blackhole-ai-server-with-32-accelerators-and-23-petaflops) | Mercatus OEM survey; ITCT quote. The same Register piece prices an eight-way DGX at "three to five times" the Galaxy, i.e. $330k-550k — one author comparing the two boxes is a better source for the *ratio* than two independent absolutes |
| rental, measured this pass | — | **$7.87/hr for 2x H200** (vast.ai, verified offer, actually billed) |
| **accelerator power on THIS job** | **135 W p50 measured sustained** per chip | **132.0 W measured mean at 72% util**, sampled every 5 s over the g1 arm |
| idle draw, measured | — | 77-79 W per card |
| datasheet TDP | — | 700 W |

**The single most important correction in this section: an H200 running this refinement draws 132 W,
19% of its 700 W datasheet TDP.** The previous pass's per-watt row used 700 W x 8 and made a DGX look
far worse than it is. **A datasheet TDP is a roof, and this program's standing rule is that a roof
must be measured on the silicon it is quoted for** — that rule was applied to Tenstorrent's numbers
throughout this lineage and not, until now, to NVIDIA's. The reason the H200 idles at 19% of TDP is
the same reason everything else in this document is what it is: the job cannot fill it.

### 7.3. The comparison, host-corrected onto one CPU class, stated plainly

Our device rows were composed against qb1's host. Putting them on the Xeon-class host the H200 arms
actually ran on — dividing every host term by the measured 1.749x — is the only way to compare walls
rather than compare hosts:

| what runs the refinement | wall, s | vs 1x H200 (129.89 s) | vs 2x H200 (98.34 s) | energy at the accelerator |
|---|---|---|---|---|
| **1x H200** | **129.89** MEASURED | 1.00x | 0.76x | **0.0048 kWh** MEASURED |
| **2x H200** | **98.34** MEASURED | 1.32x | 1.00x | **0.0070 kWh** MEASURED |
| 4x H200 | ~74.5 EXTRAPOLATED from the measured slope | 1.74x | 1.32x | — |
| 8x H200 (a DGX) | ~56.4 EXTRAPOLATED | 2.30x | 1.74x | — |
| 1x p150, coarse only — *the kernel already hooked* | 184.5 EXTRAPOLATED | **0.70x** | 0.53x | 0.0069 kWh |
| Galaxy 32 chips, coarse only | 141.5 EXTRAPOLATED | **0.92x** | 0.70x | **0.1698 kWh** |
| Galaxy, three E-step kernels | 60.6 EXTRAPOLATED | **2.14x** | **1.62x** | 0.0727 kWh |
| Galaxy, at the pipeline floor | 29.2 EXTRAPOLATED | 4.46x | 3.37x | 0.0349 kWh |

**On wall-clock: today's hooked kernel loses to a single H200 even with all 32 Galaxy chips behind it
(141.5 s against 129.89 s). Three E-step kernels would beat two H200s by 1.62x.** That is the honest
shape of it, and the first row is the one that is true today.

**On energy the Galaxy loses badly and the mechanism is idle silicon.** One H200 finishes this job on
0.0048 kWh. One p150 needs 0.0069 kWh, in the same class. **A 32-chip Galaxy needs 0.0727-0.1698 kWh,
15-36x the H200**, because it holds 4.32 kW for a job whose fanout is 38.7% of linear — 31 of its 32
chips are mostly waiting. Per-watt on this job is not close, and no kernel fixes it; only a bigger job
does.

**On price the Galaxy leads 3.4x ($110,000 against ~$370k) and that lead is real but cannot be
converted here.** Dividing either box's list price by a wall it does not earn produces a per-dollar
number that says more about the dataset than the hardware, which is the same class of error as the
8.0x ceiling this lineage already corrected once. **No per-dollar headline is quoted from this table.**

## 8. The measured/extrapolated split, in one place

**MEASURED on real silicon this pass or the ones before it:**
- Both complete refinements to convergence, walls, CPU-seconds, maxrss, rc, and RELION's own
  `3.79378 Å` from both (§2).
- Every FSC 0.143 crossing, `relion_postprocess`'s 4.033896 Å, the cross-FSC, every per-arm sha256,
  and every assignment-disagreement distribution (§3).
- The by-stage split for both arms, four follower ranks each, and RELION's own Timer walls (§4).
- The device coarse-pass gather rate, its overlap headroom, and the reader RISC's issue cost (§10).
- The 12.38x 32-chip fanout, the 404.9 GB/s DRAM roof, the 42.48 GB/s ring, the 135 W p50, the
  185.8 TFLOP/s bf16 (`relion-intercard-scaling`).
- RELION's own 4-GPU wall, 237 s whole job and 119 s over our span (§7.1).
- **E6, all of it: 1x H200 at 129.89 s, 2x H200 at 98.34 s, the same box's Xeon 8480+ CPU at
  527.12 s, all `rc=0` at `3.79378 Å`; the 1.749x host correction; the 1.321x GPU scaling slope;
  and the H200's 132.0 W mean board draw at 72% util** (§7).
- The A/A noise floor at both scales (1.1% per iteration, 11.2% per refinement) and the 0.7%
  instrument cost (§2.1).

**EXTRAPOLATED, with the assumption named at the row:**
- Every device-accelerated refinement wall in §5 and §6. The device term is measured; composing it
  with the measured host terms into a wall is arithmetic on separate measurements, and no such row is
  quoted as achieved.
- The fine pass and `storeWeightedSums` reaching the coarse pass's device efficiency (§5) —
  optimistic for the scatter.
- Every Galaxy and p150 row in §7.3, including its host correction onto the Xeon class. The device
  term is measured, the host terms are measured, composing them into a wall is arithmetic.
- **The 4x and 8x H200 rows (~74.5 s, ~56.4 s)**, carried from the *measured* 1->2 slope of 1.321x at
  the same efficiency per doubling. RELION's scaling almost certainly degrades further, so both are
  optimistic for the GPU.
- Iteration 17's `expectation_6` (~264 s), scaled from its own `expectation` share rather than timed,
  because the converged iteration exits before printing a Timer table.

**No longer on this list, because E6 ran:** an H200 of any kind, and the "different host" unknown that
made the job019 comparison an EXTRAPOLATED-comparability pair. Both are measured.

**Not measured and named as such:** a device arm of the full refinement (it needs 2 cards for a
gold-standard MPI split, or two single-halfset runs assembled); a Galaxy arm of any kind, so every
32-chip row rests on `relion-intercard-scaling`'s measured 12.38x fanout rather than on a Galaxy
running RELION; a production-scale dataset (§9); and whether the shear interpolant's error compounds,
which §10 made informational rather than blocking.

---

## 9. GO/NO-GO and the plain verdict on whether this is worth shipping

**GO on the port as a correctness result. NO-GO on shipping it as a performance product for this
class of job. GO on building the exact-trilinear device coarse kernel, gated on Moritz.**

**What is unambiguously worth shipping is the parity result.** A full RELION refinement runs to
convergence with its dominant kernel computed by `tt_bio/cryoem/relion.py`, lands on RELION's own
published resolution to the digit, and disagrees with RELION's reference by a median of exactly zero
where RELION's own CUDA backend disagrees by 0.086° on every particle. **That is a stronger fidelity
claim than any accelerator vendor in this field publishes**, it is checkable by anyone with the
tutorial data, and it is the thing a pharma or academic user is actually afraid of.

**What is not worth shipping yet is a speed claim, and E6 sharpened rather than softened that.** After
the single largest optimization this pipeline has — the coarse pass at its measured device floor — the
refinement is **84% host-bound**, and 32x more silicon buys 18%. The rented arms show the same wall
from the other side: **a single H200 gets 4.06x over its own box's CPU and a second H200 only takes
that to 5.36x, 66% of linear**, and an H200 running this job draws 132 W of a 700 W TDP. Every
accelerator pointed at this dataset, ours and NVIDIA's, is mostly waiting on RELION's host code.

**So the honest answer to "what does a real RELION refinement cost on a Galaxy against a DGX H200":**

- **On wall-clock we lose today.** All 32 Galaxy chips with the kernel we already hook land at 141.5 s
  against **129.89 s for one H200**. Three E-step kernels would put us at 60.6 s and beat two H200s by
  1.62x, but those kernels do not exist.
- **On energy we lose badly, 15-36x**, and it is not close. One H200 finishes on 0.0048 kWh; a Galaxy
  needs 0.0727-0.1698 kWh because it holds 4.32 kW at 38.7%-of-linear fanout. No kernel fixes that.
- **On price we lead 3.4x**, $110,000 against ~$370k, and that is the only column we win — but it
  cannot be converted into a per-dollar claim on a job that fills neither box.

**That is a loss on this dataset, stated with its mechanism.** It is not a kernel deficiency and it is
not a silicon deficiency; it is a 4,452-particle job below a measured ~7,060-particle crossover, and
saying so now is worth far more than a favourable framing that falls apart the first time a customer
runs the tutorial.

**What would change the verdict, and it is a real dataset rather than a real kernel:** a production
refinement is 100k-1M particles, not 4,452. Above the measured 7,060-particle crossover the fanout
improves and the host residue amortises over vastly more per-particle work, so the coarse pass's share
goes back up toward iteration 13's 90.4% and the Galaxy's 32 chips start earning their price. **That
is the arm to run next, and this pass did not run it.** Nothing here should be extended to a
production-scale job without it.

---

## 10. E4, CLOSED by measurement: the exact-trilinear gather is issue-bound, not dead

`projprobe/e4_gather_rate.py` + `projprobe/kernels/reader_e4_gather.cpp`, pc card 0, one p150, 130
cores, model 31.7 MB L1-resident at 238 kB/core, 16 B-aligned reads walking a correlated 3D line,
`barrier_every` 4. Every arm's output is nonzero, so no arm measures elided reads.

**The pre-registered kill gate, written before the kernel existed: 66 M corner-gathers/s/core, and the
route is dead if one core cannot reach that within 2x.**

| arm | ns/read | corner-gathers/s/core | chip | vs the bar | coarse E-step, 1 p150 |
|---|---|---|---|---|---|
| 32 × 8 B, one corner per read | 36.70 | 27.3 M | 3.54 G/s | 0.414x | 36.27 s |
| **32 × 16 B, a corner pair — THE HONEST ARM** | **36.85** | **54.3 M** | **7.06 G/s** | **0.824x** | **18.21 s** |
| 32 × 32 B, two pairs | 36.90 | 108.4 M | 14.09 G/s | 1.645x | 9.12 s |
| 32 × 64 B, four pairs | 36.74 | 217.8 M | 28.31 G/s | 3.305x | 4.54 s |

**16 B is the honest arm and it is not the fastest row.** A trilinear cube's 8 corners are adjacent in
x only — `(x,y,z)` and `(x+1,y,z)` share a 16 B line, but the y and z neighbours are `mdlX` and
`mdlX*mdlY` voxels away. The 32 B and 64 B rows are what a *relaid-out* model would get.

**E4b: the second dataflow RISC is free, and that is what clears the bar.** The same loop in the writer
slot on a different phase into its own scratch CB gives **1.93x: 18.96 ns per read, 105.5 M
corner-gathers/s/core, 13.71 G/s chip-wide, 1.601x the bar**, and the coarse E-step gather on one p150
falls to **9.37 s**. **The route is ALIVE and `relion-acc-backend` §4.6's dismissal is refuted by
measurement rather than by argument.**

**The mechanism, named: per-transaction issue cost on the dataflow RISC, not bytes.** Flat at
36.7-36.9 ns from 8 B to 64 B — an eightfold change in bytes moves it 0.5%. A read-count sweep at fixed
16 B fits a marginal **34.83 ns per read** on a 64.15 ns loop intercept, so the per-read number is the
read and not the loop around it. 34.83 ns at 1.35 GHz is **~47 cycles to issue one NoC read**.
Independently consistent with `s1e_bytes.json`'s 42.4 ns per 64 B read at a fixed page.
**Reproducibility:** two independent runs of the 16 B arm gave 54.3 and 54.8 M/s/core, **0.9% apart**.

**E4c, MEASURED: the arithmetic hides under the gather with 14.6x headroom**, so 9.37 s is a floor for
the whole coarse pass and not just its reads. `compute_e4_blend.cpp` runs `mul_tiles` against a CB the
reader pushes up front, so the math unit and both gather loops are live at once:

| tile ops per assembly-pair | ns | vs gather-only |
|---|---|---|
| 0 / 4 / 8 / **16** | 1204.19 / 1209.99 / 1218.11 / **1207.65** | 1.000 / 1.005 / 1.012 / **1.003x** |
| 32 / 64 / 128 | 2083.35 / 4046.30 / 7931.05 | 1.730 / 3.360 / 6.586x |

Flat to 16, then linear. **What the real kernel needs in the same currency: 1.09 tile ops per
assembly-pair against 16 free**, measured at HiFi4, the slowest fidelity, so the margin is conservative.

**E4d, MEASURED, and it is the constraint the other two hide: the reader RISC has no spare issue slot
at all.** The same harness with integer address units added to the reader's per-gather loop:

| extra ~6-instruction units per gather | 0 | **1** | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|
| ns per assembly-pair | 1209.31 | **1361.58** | 1473.71 | 2106.90 | 2806.54 | 4219.40 |
| vs zero | 1.000 | **1.126x** | 1.219 | 1.742 | 2.321 | 3.489 |

**One unit costs 4.76 ns per read, 6.4 cycles at 1.35 GHz.** The load-bearing number is the 4.76 ns,
not an instructions-per-cycle figure — what it establishes is that read issue and address arithmetic
**add rather than overlap** on that RISC. **The math unit's 14.6x of headroom says nothing about the
reader, and it would be easy to read E4c as if it did.**

**Prescriptive, for whoever builds the kernel.** The address advance is one add:
`d(address)/dx = e00 + e10*mdlX + e20*mdlX*mdlY` is constant along a scan line. The radius test against
`maxR2_padded` is per pixel, i.e. per four gathers, and the Friedel conjugate flip belongs in the blend
rather than the reader. **A reader that recomputes the address from the Euler matrix per gather would
pay several-fold, and this table is the reason to say so before anyone writes one.** Still not
measured: the shift stack, and the packing — a 16 B read lands its two corners *adjacent*, so the
blend's tile layout needs a strided weight tile or a pair reduction.

---

## 11. Decided against, so a relaunch does not relitigate it

1. **Do not quote the bridge arm's 3988.48 s as the port's performance.** §2.
2. **Do not quote a per-dollar or per-watt headline from §7.3.** It divides a whole node by a job that
   does not fill it. §7.3.
3. **Do not re-home RELION onto pc.** 15 GB free against a 36 GB tree; `pc-disk-space-critical`.
4. **Do not chase a clean wall by fixing `benchlock`.** A duration-holding lock is a fleet change.
   Report CPU-seconds beside the wall instead, which is the co-tenancy-robust metric.
5. **Do not batch particles for the coarse compare.** `relion-acc-backend` §4.9 measured 256 calls in
   one iteration with 256 distinct euler-set hashes — auto-refine builds a per-particle projection plan
   unconditionally, so there is no shared slice store to amortise. It still pays for Class2D/Class3D.
6. **Do not build the bf16 or sphere-packed reduction, or the host-mediated PCIe reduce.** Closed by
   measurement in `relion-intercard-scaling` §0.5 and §14.
7. **Do not re-derive lever O, D, F-at-LoFi, or W/R/I/P/X.** Dead with evidence in
   `relion-projection-optimize` §5 and P8. Lever F (HiFi2) is GATED at 1.127x for a 3.13x accuracy
   regression and stays out of any headline.
8. **Do not build the shear device projector.** §10 removed the reason it existed.

---

## 12. Open, in priority order

1. **The production-scale arm.** §9's verdict turns on 4,452 particles being below a measured
   ~7,060-particle crossover. A 100k-particle job is the arm that decides whether this ships as a
   performance story, and it is the single highest-value thing left in this lineage. **E6 makes it
   cheap to do properly now**: the recipe, the stock-RELION CUDA build and the teardown are in
   `perf/relion-end-to-end/e6_gpu.sh`, it cost $2.77, and $46.59 of credit remains.
2. **The exact-trilinear device coarse kernel.** Priced at 9.37 s per iteration (§10), bit-identical
   by construction, no accuracy fork. It is a new-model-port-shaped change and **cannot merge without
   Moritz**, so it is proposed, not started.
3. **The `TT_RELION_TORCH_THREADS` screen** (§5.1), launched and running on qb1 behind `ref2`'s
   benchlock hold. Confirmatory; it cannot change a headline, because the bridge's wall is not the
   port's performance claim. `ref2` itself is **DONE** and reported in §2.1 and §3.3.
4. **A device arm of the full refinement**, which needs 2 cards for a gold-standard MPI split (RELION 5
   refuses a single-process gold-standard split and its own error message advertises a flag that no
   longer parses), or two single-halfset runs assembled.
5. ~~An actual H200.~~ **DONE, §7.** 1x and 2x H200 measured, instance destroyed, $2.77 spent.

---

## 13. Durable lessons

- **Exact equality is the wrong parity metric the moment the two builds differ, and it inverts the
  verdict.** RELION's own CUDA backend scores **0/4452 identical** against RELION's own CPU backend
  while the largest disagreement over all 4,452 particles is under a degree. Graded by `==`, RELION
  disagrees with itself completely. Graded by magnitude, it agrees everywhere and is bit-identical
  nowhere, which is just what two float backends do. **Report the distribution of |Δ|, not the count of
  matches — and get the second pair into the table, because a disagreement number means nothing without
  the disagreement the reference already has with itself.** Extended by E6: build the reference's own
  accelerator too. Stock RELION's CUDA path disagrees with stock RELION's CPU path on 388 of 4,452
  particles at p99 0.37°, where our bridge disagrees on 25 at p99 exactly zero — a comparison that
  costs one rental and turns "our port is close" into "our port is 15x closer than the vendor's own.
- **A datasheet TDP is an asserted roof, and this program applied its own "measure the roof" rule to
  its silicon but not to the competitor's.** An H200 running this refinement draws **132 W of a 700 W
  TDP, 19%**, and the previous pass's per-watt row used 700 W x 8. That overstated the DGX's power by
  5x — in *our* favour, which is why it survived a review that would have caught it the other way
  round. Measure the competitor's roof on the competitor's silicon on your workload, or do not quote
  a per-watt number.
- **A cross-machine ratio silently prices the hosts, not the accelerators, and the correction was
  1.75x here.** Dividing qb1's 922.19 s CPU wall by someone else's 119 s GPU wall gave 7.75x; renting
  a box and running the SAME CPU arm on it showed the two CPUs differ by 1.749x, and the honest
  same-CPU ratio is 4.43x. **The cheapest arm on a rented box is the one that reproduces your own
  baseline** — it costs minutes, it is the only thing that makes the expensive arm comparable, and it
  is the arm most likely to be skipped.
- **Renting the competitor is cheap enough that not renting it is the expensive choice.** Three arms
  on 2x H200 — GPU, multi-GPU and the host CPU control — cost **$2.77** including build and teardown.
  A whole section of this document had been EXTRAPOLATED for want of that.
- **A vendor's own precalculated results are a timed benchmark if the mtimes survived.** RELION ships
  `Refine3D/job019` with `use_gpu Yes`, `gpu_ids 0:1:2:3`, `nr_mpi 5` and per-iteration file mtimes
  that reconstruct a same-span 4-GPU wall to the second — a better comparison arm than the rented card
  it replaced, and it was on disk the whole time. Check `note.txt` against `job.star`'s mtime to prove
  the stamps are original before trusting them.
- **After the single biggest optimization, ask what fraction is left that you do not own.** Putting
  RELION's dominant kernel on the device at its measured floor leaves the refinement **84% host-bound**,
  and 32x more silicon then buys 18%. A ceiling computed from one kernel's share is not a ceiling on the
  program; the number that matters is the share of the *residual* with no seam in any backend.
- **A benchmark can be too small to be a benchmark, and the tell is that the vendor's own accelerator
  is also only getting 6-8x.** RELION's 4 GPUs get 6.4-7.8x on the tutorial job over a 32-core CPU. When
  the reference implementation's own accelerator underperforms its class, the dataset is the constraint,
  not the port — check the measured crossover before reading any ratio from it.
- **A parity number needs the reference's disagreement with ITSELF before it means anything, and two
  independent controls beat one.** The bridge's assignments drift from RELION's over five iterations
  (0→18 of 4,452), which reads as a defect until you rerun RELION against itself and get 0→13 with
  the worst case identical to six decimals. Two controls were available almost free: a second
  reference arm (run-to-run) and RELION's own shipped CUDA results (cross-backend). Neither cost a
  kernel. **Budget a self-control arm into any trajectory-level parity claim — a bit-exact single
  iteration says nothing about a loop whose output is its own next input, and neither does a drift
  number without the reference's own.**
- **An A/A at one scale does not transfer to another.** The same host gave a 1.1% noise floor per
  iteration and **11.2%** per whole refinement, both under the same benchlock threshold. Measuring the
  A/A at the cheap scale and quoting it at the expensive one understates the bar by 10x.
- **A growing count of non-identical particles is not a growing error.** Over five iterations the
  bridge's non-bit-identical set widened 0 → 18 of 4,452 while the median and p99 disagreement stayed
  exactly zero and the converged resolution moved by 0.0001 Å. The mechanism is a tie-break set widening
  under feedback, not error accumulation, and only reporting the magnitude alongside the count tells
  them apart.
- **Headroom on one engine says nothing about the next one.** The same core gave 14.6x of slack on the
  math unit and **zero** on the dataflow RISC: one extra instruction per gather costs 12.6%. Stopping at
  the generous engine would have produced a reader that recomputes addresses per gather, priced at
  several-fold. Sweep every engine the inner loop touches, not the one the question was about.
- **A gather this program called impossible is issue-bound at ~47 cycles per NoC read, flat from 8 B to
  64 B, and the second dataflow RISC halves it for free.** The fix for a gather-bound kernel is *fewer,
  wider* reads and a second issuing RISC, not a different algorithm. Every design in this lineage that
  routed around a gather was routing around a cost that had never been measured on the real access shape.
- **RELION's accelerated E-step has no observable stage split, for two independent reasons, both traps.**
  `TIMING_ESP_DIFF1`/`DIFF2`/`WSUM` are tic'd only when `op.part_id == baseMLO->exp_my_first_part_id`,
  which never fires on the ALTCPU path, and `Timer::printTimes` only prints tags with `counts[i] > 0`, so
  they vanish rather than printing zero. Independently, the CTIC/CTOC macros are defined empty in
  `src/acc/cpu/cpu_benchmark_utils.h` with the real body inside a `/* */` comment. Giving them a
  thread-safe body instruments 51 regions at a measured 0.7%.
- **A sampling-space work estimate is not a measurement, and this one was wrong by 25x on the ratio.**
  Coarse is 1,305 sampling points per particle against fine's 41,760, which reads as "fine is 32x the
  work". Measured, coarse is 78-90% and fine is 3-7%, because the fine pass only ever evaluates the
  significant subset.
- **A converged auto-refine iteration exits before printing its Timer table**, so a parser that sums
  the tables silently drops the single most expensive iteration — here 274 s of a 922 s refinement. The
  count of `Expectation iteration` banners and the count of Timer tables must be compared, not assumed
  equal.
- **This task's DONE_CHECK passes on a planning document and on a doc that *denies* having a result.**
  Run against this doc's plan-only ancestor it printed `DONE`; and `_perf_method_gate` greps for
  `iteration[- ]level|iteration wall`, so a sentence saying "there is no iteration-level A/B" satisfies
  the clause demanding one. The gate cannot tell a claim from its negation. Flagged, not exploited —
  the honest strengthening is a clause requiring the doc to name an output tree that exists on disk
  (`e2e/ref_run_it016_data.star`, `e2e/e2e_disagree.json`), and it is orchestrator-owned. Same family as
  `donecheck-keyword-grep-lets-multistage-task-conclude-early`, one turn of the screw worse.
- **An atexit handler that reads a function-local `static` container is a use-after-free.** atexit
  handlers and static destructors share one LIFO list, so a container constructed after the handler is
  registered is destroyed before it runs. Symptom: a correct printed table then exit 139. Leak it.
- **A compute kernel's `tile_regs` cycle must be whole, and half a cycle wedges the chip rather than
  failing.** acquire+math+commit+release with no `tile_regs_wait` and no `pack_tile` leaves the packer
  signalling dest-section-done for a section it never waited on: `ops=0` ran fine because the loop was
  skipped, `ops=8` hung all 130 cores. A synthetic math loop that omits the pack is not a cheaper
  version of the real thing, it is a different and illegal thing.
