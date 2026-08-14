# RELION end to end on Tenstorrent: the coarse pass is 90.4% of the E-step, so the one kernel already built carries an 8x whole-iteration ceiling — and the wall underneath it is RELION's own host code

Task `relion-end-to-end` | worktree `/home/moritz/.coworker/wt/relion-end-to-end` on pc, branch
`wk/relion-end-to-end`. Continues `relion-acc-backend`. **Not merged.**

TASK TYPE: ACCELERATE (whole-pipeline, inside a third-party codebase) | PLAYBOOKS loaded: ALWAYS-ON +
§ACCELERATE + §VERIFY/BENCHMARK | memories read: `perf-method-floor-screen-predict-then-build`,
`roofline-roof-must-be-measured-not-asserted`, `relion-backprojection-kernel-result`,
`relion-precision-fsc-result`, `ttnn-fft-blackhole-kernel-result`,
`ttnn-scatter-gather-per-element-limited`, `tt-bio-isolated-op-timing-oversync-inflates-cost`,
`pc-disk-space-critical`, `galaxy-shared-customer`, `perf-gate-single-shot-legs-recurring-false-alarm`,
`land-task-stale-premise-and-grep-overcount`.

**STATUS: PLANNING PASS (opus5 tier). One thing is MEASURED this pass and it re-scopes the task; the
rest is an execution plan with exact commands. No DONE is claimed and no perf claim is made.**

---

## 0. What this pass measured, in one table, because it changes the answer

RELION's accelerated E-step had no observable stage breakdown. It has one now.

| region, iteration 13, cs=196, 30 threads | cpu-seconds | share of `oneParticle` |
|---|---|---|
| `oneParticle` (the whole per-particle E-step) | 730.13 / 733.65 | 100% |
| **`getAllSquaredDifferencesCoarse`** | **660.06 / 665.52** | **90.40% / 90.71%** |
| `getAllSquaredDifferencesFine` | 25.59 / 24.09 | 3.50% / 3.28% |
| `storeWeightedSums` (wavg + backproject) | 25.29 / 24.56 | 3.46% / 3.35% |
| — `maximization` (the wavg inside it) | 24.91 / 24.18 | 3.41% / 3.30% |
| — `backproject` | — / 10.48 | — / 1.43% |
| `getFourierTransformsAndCtfs` | 19.02 / 19.32 | 2.60% / 2.63% |
| `convertAllSquaredDifferencesToWeights`, both passes | 0.12 / 0.12 | 0.02% |

Two independent follower ranks, 1,104 and 1,116 particles, agreeing to 0.3 percentage points.
`prof.log`, `perf/relion-end-to-end/e2e_prof.sh`, run under `benchlock.sh` (acquired at loadavg 0.52).

**MEASURED, and it is the single number this task turns on: RELION's coarse squared-difference kernel
is 90.4% of its own accelerated E-step. The fine pass is 3.5%.** The kernel the bridge already
offloads is not a slice of the problem, it is nearly all of it.

**And the instrument is nearly free**, which is what makes the share believable rather than an
observer effect: this run's `expectation_6` is **131.826 s** against P1's un-instrumented iteration 13
at **132.756 s** (`relion-acc-backend.md` §3.1), **0.7% apart**, on the same continue from the same
`it012` optimiser. That is the A/A for the instrument.

---

## 1. Re-scope. Three predecessors land here, and one of them was wrong in the favourable direction.

The brief said to read `relion-acc-backend`, `relion-projection-optimize` and
`relion-intercard-scaling` and re-scope against what they found, following evidence rather than a
stale plan. What they carry forward:

| inherited from | claim | status after this pass |
|---|---|---|
| `relion-acc-backend` §4.5 | RELION runs a full iteration with its coarse compare in our code, **4452/4452 particle orientations bit-identical**, `relion_postprocess` **3.50195 Å** in both arms | **holds. This is the port's foundation and it is not reopened.** |
| `relion-acc-backend` §3.3 | "the whole-program ceiling from accelerating the E-step alone is ~41x" | **SUPERSEDED.** It substituted a *coarse-only* device figure (0.085 s) for the *entire* E-step particle loop. §0 makes the real split measurable: the coarse-only ceiling is **8.0x**, and 41x needs three more kernels. §2. |
| `relion-acc-backend` §4.12 | the shear interpolant's error is 6.0% median at this dataset's `--pad 1` and moves RELION's chosen orientation on 1 particle in 16 | **holds, and it is the blocker for any device projection.** §3 |
| `relion-projection-optimize` P8 | lever B is 1.25x, 51.20% of the measured floor, and needs a per-block reader offset before it is a shippable kernel | **holds. It is a kernel-level result on the shear, so it inherits §3's accuracy fork.** |
| `relion-intercard-scaling` §15.3 | 32 chips give 12.38x on the tutorial job, 38.7% of linear, crossover ~7,060 particles; reduction 117.4 ms per half-set | **holds.** |
| `relion-intercard-scaling` §8 | "per dollar cannot be sourced defensibly and should not be estimated" | **RETIRED this pass.** Both sides now have public list prices. §6. |

**The bottleneck did not move to I/O or to the M-step.** §0 settles it: the M-step is 1.362 s against
an E-step of 131.826 s, and 90.4% of that E-step is one kernel. So this task is about the E-step
after all — but the interesting question is no longer "can the coarse kernel go faster", it is **what
is left when it goes to zero**, and that is §2.

---

## 2. The pipeline floor, and it is not the sum of the primitive floors

Iteration 13, 5 MPI ranks x 6 threads = 30 threads, all rows measured this pass except where marked:

| term | wall, s | binding limit | can it be offloaded? |
|---|---|---|---|
| `getAllSquaredDifferencesCoarse` | **119.2** (90.4% of `expectation_6`) | compute-bound on CPU; the projection's trilinear **gather** is the mechanism, not the compare (`relion-acc-backend` §4.6: projection is 95.6% of the call, the `A+B−2C` compare is 4.4%) | **yes, already hooked** |
| `getAllSquaredDifferencesFine` | 4.61 | same kernel shape, significant subset only | yes, `runDiff2KernelFine` |
| `storeWeightedSums` | 4.56 | wavg + trilinear **scatter** into the destination volume | yes, `runWavgKernel` + `runBackProjectKernel` |
| `getFourierTransformsAndCtfs` | 3.43 | host FFTW per particle | no seam in any RELION backend |
| `expectation_1` + `expectation_2` | 2.11 | per-iteration host setup, never attributed by anything in this lineage | no |
| `maximization` / `reconstruction` (the M-step) | 1.362 | host FFTW + gridding. `src/backprojector.cpp` has **zero** `_CUDA_ENABLED`/`do_gpu` references, so every GPU RELION in the world pays this | no |
| `iterate: writeOutput` | 0.267 | disk | no |
| `flatten solvent` | 0.160 | host | no |
| **iteration** | **135.7** | | |

**The floor, and the three levels of it.** `D` is the device coarse E-step for one iteration, which
`relion-estep-integration` measures at 2.37 s for the whole 15-iteration trajectory on one p150, so
one cs=196 iteration is a fraction of a second and is not the term that matters.

| what is on the device | remaining host work per iteration | iteration | speedup |
|---|---|---|---|
| nothing (today, RELION's own CPU kernels) | 135.7 | **135.7 s** | 1.00x |
| **coarse only — what the bridge already hooks** | 3.90 + 4.61 + 4.56 + 3.43 = **16.50** | **16.5 s + D** | **≈8.0x** |
| all four E-step kernels | 3.90 + 3.43 = **7.33** | **7.3 s + D** | **≈18x** |
| + a device per-particle FFT (`ttnn-fft-blackhole-kernel-result`: 616 k img/s, 76.8% of the DRAM roof) | **3.90** | **3.9 s + D** | **≈35x** |

**So the pipeline floor is 3.90 s per iteration of RELION's own host code — setup, the M-step,
`writeOutput`, the solvent flatten — none of which has an accelerator seam in any of RELION's four
backends.** Over a 17-iteration refinement that is **66 s**, and it is 94% host code. That is the
answer to "derive the floor for the pipeline, which is not the sum of the primitive floors": the
primitive floors sum to a fraction of a second, and the pipeline floor is two orders of magnitude
above them, set entirely by code we do not touch.

**Residual accounting, and the one honest gap.** Of the 135.7 s iteration, 131.8 s is attributed to
named regions by direct measurement and 3.90 s to five named host terms. **Nothing is unattributed at
this altitude.** One term inside the 3.90 s has a named mechanism but no measurement:
`expectation_1` + `expectation_2` at 2.11 s is 54% of the pipeline floor and nothing in this lineage
has ever looked inside it. It is now the largest un-decomposed term in the program. §7 item 6.

**Bytes/time against the roof, both directions.** The coarse call moves at least 490 MB (231 MB of
model gather for 186 x 19,404 x 8 corners at 8 B, 260 MB of difference temporaries, 1.4 MB of shift
stack) and the host bridge does it in 151.2 ms = **3.24 GB/s**, two orders below this EPYC's socket
DRAM roof and physically consistent with the ~108%-of-600% single-core utilisation measured at the
time (`relion-acc-backend` §4.5/§4.7). In the other direction: the device coarse pass at 2.37 s for
2.660 GB of trajectory reduction plus its own model traffic is checked against the **measured**
404.9 GB/s DRAM roof and 42.48 GB/s composed ring bandwidth in `relion-intercard-scaling` §13.1/§15.2,
not against a datasheet. This lineage has published 668 GB/s on a ~400 GB/s card once; every roof
quoted here was measured on the silicon it is quoted for.

---

## 3. The fork that decides whether "end to end on Tenstorrent" can mean the same answer

The 8.0x above is a ceiling on a device kernel that **does not exist in an accuracy-preserving form**,
and this is the whole of the risk in the task. Restating it with only measured inputs:

- RELION's answer is defined by `CpuKernels::complex3D`, axis-aligned **trilinear** on the padded
  model. Our host bridge implements exactly that and is bit-identical on 4452/4452 particles.
- Every device projection kernel in this program (`tt_bio/kernels/fslice`, and the backprojection is
  its adjoint) is a **z-collapse plus per-row shear**, a different interpolant. It exists because a
  per-row NoC read is honoured at 16 B from L1 and 64 B from DRAM, so a bulk row read can express a
  shear and cannot express a gather.
- The shear costs **+0.0589 Å** at FSC 0.143 as a pipeline (n=12, CI [+0.0517, +0.0661]) — **but that
  was measured at `padding_factor` 2 and this dataset runs `--pad 1`**, where the same shear's
  reference error is **6.0% median** (range 0.3-13.3%, r=0.90 against the shear slope) and RELION's
  chosen orientation moves on **1 particle in 16**.

So there are exactly three routes to a device coarse pass, and none of them is free:

| route | speedup ceiling | accuracy | status |
|---|---|---|---|
| shear interpolant on device | the 8.0x of §2 | changes RELION's orientations on a single-digit % of particles per iteration; resolution cost at pad 1 **unmeasured** | buildable, **accuracy-gated** |
| exact trilinear on device | the 8.0x of §2 | bit-identical | **UNPRICED**, see below |
| compare-only on device, projection stays on host | **1.04x** of the bridge call, MEASURED | bit-identical | dead on arithmetic |

**The middle row is the one this lineage has not actually priced, and I am refusing to inherit its
dismissal.** `relion-acc-backend` §4.6 killed exact-trilinear-on-device by citing
`ttnn-scatter-gather-per-element-limited`'s "~10-14 cycles per element" and a "1446x worse" figure
measured on a different access pattern. Carrying that arithmetic through *this* workload gives
128.5 G corner gathers per iteration at 12 cycles over 130 cores at 1.35 GHz = **8.8 s**, against the
119.2 s of CPU coarse pass it would replace. That is a 13x, not a dead end. It is also a
`ttnn`-op-level number for a hand-written-kernel question, and the model is only 31.7 MB, which fits
across 130 cores' L1 at 244 kB each. **A roof quoted from another workload's measurement is exactly
the error this program has published three times.** §5 has the screen that settles it in one bounded
run, and it costs far less than the multi-day kernel it would justify.

---

## 4. The execution plan, in order, with the exact commands

**Host decision, decided and recorded rather than asked: every RELION run in this task happens on
qb1, not on the assigned pc card.** pc has **15 GB free of 227 GB (94% full)** and 12 cores; the
RELION tree plus the tutorial dataset is **36 GB** and already built on qb1, which has 464 GB free and
32 cores. Re-homing it would mean an 11.3 GB re-download and two cmake builds onto a disk that cannot
hold them, and `pc-disk-space-critical` is a standing fleet hazard. The RELION arms are CPU-only and
consume no card, so this does not take a card from anyone; pc card 0 stays available for the device
screen in step E4. The build and data live at `/home/ttuser/relion-scratch`, deliberately **outside**
any worktree — a gitignored `scratch/` inside a `~/.coworker/wt/` tree is deleted when the task
concludes, which already cost this lineage an 11.3 GB re-download once.

**Preconditions for every step.** `benchlock.sh` for every timed arm; commit and push after each step,
because these are 20-90 minute runs and a relaunch must not repeat one. Note that `benchlock.sh`
**gates acquisition, not the run** (`relion-acc-backend` §4.8: qb1's loadavg ranged 0.46 to 112.51
across one measurement window, and the same arm ran 2.01x faster under a clean lock). So every arm
below records `uptime` at start **and** end, and every wall is reported beside `/usr/bin/time`'s
user+sys CPU-seconds, which measures work done rather than elapsed and is the co-tenancy-robust
metric. **A wall whose loadavg trace moved by more than 2x during the run is reported as
contaminated, not quoted.**

---

### E1 — the by-stage split across the whole trajectory, not one iteration (~25 min)

§0 measured iteration 13. Iteration 1 is exhaustive (1,152 coarse orientations against 145) and
`CurrentImageSize` runs `32 76 80 128 144 146 146 150 154 154 160 160 196 196 196`, so the 90.4%
share is a late-iteration number and the trajectory average is what the deliverable needs. This is
the same instrument, one run, and it also produces RELION's own reference output to convergence.

```
cd /home/ttuser/relion-scratch
# reference arm, RELION's own kernels, profiler on, to CONVERGENCE (13 -> 17, no --auto_iter_max)
BENCHLOCK_MAXLOAD=3.0 ~/.coworker/scripts/benchlock.sh worker:relion-end-to-end -- \
  bash e2e_ref.sh          # copy of e2e_prof.sh with --auto_iter_max removed, --o e2e/ref_run
```
Accept: `rc=0` (the atexit teardown segfault is fixed — the previous binary printed a correct table
then exited 139; the statics are now leaked deliberately, see `tt_profile.h`), a complete
`ref_run_it017_*` tree, and a `TTPROF` table per follower. **Pre-registered prediction, written before
the run: the coarse share over the whole trajectory falls below the 90.4% of iteration 13, because
iteration 1's fine pass searches 9,216 orientations against 1,160, and lands in 80-90%.** If it lands
below 70% the §2 ceiling table is wrong and must be recomputed before anything else in this plan runs.

### E2 — a complete refinement to convergence with the coarse pass in our code (~60-90 min, chunked)

This is the milestone. Same shape as E1, `TT_RELION_BACKEND=torch`, so the coarse compare runs in
`tt_bio/cryoem/relion.py` for **every iteration**, not one.

**The bridge arm is ~3x the reference arm's wall** (`relion-acc-backend` §4.5 priced one call at
151.2 ms and predicted 337 s per follower per iteration, measuring 325 s — 3.6% off), so a
17-iteration from-scratch arm does not fit a 50-minute turn. **Chunk it with `--auto_iter_max`,** which
is what auto-refine already provides and what §4.5 used:

```
# chunk 1: continue from RELION's it012, stop at 15
mpirun -n 5 -x PYTHONPATH=<tt-bio checkout on qb1> -x TT_RELION_BACKEND=torch -x TT_RELION_PROFILE=1 \
  relion/build-tt/bin/relion_refine_mpi --o e2e/tt_run \
  --continue Tutorial5.0/Refine3D/job019/run_it012_optimiser.star --auto_iter_max 15 \
  --cpu --j 6 --pool 6 --dont_combine_weights_via_disc --preread_images
# chunk 2: continue from OUR it015, run to convergence
  ... --continue e2e/tt_run_it015_optimiser.star     # no --auto_iter_max
```
**`PYTHONPATH` is broken as inherited and must be fixed first.** `p3_arms.sh` points it at
`/home/ttuser/.coworker/wt/relion-acc-backend`, a worktree that no longer exists — the concluded task
took it. Clone `wk/relion-end-to-end` to a **stable path outside any worktree**
(`/home/ttuser/relion-scratch/tt-bio`) and point `PYTHONPATH` there. Verify with
`TTBridge: tt_bio.cryoem.relion loaded` from both followers and a non-zero handled-call count before
trusting a single number: **`--cpu` is mandatory and omitting it produces a healthy-looking, correct,
full-speed run that never calls the bridge once.** Three smoke runs of this integration were pure CPU.

Accept: `rc=0` on both chunks, a complete `tt_run_it017_*` tree, RELION writing its own output.

### E3 — is it the same answer? (~5 min, no device)

Against E1's reference arm, on the same data, in RELION's own currency. The battery is already
written: `p3_compare.py`.

```
relion_postprocess --i e2e/tt_run_it017_half1_class001_unfil.mrc --angpix 1.244835 --o pp/tt
relion_postprocess --i e2e/ref_run_it017_half1_class001_unfil.mrc --angpix 1.244835 --o pp/ref
python3 p3_compare.py e2e/ref_run_it017 e2e/tt_run_it017
```
| check | bar |
|---|---|
| `relion_postprocess` `FINAL RESOLUTION` | identical to the digit RELION prints (arm A/T were both **3.50195 Å** at one iteration) |
| gold-standard FSC 0.143 crossing, unmasked | **|Δ| ≤ 0.1 Å**, the program's standing bar |
| `_rlnAngleRot/Tilt/Psi`, `_rlnOriginX/YAngst` over 4,452 particles | 4452/4452 identical, as at one iteration |
| cross-FSC, ref half-k against tt half-k | min FSC over all shells ≥ 0.999 |
| **sha256 of each output half-map** | **reported per arm, and NOT used as an identity check** |

**That last row is not a formality.** `relion-acc-backend` §4.8 ran the reference arm twice and the
half-map sha256 differed while all 4,452 assignments and the FSC crossing were identical: **RELION's
ALTCPU reconstruction is not bit-reproducible run to run.** A doc that reads a sha difference here as
a bridge effect is reading noise. The gate asks for a per-arm hash; give it one, and say what it means.

### E4 — the screen that settles §3's unpriced route (~40 min, pc card 0, the only device work here)

Before anyone builds a trilinear gather kernel, measure the gather rate. `ttnn.generic_op`, model
distributed across 130 cores' L1 at 244 kB each, reader doing 16 B-aligned L1 reads along the real
access line (`xp = e0*x + e1*y` stepping x, so the walk is a 3D line, not random).

**Pre-registered kill gate, written before the build:** the route needs **128.5 G gathers per
iteration in under 15 s** to beat the 119.2 s of CPU coarse pass by 8x, i.e. **8.6 G gathers/s
chip-wide, 66 M/s/core, ~20 cycles/gather at 1.35 GHz**. Measure achieved gathers/s on one core first
and multiply by nothing — report the single-core rate and the 130-core rate as separate measurements.
**If a single core cannot reach 66 M gathers/s within 2x, the route is dead and §3's middle row closes
by measurement rather than by a borrowed citation.** Either outcome is a full result.

### E5 — the shear compounding arm, which is the only thing that can open the device path (~90 min, chunked, no device)

If E4 kills exact trilinear, the shear is the only device route, and the open question is not its
single-pass error but whether that error **compounds** over a refinement: the projection error
perturbs the scores, the scores pick the orientations, the backprojection error perturbs the map, and
that map is the next iteration's reference. **Every figure in this lineage, +0.0589 Å included, is a
single-pass score against a fixed truth.** The harness is complete: §4.12 already built the host shear
projector, so this runs on the host in exact fp32 with no device and no accuracy risk from the
hardware. Run E2 again with `TT_RELION_INTERP=shear`, then E3's battery.

**Pre-registered:** report the FSC 0.143 delta against E1 **and** the per-iteration orientation
reassignment rate against E1's assignments. A reassignment rate that grows iteration over iteration is
compounding; one that is flat at a few percent is not. **Do not report the resolution delta without
the rate, or the reverse** — §4.11 measured the argmin's tolerance, could not measure the shear's
error, bridged it with an inference, and §4.12 refuted the inference one pass later.

### E6 — the DGX H200 arm (~2-3 h, ≤$15 of the $40 cap)

The one number the program has never had: a **measured** whole-refinement GPU wall on this dataset.
Rent **one** H200 (Lambda on-demand is the cheapest live rate found this pass at $2.29/hr; vast.ai
lists ~$3.62-3.82/hr), build RELION with `-DCUDA=ON`, run the same tutorial Refine3D job to
convergence, tear down and confirm.

| what it gives | status it earns |
|---|---|
| 1x H200 whole-refinement wall | **MEASURED** |
| 8x H200 | **EXTRAPOLATED**, and RELION scales poorly past 4 GPUs, so extrapolate from RELION's own figure below, never linearly |

**Verified from primary source this pass** — RELION 5's own tutorial, for this exact Refine3D job:
*"On our computer with 4 GPUs, we used 5 MPIs and 6 threads, and this calculation took approximately
7 minutes."* That is **420 s for the whole refinement on 4 GPUs**, RELION's own claim about its own
job, and it is the anchor the comparison hangs on. The GPU model is not stated, which is exactly why
E6 rents one.

Teardown is not optional and not assumed: record the instance id, the destroy call and the final
balance in the doc.

---

## 5. The comparison, with every price and watt sourced, and the measured/extrapolated split

**Per dollar is now sourceable on both sides, which retires `relion-intercard-scaling` §8's refusal.**
That refusal was correct when written — a Blackhole UBB had no public price. It does now.

| | Tenstorrent Galaxy Blackhole | DGX / HGX H200, 8 GPU |
|---|---|---|
| **list price** | **$110,000** per 6U node, 32 chips | **$320k-420k**, ~$370k typical integrated OEM price; DGX-branded quoted $400-500k |
| source | Tenstorrent launch pricing, April 2026, reported by The Register and Dealroom | Mercatus OEM survey; ITCT quote |
| status | **public list**, not negotiated | **public survey**, spread up to 25% across OEMs, volume 8-20% below list |
| chip/GPU power | **135 W p50 measured sustained**, 32 x = 4.32 kW | 700 W datasheet, 8 x = 5.60 kW |
| bf16 throughput | **5.95 PFLOP/s measured sustained** (32 x 185.8) | 7.91 PFLOP/s datasheet peak, ~6.3 at a typical 80% achieved |
| TFLOP/s per watt | **1.38 measured** | 1.41 datasheet-peak, ~1.13 achieved |

Neither power figure includes host, board, fan or memory power, on either side. The Galaxy's 23
PFLOPS-dense-FP8 marketing number is consistent with the measured bf16 rate at the usual 2x and is
not used here.

**The comparison the deliverable owes, and the honest shape of it before E1-E6 run.** RELION's own
4-GPU figure is 420 s for the whole refinement. §2's Galaxy **floor** is 3.90 s x 17 = **66 s**, of
which 94% is RELION host code — so at its floor the Galaxy leads on wall-clock, and at $110k against
$370k it leads by a further 3.4x on price. **But that floor requires four device kernels that do not
exist, one of which is accuracy-blocked (§3), and today's measured position is that the bridge arm is
slower than RELION's own CPU kernel.** Every row of the final table must carry MEASURED or
EXTRAPOLATED, and the Galaxy row is EXTRAPOLATED until a device arm runs. Publishing the 66 s as an
achieved number would be the single worst thing this task could do.

---

## 6. Decided against, so execution does not relitigate it

1. **Do not build the device coarse kernel this task.** §3 says the only buildable route changes
   RELION's orientations and its resolution cost at `--pad 1` is unmeasured; E5 is the measurement
   that decides it, and it needs no device. Building the kernel first would be building on an
   unresolved accuracy fork, and a new-model-port-shaped change cannot merge without Moritz anyway.
2. **Do not re-home RELION onto pc.** §4: 15 GB free against a 36 GB tree.
3. **Do not chase a clean wall on qb1 by fixing `benchlock`.** §4.8 closed this: a duration-holding
   lock is a fleet change, not a task change. Report CPU-seconds beside the wall instead.
4. **Do not batch particles for the coarse compare.** `relion-acc-backend` §4.9 measured 256 calls in
   one iteration with **256 distinct euler-set hashes** — auto-refine builds a per-particle projection
   plan (`cpu_ml_optimiser.cpp:123`, `do_auto_refine` trips it unconditionally), so there is no shared
   slice store to amortise. Batching still pays for Class2D/Class3D and for the backprojection's
   shared destination volume. Not for this.
5. **Do not build the bf16 or sphere-packed reduction, and do not build the host-mediated PCIe
   reduce.** Both closed by measurement in `relion-intercard-scaling` (§0.5: 1.88x on a minority term
   whose precision half is already spent at bf16-grade; §14: 36x worse than ethernet as shipped, 2x
   worse at its measured 21.6 GB/s roof).
6. **Do not re-run the interpolant FSC at pad 2.** E5 subsumes it: it measures the reassignment's
   actual effect on resolution over a real refinement instead of its rate at one operating point.
7. **Do not quote the 43x reduction advantage.** 21x of it is OpenMPI's untuned default
   (`relion-intercard-scaling` §13.2). "Structurally cheaper" is the defensible claim.
8. **Do not re-derive lever O, lever D, lever F-at-LoFi, or lever W/R/I/P/X.** All dead with evidence
   in `relion-projection-optimize` §5 and P8. Lever F (HiFi2) is GATED at 1.127x for a 3.13x accuracy
   regression and stays out of any headline.

---

## 7. Open, in priority order

1. **E1-E3: a complete refinement through the bridge, and whether it is the same answer.** The
   milestone. Nothing else in the deliverable can be written honestly first.
2. **E4: price the exact-trilinear gather.** It is the difference between a bit-identical port and an
   accuracy-gated one, and it is currently closed by a borrowed citation rather than a measurement.
3. **E5: does the shear's error compound over a refinement.** Decides the device path if E4 kills
   exact trilinear.
4. **E6: the measured GPU wall.** The comparison is EXTRAPOLATED on both sides without it.
5. **The device arm itself**, which needs 2 cards for gold-standard MPI (RELION 5 refuses a
   single-process gold-standard split, and its own error message advertises `--debug_split_random_half`
   which no longer parses) or two single-halfset runs assembled.
6. **`expectation_1` + `expectation_2` at 2.11 s is 54% of the pipeline floor and nothing has ever
   looked inside it.** After §2 it is the largest un-decomposed term in the whole program, and it is
   bigger than the M-step everyone has been worried about.

**GO/NO-GO, stated for the decision actually in front of us: GO on E1-E3 and E6, GO on E4 as a
bounded screen, GO on E5, and NO-GO on building the device coarse kernel until E4 and E5 report.**
The reason to say it that way rather than "GO on the port": §2 shows the prize is real (8.0x from the
kernel already hooked, 35x if three more follow, against a 66 s floor that beats RELION's own 4-GPU
figure at a third of the price), and §3 shows the only currently buildable route to it silently
changes RELION's answer. Those two facts together mean the next work is measurement, not kernels.

---

## 8. Durable lessons from this pass

- **RELION's accelerated E-step has no observable stage split, for two independent reasons, and both
  are traps.** `TIMING_ESP_DIFF1`/`DIFF2`/`WSUM` are tic'd only when
  `op.part_id == baseMLO->exp_my_first_part_id` (or `thread_id == 0`), which never fires on the ALTCPU
  path, and `Timer::printTimes` only prints tags with `counts[i] > 0` — so they vanish from the table
  rather than printing zero. Independently, the CTIC/CTOC macros that bracket every interesting region
  are defined empty at the top of `src/acc/cpu/cpu_benchmark_utils.h`, and the block that would give
  them a body is inside a `/* ... */` comment. **Giving those macros a thread-safe body is a one-file
  change that instruments 51 regions at a measured 0.7% cost**, and it is why every stage share in this
  lineage before now was derived from sampling counts.
- **The sampling counts predicted the wrong answer by a factor of 5, in the favourable direction.**
  Coarse is 1,305 sampling points per particle against the fine pass's 41,760, which reads as "the
  fine pass is 32x the work". Measured, coarse is **90.4%** and fine is **3.5%**, because the fine
  pass only ever evaluates the significant subset. A work estimate from a sampling-space count is not
  a measurement, and this one was wrong by 25x on the ratio.
- **An atexit handler that reads a function-local `static` container is a use-after-free.** atexit
  handlers and static destructors share one LIFO list, so a container constructed after the handler is
  registered is destroyed before it runs. The symptom is a correct printed table followed by exit 139.
  Leak the container on purpose.
- **A ceiling built by substituting one component's device time for a whole phase is not a ceiling.**
  `relion-acc-backend` §3.3's "~41x from accelerating the E-step alone" put a coarse-only 0.085 s in
  place of the entire E-step particle loop. The real coarse-only number is 8.0x. The tell was in the
  table's own row label, and the fix needed a measurement rather than an argument.
- **A roof borrowed from another workload's measurement is an asserted roof.** The
  exact-trilinear-on-device route was closed by citing a scatter/gather figure measured on a different
  access pattern; carried through this workload's own numbers the same citation gives 13x, not a dead
  end. Same failure family as the 254.5 TFLOP/s matmul roof that turned out to be 212.7 on UBB silicon.
- **A Blackhole Galaxy now has a public list price ($110,000, 32 chips, 6U), so "per dollar has no
  defensible source" is retired.** Re-check a "cannot be sourced" verdict when the product ships.
