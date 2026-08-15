
---

# EXEC PASS 1 — 2026-08-15, qb1, opus5

What ran: W1 in full, W2 steps 0-3 as a screen on a replay harness, and the instrumented RELION
build the paired refinements need. **W1 changes the plan's own diagnosis, and W2's screen returns
2-3x more than P2 predicted.** No Tenstorrent device was used and no lease was held; every number
below is host CPU.

## E1. W1 — the roof is named, and it is not the one the plan expected

`perf/relion-full-device-deep-perf/w1_profile.sh`. `perf_event_paranoid` is 4 on qb1, which blocks
unprivileged perf entirely; the script lowers it for the profile and restores it. One worker rank of
a live 5-rank `ref` refinement (no `PYTHONPATH`, RELION's own kernels), 60 s of `cycles:P` sampling
at 499 Hz after a 200 s warm-in, then 20 s of `perf stat` on the same pid.

**Flat profile, self cycles, one worker rank (167K samples, 0 lost):**

| symbol | self | what it is |
|---|---|---|
| `runDiff2KernelCoarse` | **40.99%** | the coarse kernel's own code |
| `__sincosf_fma` (libm) | **19.94%** | 13.65 pts of it called directly from `runDiff2KernelCoarse` |
| `CpuKernels::complex3D` | **18.22%** | the model gather inside `project3Dmodel` |
| `backprojectRef3D` | 5.58% | maximisation |
| `runDiff2KernelFine` | 4.36% | the fine pass |
| `runWavgKernel` | 1.91% | `storeWeightedSums`' inner kernel |
| `sincosf@plt` | 1.30% | same call, unresolved PLT stub |
| `selectOrientationsWithNonZeroPriorProbability` + `Matrix1D/2D` ops | ~2.04% | `nonZeroProb`, W3's target |

**The coarse pass and its callees are 80.45% of a worker rank's cycles** (40.99 + 19.94 + 18.22 +
1.30). Two things follow, and both matter:

1. **The region is the kernel.** The plan carried a 273.5 ns/pair figure for
   `getAllSquaredDifferencesCoarse` and flagged that nobody had measured what binds it. There is no
   large non-kernel term hiding inside the region — no plan build, no euler generation, no
   bookkeeping worth naming. Whatever the kernel wins, the region wins.
2. **A fifth of RELION's entire refinement is libm `sincos`.** `__sincosf_fma` at 19.94% self is the
   second-largest symbol in the whole program, and it is one call: `TRANSLATE_PIXEL_2D` in
   `src/acc/cpu/cpu_kernels/helper.h:600`, invoked once per (orientation, pixel, translation).

**`perf stat` on the same pid, 20 s:**

| counter | value |
|---|---|
| cycles | 277,556,816,697 |
| instructions | 696,289,184,013 |
| **IPC** | **2.51** |
| L1-dcache-loads | 218,345,725,798 |
| **L1d miss rate** | **1.39%** |

**This is the roof, measured and not asserted: the coarse pass is issue-bound, not memory-bound.**
IPC 2.51 on a 4-wide-retire Zen4c with a 1.39% L1d miss rate is a core running near its front end
with its data resident. `roofline-roof-must-be-measured-not-asserted` asked for the roof to be named
by measurement; it is an **instruction-count** roof, so the lever is to retire fewer instructions per
pair, and no amount of DRAM bandwidth or cache blocking is relevant.

**P1's fork resolves to the instruction-count branch**, which is the branch that keeps W2 alive. The
alternative branch ("if it is backend/memory stalls on the model gather, the device kernel faces the
same wall and the 2.72x is real") is refuted by the 1.39% miss rate.

**P1's literal prediction is not gradeable as stated and that is a prediction error worth recording.**
P1 asked for instructions/pair for `diff2_coarse` against `wavg_ref3D` in one profile. `wavg` is
1.91% of cycles against the coarse pass's 80.45% — a 42:1 ratio — so a per-pair instruction
comparison between them would rest on two pair censuses of very different quality, and the profile
answered the underlying question (what binds the coarse kernel) directly instead. The 4.3x ns/pair
gap between the two kernels that motivated P1 is now explained without needing the comparison: `wavg`
does not call `sincos` per pixel-translation, and the coarse kernel does.

## E2. W2 screen — the replay harness, and the win is 3.12x not 1.15-1.6x

`perf/relion-full-device-deep-perf/coarse_replay.cpp` links the same
`build-e2e/lib/librelion_lib.a` the profiled refinement ran and replays a live dumped call
(`p8/call.2738115.1.npz`: orientation_num 180, translation_num 9, image_size 19404, model
100x199x199) through RELION's own `CpuKernels::diff2_coarse` at a chosen eulers-per-block.

Why this instrument instead of the plan's five full refinements: the kernel is a pure function of
its dumped arguments, so a replay answers "what does eulers-per-block cost" in seconds rather than
15 minutes, and it lets the sweep be widened. The plan's own W1 used the same trick
(`nzp_screen.cpp` against `librelion_lib.a`). Per
`tt-bio-isolated-op-timing-oversync-inflates-cost` the isolated rate is **not** quoted as the
refinement's rate: every arm is graded as a ratio, the ratio is then re-measured at the refinement's
exact thread topology, and the winner still has to clear a paired refinement.

### E2.1 Step 0 — the runtime confirmation

`orientation_num` read out of all 16 live `p8/call.*.npz` dumps is 174-204, every value below the
256 that `rest = orientation_num % blocks3D` divides by, so `even_orientation_num` is 0 on every
call and RELION's blocked coarse kernel never runs. The dumps also confirm the branch by execution:
the TT bridge hook sits at `acc_helper_functions_impl.h:1166` inside
`if (!do_CC && projector.mdlZ != 0 && !data_is_3D)`, so a dump existing at all proves the live run
takes that branch. §5.1's static read stands. The one-shot `fprintf` the plan asked for is built
into the instrumented binary (`TTCoarseTune::announce`) and prints on the first coarse call of the
paired arms.

### E2.2 Single thread, ns per (orientation, pixel) pair, best of 3 after a warm rep

| build | E=1 | E=2 | E=4 | E=8 | E=16 | E=1 -> E=16 |
|---|---|---|---|---|---|---|
| base, RELION's own flags | 96.42 | 70.15 | 42.93 | 35.43 | **30.81** | **3.13x** |
| `-DUSE_SINCOS_TABLE` | 60.00 | | 34.12 | | 29.20 | 2.05x |
| `-march=native` | 78.68 | | 31.85 | | **21.22** | 3.71x |
| both | 58.48 | | 26.07 | | 23.99 | 2.44x |

### E2.3 At the refinement's exact topology: 4 concurrent processes x 6 OMP threads, one shared model each

This is the 4 worker ranks x `--j 6` the refinement runs, 24 threads on 16 cores, each process's
threads sharing one 31.7 MB model volume against a 16 MiB L3.

| build | E=1 | E=16 | ratio |
|---|---|---|---|
| base | 27.75 | 8.90 | **3.12x** |
| `-march=native` | 23.67 | 6.45 | 3.67x |

base E=1 -> march E=16 is **4.30x**. **The ratio survives full contention** (3.13x isolated ->
3.12x contended), which is the check that mattered: the win is instruction count, and E2.1's 1.39%
L1d miss rate says there was no memory wall for contention to expose.

### E2.4 Parity

| variant | bit-exact vs base E=1 | max rel err |
|---|---|---|
| base, E = 2, 4, 8, 16 | **YES, 0 differing elements** | 0 |
| `-DUSE_SINCOS_TABLE`, any E | no | 2.56e-07 |
| `-march=native`, any E | no | 1.93e-07 |

**The blocking fix is bit-exact at every E.** It cannot move the resolution, which is the strongest
parity result available and it makes the §6.1 gate a formality rather than a risk. The other two
perturb by 1-2 ulp of float32 and would have to clear the gate on merit.

Harness sanity: its E=1 output matches the bridge's independent torch reference to 3.8e-07 relative
after the factor of 2 that RELION's `s_corr = g_corr * 0.5` introduces, so the harness is running the
real kernel on real arguments and not a mock.

### E2.5 The mechanism, which is not the one §5.1 named

§5.1 attributed the cost to re-walking the 467 kB `x`/`y`/`s_real`/`s_imag`/`s_corr` index tables
once per block. Reading the kernel, that is wrong: the tables are precomputed once per *call*, and
the total number of table reads is `orientation_num x pass_num x block_sz` **independent of E**.

The real mechanism is one line. `USE_SINCOS_TABLE` is **never defined anywhere in RELION's build**
(`grep -rn USE_SINCOS_TABLE src/` finds only the three `#ifdef`s in `diff2.h`), so the translation
loop compiles to the `#else` path and calls `TRANSLATE_PIXEL_2D`, which is a **`sincosf` per
(orientation, pixel, translation)**. That call sits inside the `#pragma omp simd` loop and blocks
its vectorisation as well.

Structurally, per (orientation, pixel) pair the translation loop costs
`translation_num x (F/E + J)`, where `F` is the sincos plus the phase rotation and `J` is the
per-orientation difference accumulate. At E=1 you pay `F` nine times per pair; at E=16 you pay it
0.56 times. That is why the win is ~3x and not the ~1.2x a table-re-read argument predicts, and it
is why `perf` puts `__sincosf_fma` second on the whole program.

### E2.6 What died in this screen

- **`-DUSE_SINCOS_TABLE` — DEAD.** It is a 1.61x lever at E=1 and a **1.06x** lever at E=16 (30.81 ->
  29.20), because fixing E already amortises the sincos 16-fold. Combined with `-march=native` it is
  **negative** (21.22 -> 23.99). It also costs bit-exactness. Once E is fixed there is nothing left
  for it to win, so it is not worth a parity argument.
- **Sweeping `D2C_BLOCK_SIZE_REF3D` — still dead**, §9 item 4, and E2.5 gives the reason
  independently: the pixel block size does not appear in the `F/E` term at all.
- **The 467 kB table-re-read mechanism — dead as a mechanism**, E2.5. The lever survives, the
  explanation for it does not.

### E2.7 What survives

- **The `rest % E` fix at E=16.** Bit-exact, 3.12x on the coarse kernel at the refinement's topology,
  one line.
- **`-march=native`.** A further 1.38x on top (8.90 -> 6.45), 1-2 ulp, needs the §6.1 gate. RELION's
  CMake passes no `-march` at all, so the whole `acc` backend is built for baseline x86-64 on a
  Zen4c that has AVX-512.

## E3. What this does to the program's headline, and it is not good news for the device

Composing E2.3 with §2's floor table. The coarse term is 681.3 s of a 922.19 s reference refinement,
and E1 established that the region is the kernel, so the region scales with the kernel's ratio.

| scenario | coarse term, s | reference refinement, s | device coarse (250.8 s) vs it |
|---|---|---|---|
| RELION today | 681.3 | 922.19 | **2.72x** |
| + the bit-exact `rest % E` fix at E=16 | 218.4 | **459.3** | **0.87x — the device loses** |
| + `-march=native` as well | 158.4 | **399.3** | **0.63x** |

**This is §5.3's bottom row arriving, and it arrives from a one-line bit-exact change.** The device
coarse kernel's 2.72x was never a property of the silicon; it was the distance to a RELION CPU
kernel calling libm `sincos` nine times per pixel with the blocking that would have amortised it
switched off by a modulus against the wrong variable. Fix that and one p150's coarse kernel is
**slower** than the CPU it was beating.

**Held as predicted, not concluded:** the table above is arithmetic on a screen, not a refinement.
The paired arms in E4 are what settle it, and the verdict line does not get written until they run.

**W5 stays held and its kill line did not fire.** §7 said W5 runs "only if W2's kill line fires — i.e.
only if RELION's coarse kernel turns out to be as expensive as it looks". It does not. Not starting
the multi-day device build is the single most valuable thing this pass did, and it is exactly what
`perf-method-floor-screen-predict-then-build` exists to produce.

## E4. Built and ready: the instrumented binary, and P4 pre-registered

`perf/relion-full-device-deep-perf/tt_coarse_tune.h` plus `apply_tune_patch.py`, applied to
`/home/ttuser/relion-scratch/relion/src/acc/`, and `build-e2e` rebuilt. Everything is behind
`#ifdef TT_COARSE_TUNE`, and `TT_COARSE_E` defaults to **0 = leave RELION's dispatch exactly as it
was**, so the binary without the env var is behaviourally the old one plus a timer. The source tree
is a git checkout, so it is one `git checkout` from pristine.

It adds three things: the one-shot `announce` (step 0's runtime confirmation), an exact atomically
accumulated timer around `runDiff2KernelCoarse` printed at exit (so the paired arms report kernel
thread-seconds directly, no sampling), and `TT_COARSE_E` to force eulers-per-block so **one binary
runs both arms** — which removes the compiler and the link from the comparison entirely.

**P4, pre-registered before the arms run.** Paired full it13-17 refinements under `benchlock`,
`TT_COARSE_E=1` (control, byte-identical dispatch to RELION today) against `TT_COARSE_E=16`:

- `[tt_coarse_tune]` kernel thread-seconds fall **3.0-3.3x**, from E2.3's 3.12x at the same topology.
- The refinement **wall** falls **1.8-2.1x**, to **440-510 s** from 922 s.
- `diff2s` is bit-identical, so the resolution digit is unchanged, `relion_postprocess` prints
  **4.033896 Å**, and `e2e_disagree.py` stays at median 0.000000 deg / p99 0.000000 deg.

Unlike W2 and W3, this prize is **far outside** the 11.2% whole-refinement noise floor, so the wall
is gradeable here and §6's "grade on region seconds not the wall" restriction does not bind. The
`[tt_coarse_tune]` line is reported anyway because it is the tighter measurement.

**Kill line for P4:** if the E=16 arm's kernel thread-seconds do not improve by at least 2x, the
replay harness is not representative of the refinement and E2/E3 are withdrawn.
