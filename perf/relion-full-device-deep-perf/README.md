# The coarse-blocking fix — screen, arms, parity

RELION's blocked coarse `diff2` kernel never runs on an auto-refine job with fewer coarse
orientations than `D2C_BLOCK_SIZE_REF3D` (256). `acc/acc_helper_functions_impl.h` computes
`rest = orientation_num % blocks3D`, where `blocks3D` is a **pixel** block size but the quantity the
blocked kernel needs `orientation_num` to be divisible by is the **orientation** block size
`D2C_EULERS_PER_BLOCK_REF3D` (16). So `even_orientation_num` is 0 and every coarse call falls to the
one-orientation remainder path.

That matters because `USE_SINCOS_TABLE` is never defined in RELION's build, so the translation loop
calls libm `sincosf` once per (orientation, pixel, translation). Blocking amortises that call
16-fold; unblocked, it is paid nine times per (orientation, pixel) pair. `perf` on a live worker rank
puts `__sincosf_fma` second in the whole program at 19.94% of cycles.

The fix is `integrations/relion/relion5-coarse-blocking.patch`, one line.

## Result — three full auto-refines, one binary, `TT_COARSE_E` the only variable

| arm | `TT_COARSE_E` | wall, s | coarse kernel, thread-s | ns/pair | user CPU, s | postprocess |
|---|---|---|---|---|---|---|
| e1 | 1 (= RELION today) | 878.18 | 14,507.78 | 164.81 | 19,751.26 | 4.033896 Å |
| e1b | 1 (control, rerun) | 876.26 | 14,522.67 | 164.96 | 19,761.72 | 4.033896 Å |
| e16 | 16 (the fix) | **442.18** | 4,860.70 | 55.21 | 9,651.87 | 4.033896 Å |

**1.986x on the wall, 2.985x on the kernel**, against a run-to-run control that reproduces to 0.22%
on the wall and 0.10% on the kernel.

## Parity

`relion_postprocess` unmasked at `--angpix 1.244835` gives **4.033896 Å on all three arms**, the
standing gate value. The it013 `_data.star` — the first E-step after the change — is **byte-identical
across all three runs**, and the replay harness has `diff2s` bit-exact at every E.

From it014 the star files differ between arms **and equally between e1 and e1b**, so the divergence
is RELION's own: `backprojectRef3D` accumulates under a `tbb::spin_mutex` in thread-arrival order,
which makes the reconstruction non-reproducible run to run. An auto-refine's later-iteration outputs
are therefore not usable as a parity oracle for anything. The first E-step and the final resolution
are.

## Screen — the replay harness

`coarse_replay.cpp` links the same `librelion_lib.a` the profiled refinement ran and replays a live
dumped call. Single thread, ns per (orientation, pixel) pair, best of 3 after a warm rep:

| build | E=1 | E=2 | E=4 | E=8 | E=16 |
|---|---|---|---|---|---|
| base, RELION's own flags | 96.42 | 70.15 | 42.93 | 35.43 | **30.81** |
| `-DUSE_SINCOS_TABLE` | 60.00 | | 34.12 | | 29.20 |
| `-march=native` | 78.68 | | 31.85 | | **21.22** |
| both | 58.48 | | 26.07 | | 23.99 |

At the refinement's topology (4 processes x 6 OMP threads, one shared model each): base 27.75 -> 8.90
(3.12x), `-march=native` 23.67 -> 6.45.

**The harness predicted the refinement to 1%**: 27.75 ns/pair across 6 threads is 166.5
thread-ns/pair against the refinement's measured 164.81.

| variant | bit-exact vs base E=1 | max rel err |
|---|---|---|
| base, E = 2, 4, 8, 16 | **YES** | 0 |
| `-DUSE_SINCOS_TABLE`, any E | no | 2.56e-07 |
| `-march=native`, any E | no | 1.93e-07 |

`-DUSE_SINCOS_TABLE` is dead: 1.06x once E is fixed, negative combined with `-march=native`, and it
costs bit-exactness. `-march=native` is worth a further 1.38x at 1-2 ulp; RELION's CMake passes no
`-march` at all.

## Other backends

The same line is compiled by `cuda_helper_functions.cu` and `hip_helper_functions.hip.cpp`, where
`D2C_BLOCK_SIZE_REF3D` is 128. At this job's 174-204 orientations that leaves ~29% of orientations on
the unblocked path rather than 100%. Their cost profile differs — a GPU's `sincosf` is a hardware
instruction — and **the GPU impact was not measured here and is not claimed.**

## Files

- `integrations/relion/relion5-coarse-blocking.patch` — the fix, `git apply --check` clean on pristine
- `coarse_replay.cpp`, `build_replay.sh`, `coarse_dump2bin.py` — the screen
- `tt_coarse_tune.h`, `apply_tune_patch.py` — the instrumented build (`TT_COARSE_TUNE`, `TT_COARSE_E`)
- `w1_profile.sh`, `w1/` — the perf attribution
- `w2_arms.sh`, `w2_parity.sh`, `w2/` — the three arms and the gate
