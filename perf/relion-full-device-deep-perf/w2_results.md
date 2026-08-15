# W2 screen — the replay harness, single thread and at the refinement's topology

`perf/relion-full-device-deep-perf/coarse_replay.cpp`, linking the same `build-e2e/lib/librelion_lib.a`
the profiled refinement ran, replaying `p8/call.2738115.1.npz` (a live `relion_refine_mpi` call:
orientation_num 180, translation_num 9, image_size 19404, model 100x199x199).

## Single thread, ns/(orientation,pixel) pair, best of 3 after a warm rep

| build | E=1 | E=2 | E=4 | E=8 | E=16 | E=1 -> E=16 |
|---|---|---|---|---|---|---|
| base (RELION's own flags) | 96.42 | 70.15 | 42.93 | 35.43 | **30.81** | **3.13x** |
| + `-DUSE_SINCOS_TABLE` | 60.00 | | 34.12 | | 29.20 | 2.05x |
| + `-march=native` | 78.68 | | 31.85 | | **21.22** | 3.71x |
| + both | 58.48 | | 26.07 | | 23.99 | 2.44x |

## At the refinement's exact topology: 4 processes x 6 OMP threads, one shared model per process

| build | E=1 | E=16 | ratio |
|---|---|---|---|
| base | 27.75 | 8.90 | **3.12x** |
| `-march=native` | 23.67 | 6.45 | 3.67x |

base E=1 -> march E=16 is **4.30x**. The ratio survives full 24-thread contention on 16 cores, so the
win is not an artifact of an uncontended L3.

## Parity

| variant | bit-exact vs base E=1 | max rel err |
|---|---|---|
| base, E = 2, 4, 8, 16 | **YES** | 0 |
| `-DUSE_SINCOS_TABLE`, any E | no | 2.56e-07 |
| `-march=native`, any E | no | 1.93e-07 |

The blocking fix on its own is **bit-exact**, at every E, which is the strongest possible parity
result: it cannot move the resolution. The other two perturb by 1-2 ulp of float32 and would have to
clear the §6.1 gate on merit.
