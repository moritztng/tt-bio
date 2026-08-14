# RELION 5 on Tenstorrent

This directory holds everything needed to make RELION 5 call tt-bio kernels for its expectation step.
It is a fork overlay, not a vendored copy: `relion5-tt.patch` applies to a clean `3dem/relion` `ver5.0`
checkout and `src_acc_tt/` drops into `src/acc/tt/`.

Nothing here is merged into RELION upstream and nothing here is enabled by default.

## What it does

RELION's accelerator seam, `src/acc`, already carries four backends (`cuda`, `hip`, `sycl`, `cpu`).
The patch adds a fifth entry point at the layer where RELION passes whole tensors rather than launch
geometry: `runDiff2KernelCoarse` in `src/acc/acc_helper_functions_impl.h`. When the shape is a 3D
reference against 2D data, the call is handed to `TTBridge::diff2Coarse` and RELION's own kernel is
skipped. Every other shape, and any failure, falls through to the existing code, so a broken device
degrades to CPU instead of failing the refinement.

`src/acc/tt/tt_bridge.cpp` is the only file in RELION that knows about Python. It embeds one CPython
interpreter per process and calls `tt_bio.cryoem.relion`, passing RELION's own buffers as memoryviews
so nothing is copied. The interface is plain C++ over plain arrays, so replacing the embedded
interpreter with a C++ libttnn implementation later needs no change anywhere else in RELION.

## Build

```sh
git clone --branch ver5.0 --depth 1 https://github.com/3dem/relion
cd relion && git apply /path/to/relion5-tt.patch
mkdir -p src/acc/tt && cp /path/to/src_acc_tt/* src/acc/tt/
mkdir build && cd build
cmake .. -DALTCPU=ON -DTT=ON -DCUDA=OFF -DGUI=OFF -DMKLFFT=OFF -DFETCH_WEIGHTS=OFF
make -j
```

`TT=ON` requires `ALTCPU=ON`: the kernels that are not offloaded yet compile from `src/acc/cpu`, and
`ALTCPU` is also what defines `PROJECTOR_NO_TEXTURES`, which makes the padded Fourier reference a plain
array instead of a texture object. `TT=ON` without `ALTCPU=ON` is a configure-time error.

Needs Python development headers (`python3-dev`).

## Run

```sh
export PYTHONPATH=/path/to/tt-bio
export TT_RELION_BACKEND=torch        # or ttnn
mpirun -n 3 -x PYTHONPATH -x TT_RELION_BACKEND \
    build/bin/relion_refine_mpi --cpu --j 6 ... 
```

`--cpu` is required. Without it RELION uses its plain scalar path and never enters `src/acc` at all,
so the bridge is never called and the run silently produces a pure-CPU answer.

One MPI rank per card. Gold-standard 3D auto-refine needs at least three ranks (a leader plus one
follower per half set), so it needs two cards. RELION's own error message suggests
`--debug_split_random_half` as a single-process escape; that option no longer parses in RELION 5.

Environment read by the Python side:

| variable | meaning |
|---|---|
| `TT_RELION_BACKEND` | `torch` (host, exact, slow) or `ttnn` (device) |
| `TT_RELION_ORI_CHUNK` | orientations per chunk, bounds peak host memory |
| `TT_RELION_CHECK` | keep the computed scores for a residual check |
