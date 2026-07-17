# BoltzGen diffusion trace replay — lossless, no measurable wall-clock win

**Goal:** port Protenix's ttnn trace-replay machinery
(`tt_bio/protenix.py:_capture_trace` / `denoise_traced`, `fold(trace=True)`,
measured -22% warm diffusion @L256) to BoltzGen's diffusion sampling loop.
BoltzGen's per-step device graph is
shape-stable across all 500 sampling steps (only the scalar `times` and the `r`
coords change; schedule phases are host-side scalars), the same precondition
Protenix's trace relies on, so the mechanism transfers directly.

Single card qb2, `examples/binder.yaml --steps design` (the scout's recipe),
`TT_MESH_GRAPH_DESC_PATH=/tmp/single_bh_mesh_graph_descriptor.textproto`.

## What shipped

Opt-in (`gen run --diffusion_trace`, threaded as `override.diffusion_trace=true`
→ `Boltz.__init__(diffusion_trace=True)`) trace replay of the per-step DiT device
stream, mirroring Protenix's idiom on the shared `TTDiffusionModule`
(`tt_bio/tenstorrent.py`):

- `Boltz.__init__(diffusion_trace=False)` reserves the trace region by calling
  `get_device(trace_region_size=1 << 30)` before any module opens the device
  (the first `get_device()` opens; subsequent calls return the cached device),
  and passes the flag to `AtomDiffusion`. `AtomDiffusion._trace` threads through
  `preconditioned_network_forward` to `TTScoreModelAdapter.forward(trace=...)`.
  Mirrors the `use_resident_trunk` kwarg pattern; no env-var toggle
  (`prefer-args-over-envvars`), matching the Protenix `fold(trace=)` /
  `get_device(trace_region_size=)` precedent.
- `TTDiffusionModule.forward` is refactored into `_populate_diffusion_cache`
  (hoists the per-step-invariant conditioning once) and `_run_diffusion_device`
  (the pure on-device DiT over the cached conditioning). `forward_traced`
  captures the `_run_diffusion_device` graph once per `(B, N_padded)` — staging
  `r` and `times` into persistent input buffers, warming the lazy caches, then
  `begin_trace_capture` / `end_trace_capture` — and replays it each step with
  `copy_host_to_device_tensor` + `execute_trace`. The conditioning
  (`s_inputs`, `s_trunk`, `q`, `c`, atom/token biases, `keys_indexing`, masks)
  is loop-invariant and baked into the capture; only `r` + `times` vary.
- `get_device` honors a dev-only `TT_BIO_TRACE_REGION_SIZE` escape hatch (used by
  the parity harness below) so a single-BH open can reserve a trace region
  without the CLI flag.

Default off; no cost to runs that don't opt in.

## Correctness — provably lossless

Trace replay reuses the exact captured device program with new input buffer
contents, so the per-step output is bit-identical to the untraced `forward`.
Verified directly at the device level on a real design
(`perf/boltzgen_trace_step_parity/sitecustomize.py`): on the first per-step
score-model call, run BOTH `forward(trace=False)` and `forward(trace=True)` on
the identical `(r_noisy, times, conditioning)` — same weights, same resident
cache — and compare `r_update`:

    [TRACE_PARITY] per-step r_update maxdiff=0.0 exact=True shape=(1, 2656, 3)

`r_update` is bit-for-bit identical (not just close). This is the right bar for
"trace replay is lossless": the per-step device call is unchanged.

**End-to-end designs are NOT bit-deterministic on ttnn** — two untraced runs
(same global seed, same length) drift by ~10 A over 500 steps (reduction order
in the shared Pairformer/DiT is not bit-stable run-to-run), so an end-to-end
trace-on vs trace-off coord comparison cannot isolate the trace path. The
per-step parity above is the rigorous proof; the end-to-end drift is pre-existing
device nondeterminism, not the trace.

Designability gate (`scripts/boltzgen_designability.py --num_designs 8
--diffusion_trace`, single card) — no regression:

    n=8  scRMSD  min 0.49 / median 0.79 / max 1.83 A
    designable  <=2A: 100.0%   <=4A: 100.0%

8/8 strict pass — cleaner than the resident-trunk n=8 precedent (87.5%, median
0.84, max 5.90 A; `boltzgen-resident-trunk` memo), whose one failure was a
hard-target outlier. Trace adds no designability error.

## Wall-clock — honest negative: ~0% win, not the projected 13-19%

Measured with the scout's harness (`perf/boltzgen_fusion_scout/sitecustomize.py`,
warm designs 2&3, `--num_designs 3 --steps design`), trace OFF vs ON
(`--diffusion_trace`). Trace was confirmed active: `forward_traced=500,
forward(untraced)=0` per design (`perf/boltzgen_trace_count/sitecustomize.py`).

| component (warm, 2 designs) | trace OFF | trace ON (`--diffusion_trace`) | delta |
|---|---|---|---|
| diffusion sample (500 steps) | 11.015 s | 10.972 s | -0.4% |
| total design forward | 16.751 s | 16.725 s | -0.15% |

No measurable win — within run-to-run noise (±0.5%). The scout's 13-19%
projection did not materialize. Two independent reasons:

1. **BoltzGen's diffusion DiT is compute-bound at design lengths (~80-120 res),
   not dispatch-bound.** The scout extrapolated "dispatch-bound" from the trunk
   (`boltzgen-resident-trunk` — the trunk's pairformer IS dispatch-bound at these
   lengths) to the diffusion, but the atom-attention DiT does real matmul work
   (32q x 128k windows over ~2600 atoms, 3 atom transformers + 1 token
   transformer per step). Collapsing host dispatch saves little when device
   compute dominates the step. Protenix's -22% at L=256 is a different
   op-mix/shape regime, not a length scaling.
2. **The trace is re-captured per design.** BoltzGen samples a random binder
   length per design (`sequence: 80..120` in `examples/binder.yaml`, drawn by
   `np.random.randint` in `data/parse/schema.py`), so `N_padded` changes each
   design and `forward_traced` re-captures (2 warmup DiT passes + capture, ~1-2 s)
   per design. For a short run this eats the small steady-state dispatch saving.
   Caching one trace per `N_padded` would avoid it but costs trace memory, and
   reason 1 leaves little steady-state saving to recover.

**Verdict:** the mechanism is correct and provably lossless, but the lever is
not worth turning on for BoltzGen design today — shipped opt-in (default off)
for transparency and future reuse (longer/different shapes, or a ttnn build
where dispatch is a larger fraction). Do NOT expect the projected 13-19% on the
binder.yaml design recipe.

## Artifacts

- `perf/boltzgen_trace_step_parity/sitecustomize.py` — per-step device parity
  (the bit-exactness proof; `forward(trace=False)` vs `forward(trace=True)` on
  identical inputs, asserts `maxdiff == 0`).
- `perf/boltzgen_trace_count/sitecustomize.py` — confirms `forward_traced` is
  exercised (500/500 steps) when `--diffusion_trace` is passed.
