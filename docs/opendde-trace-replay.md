# OpenDDE diffusion trace replay — lossless, ~1% wall-clock win

**Goal:** port Protenix's ttnn trace-replay machinery
(`tt_bio/protenix.py:_capture_trace` / `denoise_traced`, `fold(trace=True)`,
measured -22% warm diffusion @L256) to OpenDDE's diffusion sampler. OpenDDE
reuses the Protenix-v2 diffusion stack verbatim (`from .protenix import
_KeyedWeights`; `OpenDDE.fold` calls the shared `edm_sample`), so the per-step
device graph is already shape-stable across all 200 sampling steps (only the
scalar `t_hat` and the scaled coords change) — the same precondition Protenix's
trace relies on. The win size was the open question: Protenix is a fold task
(dispatch-bound, -22%); BoltzGen is a design task (compute-bound at binder
lengths, ~0%). OpenDDE is a fold task like Protenix, so it was the next real
candidate.

Single card qb2 (Blackhole p300c), 7ROA (117 res, Ns=229 structural tokens,
900 atoms), `examples/prot.yaml --model opendde` recipe, 10 recycling / 200
diffusion steps, `TT_MESH_GRAPH_DESC_PATH=/tmp/single_bh_mesh_graph_descriptor.textproto`.

## What shipped

Opt-in (`tt-bio predict --model opendde --trace`, threaded `cfg["trace"]` ->
`OpenDDE.fold(trace=...)` -> the shared `edm_sample(trace=...)` ->
`DiffusionModule.denoise_traced`) trace replay of the per-step denoise device
stream. The trace machinery itself (`_capture_trace` / `denoise_traced` /
`_denoise_device`) is the existing shared code in `tt_bio/protenix.py` — no new
device code, just threading the flag through OpenDDE's call site:

- `OpenDDE.fold(trace=False)` adds the same precondition check as
  `Protenix.fold(trace=True)`: the device must have been opened with a trace
  region (`get_device(trace_region_size=1 << 30)`), else a clear `ValueError`.
- `main.py` reserves the trace region when `--trace` is set by setting
  `TT_BIO_TRACE_REGION_SIZE=1<<30` in the parent env before workers spawn (each
  worker's first `get_device()` then reserves the region up front; a later
  reopen is unstable on TT). `worker._predict_opendde_one` forwards
  `cfg.get("trace")` into `model.fold(...)`.
- `get_device` already honors `TT_BIO_TRACE_REGION_SIZE` as a dev-only escape
  hatch (used by the parity harness below), so a single-BH open can reserve a
  trace region without the CLI flag.

Default off; no cost to runs that don't opt in. Protenix-v2's own
`_predict_protenix_one` does not yet read `cfg["trace"]` (out of scope for this
stream); the `--trace` flag now reaches OpenDDE end-to-end.

## Correctness — provably lossless

Trace replay reuses the exact captured device program with new input buffer
contents, so the per-step output is bit-identical to the untraced `denoise`.
Verified directly at the device level on a real OpenDDE fold
(`perf/opendde_trace_step_parity/sitecustomize.py`): on the first per-step
denoise of a traced fold, run BOTH `denoise` (untraced) and `denoise_traced` on
the identical `(x_noisy, t_hat, cond)` — same weights, same resident cache —
and compare the returned coords:

    [TRACE_PARITY] per-step denoise maxdiff=0.0 exact=True shape=(1, 900, 3)

Bit-for-bit identical (not just close). This is the right bar for "trace replay
is lossless": the per-step device call is unchanged.

**End-to-end coords are also bit-identical here** (stronger than the per-step
gate, and unlike BoltzGen): running `scripts/opendde_e2e_smoke.py` at production
settings (10 cycles / 200 steps, seed 0) trace OFF, trace ON, and trace OFF
again, the three coord tensors are pairwise `torch.equal` (maxdiff 0.0,
including OFF-vs-OFF — the OpenDDE fold is bit-deterministic run-to-run on this
card, so the end-to-end comparison cleanly isolates the trace path). The
accuracy gate (`scripts/opendde_e2e_rmsd.py`, Ca-RMSD vs the 7ROA ground truth)
is unchanged: 3.096 A / TM 0.720 either way. No regression.

## Wall-clock — honest: ~1% win, not Protenix's -22%

Measured with `perf/opendde_trace_step_parity/bench.py`: same process, same
seed, same resident cache, warm (both paths warmed before measurement), 5
alternating OFF/ON trials at production settings (10 cycles / 200 steps).
Diffusion-only time isolates the dispatch collapse; total is the user-facing
number.

| component (warm, median of 5) | trace OFF | trace ON | delta |
|---|---|---|---|
| diffusion sampler (200 steps) | 3.440 s | 3.375 s | -1.9% |
| total fold | 11.754 s | 11.652 s | -0.9% |

A real but small win, and noisy at this scale (one of the five ON trials ran
slower than its OFF pair; the per-step diffusion saving is ~0.3 ms / step,
inside the run-to-run jitter envelope). This is BoltzGen's ~0% regime, not
Protenix's -22% regime. The reason: on Blackhole with ETH dispatch the per-step
host dispatch is already a small fraction of the DiT compute at this size
(~17 ms / step, of which trace collapses only ~0.3 ms), so the diffusion DiT is
compute-bound here, not dispatch-bound. Protenix's -22% was a different
dispatch / op-mix regime (and likely a slower-dispatch device config); it does
not transfer to OpenDDE on this card.

**Verdict:** the mechanism is correct and provably lossless, the accuracy gate
is unchanged, and the win is real but small (~1% of total, ~2% of diffusion).
Shipped opt-in (default off) — turn it on with `--trace` when every percent
matters and the 1 GiB trace region is spare; do not expect a Protenix-scale
speedup.

## Artifacts

- `perf/opendde_trace_step_parity/sitecustomize.py` — per-step device parity
  (the bit-exactness proof; `denoise` vs `denoise_traced` on identical inputs,
  asserts `maxdiff == 0`).
- `perf/opendde_trace_step_parity/run.py` — runs a real traced OpenDDE fold so
  the parity hook fires.
- `perf/opendde_trace_step_parity/bench.py` — the warm OFF-vs-ON wall-clock
  benchmark (median-of-K alternating trials).
