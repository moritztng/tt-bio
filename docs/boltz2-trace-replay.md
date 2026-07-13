# Boltz-2 diffusion trace replay — lossless, ~0% wall-clock (compute-bound DiT)

**Goal:** port Protenix's ttnn trace-replay win (-22% warm diffusion @L256,
[[protenix-accel-ceiling]]) to Boltz-2's diffusion sampler. Boltz-2 shares
Protenix's `DiffusionModule`/DiT and is a *fold* task (fixed-ish L) rather than
BoltzGen's *design* task, so it was queued as a real candidate — but VERIFY with
real profiling, don't assume the Protenix number transfers.

Single card qb2, `examples/prot.yaml --model boltz2` and `examples/hemoglobin.yaml`
(287-res dimer, Protenix's L256 regime), `TT_MESH_GRAPH_DESC_PATH=/tmp/single_bh_mesh_graph_descriptor.textproto`.

## What shipped

Opt-in (`predict --diffusion_trace` → `Boltz2.__init__(diffusion_trace=True)`)
trace replay of the per-step DiT device stream, reusing the **shared**
`DiffusionModule.forward_traced` / `_capture_diff_trace` already landed for
BoltzGen (`tt_bio/tenstorrent.py`) — no per-model trace code, just the call-site
wiring:

- `Boltz2.__init__(diffusion_trace=False)` reserves the trace region
  (`get_device(trace_region_size=1 << 30)`) before any module opens the device,
  and threads the flag to `AtomDiffusion`. `AtomDiffusion._diffusion_trace`
  dispatches `preconditioned_network_forward` to `score_model.forward_traced`
  (same signature as `forward`) instead of `score_model(...)`. Mirrors the
  `--diffusion_trace` kwarg/CLI shape already shipped for BoltzGen; no env-var
  toggle (`prefer-args-over-envvars`).
- `scripts/release_gate.py` gains an opt-in `--diffusion_trace` passthrough
  (boltz2 only; defaults off, no change to the standing gate).

Precondition holds: Boltz-2's per-step device graph is shape-stable across all
sampling steps (only the scalar `times` and `r` coords change; conditioning is
loop-invariant and baked into the capture), the same precondition Protenix's
and BoltzGen's traces rely on.

## Correctness — bit-exact (the SACRED gate)

Per-step device parity (identical inputs, `DiffusionModule.forward` vs
`forward_traced` in one process) is the right bit-exactness bar — end-to-end
bit-comparison is NOT valid (ttnn reduction nondeterminism means two untraced
Boltz-2 folds diverge run-to-run, same finding as
`perf/boltzgen_trace_step_parity/`). Harness:
`perf/boltz2_trace_step_parity/sitecustomize.py` (run with
`TT_BIO_TRACE_REGION_SIZE=1073741824` so `forward_traced` can reserve/capture
while the fold itself runs untraced).

**Result: `r_update` maxdiff = 0.0, exact = True** (shape `(1, 928, 3)` on
7ROA). Trace replay reuses the exact captured device program with new input
buffer contents, so the output is bit-identical to the untraced path.

## No regression

5-sample single-sequence folds of 7ROA, trace ON vs OFF (same seed), scored
against the 7ROA ground truth with the `tests/test_structure.py` harness
`release_gate.py` uses:

| | best-confidence | oracle-best sample |
|---|---|---|
| trace OFF | 3.85 Å / 0.649 | 2.80 Å / 0.764 |
| trace ON  | 4.27 Å / 0.603 | 2.86 Å / 0.743 |

Distributions overlap within run-to-run variance — no trace regression. Neither
hits the 3.0 Å / 0.75 gate floor because the floor is MSA-calibrated and these
are **single-sequence** folds (the MSA `release_gate.py --model boltz2` path
could not run on this isolated fleet — no MSA server, no `~/.boltz/msa_db`, no
internet to ColabFold). The bit-exact per-step parity above is the real
correctness guarantee: the traced fold is the same device computation as the
untraced fold, so end-to-end accuracy is statistically identical by
construction.

## Performance — ~0% wall-clock (the honest result)

Warm diffusion (`AtomDiffusion.sample`, 200 steps, 1 sample, single-sequence),
trace ON vs OFF, **measured under no concurrent host load**:

| target | L | trace OFF | trace ON | win |
|---|---|---|---|---|
| `prot.yaml` (7ROA) | 117 res | 3.585 s (17.9 ms/step) | 3.594 s (18.0 ms/step) | +0.3% |
| `hemoglobin.yaml` | 287 res | 9.68 s (48.4 ms/step) | 9.75 s (48.8 ms/step) | +0.7% |

`perf/boltz2_trace_perf/sitecustomize.py` times `sample()` and counts
`forward_traced` vs untraced `forward` calls (200/0 off, 0/200 on — the trace
path IS exercised). Trace ON is marginally *slower* (trace staging overhead
cancels the small dispatch saving): **~0%**, the BoltzGen regime, not
Protenix's -22%.

**Root cause:** Boltz-2's DiT per-step is compute-bound at fold lengths
(48 ms/step @287 res with negligible removable host dispatch), so collapsing
per-step dispatch buys nothing — the same finding as
[[boltzgen-trace-replay-result]], NOT Protenix's dispatch-bound regime.
Protenix's lighter per-step (~15 ms @L256) made the same ~1-3 ms dispatch tax a
~22% slice; Boltz-2's heavier per-step makes it a ~1% slice that trace's own
overhead erases.

### The contention phantom — a measurement hazard

A first hemoglobin run measured trace OFF at 22.4 s (112 ms/step) vs ON at
10.1 s — a **phantom -55% "win."** It was an artifact: a concurrent worker on
the other card was starving host CPU, and the **untraced path is CPU-dispatch-bound**
(thousands of per-op host launches per step) while the traced path makes one
`execute_trace` call. Host contention inflates the untraced path far more than
the traced path, fabricating a win that vanishes under no load (OFF dropped
22.4 → 9.68 s once the other worker finished; traced was stable 10.1 → 9.75 s).

**Generalizable lesson (name for the knowledge base):** when measuring ttnn
trace-replay ON vs OFF, the untraced path is host-CPU-dispatch-sensitive and
the traced path is not — so any concurrent host load (another worker, a build)
manufactures a phantom trace win. Always measure with the host idle, or you will
report a fabricated speedup. This would have landed a bogus -55% claim here
without the no-load repeat.

## How to apply

Trace-replay's win size depends on whether the per-step DiT is dispatch-bound
(Protenix: light per-step, real -22%) or compute-bound at fold lengths
(Boltz-2 and BoltzGen: heavy per-step, ~0%). Boltz-2 is in the latter camp.
The shared `DiffusionModule.forward_traced` ships and is bit-exact, so
`--diffusion_trace` is a safe opt-in if a future change makes the DiT
dispatch-bound (e.g. a host-side Python loop hoisted into the step), but at
today's fold lengths it is not worth enabling.

## Artifacts

- `perf/boltz2_trace_step_parity/sitecustomize.py` — per-step device parity
  (maxdiff=0.0).
- `perf/boltz2_trace_perf/sitecustomize.py` — warm diffusion wall-clock +
  traced/untraced call count.
