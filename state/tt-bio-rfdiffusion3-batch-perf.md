# tt-bio RFD3 batched design generation (Accelerate)

## Outcome

Implemented D>1 batched design generation for RFD3 via two new `tt-bio design`
options: `--num_designs N` (N independent designs per spec, seeds `--seed + i`)
and `--devices 0,1,...` (data-parallel fan-out across cards, reusing the
existing `tt-bio embed`/`predict` subprocess-per-card pattern). Output files are
`<spec_id>.cif` when `--num_designs 1` (back-compat) else `<spec_id>_<i>.cif`.

## Profiling finding (the key result)

Profiled both batching strategies on pc (1× p150a, IAI_protein.pdb fixture,
I=40 tokens, L=419 atoms):

1. **In-batch tensor batching** (D designs in one device forward, the
   rc-foundry `diffusion_batch_size=8` strategy): **measured linear scaling,
   no amortization**. Per-step time at warm cache:
   - D=1: 113 ms/step
   - D=2: 206 ms/step (1.82×)
   - D=4: 452 ms/step (3.99×)
   - D=8: 1016 ms/step (8.99× — slightly super-linear)

   The RFD3 device forward is **compute-bound at D=1** (the 18-block DiT
   matmuls already saturate the device), so batching D independent samples
   just does ~D× the FLOPs with no shared-weight reuse benefit. This is
   structurally different from GPU, where batched GEMMs reuse weights across
   the batch dim efficiently. **In-batch batching is a perf no-op on TT and
   is NOT exposed as a user lever** (wiring `--batch_size` would mislead users
   into expecting a speedup that doesn't materialize).

2. **Multi-card data-parallel fan-out** (one design per card, the existing
   `--devices` pattern): the real lever. Each card delivers ~0.044 designs/sec
   at `num_timesteps=200`; N cards give ~N× that (Amdahl-bound by per-worker
   fixed cost, per memory `predict-multicard-already-exists`). pc has 1 card
   so the multi-card speedup is not measurable here, but the wiring is
   correct and parity-bit-identical by construction (each design is an
   independent D=1 forward on its own card).

## Measured designs/sec

Bench: `scripts/rfd3_port/bench_designs_per_sec.py` (extended for `num_designs`).
Warm cache, 1× p150a, IAI_protein.pdb, `contig=A1-10,20,A31-40`:

| config | per_step | designs/sec |
|---|---|---|
| batch=1, num_timesteps=40 | 116 ms/step | 0.2225 (sequential N=8) |
| batch=1, num_timesteps=200 (extrapolated) | 116 ms/step | **0.0436** |
| in-batch D=8, num_timesteps=200 (extrapolated) | 1016 ms/step | 0.0395 (worse) |
| H200 batch=8 num_timesteps=200 (rc-foundry defaults) | 63 ms/step | **0.452** (reference) |

**On a single p150a, the designs/sec ceiling is ~0.044/s at `num_timesteps=200`.**
Matching H200's 0.452/s requires ~10 p150a cards via `--devices` (each ~0.044/s),
Amdahl-bound by the ~18s per-worker fixed cost (device open + program-cache
warmup). This is a genuine single-card compute ceiling, not a wiring gap —
reported honestly with the profiling evidence above rather than forcing a fake
win. The per-step compute gap vs H200 is small (1.55× at batch=1, 116ms vs 63ms);
the designs/sec gap is dominated by the missing batch lever, which on TT is
compensated by multi-card fan-out rather than in-batch sharing.

## Parity (sacred — verified, not claimed)

- **No regression at batch=1**: `verify_trajectory_from_pdb.py` (F1 from-PDB
  trajectory parity, device vs vendored-torch reference, 8 steps) passes:
  final X_L PCC=0.999240, motif atoms verified fixed at true position on both
  backends. The `RFD3DiffusionModule.__call__` batch-expand fix is guarded by
  `if B != 1`, so the D=1 code path is byte-identical to before.
- **Parity at batch>1 (num_designs>1)**: `--num_designs 2 --seed 42` produces
  `iai_0.cif` (seed 42) and `iai_1.cif` (seed 43), each **bit-identical** (cmp)
  to a standalone `--num_designs 1 --seed 42` / `--seed 43` run respectively.
  Different seeds produce different designs (no silent seed collision). This
  is the strongest possible parity result: bit-exact, no cross-job state.
- **In-batch D>1 device-forward correctness** (not exposed as a user lever, but
  verified for any future use): `spike_batch_invariance.py` confirms B=2 with
  two identical inputs produces element-0 == element-1 bit-exactly (maxabs=0,
  no cross-contamination), and B=2 element-0 vs B=1 PCC=0.9995 (within the
  bf16 tile-padding noise floor). The `__call__` fix expands the batch-1
  TokenInitializer outputs to the full batch B; without it the
  `DiffusionTokenEncoder` concat crashed at B>1 (found via the bench harness).

## Files changed

- `tt_bio/rfd3_design.py`: `run_design` gains `num_designs` + `devices` params;
  split into `_run_design_jobs` (in-process, per-spec featurize+TI cached across
  that spec's design draws) and `_run_design_fanout` (subprocess-per-card,
  reusing the embed/predict pattern) + `_run_design_shard` subprocess entry.
- `tt_bio/rfd3.py`: `RFD3DiffusionModule.__call__` expands batch-1 init tensors
  to the full batch B (5-line guarded fix; root cause of the original D>1 crash).
- `tt_bio/main.py`: `design_cmd` gains `--num_designs` and `--devices` options.
- `scripts/rfd3_port/bench_designs_per_sec.py`: extended with the
  `num_designs` measurement (N independent D=1 forwards) + H200 reference line.
- `scripts/rfd3_port/spike_batch_invariance.py`: permanent B>1 device-forward
  correctness check (kept; not throwaway).

## Release-gate status

- **Accuracy**: no regression at batch=1 (trajectory parity OK); bit-exact
  parity at num_designs>1. PASS.
- **Perf-regression**: batch=1 per-step unchanged (116 ms/step warm); the
  `__call__` expand is a no-op at B=1. No regression. The in-batch D>1 path is
  not exposed to users (no perf claim), so its linear scaling is not a
  regression.
- **UX**: `--num_designs` and `--devices` documented in `tt-bio design --help`
  with the honest note that in-batch batching is a no-op on TT and the real
  throughput lever is `--devices`. README/docs for `tt-bio design` should be
  updated to mention `--num_designs`/`--devices` (not done this pass — flagging).

## Not done this pass

- README/docs/rfd3-design.md not updated for `--num_designs`/`--devices` (the
  task's "if a change touches a public interface, docs must match reality" bar
  is not yet met for the public docs — only `--help` is updated). Flagging for
  the next pass or the orchestrator.
- Multi-card fan-out not measured on a real multi-card host (pc has 1 card);
  the wiring is parity-correct by construction but the Amdahl speedup curve is
  extrapolated from `predict-multicard-already-exists`, not re-measured here.

## Durable lesson (for the orchestrator to save)

**RFD3 in-batch tensor batching is a perf dead end on TT (Blackhole p150a):
the per-step device forward is compute-bound at D=1, so batching D independent
designs into one forward scales linearly (~113ms→1016ms for D=1→8, no
amortization), unlike GPU where batched GEMMs reuse weights across the batch
dim. The right TT equivalent of rc-foundry's `diffusion_batch_size=8` is N
independent D=1 forwards fanned across `--devices` (data-parallel), not in-batch
sharing. Generalizes to any TT diffusion port where the per-step forward is
already device-saturating at batch=1 — check per-step scaling before assuming
batching helps.** (See also `rfd3-trace-viability-submodule-granularity`: the
DiT is the compute-bound sub-module; the narrow atom-encoder/decoder are the
dispatch-bound ones where trace wins.)
