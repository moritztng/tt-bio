# BoltzGen fusion/dispatch-collapse scout — honest negative on fusion, one real dispatch-collapse candidate

**Goal (GOALS.md Workstream 3, Moritz 2026-07-11):** give BoltzGen the systematic
`finding-missed-fusions`-skill audit that Boltz-2 and Protenix-v2 got this tick
(both honest dead ends — `docs/boltz2-fusion-scout.md`, `docs/protenix-conditioning-scout.md`,
memories `atomattention-kernel-scout`, `protenix-accel-ceiling`, `dit-attention-kernel-scout`).
BoltzGen's own prior perf work (`boltzgen-resident-trunk`, merged `e33cf33c`) collapsed
host-dispatch via device residency — a different lever, not a component-level missed-fusion
audit. Single card, `examples/binder.yaml` (protein-anything protocol, ~80-120 res).

## Methodology — profiled a real device run, did not guess

`perf/boltzgen_fusion_scout/sitecustomize.py` patches
`tt_bio.boltzgen.adapter.load_boltz_checkpoint` at interpreter startup (so it
also applies inside the per-device subprocess `gen run` spawns) and wraps the
design-forward components with `ttnn.synchronize_device`-timed bounds:

- `Boltz.forward` — per-design total
- `Boltz._tt_trunk_module()` → resident `TrunkModule.__call__` — trunk
- `DiffusionConditioning.forward` — the design-network conditioning (once/design)
- `AtomDiffusion.sample` — the 500-step diffusion loop

Run: `TT_VISIBLE_DEVICES=1 TT_MESH_GRAPH_DESC_PATH=/tmp/single_bh_mesh_graph_descriptor.textproto
PYTHONPATH=$PWD/perf/boltzgen_fusion_scout:$PWD /home/ttuser/tt-bio-dev/env/bin/python3 -m
tt_bio.main gen run examples/binder.yaml --output <out> --num_designs 3 --devices 1 --budget 3
--steps design`. (Design step alone: `recycling_steps=3`, `sampling_steps=500`,
`diffusion_samples=1`; affinity/confidence are separate pipeline steps with their own
ckpts and are OFF here.) Warm = designs 2 and 3 (design 1 is cold compile).

### Measured (warm, single card, binder.yaml)

| component | design 2 | design 3 | % of design forward |
|---|---|---|---|
| diffusion sample (500 steps) | 11.02 s | 11.42 s | ~64-66% |
| trunk (4 recycling iters, resident) | ~5.60 s | ~5.58 s | ~33% |
| diffusion conditioning (once) | 0.32 s | 0.30 s | ~1.8% |
| **total forward** | **16.94 s** | **17.30 s** | 100% |

(trunk = total − diffusion − conditioning; cold-compile design 1 = 32.19 s total.)

This reproduces the resident-trunk doc's split (trunk ~29%, diffusion ~71%) on an
independent run and adds the conditioning row the prior measurement didn't break out.

## Fusion audit — clean negative: the hot path is shared primitives, all already closed

The design forward's two cost centers are NOT BoltzGen-specific:

1. **Diffusion (65%) = the shared `tt_bio.tenstorrent.DiffusionModule` /
   `DiffusionTransformer` / `AttentionPairBias` DiT.** BoltzGen's
   `TTScoreModelAdapter` (`tt_bio/boltzgen/adapter.py`) subclasses the SAME
   `TTDiffusionModule` that `tt_bio/boltz2.py` imports and uses. This is the exact
   graph `docs/boltz2-dit-attention-kernel-scout.md` audited and closed: QKV is
   already one packed linear, the pair-bias is precomputed once and replayed as an
   SDPA mask, and the only residual pack (gate → packed QKV) **regresses 0.97x
   e2e and breaks parity (5.2 A coord RMSD @200 steps)**. No BoltzGen-specific
   attention/transition variant exists in the per-step path — BoltzGen feeds
   precomputed conditioning biases into the shared DiT and adds nothing per-step.

2. **Trunk (33%) = the shared `PairformerModule` (matmul-bound, per
   `docs/msa-pwa-scout.md`) + BoltzGen's `token_distance_module`/`template_module`,
   already collapsed to device-resident recycling by the merged
   `boltzgen-resident-trunk` change (`e33cf33c`, ~31% e2e win).** No further
   fusion headroom — the pairformer is matmul-bound, not dispatch-bound, at this
   component.

3. **Diffusion conditioning (1.8%) — the one genuinely BoltzGen-specific component
   in the design forward — is a non-candidate on two independent grounds:**
   - It is **host-side PyTorch**, not ttnn. `DiffusionConditioning`
     (`tt_bio/boltzgen/model/modules/diffusion_conditioning.py`) composes stock
     `nn.LayerNorm`/`nn.Linear` `nn.Sequential`s (24× for the token-transformer
     AdaLN biases, 3× each for atom-enc/atom-dec) + `PairwiseConditioning` (host
     `Transition`s) + `AtomEncoder`. No `ttnn.*` op is launched, so there is no
     device op sequence to fuse.
   - It runs **once per design** (precomputed, then replayed as static biases
     across all 500 diffusion steps). At 0.30 s vs 11 s diffusion it is ~1/37 of
     the hot path — below the ~5% threshold for any lever, fusion or otherwise.
   Its device-side piece (the `AtomEncoder`/`AtomDecoder` windowed SDPA) is the
   shared atom-attention primitive `docs/atomattention-kernel-scout.md` already
   closed (dispatch-bound, gate-pack regresses + breaks parity) — and it runs in
   this once-per-design precompute, not per step.

4. **Affinity / confidence heads — not in the design forward.** They are separate
   pipeline steps (`affinity` uses `boltz2_aff.ckpt`, `design_folding`/
   `confidence` uses `boltz2_conf_final.ckpt`), built on the shared
   `TTPairformerModule` (matmul-bound) plus host glue (cdist, distogram bins,
   PAE/PDE/pLDDT linear heads). Not a BoltzGen-specific device fusion target.

**Verdict (fusion): no missed fusion in BoltzGen's design forward.** Every
device-op sequence in the hot path is a shared primitive already audited and
closed for the sibling models; the one BoltzGen-specific component is host-side
PyTorch running once per design at 1.8%. Silence is the success signal per the
`finding-missed-fusions` skill — no `fusion_todo.yml` produced.

## One real dispatch-collapse candidate (NOT fusion): ttnn trace replay, not shipped for BoltzGen

The diffusion loop's only device op per step is the single
`TTDiffusionModule.forward` DiT call; all schedule math (`step_scale`/`noise_scale`
phases), random augmentation, and `weighted_rigid_align` run host-side on CPU
(`AtomDiffusion.device == cpu`). The per-step device graph is therefore
**shape-stable across all 500 steps** (same atom/token count per design; only the
scalar `sigma`/`t_hat` and the `r_noisy` tensor change) — exactly the structure
that makes ttnn trace replay applicable.

**Protenix already ships this lever** (`tt_bio/protenix.py:_capture_trace`/
`_execute_trace`, `fold(trace=True)`, `get_device(trace_region_size=1<<30)`):
measured **-22% warm diffusion @L256** (`protenix-accel-ceiling` memory), lossless
by construction (replays the exact captured device program with new input tensors).
**Boltz-2 and BoltzGen do NOT ship it** — `capture_trace`/`execute_trace` appear
only in `protenix.py`. BoltzGen's diffusion is 65% of the design forward and is
**dispatch-bound at design lengths** (~80-120 res, `boltzgen-resident-trunk` doc),
so the dispatch fraction is higher than at Protenix's L=256 — the win should be at
least the ~22% Protenix sees. Conservative projection: **~20-30% of the 11 s
diffusion = ~2-3 s/design ≈ 13-19% of the 17 s design forward**, above the 5%
threshold.

Parity risk is LOW (unlike the proven-lossy distinct-structure batching,
`boltzgen-batch-threshold-dead-end`): trace replay reuses the identical device
tiling every step, so it is bit-identical to the untraced per-step call. The
schedule phase changes do not touch the device graph (host-side scalars).

### Next-turn implementation plan (deferred — exceeds one bounded turn)

1. Open the BoltzGen device with a trace region: thread `trace_region_size=1<<30`
   through `get_device()` for the design path (mirror Protenix's `fold(trace=True)`
   knob; default off, opt-in).
2. Port Protenix's `_capture_trace`/`_execute_trace` to `TTScoreModelAdapter`
   (`tt_bio/boltzgen/adapter.py`): capture the `TTDiffusionModule.forward` DiT
   graph once per design (after the first warm step, when shapes are fixed), then
   for steps 2..500 copy the new `r_noisy` + scalar `sigma`/`t_hat` into the
   captured input tensors and `ttnn.execute_trace`. The conditioning biases,
   `s_inputs`, `s_trunk`, `keys_indexing`, and masks are loop-invariant → baked
   into the capture.
3. Verify accuracy unchanged with `scripts/boltzgen_designability.py --num_designs 8`
   (same n=8 bar as the merged resident-trunk change; bit-identical coords expected,
   so scRMSD distribution must be unchanged).
4. Measure real wall-clock delta with the harness in
   `perf/boltzgen_fusion_scout/sitecustomize.py` (before/after diffusion_sample
   row, warm designs 2&3, same binder.yaml run).

## Env note — qb2 single-card open requires the mesh-graph descriptor (ttnn 0.68)

`ttnn.open_device(device_id=0)` on qb2 (ttnn 0.68.0, 4x Blackhole P150 meshed box)
fatals with `Custom fabric mesh graph descriptor path must be specified for CUSTOM
cluster type` unless `TT_MESH_GRAPH_DESC_PATH=/tmp/single_bh_mesh_graph_descriptor.textproto`
is set. The `gen run` per-device subprocess inherits `os.environ`, so setting it on
the parent `gen run` invocation propagates. (The cli already does this for p300
devices via `_find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")`
in `cli/boltzgen.py:762`, but only for detected p300 cards — a plain single-BH open
on qb2 still needs it.) Worth a small `get_device()` fallback in a follow-up;
out of scope for this scout and not merged here.

## Artifacts

- `perf/boltzgen_fusion_scout/sitecustomize.py` — the spawn-safe profiling harness
  (reusable for any future BoltzGen per-component timing).
- `perf/boltzgen_fusion_scout/prof_*.json` — raw per-design timings (this run).
