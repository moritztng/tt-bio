# BoltzGen resident trunk: real but modest win (~5% e2e), not yet merge-ready

**Goal:** Boltz-2's device-resident trunk recycling loop (`TrunkRecycle`+`TrunkModule`
in `tenstorrent.py`, see [[boltz2-fast-perf-2026-06]]) removed per-recycling-iteration
host<->device round-trips and won ~16% e2e at L=512 (compute-bound regime, O(L³)
triangle-mult dominant). BoltzGen (`tt_bio/boltzgen/`) has its own trunk/pairformer
stack (`tt_bio/boltzgen/model/models/boltz.py:Boltz.forward`) that never adopted this —
this investigated whether an equivalent lever exists there.

## Profiling: yes, a real bottleneck, but a different regime than Boltz-2

`examples/binder.yaml`, `--steps design`, single card (warm, 2nd design of a
`--num_designs 2` run, ~100-120 residues, `recycling_steps=3` i.e. 4 iterations):

| stage | warm time | % of design forward |
|---|---|---|
| trunk (4 recycling iters) | ~6.5–7.0 s | ~29% |
| diffusion sample (500 steps) | ~15.8–17.3 s | ~71% |

Per-iteration breakdown (warm, `recycle_glue`/`token_distance`/`template`/`msa`/`pairformer`):
`0.006 / 0.21 / 0.05 / 0.37 / 0.99` s — sums to the ~1.6 s/iter measured directly.

**Key difference from Boltz-2:** at Boltz-2's L=512 benchmark the trunk is *compute-bound*
(`docs` / [[boltz2-fast-perf-2026-06]]: "trunk already heavily fused, compute-bound", recompute-hoist
≈ 0 win). At BoltzGen's design lengths (~80–120 residues), O(L³) triangle-mult compute is
~(100/512)³ ≈ 0.008× of the L=512 case — if the trunk were similarly compute-bound, one
iteration should cost tens of ms, not ~1.6 s. It doesn't scale down with L³, meaning this
regime is **dispatch-bound**: each of the 4 sub-modules (token-distance, template, MSA,
pairformer) is a separate `TorchWrapper` call that round-trips through
`from_torch`/`to_torch` at its boundary, and at this L that per-call host overhead
dominates over the (small) actual device compute. This is the same category of problem
Boltz-2's resident trunk solved, just for a different reason (dispatch floor, not
recycle-glue fp32 host drift) and in a regime Boltz-2's own benchmark didn't cover.

BoltzGen's trunk also does *more* per iteration than Boltz-2's: `token_distance_module`
(BoltzGen-only) and `template_module` are called **unconditionally** every iteration —
unlike Boltz2Model, which gates `template_module` behind `has_templates` (skipped when no
template is present). This is a pre-existing BoltzGen behavior (not introduced here) —
templates still cost ~0.05s/iter even for a template-less design.

## Prototype built (this branch, not yet merged)

- **`tenstorrent.TokenDistanceRecycle`** (new): mirrors `TemplateRecycle` — the
  `a_ij` distance/feature tensor (`TokenDistanceModule.forward`, from `center_coords`
  and `relative_position_encoding`) is loop-invariant, so it's computed once on host
  and only the z-dependent path (`z_proj`→pairformer→`v_norm`→`u_proj`) runs resident
  on device via the module's own inner ttnn Pairformer.
- **`tenstorrent.TrunkModule`** extended with an optional `token_distance_recycle` hook
  (wired before the template stage, matching host ordering) and an optional
  `template_module_torch` hook: BoltzGen's `TemplateModule` computes its template
  geometry (frame rotation/translation, visibility, CB/CA distances) inline rather
  than through a factored `template_features()` helper like Boltz2's `TemplateV2Module`
  — porting it to run resident was out of scope for this pass. It runs unchanged via
  **one** host round-trip per iteration instead (still collapsing what would otherwise
  be 4 separate module-boundary round-trips down to 2: down-then-up around the template
  call, with everything else chained on-device).
- `Boltz._tt_trunk_module()` in `boltz.py` builds this driver, mirroring
  `Boltz2Model._tt_trunk_module` — but **does not cache it across calls**. BoltzGen's
  design pipeline hot-swaps checkpoint weights mid-run (`load_checkpoint_weights`, the
  multi-ckpt schedule); Boltz-2 never reloads weights mid-run so its cached-`_tt_trunk`
  pattern is safe there but is not here — caching would silently reuse stale weights
  after a swap, and (worse) a plain `self._tt_trunk = trunk` assignment auto-registers
  `TrunkModule` as an `nn.Module` submodule, which the next `load_state_dict` call then
  tries to load into and crashes (`TrunkModule` has no state dict — it borrows already-
  loaded weights from its host modules). Found by testing multi-design runs, not by
  inspection — worth flagging since the same trap would bite any future resident-driver
  reuse across a hot-swapping pipeline.

## Measured result (controlled: same fixed design length=100, warm, single card 0)

| | trunk stage (warm) | Δ |
|---|---|---|
| host (main) | 6.99 s | — |
| resident (this branch) | 5.88 s | **-16%** |

Trunk is ~29% of design forward (§ above), so this is **~5% of design-forward wall-clock**
— real, but right at the task's stated merge threshold, and far short of the >50% one
might expect from "dispatch-bound" alone (the template stage's host round-trip and the
inherent per-call overhead of the resident driver itself eat into the theoretical
ceiling). Design-folding (refold) uses the same trunk loop, so the saving applies twice
per design in the full `design` → `design_folding` pipeline.

## Parity: no regression signal, but only n=2 (not a statistically robust check)

Trunk output tensors (`s`, `z`) differ resident-vs-host by ~6–13% relative mean-abs-diff
after 4 iterations — expected, not a bug: the resident path stays in bf16 on-device across
all 4 iterations with no intermediate fp32 host round-trip, the same category of drift
[[boltz2-fast-perf-2026-06]] measured and accepted for Boltz-2's resident trunk ("NOT
bit-identical... accuracy improves"). Per that memory and [[gen-multicard-already-exists]],
diffusion is seed-stochastic anyway, so the right bar is designability distribution, not
bit-exact diff.

`scripts/boltzgen_designability.py`, fixed length=100 (removes the length-randomization
confound), `--num_designs 2`, single card:

| | scRMSD median | ≤2Å pass | wall-clock (design+refold+confidence+analysis+filtering) |
|---|---|---|---|
| host (main) | 3.64 Å | 50% | 155 s |
| resident (this branch) | 0.89 Å | 100% | 152 s |

No regression — resident is comparable-or-better here, and e2e wall-clock (not just the
isolated trunk) also shows the modest expected improvement. But **n=2 per side is far
too small to be a statistically meaningful designability comparison** (one host design
scoring 6.38 Å is plausibly just sampling variance — `docs/boltzgen-designability.md`'s
own baseline table shows main-branch pass rates ranging 75–100% at n=2–4). This needs an
n≥8 run per side before it's a real parity gate, which didn't fit this pass's time
budget.

## Status: real modest win, recommend a larger validation pass before merge

Not a dead end (unlike [[boltzgen-batch-threshold-dead-end]] / [[protenix-accel-ceiling]])
— there is a genuine, reproducible ~5% design-forward win with no crash and no regression
signal so far. But given the modest margin (right at the merge threshold) and the small
parity sample, this should not be merged as-is. Code lives on
`wk/tt-bio-boltzgen-resident-trunk` (not on `main`): `tenstorrent.TokenDistanceRecycle` +
`TrunkModule` template-hybrid extension + `Boltz._tt_trunk_module`/resident forward gate.
Follow-up before merge: an n≥8 designability comparison at fixed length (harness and
fixed-length spec pattern already used above), and ideally profiling whether the
template host round-trip or the token-distance/msa/pairformer chaining is closer to the
per-iteration floor (further squeeze potential vs. diminishing returns).
