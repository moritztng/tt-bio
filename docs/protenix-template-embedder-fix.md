# Protenix-v2 template-embedder fix — implemented, measured

Follow-up to `docs/protenix-hemoglobin-trunk-drift-investigation.md`, which root-caused but
deliberately did not fix: the on-device Trunk never ran the template-embedder pass in any real
predict call, because `tt_bio/protenix_data.py`'s featurizer never emitted `template_aatype`
(so `Trunk.__call__`'s `nt = feat["template_aatype"].shape[0] if ... else 0` always took
`nt=0`) — independent of the CLI's `--use_template` flag, since upstream's own model config
(`template_embedder.n_blocks=2`) runs the template embedder unconditionally.

## Fix

`tt_bio/protenix_data.py`: new `dummy_template_features(n_token, max_templates=4)`, wired into
`build_complex_features`'s returned feature dict (the single call site feeding
`tt_bio/worker.py`'s protenix predict path). Reproduces upstream's
`TemplateFeatureAssemblyLine(max_templates=4)` output for the `use_template=False` path (traced
in `/tmp/protenix-src/protenix/data/template/{template_featurizer,template_utils}.py`):

- `template_aatype`: slot 0 = gap token (`STD_RESIDUES_WITH_GAP["-"] == 31`) over every
  position (`TemplateFeatures.empty_template_features`); slots 1-3 = `0` (ALA) — `np.pad`'s
  default `constant_values=0` zero-fill of the assembly line's per-chain padding to
  `max_templates`.
- `template_distogram` / `template_pseudo_beta_mask` / `template_unit_vector` /
  `template_backbone_frame_mask`: exactly zero for every slot. Confirmed by tracing
  `TemplateFeatures.dgram_from_positions` / `compute_template_unit_vector`: with all-zero
  atom positions and an all-zero atom mask, every position-derived quantity reduces to a clean
  `0` (the unit-vector code's `norm + epsilon` denominators avoid a `0/0` NaN). So
  `template_aatype`'s one-hot (gap vs ALA) is the *entire* template-embedder signal under
  `use_template=False` — matches the investigation doc's finding exactly.

This is the default behavior now (no new CLI flag), matching the MSA-on-by-default precedent:
a correctness fix, not an opt-in.

`tt_bio/protenix.py`'s `Trunk` already implemented the matching consumption side correctly
(`_template`, and the per-cycle `te_at` concat in `__call__`) — it just never received `nt>0`
input from a real predict call. No model-side changes were needed.

## Trunk z-PCC recovery (`scripts/protenix_hemo_device_trunk_pcc.py`)

Same reference capture (hemoglobin, 574 tok, `N_cycle=3`) as the investigation doc, now with
`dummy_template_features` added to the fed `feat` (that harness's `feat_small` deliberately
omitted `template_*` to reproduce the pre-fix `nt=0` bug — the CPU reference that produced the
`cycles` capture ran upstream's real, always-on template embedder, so this is the correct input
to compare against it):

| cycle | z_PCC before | z_PCC after | s_PCC before | s_PCC after |
|------:|-------------:|------------:|-------------:|------------:|
| 1 | 0.768 | **0.992** | 0.997 | 0.9999 |
| 2 | 0.765 | **0.967** | 0.998 | 0.9999 |
| 3 | 0.677 | **0.952** | 0.998 | 0.9998 |

z-PCC recovers dramatically toward the 38-token validation's 0.98967 (@10 cycles) benchmark —
cycle 1 already exceeds it. The prior monotonic *decline* with more cycles (0.768→0.765→0.677)
is gone; z-PCC now declines only slowly (0.992→0.967→0.952), consistent with ordinary
compounding of small per-cycle bf16/tile differences rather than a missing term.

## Delivered ground-truth RMSD (production settings, same targets as the investigation thread)

`--sampling_steps 200 --diffusion_samples 5 --seed 0 --recycling_steps 3` (current shipped
default — this fix alone, no recycling_steps change), `tests/test_structure.evaluate`:

| target | delivered before (baseline) | delivered after (this fix) | oracle before | oracle after |
|--------|---:|---:|---:|---:|
| 7ROA (`examples/prot.yaml`, monomer, shallow MSA) | 3.472 Å | **1.842 Å** | 2.391 Å | 1.648 Å |
| hemoglobin (`examples/hemoglobin.yaml`, α₂β₂ tetramer) | 1.349 Å | **0.595 Å** | 0.882 Å | 0.549 Å |

(Before-numbers from `wk/tt-bio-protenix-confidence-fix`'s `docs/protenix-confidence-head-rootcause.md`,
same settings/seed/harness.) Both targets improve substantially and neither regresses — 7ROA by
1.63 Å, hemoglobin by 0.75 Å.

## Relevance to `wk/tt-bio-protenix-confidence-fix` (open, not merged)

That branch's `--recycling_steps` default bump (3→10) closed 7ROA's gap (3.47→2.35 Å) but
**regressed** hemoglobin (1.35→1.96 Å, +0.61 Å, mixed sign, release-gated pending this decision).
This fix closes 7ROA even further (→1.84 Å) at the *unchanged* default of 3 cycles, and
*improves* hemoglobin rather than regressing it (→0.60 Å). That strongly suggests the missing
template embedder — not under-recycling — was the dominant cause of both targets' inaccuracy,
and that the confidence-fix branch's hemoglobin regression was measured against a trunk that
was itself missing this term. Recommend the orchestrator re-evaluate the recycling_steps 3→10
change once this fix is in: it may now be unnecessary, or its hemoglobin tradeoff may look
different when re-measured with the template fix present (not re-measured here — out of scope
for this task, which only re-measures at the unchanged default of 3).

## Release gate (no cross-model regression)

`scripts/release_gate.py` (200 steps / 5 samples / seed 0, `examples/prot.yaml`):

| model | RMSD (Å) | TM | floor | result |
|---|---:|---:|---|---|
| protenix-v2 | 1.842 | 0.883 | ≤6.0 / ≥0.5 | **PASS** |
| boltz2 | 1.905 | 0.903 | ≤3.0 / ≥0.75 | PASS (unaffected) |
| esmfold2 | 2.730 | 0.797 | ≤4.0 / ≥0.65 | PASS (unaffected) |

`protenix_data.py` is only imported by protenix-specific code (`tt_bio/main.py`,
`tt_bio/worker.py`'s protenix predict path) — boltz2/esmfold2 don't share this featurizer code
path, so the clean boltz2/esmfold2 passes are confirmation, not surprise.

## Not done here (deliberately, out of scope)

Re-measuring hemoglobin/7ROA at `--recycling_steps 10` with this fix present (to update the
confidence-fix branch's mixed-sign verdict) — that's the confidence-fix branch's decision to
revisit, not this one's.
