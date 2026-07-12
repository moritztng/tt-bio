# OpenFold3 → tt-bio port

Resume anchor for the OpenFold3 (OF3) port. Branch `wk/tt-bio-openfold3-port`.
Companion artifact: `docs/openfold3-weight-keys.txt` (block-collapsed manifest of the
real checkpoint, 4935 params → 559 representative keys+shapes).

## 1. Maturity verdict (re-verified 2026-07-12, fresh)

OF3 has matured well past the Oct-2025 "research preview / not AF3-parity" read that
sent the earlier task to classic OpenFold. Current state:

- **Consortium major release 2026-03-13**: end-to-end open cofolding stack — training
  datasets (AWS Registry of Open Data), model weights, training + inference code, eval
  scripts, all under permissive licenses. White paper reports performance competitive
  with AlphaFold3 across most evaluated modalities.
- **Current code state (as of 2026-07): preview2, package v0.4.3**, Apache-2.0
  (AlQuraishi Lab + AMD). Default checkpoint `openfold3-p2-155k` (`of3-p2-155k.pt`,
  155k steps). CLI `run_openfold predict --query_json=...`, JSON input.
- **Accuracy** (OF3 white paper / independent write-ups): matches AF3 on monomeric RNA
  (the only open model to do so); competitive on FoldBench, Runs-and-Poses, AF3 Ab-Ag.
  Still below AF3 on CASP16 monomers and protein–protein complexes (largest gap:
  antibody–antigen). Full-PDB retrain for complete AF3 parity is still in progress; the
  released preview2 is a genuinely usable, competitive AF3-class model, not a toy.

Verdict: **mature enough to port.** Real weights, real accuracy, permissive license.

## 2. Redundancy verdict vs boltz2 / protenix-v2

tt-bio already ships two AF3-family models. OF3 is the **same architecture family**
(Pairformer trunk + EDM diffusion + confidence heads), so in raw compute it is largely
redundant — which is exactly what makes the port cheap (near-total primitive reuse).

Non-redundant value it does add:
- **The standard-bearer open AF3 reproduction** from the OpenFold Consortium (OMSF), the
  most-adopted / most community-trusted open-source folding lineage.
- **Fully public training data** (AWS Open Data) — reproducibility no other AF3 clone in
  tt-bio offers.
- **Distinct accuracy point**: RNA parity with AF3, which is a real differentiator vs
  boltz2/protenix.
- Apache-2.0, clean commercial terms.

Honest call: worth having as a third AF3 option for the RNA/reproducibility/adoption
reasons, not because it fills a compute gap. The port is justified by low marginal cost
(reuse) × real user-facing differentiation, not by architectural novelty.

## 3. Architecture — dims identical to what tt-bio already ports

From `openfold3/projects/of3_all_atom/config/model_config.py`:

| | value |
|---|---|
| c_s / c_z / c_m | 384 / 128 / 64 |
| c_atom / c_atom_pair | 128 / 16 |
| c_s_input | 449 (384 + 65) |
| Pairformer blocks | 48 (heads_pair 4, heads_pair_bias 16, c_hidden_mul 128) |
| MSA module blocks | 4 (OuterProductMean + PairWeightedAveraging) |
| Template pair stack | 2 blocks |
| Diffusion transformer | 24 blocks, 16 heads, c_token 768 |
| Diffusion atom enc/dec | 3 blocks, 4 heads, c_hidden 32 |
| Confidence pairformer | `aux_heads.pairformer_embedding` (its own small Pairformer) |

These are **the AF3 dims** — bit-for-bit what boltz2.py and protenix.py already run
(c_s=384, c_z=128, 48-block Pairformer, 24-block DiT). No new tensor-shape regime.

Triangle multiplicative update is **AF3-style gated** (`linear_g` sigmoid gate +
`layer_norm_in/out` + biased `linear_a_p/a_g/b_p/b_g`). => The abandoned classic-OpenFold
branch's AF2 "bias-free vs biased triangle linear" gating fix **does NOT transfer**;
OF3 matches the boltz2/protenix gating exactly, as hypothesized in the task brief.

## 4. Reuse map — OF3 weight tree → existing tt-bio primitives

Nearly 1:1 onto `tt_bio.tenstorrent` + `boltz2.py` + `protenix.py`. Port is a
**weight-remap + data-pipeline vendor**, not new compute.

| OF3 checkpoint prefix | tt-bio primitive |
|---|---|
| `pairformer_stack.blocks.*.pair_stack.tri_mul_{in,out}` | `tenstorrent.TriangleMultiplication` |
| `...pair_stack.tri_att_{start,end}` | `tenstorrent.TriangleAttention` |
| `...pair_stack.pair_transition` (SwiGLU) | `tenstorrent.Transition` |
| `...attn_pair_bias` | `tenstorrent.AttentionPairBias` |
| `...single_transition` | `tenstorrent.Transition` |
| `pairformer_stack` (48×) | `tenstorrent.Pairformer` |
| `msa_module.blocks` (4×) + OPM | `tenstorrent.MSA` / `MSALayer` / `OuterProductMean` / `PairWeightedAveraging` |
| `template_embedder.template_pair_stack` | `boltz2.TemplateModule` (or `tenstorrent` template recycle) |
| `input_embedder.atom_attn_enc` | `protenix.AtomAttentionEncoder` / `boltz2.AtomEncoder` |
| `input_embedder.linear_{s,z_i,z_j,relpos,token_bonds}` | `boltz2.InputEmbedder` glue |
| `diffusion_module.diffusion_conditioning` | `protenix`/`boltz2` `DiffusionConditioning` |
| `diffusion_module.atom_attn_{enc,dec}` | `protenix` atom enc/dec |
| `diffusion_module.diffusion_transformer` (24× AdaLN DiT) | `tenstorrent.DiffusionTransformer` + `AdaLN` + `ConditionedTransitionBlock` |
| `aux_heads.{pae,pde,plddt,distogram,experimentally_resolved}` | `protenix.ConfidenceHead` / `boltz2.ConfidenceHeads` |
| `aux_heads.pairformer_embedding` | reuse `Pairformer` (confidence trunk) |

Deltas to reconcile during remap:
1. **`sample_diffusion.diffusion_module.*` is bit-identical to `diffusion_module.*`**
   (verified `torch.equal`) — a tied duplicate. Load one, ignore the other.
2. `aux_heads.plddt.linear` is (1150, 384) = 50 bins × 23 atom groups — check vs
   protenix pLDDT head layout (protenix pLDDT/resolved were the weakest PCC there:
   0.93/0.77; expect the same head to be the parity-sensitive spot here).
3. OF3 `linear_z` bias-proj head counts: tri-att 4, attn_pair_bias 16 — confirm against
   tt-bio's `no_heads` config paths.
4. Naming is mechanical (`tri_mul_out.linear_a_p` etc.) — build a single remap table in
   an `openfold3_weights.py` module, mirroring `protenix_weights.py`.

## 5. Assets already obtained (persist outside worktree, survive restart)

- Real checkpoint: `~/of3-weights/of3-p2-155k.pt` (2.29 GB, public S3, **UNSIGNED / no
  gating**: `s3://openfold3-data/openfold3-parameters/of3-p2-155k.pt`). NOT HF-gated.
- Full key manifest: `~/of3-weights/keys_full.txt` (4935 lines).
- Reference clone: `/tmp/of3-ref` (shallow; re-clone `github.com/aqlaboratory/openfold-3`
  if wiped — scratch, not persistent).

## 6. Component-by-component plan (PCC-gate each vs real-weight reference before advancing)

Per the port-bio-model-to-tenstorrent playbook. Reference deps are pure-pip
(torch, numpy, scipy, biotite, rdkit<2026, pdbeccdutils, ml-collections, lightning);
cuequivariance/deepspeed are optional CUDA-only — skip for CPU golden generation.

- [ ] **P0 reference harness**: CPU venv, `pip install openfold3`, load `of3-p2-155k.pt`,
      run `query_ubiquitin.json` end-to-end on CPU, capture golden activations per module.
- [x] **P1 vendor** host-side data pipeline (JSON query → feats dict, CCD/ligand,
      relpos, token bonds) into `tt_bio/_vendor/openfold3/` — inference-only, strip
      training/lightning/losses/optimizers. No runtime git-clone, no sys.path shims.
      Done on `wk/tt-bio-openfold3-port-p1` — see "P1 status" below.
- [ ] **P2** `openfold3_weights.py` remap table (OF3 keys → shared-primitive keys).
- [ ] **P3 PCC gate, smallest first**: TriangleMultiplication → TriangleAttention →
      AttentionPairBias → one Pairformer block → 48-block trunk → MSA block → template →
      atom encoder → InputEmbedder → DiffusionConditioning → DiT block → atom decoder →
      confidence heads. Threshold PCC > 0.98 per module vs real-weight golden.
- [ ] **P4 assemble** `OpenFold3` class (`load_from_checkpoint` + `fold`), EDM sampler.
- [ ] **P5 integrate**: `--model openfold3` in CLI/worker/scheduler, `--fast` block-fp8,
      `--devices` fanout — consistent with predict precedent. ONE unified README --model
      table row (no parallel prose block; bio audience, no ttnn/driver detail).
- [ ] **P6 HARD GATE**: `examples/prot.yaml` end-to-end → parsed output → vs-ground-truth
      Kabsch RMSD via `scripts/release_gate.py`'s method. No fabricated numbers.

Closest existing model to diff against at every step: **protenix.py** (same v2 atom
transformer + EDM + confidence structure). Start remap from `protenix_weights.py`.

## Status log

- 2026-07-12 (tick 1): maturity + redundancy verified; real weights downloaded
  (public S3, no gating); full architecture + weight-tree analysis done; reuse map +
  component plan written; confirmed AF2 bias-gating fix does not transfer. No device
  code yet — next tick starts P0 reference harness + P2 remap table.

## P1 status (branch `wk/tt-bio-openfold3-port-p1`)

Done, verified end-to-end against the real `pip install openfold3` (v0.4.3) reference.

**What's vendored** — `tt_bio/_vendor/openfold3/` (75 files, see NOTICE #5): the
query → feature-dict path only — `Query`/`InferenceQuerySet` schema, CCD/ligand
lookup (`BiotiteCCDWrapper`, rdkit/pdbeccdutils), tokenization, structure + reference
-conformer + MSA + template featurization. Dropped everything not on that path: the
Lightning `Dataset`/`DataModule`/dataset-registry framework (`register_dataset`,
`abstract_single`, `data_module.py`, `stochastic_sampler_dataset.py` — training-only,
pulls in `pytorch_lightning`), the LMDB-backed training dataset-cache formats
(`lmdb`, `boto3`/S3), and the PDB/S3 template-cache *build* pipeline
(`func_timeout`-wrapped fetch/precache/multiprocessing in
`pipelines/preprocessing/template.py` — trimmed to just its `TemplatePreprocessorSettings`
config class, which `InferenceDataset` actually reads). Two files got a matching
trim for the same reason: `primitives/caches/format.py` (kept only the
`DatasetChainData`/`DatasetReferenceMoleculeData` type-hint dataclasses, dropped the
LMDB dataset-cache classes) and `primitives/quality_control/logging_utils.py` (the
`memory_profiler` import was made lazy — it's an off-by-default profiling decorator,
never on the inference path). New pip deps (pure-Python, no CUDA): `pydantic`,
`pdbeccdutils`, `func_timeout`, `networkx` (the last for CCD bond-graph connected
components, a genuine runtime need, not training cruft).

**Driver**: `tt_bio/openfold3_data.py::build_openfold3_features(query)` — replicates
`InferenceDataset.create_all_features` as a plain function (no Dataset/DataModule
needed to featurize one query).

**Verification method**: pip-installed real `openfold3==0.4.3` into a scratch venv,
called the *unmodified* upstream `InferenceDataset` directly (bypassing the
Lightning/checkpoint-download CLI, which needs a model for a data-only question) on
`examples/example_inference_inputs/query_ubiquitin.json`, and diffed every tensor
against `tt_bio.openfold3_data`'s output. All 34 feature-dict keys match in shape,
dtype, and value (`torch.equal`/`allclose`) — **except** `ref_pos` (RDKit reference-
conformer 3D coordinates), which differs `run-to-run` even for two calls into the
*same* unmodified upstream code (confirmed: reran the real reference pipeline twice,
`ref_pos` differed between those two runs by as much as the diff against tt-bio's
output, while every other key was identical across the two reference runs). Bond
lengths in `ref_pos` are chemically valid in all three runs, confirming this is
upstream RDKit ETKDG conformer-embedding stochasticity (no fixed seed), not a
vendoring bug — every deterministic feature is bit-exact.

**Resume anchor for P2/P3**: nothing here blocks the weight-remap leg. When wiring
`OpenFold3.fold()`, import features via `tt_bio.openfold3_data.build_openfold3_features`
— it returns the exact same dict shape as protenix_data.py's featurizer, so the model
assembly step can treat it as a drop-in `input_feature_dict`. `pyproject.toml` needs
the 4 new deps installed (`pip install -e .` after merge) before the shared dev env
picks them up.
