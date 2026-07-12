# OpenDDE port

Resume anchor for porting OpenDDE onto Tenstorrent inside tt-bio. OpenDDE's
compute graph is Protenix-v2's (already fully ported in `tt_bio/tenstorrent.py` +
`tt_bio/protenix.py`) plus exactly one novel module, so the port reuses that
entire ttnn stack and adds the one block. Status: identity + architecture +
redundancy **measured** (not assumed); the novel block (`StructuralTokenExpander`)
is **ported to ttnn and on-device parity-verified** (PCC ≥ 0.99999 vs the Phase-0
golden, qb2 card 0); the real `opendde.pt`/`opendde_abag.pt` checkpoints are **pulled**,
the **weight remap + expander→refiner pipeline assembly are done and on-device verified
with real weights**; the structural-token tokenizer/featurizer is **ported and done**
(bit-exact vs the real upstream tokenizer); the shared `Trunk` is **c_z-parametric** and
the diffusion `z_trunk` conditioning branch is **wired**; a first **end-to-end co-fold
ran and is verified finite** on real weights (P3); and (P4, 2026-07-12) a **real
production-setting accuracy number was measured** (honest, and not yet matching
Protenix-v2's own no-MSA floor on the same target — see "Production-setting accuracy"),
**confidence-head best-of-N selection is wired** (c_z-parametrized, residue-axis, reusing
Protenix-v2's `ConfidenceHead` verbatim), and **CLI/predict integration is done and
verified end-to-end** (`tt-bio predict --model opendde` runs the same scheduler/worker
path as every other model, real weights, real device) — see "Remaining" for what's
still open.

## Identity (re-verified 2026-07-12)

Live re-check this date: GitHub `aurekaresearch/OpenDDE` (main HEAD `a0d5134`,
the pin used below), HF `aurekaresearch/OpenDDE`, arXiv:2607.03787 all resolve.
No point-release past the 2026-07-06 preview (main is still `a0d5134`; PRs #3/#4
open, unmerged). Apache-2.0.

- **What:** OpenDDE = "Open Drug Discovery Engine", Aureka AI Research. All-atom
  AF3-lineage co-folding model; `pyproject.toml` keywords list `alphafold3`, and
  the code ships the same OpenFold/Protenix CUDA `FusedLayerNorm` kernel.
- **Scale:** `opendde_v1`, 656M params (`config/model_registry.py`).
- **Release scope:** the shipped preview is **co-folding (structure prediction)
  only**. Design / affinity are roadmap, not in the checkpoints. (Corrects the
  task brief's fold+design premise.)
- **Headline result:** best *open* antibody-antigen co-folding (rank DockQ
  PXMeter-AB 51.0 / FoldBench-AB 70.0 / 2026ARK-AB 66.4). Ships a dedicated
  `opendde_abag.pt` alongside the general `opendde.pt`.

## Redundancy: measured, not assumed

The trunk/diffusion/confidence classes in OpenDDE are the AF3/Protenix set, and
**every one already has a ttnn implementation in tt-bio** (`grep '^class'`, both
sides, 2026-07-12):

| OpenDDE class | Already ported in tt-bio |
|---|---|
| `InputFeatureEmbedder`, `RelativePositionEncoding`, `FourierEmbedding` | `protenix.py` `TrunkInput` / `AtomFeaturization` |
| `AtomAttentionEncoder`/`Decoder`, `AtomTransformer` | `protenix.py` `AtomAttentionEncoder`, `AtomTransformer` (+ decoder in `DiffusionModule`) |
| `MSAModule`, `OuterProductMean`, `MSAPairWeightedAveraging` | `tenstorrent.py` `MSAModule` / `MSA` / `OuterProductMean` / `PairWeightedAveraging` |
| `PairformerStack`/`Block`, `TriangleMultiplication*`, `TriangleAttention`, `AttentionPairBias`, `Transition` | `tenstorrent.py` `Pairformer` / `PairformerLayer` / `TriangleMultiplication` / `TriangleAttention` / `AttentionPairBias` / `Transition` |
| `DiffusionModule`, `DiffusionTransformer(Block)`, `DiffusionConditioning`, `ConditionedTransitionBlock`, `AdaptiveLayerNorm` | `tenstorrent.py` `DiffusionModule` / `DiffusionTransformer(Layer)` / `Diffusion` / `ConditionedTransitionBlock` / `AdaLN` |
| `TemplateEmbedder` | `tenstorrent.py` `TemplateRecycle` |
| `ConfidenceHead`, `DistogramHead` | `protenix.py` `ConfidenceHead` (+ distogram) |
| **`StructuralTokenExpander`** | **nothing — the only novel compute** |

Dims (measured 2026-07-12 from `opendde.pt` shapes + `config/model_base.py`, not
assumed): `c_s=c_s_inputs=449`/`c_s=384`, Pairformer 48 blocks / single-attn 16 heads,
MSA 4, template 2, DiT 24 / 16 heads, atom tx 3 / 4. **Correction to an earlier note:**
the pair channel is `c_z=384` and triangle attention has **12 heads**, *not* the tt-bio
Protenix-v2 checkpoint's `c_z=256` / 8 heads — same architecture, wider pair. The shared
subtree's key *names* are still byte-identical to `protenix-v2.pt` (0 keys missing, both
directions), so it routes straight into the Protenix stack; the tt-bio `Trunk` just has to
be made `c_z`-parametric (it currently hardcodes 256) to accept the wider weights. OpenDDE
also adds a diffusion-conditioning `z_trunk` branch (`diffusion_module.diffusion_conditioning.*_z_trunk`)
absent in Protenix-v2. The `structural_token_refiner` is a 4-block `PairformerStack`
(reused, not new; single-attn 8 heads, 12 triangle heads).

**Verdict — additive via one block + a checkpoint, not a redundant second
engine.** The compute is ~100% Protenix-v2's already-ported graph; what is new
is (1) `StructuralTokenExpander` and (2) the antibody-antigen checkpoint
`opendde_abag.pt`, which delivers the best open Ab-Ag accuracy (Boltz-2 /
Protenix-v2 are materially weaker there). Ab-Ag is the most-requested therapeutic
co-folding regime, so the capability is a real gain riding on already-ported
compute. Nucleic-acid coverage is *not* the differentiator (Protenix-v2 has it).
It does **not** add a design stack (design is roadmap), so no overlap with
`boltzgen`.

## The one novel block: `StructuralTokenExpander`

Expands the residue-level trunk (`s_inputs`, `s`, `z`) onto the structural-token
axis before diffusion, adding role conditioning and same-residue pair structure.
The rest of the pipeline (diffusion, confidence) then runs unchanged on the
structural-token axis — ttnn ops are axis-agnostic, so no primitive changes are
needed, only feeding them the expanded tensors.

Forward (`opendde/model/modules/structural_tokens.py`, `opendde_v1` config):

- **single:** `s_inputs_struct = gather(s_inputs_res, parent) + role_emb(role)`;
  `s_struct = s_parent + split_MLP(s_parent) + role_emb(role)` where
  `split_MLP = LayerNorm -> LinearNoBias(c_s->2c_s) -> SiLU -> LinearNoBias(->c_s)`.
  All host-gather + reused ttnn LN/Linear/SiLU.
- **pair (the hard part):** `pair_projection_mode="full"` = `n_roles*n_roles = 49`
  separate `LinearNoBias(c_z->c_z)` selected per (row_role, col_role), assembled
  in `pair_chunk_size=128` chunks (`_make_structural_pair_activations_chunked`),
  plus additive role/adjacency pair-init biases and a scalar attention bias.
  `pair_output_space="residue"`. This is the genuinely new ttnn work.

The index gathers (parent, prev/next-parent adjacency, role-pair-type maps) are
integer-only and precomputable host-side; only the projections/MLP/bias adds run
on device.

## Reference harness (Phase 0 — done, runs on qb2)

`scripts/opendde_structtoken_ref.py` builds `StructuralTokenExpander` at the real
`opendde_v1` config (full projection, chunked, `init_mode="scratch"`), randomizes
all weights (fixed seed), runs a deterministic forward on synthetic residue-trunk
inputs (`N_RES=32`, `N_STRUCT=64`), and saves inputs + golden outputs for the PCC
gate. FoldCP (context-parallel) and `optree` are stubbed; no data pipeline, no
CUDA, no checkpoints (per-module parity methodology). Set `OPENDDE_SRC` to a
checkout pinned at `a0d5134`. Verified output (2026-07-12): `s_struct (64,384)`,
`z_struct (64,64,384)`, `structural_pair_attn_bias (64,64)`.

Reference-build precedent: follows Protenix (`scripts/protenix_ref_build.py`
builds from an external `PROTENIX_SRC` checkout, stubbing `FusedLayerNorm`)
rather than copying model code into `_vendor/`; OpenDDE is the same AF3-family
situation.

## ttnn port (P1 — done, on-device parity-verified)

`tt_bio/opendde.py` `StructuralTokenExpander` ports the block, reusing tt-bio's
`LayerNorm`/`Linear`/`SiLU` via `protenix._KeyedWeights` (no primitive duplication).
Split of work per the plan: the integer index gathers + role/adjacency masks +
`role_pair_type` map are precomputed host-side; only the split-MLP, the 49
role-pair pair projections, and the bias adds run on device. The
`pair_projection_mode="full"` path (the genuinely-new compute) is done by grouping
the flattened pair positions by `role_i*7+role_j` (host permute), running one
device matmul per non-empty group, and scattering back with a single device gather
(`ttnn.embedding` on the inverse permutation) — numerically identical to OpenDDE's
masked per-`(role_i,role_j)` projection, just reordered so each group is contiguous.

Gate: `scripts/opendde_structtoken_parity.py` (qb2 card 0, same random-weight
golden as Phase 0, apples-to-apples). Threshold PCC > 0.98; measured 2026-07-12:

| output | PCC | pathway |
|---|---|---|
| `s_inputs_struct` | 1.00000 | single (gather + role emb) |
| `s_struct` | 1.00000 | single split-MLP |
| `z_struct` | 0.99999 | full chunked 49-role-pair projection |
| `structural_pair_attn_bias` | 1.00000 | scalar-weighted mask sum |

Multi-chunk self-consistency (forced `pair_chunk_size=16`, 4 row-blocks): z_struct
0.99999, attn_bias 1.00000 — the chunk loop + concat assemble correctly (the golden
itself used `pair_chunk_size=128`, a single chunk at `N_STRUCT=64`).

qb2 note: this block's device open needs `TT_MESH_GRAPH_DESC_PATH` pointed at
ttnn's `p150_mesh_graph_descriptor.textproto` (the P300-misdetection quirk; see
memory `ttatom-qb2-multicard-fanout`); the predict/worker path sets it
automatically, standalone scripts must export it.

## Pipeline assembly + real-weight load (P2 — done, on-device verified)

`tt_bio/opendde.py` now has the assembly + real-weight path:

- `load_opendde_checkpoint(path=None, abag=False)` — fetch (HF `aurekaresearch/OpenDDE`)
  + load `opendde.pt` / `opendde_abag.pt` to a flat state_dict.
- `route_opendde_weights(sd)` — **the "remap"**: splits all 4482 keys into
  `expander` (identity under prefix strip), `refiner` (4× the reused
  `protenix_weights.remap_pairformer_block`), and `shared` (Protenix-v2-family, keys
  byte-identical to `protenix-v2.pt`). Asserts full coverage — no dropped keys.
- `class OpenDDE` — builds the expander + 4-block refiner from real weights and holds
  the shared subtree for the Protenix stack. `expand_and_refine(...)` is the fully-wired
  novel seam (expander → refiner, with `structural_pair_attn_bias` fed to the refiner's
  pair/triangle attention).

Gate `scripts/opendde_assembly_verify.py` (qb2 card 0, **real `opendde.pt` weights**),
measured 2026-07-12:

| check | result |
|---|---|
| weight routing coverage | 4482 keys → 65 expander + 228 refiner + 4189 shared, 0 dropped |
| shared subtree vs `protenix-v2.pt` | 0 of 4174 protenix keys missing (names identical) |
| `opendde_abag.pt` routing | identical split (4482 → 65 + 228 + 4189) |
| expander→refiner seam on device | finite; `s_inputs (64,449)`, `s (64,384)`, `z (1,64,64,384)` |

This is a **wiring + finiteness** result with real weights, *not* an accuracy claim: there
is no real-weight golden without an upstream (CUDA) OpenDDE forward, so no PCC/RMSD is
reported for the shared path. The expander block alone stays parity-verified (PCC
≥ 0.99999) against the Phase-0 random-weight golden (`opendde_structtoken_parity.py`).

## Structural-token featurizer (P3 -- done, bit-exact verified)

`tt_bio/opendde_data.py` `build_structural_token_features` ports
`opendde/data/tokenizer.py`'s residue->structural-token expansion natively onto tt-bio's
own residue-token feature dict (`tt_bio.protenix_data`) -- no biotite `AtomArray`
dependency: tt-bio's per-residue atom names already come from
`tt_bio.data.const.ref_atoms` (the same table `protein_atom_features` uses for `ref_pos`),
so the backbone/sidechain split and atom<->structural-token maps are derived independently
from the identical source, in the identical iteration order. Produces
`parent_residue_idx`, `subtoken_role_id`, `twin_token_idx`, `prev/next_parent_residue_idx`,
`atom_to_structural_token_idx`, `atom_to_structural_tokatom_idx` -- everything
`StructuralTokenExpander.__call__` and the diffusion module's atom<->token broadcast need.

Scope: **protein chains only** (the target case for a first real co-fold). Glycine and
any non-protein/ligand residue degenerate to a single "atom"-role token -- the same
fallback the upstream tokenizer itself uses whenever the sidechain atom group is empty.
Nucleic-acid backbone/base splitting follows the identical pattern and is a followup for
when a nucleic co-folding target is on the critical path.

Gate: `scripts/opendde_structtoken_featurizer_parity.py` builds a synthetic sequence
(`"AGWKSG"` -- exercises a sidechain-bearing residue, glycine's single-token fallback, and
repeats) from tt-bio's own atom-name table, tokenizes it with the REAL upstream
`opendde.data.tokenizer.AtomArrayTokenizer` (via a minimal compatible `biotite.AtomArray`)
and independently with `build_structural_token_features`, and diffs every annotation.
Measured 2026-07-12: **bit-exact match** on `subtoken_role_id`, `parent_residue_idx`,
`twin_token_idx`, and atom->structural-token role consistency.

## `Trunk` c_z-parametric (P3 -- done)

`protenix.Trunk.__init__` takes `c_z=None` (defaults to 256, Protenix-v2's) and derives
`n_tri_heads = c_z // TRI_HEAD_DIM` (head dim fixed at 32 across both variants -- measured
from real checkpoint tensor shapes: Protenix-v2 `tri_att_start.linear.weight` is `(8,256)`,
OpenDDE's is `(12,384)`, both `/32`) for the main 48-block `Pairformer` AND the MSA
module's `pair_stack` `PairformerLayer`s. The template embedder's pair stack stays fixed at
its own 64-dim/2-head channel regardless of the main `c_z` (verified against both
checkpoints' real tensor shapes -- it is architecturally independent of the main pair
width, not a `c_z`-derived quantity). `Protenix.__init__` threads `c_z` through to `Trunk`.
Smoke-verified on-device with REAL weights both ways: Protenix-v2 checkpoint builds
`C_Z=256`/8 heads and runs finite (unchanged from before this change); OpenDDE's routed
shared subtree builds `C_Z=384`/12 heads and runs finite, correct `(1,N,N,384)` shape.

## Diffusion `z_trunk` conditioning branch (P3 -- done)

`protenix.Protenix._diffusion_pair_cond` now checks for
`diffusion_module.diffusion_conditioning.linear_no_bias_z_trunk.weight`: when present
(OpenDDE: `c_z_pair_diffusion=128` compressed below the shared Trunk's `c_z=384`), it
LN+projects `z_trunk` down to `c_z_pair_diffusion` (`layernorm_z_trunk` +
`linear_no_bias_z_trunk`, reference `DiffusionConditioning._project_z_trunk` /
`compress_pair_z`) before concatenating with `relpe`, exactly matching the reference's
`prepare_cache`. When absent (Protenix-v2: `c_z_pair_diffusion == c_z == 256`, no
compression), behavior is byte-identical to before this change -- gated on key presence,
no duplicated method. Verified on-device with real weights both ways: Protenix-v2 path
outputs `(1,N,N,256)` finite (unchanged); OpenDDE path outputs `(1,N,N,128)` finite.

## End-to-end co-fold (P3 -- first real run, reduced settings)

`OpenDDE.__init__` now builds a `tt_bio.protenix.Protenix` instance from the routed shared
subtree at `c_z=384` (reused verbatim -- no duplicated orchestrator), giving OpenDDE the
input embedder / trunk / diffusion module for free. `OpenDDE.fold(feats, n_step=, n_cycles=)`
implements the reference `get_pairformer_output -> expand_to_structural_tokens ->` EDM
diffusion pipeline: (1) input embedder + trunk at the **residue** axis (identical to
`Protenix.fold`'s first steps); (2) `expand_and_refine` -- the novel seam -- onto the
**structural-token** axis; (3) diffusion pair conditioning (`_diffusion_pair_cond` on
`z_struct` + a relp recomputed at structural-token granularity) and the EDM sampler, with
the atom<->token broadcast (`S`, `_plm_z_term`) switched to
`atom_to_structural_token_idx` -- matching `opendde/model/opendde.py`'s
`select_pair_output_branch` (`pair_output_space="residue"` only pools the PAIR branch for
confidence; diffusion itself runs on the structural axis, verified by reading the
reference source, not assumed).

`scripts/opendde_e2e_smoke.py` ran this on PDB 7ROA (117 residues, the same target
`scripts/release_gate.py` uses for Protenix-v2/Boltz-2/ESMFold2), REAL `opendde.pt`
weights, reduced settings (2 trunk recycles, 10 diffusion steps, 1 sample -- not
production's 10/200/5): **ran to completion, finite `(1,900,3)` coords, no crash.**
`scripts/opendde_e2e_rmsd.py` then computed Ca-RMSD directly against
`examples/ground_truth_structures/prot.cif` (reusing `tests/test_structure.py`'s
Kabsch/TM harness, not re-derived): **15.58 A RMSD, TM 0.123** at these reduced settings
(P3). See below (P4) for the production-setting number.

## Production-setting accuracy (P4, 2026-07-12)

Ran `scripts/opendde_e2e_smoke.py` at production settings (`OPENDDE_NCYCLES=10
OPENDDE_NSTEP=200`) across 5 seeds on 7ROA, single-sample (no MSA, no confidence
selection yet at the time of this specific sweep):

| seed | Ca-RMSD (A) | TM |
|---|---|---|
| 0 | 14.219 | 0.137 |
| 1 | 13.631 | 0.177 |
| 2 | 14.273 | 0.137 |
| 3 | 13.249 | 0.197 |
| 4 | 12.785 | 0.158 |

Mean 13.6 A, range 12.8-14.3 A -- tight (low seed-to-seed variance), essentially
**unchanged from the P3 reduced-setting run (15.58 A at 2 cycles/10 steps)**. Going from
2->10 cycles and 10->200 diffusion steps bought about 2 A, not the large jump a genuine
undersampling fix would predict. This is an **honest, real number, not a fabricated one**
-- and it does not yet match expectations: `docs/protenix-accuracy-investigation.md`
measured Protenix-v2's own no-MSA floor on this *exact* target at 5.44-7.84 A (n_step=200,
single-sample) and 3.13 A with a real MSA, best-of-5. OpenDDE's ~13.6 A is roughly 2x
Protenix-v2's no-MSA number on the same input.

Two candidate explanations, **neither confirmed this tick**:
1. **No MSA fed** (`build_complex_features([(SEQ, None, "protein")])` -- a3m is `None`),
   the same handicap Protenix-v2 carries on this target; `examples/msa/seq2.a3m` looks
   like the right alignment (same organism/length range) but its row-0 query does not
   character-match `SEQ` exactly (a leading residue difference), so `_parse_a3m_to_msa`'s
   exact-length match silently returns `None` -- feeding it needs either a re-aligned a3m
   or a fresh search against `SEQ` itself. Not attempted this tick.
2. A real, OpenDDE-specific gap: the structural-token expand/refine seam sits between
   trunk and diffusion for OpenDDE (absent in Protenix-v2), so exact parity with
   Protenix-v2 numbers isn't guaranteed even controlling for MSA -- this needs the
   with-MSA number (1) before it can be distinguished from "OpenDDE is just a harder
   model on this target."

**Best-of-N does not rescue this** (see next section): confidence-selected best-of-5 on
the same no-MSA input landed at 13.63 A, inside the single-sample seed range above --
the underlying per-seed distribution is tight, so there's no lucky outlier for
confidence to find. No DockQ / antibody-antigen read was attempted this tick: no
antibody-antigen ground-truth example exists in `examples/` today (only single-protein
targets), so measuring `opendde_abag.pt`'s headline differentiator needs sourcing one
first.

## Confidence-head best-of-N selection (P4, 2026-07-12)

`ConfidenceHead.confidence()` (`tt_bio/protenix.py`) had two LayerNorms hardcoded to
Protenix-v2's `c_z=256` (`pae_ln`/`pde_ln`, operating directly on the z-pair tensor) --
a real bug that would crash at OpenDDE's `c_z=384` (LayerNorm's `normalized_shape` must
match the tensor's last dim). Fixed by deriving the shape from `zf.shape[-1]` instead of
the literal `256`; Protenix-v2 is unaffected (its own `zf.shape[-1]` is still 256, so
behavior there is byte-identical).

Read `opendde/model/opendde.py`'s `select_pair_output_branch` (not assumed) to find the
right call shape: OpenDDE's shipped config has `pair_output_space="residue"`, so
confidence is computed on the **pre-expansion residue-axis** `s_inputs`/`s_trunk`/
`z_trunk` from fold() step 1 -- not the structural-token axis, and not any pooled
version of it. So `OpenDDE.fold()` calls
`self._protenix.confidence_head.confidence(s_inputs, s_trunk, z_trunk, coords[k], feats)`
with the exact same tensors/`feats` shape `Protenix.fold()` already uses -- no new
structural-token distogram-rep-atom-mask machinery needed (that path only matters for
the *other*, unconfigured `pair_output_space="structural"` branch).

`OpenDDE.fold()` now takes `n_sample=`/`return_confidence=`, mirroring
`Protenix.fold()`'s signature and per-sample-seed-offset loop exactly.
`scripts/opendde_confidence_verify.py` ran this at production settings (5 samples):
pLDDT 0.496-0.504, pTM 0.352-0.375 (sane range, discriminating between samples), best-of-5
picked sample 1 (pTM 0.375) -- Ca-RMSD of the picked sample was 13.63 A, consistent with
the single-sample distribution (see above): confidence selection works mechanically but
the input distribution is too tight for it to matter at these settings/no-MSA.

## CLI/predict integration (P4, 2026-07-12)

`--model opendde` / `opendde-abag` now ride the exact same scheduler/worker path as every
other model (`_local_workers` + `_dispatch_run`, or `--controller`): `tt_bio/worker.py`
gained `_predict_opendde_one` (mirrors `_predict_protenix_one` -- same
`build_complex_features` + `_write_protenix_structure` -- minus the MSA stage, since
OpenDDE is single-sequence only for now) plus `load_model`/`_prepare_run_cache` branches
(`OpenDDE.load_from_checkpoint`, auto-fetches from HF exactly like `Protenix.
load_from_checkpoint`). `tt_bio/main.py`'s former `raise click.ClickException` for
`--model opendde` is gone; `_resolve_recycling_steps` now defaults opendde to the trunk's
spec of 10 (was falling through to 3).

Verified end-to-end on real weights, real device (qb2 card 0), both checkpoints:
```
tt-bio predict examples/prot.yaml --model opendde --devices 0 \
    --recycling_steps 2 --sampling_steps 10 --diffusion_samples 2
tt-bio predict examples/prot.yaml --model opendde-abag --devices 0 \
    --recycling_steps 1 --sampling_steps 2 --diffusion_samples 1
```
Both ran to `Done: 1 ok, 0 failed`, wrote a valid `structures/prot.cif` (+
`prot_model_1.cif` for the second sample) and a `results.json` with per-sample
pLDDT/pTM/confidence_score and an `all_runs` ranking -- the identical output shape
Protenix-v2/Boltz-2 produce. Non-protein input chains raise a clear error (OpenDDE is
protein-only for now) instead of failing deep in the data pipeline.

## Remaining (P5)

- **Root-cause the no-MSA accuracy gap**: run OpenDDE with a real MSA on 7ROA (fix
  `examples/msa/seq2.a3m`'s query-sequence mismatch against `SEQ`, or search a fresh a3m)
  and compare against Protenix-v2's with-MSA 3.13 A on the same target -- this is what
  distinguishes "OpenDDE needs an MSA like every AF3-family model" from "there's a real
  port bug" (see "Production-setting accuracy" above).
- **DockQ / antibody-antigen read**: source an antibody-antigen input + ground-truth
  complex (none exists in `examples/` today) to measure `opendde_abag.pt`'s actual
  differentiator -- the whole reason to ship the model.
- **MSA wiring for OpenDDE's CLI path**: `_predict_opendde_one` is single-sequence only;
  once (1) shows MSA matters, wire the same MSA search stage `_predict_protenix_one` has.
- **Nucleic-acid / ligand structural tokens**: `opendde_data.py` is protein-only; extending
  to DNA/RNA backbone/base splitting and ligand atom-tokens follows the identical pattern
  once a mixed-modality co-folding target is on the critical path.
- `--fast` + multi-card `--devices` ride the existing predict scheduler (memory
  `predict-multicard-already-exists` -- no new fanout path) now that CLI integration has
  landed, but this specific combination (`--model opendde --fast` / `--devices 0,1,2,3`)
  has not been explicitly exercised yet.

## Accuracy gate

- **Metric:** Ca-RMSD vs ground truth (`scripts/release_gate.py` method) for
  co-folding, plus **DockQ on antibody-antigen** complexes (the whole reason to
  add the model, and what the paper reports). No designability gate -- that path
  does not exist in the release.
- **Stochasticity:** diffusion is seed-stochastic and the repo warns outputs are
  not reproducible across releases, so parity is per-target Ca-RMSD/DockQ within
  sample variance (as for Boltz-2 / Protenix-v2), not bit-exact.

## Status

- Identity, measured redundancy, architecture mapping, gate choice: **done.**
- Novel-block torch reference + golden captured (`scripts/opendde_structtoken_ref.py`).
- **ttnn `StructuralTokenExpander` ported and on-device parity-verified**
  (`opendde_structtoken_parity.py`, PCC >= 0.99999).
- **Real checkpoints pulled** -- `opendde.pt` + `opendde_abag.pt` (2.6 GB each) on qb2 in
  the HF cache.
- **Weight remap + pipeline assembly done and on-device verified with real weights**
  (`route_opendde_weights`, `OpenDDE`, `scripts/opendde_assembly_verify.py`).
- **Structural-token tokenizer/featurizer: done**, bit-exact vs the real upstream
  tokenizer (`opendde_structtoken_featurizer_parity.py`).
- **`Trunk` c_z-parametric: done**; **diffusion `z_trunk` conditioning branch: done**.
- **End-to-end co-fold: verified finite** on real weights, both reduced (P3) and
  **production settings (P4)**: real Ca-RMSD measured at production settings, 5 seeds,
  mean 13.6 A / range 12.8-14.3 A on 7ROA -- honest number, not yet matching
  Protenix-v2's own no-MSA floor on the same target (5.44-7.84 A); root cause
  (no-MSA vs a real port gap) not yet distinguished.
- **Confidence-head best-of-N selection: done** (`ConfidenceHead` c_z-parametrized,
  residue-axis call verified against the reference's `select_pair_output_branch`;
  `scripts/opendde_confidence_verify.py` sane pTM/pLDDT, working selection).
- **CLI/predict integration: done and verified end-to-end** (`tt-bio predict --model
  opendde`/`opendde-abag`, real weights, real device, valid CIF + results.json).
- **Not yet**: MSA-on comparison run (the top accuracy lever), DockQ / antibody-antigen
  read (no ground-truth example available), nucleic-acid/ligand structural tokens,
  explicit `--fast`/multi-card verification for opendde.

**Next action (P5):** get an MSA onto 7ROA for OpenDDE (fix or regenerate the a3m) and
re-run the production-setting accuracy comparison against Protenix-v2's with-MSA 3.13 A
-- this is the single highest-value next step, since it decides whether the current ~13.6
A is expected (no-MSA AF3-family models fold badly) or a real bug to chase. In parallel,
source an antibody-antigen example for the DockQ read `opendde_abag.pt` exists to deliver.
