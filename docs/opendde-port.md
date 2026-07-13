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
path as every other model, real weights, real device); (P7, 2026-07-13) **MSA search is
wired into the OpenDDE CLI path** (reuses the Protenix-v2 MSA stage) and **re-measured on
PDB 9dsg with MSA + best-of-5** — antibody-antigen DockQ stays 0.011 (a genuine port/model
issue, not the missing-input gap; the Fab assembles and confidence rises, but the antigen
is mis-docked relative to the Fab across all samples) — see "MSA + best-of-N wired into
the CLI path" — see "Remaining" for what's still open. (P8, 2026-07-13) **a real
multi-chain MSA assembly bug is fixed** (`build_complex_features` now merges per-chain MSAs
the reference way and computes `profile`/`deletion_mean` per chain; single-chain
bit-exact, Protenix-v2 gate still passes) — it improves the Fab internal dock (H-L fnat
0.72 -> 0.82) but **the Ab-Ag DockQ stays 0.011 / fnat 0 across all 5 samples**, so the
Ab-Ag gap is not the multi-chain encoding; the most likely remaining cause is the missing
real complex templates / paired MSA — see "Multi-chain MSA assembly fix + Ab-Ag
re-measure (P8)".

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
`build_complex_features` + `_write_protenix_structure` + the same MSA search stage as
Protenix-v2 / ESMFold2, wired in P7) plus `load_model`/`_prepare_run_cache` branches
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

## Accuracy gap root cause (P5, 2026-07-12)

The reference supports MSA input and uses a Boltz-style MSA block. A controlled 7ROA run
used the complete 136-residue query from `examples/msa/seq2.a3m`, with 76 deduplicated MSA
rows. All figures below use 10 recycling cycles, 200 diffusion steps, five seeds, and
score the 117 resolved residues.

| Configuration | Mean Ca-RMSD | Range | Confidence-selected |
|---|---:|---:|---:|
| Corrected OpenDDE, no MSA | 3.832 A | 3.316-4.320 A | 3.809 A |
| Corrected OpenDDE, MSA | **2.714 A** | **2.613-2.804 A** | **2.680 A** |

MSA input helps, but missing MSA was not the root cause of the 13.6 A result. The real
activation bisect stayed at PCC 0.999963 through template conditioning, then fell to
0.067526 at the first residue-trunk MSA module output. OpenDDE refreshes the MSA state
before its outer-product pair update. The shared port used the Protenix-v2 ordering,
which computes that pair update from the stale MSA state. Applying the OpenDDE ordering
raises first-cycle trunk output PCC from 0.950609 to 0.997102 for `s` and from 0.389377
to 0.947432 for `z`. Protenix-v2 retains its original checkpoint-specific ordering.

The structural-token expander and refiner were not the dominant gap. Their real-input
outputs remain at PCC 0.99849 or better. The bisect also found that the port dropped
`structural_pair_attn_bias` from the diffusion transformer. That branch is now routed
like the reference, although restoring it alone did not materially change RMSD.

A DockQ read is now possible and done -- see "DockQ / antibody-antigen read (P6)" below.
`examples/9dsg_abag.yaml` + `examples/ground_truth_structures/9dsg.cif` add the first
antibody-antigen input/ground-truth pair.

## DockQ / antibody-antigen read (P6, 2026-07-12)

Sourced the first antibody-antigen input + public ground truth from the OpenDDE
benchmark set itself: **PDB 9dsg** (SARS-CoV-2 spike RBD antigen + a neutralizing
Fab), one of the targets in `benchmarks/2026ARK_AB/common_targets.txt` in the OpenDDE
repo. Input `examples/9dsg_abag.yaml` carries the three resolved protein chains --
antigen A (196), Fab heavy H (248), Fab light L (212) -- sequenced from the 9dsg
structure, so predicted and native chains align 1:1; the ground truth is
`examples/ground_truth_structures/9dsg.cif` (the released 9dsg mmCIF). The OpenDDE CLI
path has no MSA stage yet, so the input is single-sequence.

A multi-chain Ab-Ag input first crashed the structural-token featurizer with a 2-atom
mismatch (4969 vs 4967): `opendde_data.build_structural_token_features` decided the
C-terminal OXT carrier from the *global* last residue, but `protenix_data.
protein_atom_features` is called once per chain and appends OXT to each chain's
C-terminus. For 3 chains that is N_chain-1 = 2 OXT atoms off, breaking
`atom_to_structural_token_idx` alignment. Fixed by deriving `is_c_terminal` per
`asym_id`; single-chain behavior is unchanged (the structural-token parity gate still
passes).

Fold: `tt-bio predict examples/9dsg_abag.yaml --model opendde-abag --recycling_steps 10
--sampling_steps 200 --diffusion_samples 1` on qb2 card 0, real `opendde_abag.pt`
weights, ~245 s. Confidence: pLDDT 0.839, pTM 0.608, ipTM 0.549.

DockQ via the reference tool (`DockQ==2.1.3`, the Wallner-lab implementation that
defines the metric; installed into the run venv as an eval-time requirement, not a
project runtime dependency) -- `scripts/opendde_dockq.py`. DockQ maps model chains to
native by sequence (model A,B,C -> native A,H,L) and scores every native interface:

| native interface | meaning | DockQ | Fnat | Fnonnat | clashes |
|---|---|---:|---:|---:|---:|
| A-H | antigen - Fab heavy (the paratope-epitope) | **0.011** | 0.00 | 1.00 | 2 |
| H-L | Fab heavy - light (internal) | 0.377 | 0.72 | 0.33 | 1 |

9dsg has no A-L native interface (the light chain does not contact the antigen), so
A-H is the complete antibody-antigen interface. GlobalDockQ (mean over native
interfaces) = 0.194.

**Honest verdict -- a genuine negative result for this regime.** The Fab assembles
correctly (H-L DockQ 0.377, Fnat 0.72) and the fold is confident (ipTM 0.549), but the
antigen is not placed in the paratope: antibody-antigen DockQ is 0.011 with zero native
contacts reproduced. This is single-sequence, one sample, no MSA -- the paper's
headline Ab-Ag DockQ (PXMeter-AB 51.0 / FoldBench-AB 70.0 / 2026ARK-AB 66.4, rank
DockQ) uses MSA for the antigen and best-of-N confidence ranking, neither of which the
OpenDDE CLI path wires at P6. P5 already showed OpenDDE single-sequence underperforms
its with-MSA number on a simpler target; for Ab-Ag the gap is larger because correct
paratope-epitope placement is the whole task. The follow-on is the MSA CLI stage (the
antigen has many homologs) plus best-of-N -- done in P7 below.

## MSA + best-of-N wired into the CLI path (P7, 2026-07-13)

`_predict_opendde_one` now reuses the Protenix-v2 / ESMFold2 MSA stage verbatim
(`_generate_esmfold2_a3m` + `_resolve_a3m_text` + `build_complex_features`'
block-diagonal MSA) instead of folding single-sequence, and `opendde` / `opendde-abag`
are added to `_resolve_msa_default` so a source is resolved (local DB > online) rather
than silently single-sequence, matching protenix-v2 / boltz2. Best-of-N needs no extra
wiring: `OpenDDE.fold` already takes `n_sample` / `return_confidence` (P4) and the
worker path ranks samples by the same AF-style `0.8*ipTM+0.2*pTM` score, so
`--diffusion_samples 5` is the paper's default `N_sample=5` end-to-end.

Re-ran the P6 benchmark with MSA now wired (PDB 9dsg, `opendde_abag.pt`, 10 recycles /
200 diffusion steps, real MSAs searched via the ColabFold API for all three chains:
antigen 12651 rows, Fab heavy 117, Fab light 10950). Two configs:

| config | samples | A-H DockQ (Ab-Ag) | A-H fnat | H-L DockQ (Fab) | GlobalDockQ | ipTM |
|---|---:|---:|---:|---:|---:|---:|
| P6, no MSA | 1 | 0.011 | 0.00 | 0.377 | 0.194 | 0.549 |
| P7, MSA | 1 | 0.011 | 0 | 0.494 | 0.253 | 0.712 |
| P7, MSA + best-of-5 | 5 (conf. rank 0) | 0.011 | 0 | 0.497 | 0.254 | 0.715 |

DockQ across all 5 samples (oracle view, since the paper's Figure 5 separates
ranking- vs oracle-based selection): A-H 0.0109-0.0113, **fnat 0 in every sample**;
H-L 0.477-0.497; GlobalDockQ 0.244-0.254. Confidence-selected best (rank 0, ipTM 0.7147)
== the oracle best on A-H (0.0113, tied with sample 4) -- the 5-sample distribution is
degenerate at fnat=0, so neither ranking- nor oracle-based selection can find a
paratope-placement that none of the samples produced.

**Honest verdict -- MSA is wired and consumed, but the Ab-Ag DockQ gap is NOT the
missing-input gap.** MSA clearly feeds the model (whole-complex ipTM 0.549 -> 0.712,
Fab H-L DockQ 0.377 -> 0.494, pLDDT 0.839 -> 0.892), matching the P5 7ROA pattern where
MSA lifted RMSD from 3.83 A to 2.71 A. But the antibody-antigen interface is unchanged
at DockQ 0.011 / fnat 0 across one and five samples: the antigen folds confidently and
the Fab assembles, yet the two are docked in the wrong relative orientation (zero native
paratope-epitope contacts). This is a genuine port/model issue distinct from missing
MSA, not a lower bound the MSA stage was expected to close on its own. Frame vs the
paper's Figure 2 headline (PXMeter-AB 51.0 / FoldBench-AB 70.0 / 2026ARK-AB 66.4 success
rates): a single-target DockQ is not comparable to a benchmark-wide success rate, so
this is one honest data point, not a reproduction of Figure 2/3. Remaining levers that
the paper's standard pipeline also includes but this port does not wire yet: template
features (Section 3's standard input list), and far more seeds under oracle selection
(Figure 5 scales 1 -> 500 -- but here the 5-seed distribution is degenerate at fnat=0,
so more seeds of the same distribution are unlikely to help without a distribution
change). Tracked in "Remaining".

## P8 — multi-chain MSA assembly fix + Ab-Ag re-measure (2026-07-13)

P7 ranked the remaining Ab-Ag levers (1) template features, (2) a multi-chain
relative-pose/asymmetry bug, (3) more seeds. P8 pursued them in that order.

**(1) Templates — already wired (dummy), and not the Ab-Ag lever.** The Trunk template
embedder runs every OpenDDE predict call: `build_complex_features` emits
`dummy_template_features` (4 slots: gap + 3 zero-geometry ALA, the reference's
`use_template=False` padding, merged in v0.2.5), so `Trunk.__call__` takes `nt=4`, not 0
— correcting P7's stale "runs with `nt=0`" line. Real template **search** is a separate,
larger lift: tt-bio has no template-search stage anywhere (Protenix-v2 / Boltz-2 / OF3 all
consume *provided* templates, never search), and OpenDDE's own pipeline is a 509-line
`data/tools/search.py` (HMMER/Kalign binary wrappers) + ~1900 lines of
`data/template/{parser,featurizer,utils}.py` + a PDB template database
(`scripts/download_opendde_data.sh`) — a multi-day data-pipeline port, not a wire-up.
It is also not the Ab-Ag lever: the trunk masks every template feature to the same-`asym`
block (`template_distogram * mc`, `protenix.py` trunk), so templates carry only
**intra-chain** structural hints. They reinforce a chain's own fold (already confident on
9dsg) and do not encode the cross-chain docking orientation, which is the failure mode.
Flagged as the next real lift in "Remaining"; not attempted this tick.

**(2) Multi-chain encoding — found and fixed a real MSA bug; it is not the Ab-Ag cause.**
`build_complex_features` assembled the multi-chain MSA wrong versus both the OpenDDE and
Protenix-v2 references (`data/msa/msa_featurizer.py` + `MSAPairingEngine.merge_chain_features`):

- **Row structure** was "tall": one full-width row per chain per alignment row
  (sum-of-depths rows). The reference pads each chain's MSA to the max chain depth with GAP
  and concatenates **column-wise** into one `(max_d, N_tot)` tensor (one horizontal slice
  per alignment index).
- **`profile` and `deletion_mean`** were computed over the whole merged MSA, so every
  column was diluted by the other chains' GAP rows. On 9dsg (antigen 12651 / heavy 117 /
  light 10950 rows, ~23719 total) the Fab-heavy profile collapsed to ~99.5% GAP and the
  antigen's to ~47% GAP. The reference computes both **per chain** (over that chain's rows
  only, query included) and concatenates.

`relp` itself (residue- and structural-token axis) was checked against the reference and is
correct; `sym_id`/`entity_id`/`asym_id`/`residue_index` assignment is correct for 9dsg
(three distinct entities, residue_index restarts per entity). The MSA assembly was the only
multi-chain bug. Fix in `tt_bio/protenix_data.py`: per-chain `profile`/`deletion_mean`,
max_d-padded column-concatenated `msa`/`deletion_matrix`. Single-chain is bit-exact
(`max_d == m`, per-chain == whole) — verified directly, and the Protenix-v2 release gate
on 7ROA still PASSES (1.417 A, TM 0.936, floor <=6.0/>=0.5), so the shared trunk/MSA
machinery is not regressed.

Re-measured 9dsg with the fix (`opendde_abag.pt`, 10 recycles / 200 steps, the same
per-chain MSAs as P7, real device):

| config | samples | A-H DockQ (Ab-Ag) | A-H fnat | H-L DockQ (Fab) | H-L fnat | GlobalDockQ | ipTM |
|---|---:|---:|---:|---:|---:|---:|---:|
| P7, MSA + best-of-5 (old assembly) | 5 | 0.011 | 0 | 0.497 | 0.72 | 0.254 | 0.715 |
| P8, MSA + best-of-5 (fixed assembly) | 5 (conf. rank 0) | 0.011 | 0 | 0.497 | 0.825 | 0.254 | 0.706 |

DockQ across all 5 samples (oracle view): A-H 0.0110-0.0113, **fnat 0 in every sample**;
H-L 0.473-0.497, fnat 0.81-0.825.

**Honest verdict.** The fix is a real correctness gain: the Fab internal dock rises
(H-L fnat 0.72 -> 0.81-0.825 across all 5 samples, consistent) and the antigen's MSA
profile is no longer half-GAP. But the antibody-antigen interface is unchanged at DockQ
0.011 / fnat 0 across all 5 samples — the same degenerate distribution as P7. A stronger
antigen MSA (now full-depth, no GAP dilution) did not move A-H, which is consistent with
unpaired MSA carrying no cross-chain co-evolution signal: it cannot tell the model where
the antigen meets the paratope. The Ab-Ag mis-docking is therefore **not** a multi-chain
encoding bug; the most likely remaining cause is the input the paper's standard pipeline
uses and this port lacks: real **complex templates** (which DO encode relative orientation,
intra-chain-masked dummy ones do not) and/or **paired MSA** (the OpenDDE/Protenix
`MSAPairingEngine` is not wired into tt-bio's predict path). Both are larger lifts (see
"Remaining"). Lever (3) more seeds is confirmed unhelpful while the 5-sample distribution
is degenerate at fnat=0.

## Remaining

- **MSA search in the OpenDDE CLI path: done (P7).** `_predict_opendde_one` reuses the
  Protenix-v2 MSA stage; `--use_msa_server` / `--msa_db_path` / `--msa_endpoint` all
  work; `opendde` / `opendde-abag` resolve a source via `_resolve_msa_default`.
- **Multi-chain MSA assembly: fixed (P8).** `build_complex_features` now builds the
  block-diagonal MSA the way the reference merges per-chain MSAs (max_d-padded,
  column-concatenated) and computes `profile`/`deletion_mean` per chain. Single-chain is
  bit-exact; Protenix-v2 release gate still passes. Improves the Fab internal dock
  (H-L fnat 0.72 -> 0.81-0.825) but does not close the Ab-Ag interface.
- **Best-of-N for the Ab-Ag DockQ read: done (P7).** `--diffusion_samples N` flows
  through `OpenDDE.fold(n_sample=N)` and the worker's confidence ranking; measured at
  N=5 (paper default). Does not rescue the Ab-Ag interface (degenerate at fnat=0).
- **Antibody-antigen DockQ still 0.011 with MSA + best-of-5 + the P8 MSA fix -- the
  gap is not the multi-chain encoding.** P8 cleared lever (2): `relp` (residue and
  structural-token axis), `sym_id`/`entity_id`/`asym_id`/`residue_index`, and the MSA
  assembly were checked against the reference; the one real bug (multi-chain MSA
  profile/row-structure) is fixed but does not move A-H. The antigen folds confidently
  and the Fab assembles (H-L fnat ~0.82), yet the antigen is mis-docked relative to the
  Fab across all 5 samples. The most likely remaining cause is the input the paper's
  standard pipeline uses and this port lacks -- real **complex templates** (encode
  relative orientation; the dummy-template embedder already runs at `nt=4`, but real
  template *search* is not ported: tt-bio has no template-search stage, and OpenDDE's is
  a 509-line `search.py` + ~1900 lines of template parser/featurizer + a PDB template DB,
  a multi-day data-pipeline port) and/or **paired MSA** (the `MSAPairingEngine` is not
  wired into tt-bio's predict path). Both are release-gated larger lifts.
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
  does not exist in the release. DockQ is computed with the reference `DockQ==2.1.3`
  tool (eval-only, installed into a throwaway `--target` lib for the eval, not a
  project runtime dependency) via `scripts/opendde_dockq.py`; measured on PDB 9dsg at
  **0.011 antibody-antigen DockQ** both single-sequence 1-sample (P6) and with MSA +
  best-of-5 (P7) -- the gap is not closed by wiring the paper's standard MSA input, nor by
  the P8 multi-chain MSA assembly fix (Ab-Ag fnat stays 0 across all 5 samples; the fix
  does improve the Fab internal dock, H-L fnat 0.72 -> 0.82).
- **Stochasticity:** diffusion is seed-stochastic and the repo warns outputs are
  not reproducible across releases, so parity is per-target Ca-RMSD/DockQ within
  sample variance (as for Boltz-2 / Protenix-v2), not bit-exact.

## Status

- Identity, measured redundancy, architecture mapping, gate choice: **done.**
- **Novel-block torch reference + golden captured (`scripts/opendde_structtoken_ref.py`).
- **ttnn `StructuralTokenExpander` ported and on-device parity-verified**
  (`opendde_structtoken_parity.py`, PCC >= 0.99999).
- **Novel-block kernel scout: closed.** StructuralTokenExpander is 2.21% of a production
  fold, host/upload-bound (device matmul 0.6% of the block), no fusion lever, Amdahl ceiling
  1.023x. See `docs/opendde-kernel-scout.md`.
- **Real checkpoints pulled** -- `opendde.pt` + `opendde_abag.pt` (2.6 GB each) on qb2 in
  the HF cache.
- **Weight remap + pipeline assembly done and on-device verified with real weights**
  (`route_opendde_weights`, `OpenDDE`, `scripts/opendde_assembly_verify.py`).
- **Structural-token tokenizer/featurizer: done**, bit-exact vs the real upstream
  tokenizer (`opendde_structtoken_featurizer_parity.py`).
- **`Trunk` c_z-parametric: done**; **diffusion `z_trunk` conditioning branch: done**.
- **Production accuracy gap: fixed.** The OpenDDE-specific MSA ordering restores 7ROA
  mean Ca-RMSD to 2.714 A with MSA and 3.832 A without MSA at production settings.
- **Confidence-head best-of-N selection: done** (`ConfidenceHead` c_z-parametrized,
  residue-axis call verified against the reference's `select_pair_output_branch`;
  `scripts/opendde_confidence_verify.py` sane pTM/pLDDT, working selection).
- **CLI/predict integration: done and verified end-to-end** (`tt-bio predict --model
  opendde`/`opendde-abag`, real weights, real device, valid CIF + results.json).
- **DockQ / antibody-antigen read: done (P6), honest negative.** On PDB 9dsg (Fab +
  SARS-CoV-2 RBD, from the OpenDDE 2026ARK_AB benchmark set), single-sequence 1-sample
  at production settings: antibody-antigen DockQ 0.011 (fnat 0, antigen mis-docked),
  internal Fab DockQ 0.377, GlobalDockQ 0.194. Multi-chain input needed a per-chain
  C-terminal OXT fix in `opendde_data.py` (single-chain parity unchanged).
- **MSA search in the OpenDDE CLI path: done (P7).** `_predict_opendde_one` reuses the
  Protenix-v2 MSA stage (`_generate_esmfold2_a3m` + `_resolve_a3m_text` +
  `build_complex_features`); `opendde` / `opendde-abag` resolve a source via
  `_resolve_msa_default` (local DB > online). Verified end-to-end on 9dsg with real
  per-chain MSAs (antigen 12651 rows, heavy 117, light 10950).
- **Best-of-N for the Ab-Ag DockQ read: done (P7).** `--diffusion_samples N` flows
  through `OpenDDE.fold(n_sample=N)` and the worker's confidence ranking; measured at
  N=5 (the paper's `N_sample=5` default). Ab-Ag DockQ stays 0.011 / fnat 0 across all 5
  samples (degenerate distribution; confidence-selected rank 0 == oracle best on A-H).
  MSA is consumed (ipTM 0.549 -> 0.712, Fab H-L 0.377 -> 0.497) but does not place the
  antigen in the paratope -- a genuine port/model issue, not the missing-input gap.
- **Multi-chain MSA assembly: fixed (P8).** `build_complex_features` now merges per-chain
  MSAs the way the reference does (max_d-padded, column-concatenated) and computes
  `profile`/`deletion_mean` per chain (was: whole-merged-MSA, diluting every chain's
  columns with the other chains' GAP rows -- on 9dsg the Fab-heavy profile was ~99.5%
  GAP). Single-chain bit-exact; Protenix-v2 release gate still passes (1.417 A / TM 0.936
  on 7ROA). On 9dsg the Fab internal dock improves (H-L fnat 0.72 -> 0.81-0.825 across all
  5 samples) but the Ab-Ag interface stays DockQ 0.011 / fnat 0 (degenerate, unchanged) --
  the Ab-Ag gap is not the multi-chain encoding; most likely the missing real complex
  templates / paired MSA.
- **Not yet**: real template **search** (the dummy-template embedder already runs at
  `nt=4`; real templates need porting OpenDDE's HMMER/Kalign search pipeline + a PDB
  template DB, a multi-day data-pipeline lift with no reusable search stage in tt-bio)
  and **paired MSA** (the `MSAPairingEngine` is not wired into the predict path) -- the
  two inputs the paper's standard pipeline uses for Ab-Ag and the most likely remaining
  cause of the 0.011 Ab-Ag DockQ now that the multi-chain MSA assembly is fixed (P8);
  nucleic-acid/ligand structural tokens; and explicit `--fast`/multi-card verification.
  OpenDDE is deliberately not in the README `--model`
  table yet -- its Ab-Ag differentiator is measured but not at parity.
