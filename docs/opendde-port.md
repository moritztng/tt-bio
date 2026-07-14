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
the CLI path" — see "Remaining" for what's still open. **(P11, 2026-07-14) this is
confirmed NOT a port bug:** the reference OpenDDE (CUDA) reproduces 0.011 / fnat 0 on 9dsg
identically, and scores 0.86 on the standard Ab-Ag complex 1ahw — the opendde_abag
checkpoint works on standard targets and 9dsg is just hard for it. See "P11 —
reference-vs-device Ab-Ag parity (decisive)". (P8, 2026-07-13) **a real
multi-chain MSA assembly bug is fixed** (`build_complex_features` now merges per-chain MSAs
the reference way and computes `profile`/`deletion_mean` per chain; single-chain
bit-exact, Protenix-v2 gate still passes) — it improves the Fab internal dock (H-L fnat
0.72 -> 0.82) but **the Ab-Ag DockQ stays 0.011 / fnat 0 across all 5 samples**, so the
Ab-Ag gap is not the multi-chain encoding. (P9, 2026-07-13) **paired MSA is wired and
ruled out**: the reference `MSAPairingEngine` species-pairing path is wired via ColabFold's
pair endpoint and `build_complex_features` stacks the paired block on top of the unpaired
block; on 9dsg 1008 species-paired heavy-light rows are consumed (Fab internal DockQ
0.48 -> 0.56) but the antigen has zero paired rows with the antibody (no genomic
co-occurrence), so Ab-Ag DockQ stays 0.011 / fnat 0 across all 5 samples. Paired MSA
cannot close Ab-Ag for any Ab-Ag complex (antibodies do not co-evolve genomically with
antigens); the gap is a model/inference gap in the structural-docking prior, not an
MSA-side input gap. Protenix-v2 7ROA gate re-run PASS (1.428 A / TM 0.947). See "P9 —
paired MSA wired + Ab-Ag re-measure". (P10, 2026-07-13) **diffusion trace
replay is wired** (`tt-bio predict --model opendde --trace` -> `OpenDDE.fold(trace=)`
-> the shared `edm_sample`/`denoise_traced`): provably lossless (per-step device
parity maxdiff=0.0, and end-to-end coords bit-identical OFF vs ON), accuracy gate
unchanged (3.096 A / TM 0.720), but only ~1% total / ~2% diffusion wall-clock on
Blackhole (compute-bound at this scale, not dispatch-bound like Protenix @L256) —
see `docs/opendde-trace-replay.md`. (P11, 2026-07-14) **the Ab-Ag 0.011 is NOT a port
bug — settled by running the reference OpenDDE (CUDA) on the same 9dsg input + settings.**
The reference itself scores A-H DockQ 0.011 / fnat 0 on 9dsg (best-of-5, opendde_abag.pt,
MSA on, 10 recycles / 200 steps) — identical to the device — while scoring global DockQ
0.83-0.86 on the standard Ab-Ag complex 1ahw, so the opendde_abag checkpoint's Ab-Ag prior
works in general and 9dsg is simply a hard target for it. The earlier "genuine port/model
gap (unknown which)" framing is resolved to model/checkpoint/target reality, not a port
defect. See "P11 — reference-vs-device Ab-Ag parity (decisive)".

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
antigen 117 rows, Fab heavy 12651, Fab light 10950 (the heavy/light immunoglobulin MSAs are the deep ones; the viral RBD has few unique homologs after dedup)). Two configs:

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
9dsg) and do not encode the cross-chain docking orientation, which is the failure mode. De-prioritized as the Ab-Ag lever in "Remaining"; not attempted this tick.

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
encoding bug; the most likely remaining cause is the cross-chain co-evolution input the
paper's standard pipeline uses and this port lacks: **paired MSA** (the OpenDDE/Protenix
`MSAPairingEngine` species-pairing path, which produces a `pairing.a3m` per chain and is
the standard AF3/Protenix docking mechanism for multi-chain complexes; tt-bio wires only
unpaired MSA, which carries no cross-chain signal). Real **complex templates** are **not**
the lever: the reference `TemplateEmbedder` masks every template pair feature (distogram,
unit vector, backbone) to the same-chain block (`multichain_mask = asym_id[:,None] ==
asym_id[None,:]`, `pairformer.py`), exactly as this port does (`template_distogram * mc`),
so templates carry only intra-chain geometry and reinforce a chain's own fold (already
confident on 9dsg), not the cross-chain orientation. Paired-MSA wiring is the larger lift
(see "Remaining"). Lever (3) more seeds is confirmed unhelpful while the 5-sample
distribution is degenerate at fnat=0.

## P9 — paired MSA wired + Ab-Ag re-measure (2026-07-13, clean negative)

P8 named paired MSA (the reference `MSAPairingEngine` species-pairing path) as the most
likely remaining Ab-Ag lever. P9 wires it in and measures it.

**Reference behavior (confirmed from `opendde/data/msa/msa_utils.py`).** `MSAPairingEngine.
pair_chains_by_species` aligns per-chain MSA rows by UniProt/UniRef species ID
(`_UNIPROT_REGEX`/`_UNIREF_REGEX` on description lines) and produces `msa_all_seq` per
chain, column-concatenated across chains so row j of every chain corresponds to the same
species/genome. `FeatureAssemblyLine.assemble` step 6 then stacks this on top of the
unpaired block: `merged[msa] = concat([msa_all_seq, msa], axis=0)`. `profile`/`deletion_mean`
are computed per chain over the UNPAIRED block before pairing; `cleanup_unpaired_features`
dedups the query out of the unpaired block. The model's `MSAModule` consumes the combined
rows as ordinary MSA rows, there is no separate pair-mask, the cross-chain co-evolution
signal is intrinsic to the paired rows (each paired row carries all chains' residues for
one genome).

**Wiring (data pipeline, no new kernels).** Reuses the repo's existing paired-MSA utility,
ColabFold's `ticket/pair` endpoint (`run_mmseqs2(use_pairing=True)` in `tt_bio/data/msa.py`,
already used by Boltz-2), which does the species pairing server-side and returns one
species-aligned a3m per chain, the same artifact `MSAPairingEngine` produces client-side.
No reimplementation of the pairing algorithm.

- `tt_bio/main.py:_generate_opendde_paired_a3m` runs the paired search and returns
  `{seq_hash: paired_a3m_text}`.
- `tt_bio/protenix_data.py` adds `_parse_paired_a3m_to_msa` (a non-dedup, order-preserving
  parser, dedup would break cross-chain row alignment, e.g. the antigen's all-GAP paired
  rows would collapse) and a `paired_a3ms` arg to `build_complex_features`. The paired
  rows are column-concatenated into one `(max_pd, N_tot)` block, truncated to the min
  per-chain row count (alignment guard), query (row 0) dropped (already in the unpaired
  block, the reference's `cleanup_unpaired_features` equivalent), and stacked ON TOP of
  the unpaired block. `profile`/`deletion_mean` stay per-chain over the unpaired block,
  byte-identical to the unpaired-only path. Single-chain / all-None / every-paired-a3m-
  query-only callers get `max_pd == 0` and byte-identical output to before.
- `tt_bio/worker.py:_predict_opendde_one` runs the paired search for multi-chain protein
  complexes (best-effort: a failed search falls back to unpaired-only) and passes the
  paired a3ms through. Protenix-v2 / single-chain paths are untouched.

**Unit-verified (no device).** Single-chain with a query-only paired a3m is byte-identical
to unpaired-only; multi-chain stacks the paired block at the right depth; a 9dsg-style case
(one chain's paired block all-GAP, two chains real) preserves row alignment without dedup
collapse; `profile`/`deletion_mean` are byte-identical with vs without the paired block.

**9dsg re-measure (`opendde_abag.pt`, 10 recycles / 200 steps / 5 samples, real MSAs via
the ColabFold API, qb2 card 2).** The paired search returns 1009 rows per chain (query +
1008), all chains row-aligned by genome. Per-chain real (non-GAP) paired rows after parse:
antigen 0, Fab heavy 1008, Fab light 1008. The antigen has NO paired homolog with either
antibody chain (no genome encodes both a coronavirus spike and an immunoglobulin), its
paired block is all-GAP; the Fab heavy and light share 1008 UniRef100 species-paired rows
(immunoglobulins co-occur across vertebrate genomes). Featurization confirmed: unpaired
MSA depth 12636 -> 13644 with the paired block (+1008 rows stacked on top), `profile`/
`deletion_mean` byte-identical.

| config | samples | A-H DockQ (Ab-Ag) | A-H fnat | H-L DockQ (Fab) | H-L fnat | GlobalDockQ | ipTM |
|---|---:|---:|---:|---:|---:|---:|---:|
| P8, MSA + best-of-5 (no paired MSA) | 5 (conf. rank 0) | 0.011 | 0 | 0.473-0.497 | 0.81-0.825 | 0.244-0.254 | 0.706 |
| P9, MSA + paired MSA + best-of-5 | 5 (conf. rank 0) | 0.011 | 0 | 0.492-0.562 | 0.817-0.825 | 0.251-0.295 | 0.702 |

DockQ across all 5 samples (oracle view): A-H 0.0113-0.0285, **fnat 0 in every sample**;
H-L 0.492-0.562, fnat 0.817-0.825. Confidence-selected rank 0: A-H DockQ 0.0113 / fnat 0
(essentially unchanged from P8's 0.011).

**Honest verdict, a clean negative.** Paired MSA is wired and confirmed consumed: 1008
species-paired heavy-light rows are stacked on top of the unpaired block, the Fab internal
dock firms up slightly (H-L DockQ 0.48 -> 0.56), and whole-complex confidence is unchanged
(ipTM 0.706 -> 0.702). But the antibody-antigen interface stays at DockQ 0.011 / fnat 0
across all five samples, the same degenerate distribution as P7/P8. The root cause is
biological, not a port bug: paired MSA carries cross-chain co-evolution only between chains
that co-occur in the same genome. The Fab heavy and light chains do (immunoglobulins across
vertebrate genomes, 1008 paired rows), so the H-L signal is fed and helps the Fab internal
dock. The antigen and the antibody do not, no genome encodes both a coronavirus spike and
an immunoglobulin, so the antigen's paired block is empty and there is NO co-evolution
signal telling the model where the antigen meets the paratope. This is true of every
antibody-antigen complex (antibodies do not co-evolve genomically with their antigens), so
paired MSA cannot be the Ab-Ag lever for this target class, in this port or in the
reference. The reference OpenDDE's headline Ab-Ag accuracy must therefore rest on the
model's trained structural prior for paratope-epitope geometry (plus correct Fab assembly
and antigen fold), not on MSA co-evolution. The 0.011 gap in this port is a model/inference
gap in that structural-docking prior, not a missing-paired-MSA gap. This rules out the last
MSA-side lever; the remaining candidates are model-side (the structural-token refiner's
cross-chain conditioning, the diffusion sampler's docking-mode settings, or a weight/checkpoint
mismatch in the Ab-Ag-specific `opendde_abag.pt` routing) and are tracked in "Remaining".

**Protenix-v2 release gate (shared data path), re-run on card 2.** `scripts/release_gate.py
--model protenix-v2` on 7ROA: RMSD 1.428 A, TM 0.947, **PASS** (floor <=6.0/>=0.5). The
`build_complex_features` change is gated on `paired_a3ms` (None for Protenix-v2 / single-chain),
so the shared trunk/MSA featurization is byte-identical there; the gate confirms no
regression on the real device.

## Speed vs Boltz-2 on a single protein (2026-07-13)

OpenDDE is ~2.4x slower than Boltz-2 end-to-end on `examples/prot.yaml` (117-residue
single protein, default `recycling_steps`/`sampling_steps`, warm MSA, qb2 card 2):
**13.2 s vs 5.6 s** worker-side (22.9 s vs 14.9 s wall-clock incl. checkpoint load).
This is correct-by-design, not a bug, and breaks down as follows (all measured on the
real device, idle):

| stage | OpenDDE (r10) | Boltz-2 (r3) |
|---|---|---|
| trunk (Pairformer) | 7.98 s | 1.79 s |
| expand_and_refine (OpenDDE-only seam) | 0.47 s | — |
| diffusion (200 steps) | 3.69 s | 3.55 s |
| confidence | 0.11 s | 0.16 s |

The 7.6 s gap is ~71% the **recycling-step default** (OpenDDE 10 vs Boltz-2 3, per
`_resolve_recycling_steps` -> +5.43 s; running OpenDDE at 3 under-recycles the trunk
and mis-ranks the confidence ensemble, see `docs/protenix-recycling-revisit.md`), ~10%
the **wider Pairformer** (c_z=384 vs 128, measured 1.91x per recycle, not the paper's
theoretical ~9x pair-compute multiple — OpenDDE's ttnn Pairformer kernel is slightly
more efficient per pair-FLOP here and trunk cost isn't purely pair-compute), ~6% the
OpenDDE-only structural-token seam, and the rest diffusion+overhead (diffusion is
effectively equal between the two models, so it dilutes the e2e ratio). No fixable
inefficiency was found — no duplicate compute, no fusion missing on OpenDDE, no wrong
default. Numbers are only stable on an idle device (a contended card inflates Boltz-2
disproportionately; one boltz2@3 run hit 54.3 s vs the stable 5.6 s). Full attribution:
`~/.coworker/state/opendde-vs-boltz2-speed.md`.

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
- **Paired MSA: wired and ruled out as the Ab-Ag lever (P9).** The reference
  `MSAPairingEngine` species-pairing path is now wired into the predict path via
  ColabFold's pair endpoint (the repo's existing paired-MSA utility); `build_complex_features`
  stacks the paired block on top of the unpaired block. On 9dsg the paired search returns
  1008 species-paired heavy-light rows (consumed, Fab internal DockQ 0.48 -> 0.56) but zero
  antigen-antibody paired rows (no genomic co-occurrence), so Ab-Ag DockQ stays 0.011 / fnat 0
  across all 5 samples. Paired MSA cannot close Ab-Ag for any Ab-Ag complex (antibodies do
  not co-evolve genomically with antigens); the gap is a model/inference gap in the
  structural-docking prior, not a missing-input gap. See P9 above.
- **Antibody-antigen DockQ still 0.011 with MSA + paired MSA + best-of-5 -- and the
  reference reproduces it, so this is NOT a port bug (P11, decisive).** P8 cleared the
  multi-chain encoding (relp on both axes, `sym_id`/`entity_id`/`asym_id`/`residue_index`,
  and the MSA row/profile assembly) and P9 cleared paired MSA (the last MSA-side lever):
  the `MSAPairingEngine` species-pairing path is wired and consumed, but the antigen has no
  paired rows with the antibody (no genomic co-occurrence), so there is no Ag-Ab
  co-evolution signal for any Ab-Ag complex and A-H stays fnat 0. The antigen folds
  confidently and the Fab assembles (H-L fnat ~0.82), yet the antigen is mis-docked
  relative to the Fab across all 5 samples. P11 then ran the reference OpenDDE (CUDA) on
  the same 9dsg input + regime: the reference ALSO scores A-H 0.011 / fnat 0 (best-of-5),
  identical to the device, and scores global DockQ 0.83-0.86 on the standard Ab-Ag complex
  1ahw — so the opendde_abag checkpoint's Ab-Ag prior works on standard targets and 9dsg is
  specifically hard for it. The model-side candidates (structural-token refiner cross-chain
  conditioning, diffusion docking-mode/sampler settings, opendde_abag.pt routing/loading)
  are all exonerated for 9dsg: the reference, which has none of the port's wiring, fails
  identically, and the abag checkpoint is verified loaded (strict, 655.79M) and works on
  1ahw. Real **complex templates** are **not** the lever and are de-prioritized: the
  reference `TemplateEmbedder` masks every template pair feature to the same-chain block
  (`multichain_mask = asym_id[:,None] == asym_id[None,:]`), exactly as this port does, so
  templates carry only intra-chain geometry (reinforce a chain's own fold, already confident
  on 9dsg), not the cross-chain orientation. Real template *search* is also a separate
  multi-day data-pipeline port (tt-bio has no template-search stage; OpenDDE's is a
  509-line `search.py` + ~1900 lines of template parser/featurizer + a PDB template DB).
- **Nucleic-acid / ligand structural tokens**: `opendde_data.py` is protein-only; extending
  to DNA/RNA backbone/base splitting and ligand atom-tokens follows the identical pattern
  once a mixed-modality co-folding target is on the critical path.
- `--fast` + multi-card `--devices` ride the existing predict scheduler (memory
  `predict-multicard-already-exists` -- no new fanout path) and are now verified for
  OpenDDE (see the Status section). `--fast` only bf8s the trunk, which is not the
  bottleneck for OpenDDE (the structural-token diffusion stays bf16), so it is
  correctness-neutral but perf-neutral on this model; multi-card fanout is lossless and
  bit-identical to single-card at fixed seed.

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
- **Reference parity (P11, decisive):** the 0.011 is NOT a port bug. The reference OpenDDE
  (CUDA) run on the same 9dsg input + regime scores A-H DockQ 0.011 / fnat 0 (best-of-5,
  range 0.0107-0.0116) — identical to the device — and scores global DockQ 0.83-0.86 /
  fnat 0.87-1.0 on the standard Ab-Ag complex 1ahw (best-of-3), so the opendde_abag
  checkpoint's Ab-Ag prior works on standard targets and 9dsg is specifically hard for it.
  See "P11 — reference-vs-device Ab-Ag parity (decisive)".
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
  antigen in the paratope. P11 (2026-07-14) settled this: the reference OpenDDE reproduces
  the 0.011 / fnat 0 on 9dsg identically, so it is checkpoint/target reality for 9dsg, not
  a port defect — see "P11 — reference-vs-device Ab-Ag parity (decisive)".
- **Multi-chain MSA assembly: fixed (P8).** `build_complex_features` now merges per-chain
  MSAs the way the reference does (max_d-padded, column-concatenated) and computes
  `profile`/`deletion_mean` per chain (was: whole-merged-MSA, diluting every chain's
  columns with the other chains' GAP rows -- on 9dsg the Fab-heavy profile was ~99.5%
  GAP). Single-chain bit-exact; Protenix-v2 release gate still passes (1.417 A / TM 0.936
  on 7ROA). On 9dsg the Fab internal dock improves (H-L fnat 0.72 -> 0.81-0.825 across all
  5 samples) but the Ab-Ag interface stays DockQ 0.011 / fnat 0 (degenerate, unchanged) --
  the Ab-Ag gap is not the multi-chain encoding. Paired MSA was the suspected next lever
  and is now wired and ruled out (P9, next bullet) -- no Ag-Ab genomic co-occurrence, so
  paired MSA cannot close Ab-Ag for any Ab-Ag complex. Templates are not the lever (the
  reference masks template pair features to same-chain, intra-chain geometry only).
- **--fast + multi-card verification: done (2026-07-13, qb2, 4x Blackhole p300c).**
  `tt-bio predict examples/9dsg_abag.yaml --model opendde-abag --fast` runs end-to-end on
  one card and writes a valid CIF + results.json; confidence/DockQ match the non-fast
  baseline within single-sample variance (single-sequence, recycling 4 / sampling 40, seed
  0, card 3: ipTM 0.408 vs 0.334, pLDDT 0.791 vs 0.782, global DockQ 0.289 vs 0.264, Ab-Ag
  DockQ 0.103 vs 0.071, internal Fab DockQ 0.476 vs 0.457). `--fast` gives no wall-clock win
  here (109.8s vs 109.6s): it only bf8s the trunk, while OpenDDE's structural-token
  diffusion (always bf16) dominates the fold. Multi-card `--devices 0,1,2,3` fans 8
  protein targets (4x prot-117 + 4x trpcage-20, --fast, single-sequence, recycling 3 /
  sampling 80, seed 0) across 4 cards with per-target CIF + confidence bit-identical (md5
  match) to a single-card run, 24.8s vs 35.0s wall (1.41x; the fixed per-worker device-open
  cost, serialized by the host-wide open-lock, caps the win for this small a target set --
  it climbs toward the card-count ceiling as fold compute grows, per memory
  `predict-multicard-already-exists`). No --fast codepath or device-mesh gap (cf. the
  `esmc-embed-p300-mesh-gap` precedent): the predict scheduler sets the P300 mesh-graph
  descriptor per worker for OpenDDE too.
- **Paired MSA: done and ruled out (P9).** The `MSAPairingEngine` species-pairing path is
  wired into the predict path (ColabFold pair endpoint + `build_complex_features` paired
  block); it is consumed but does NOT close the 0.011 Ab-Ag DockQ (no antigen-antibody
  genomic co-occurrence, so no Ag-Ab co-evolution signal exists for any Ab-Ag complex).
  The remaining Ab-Ag candidates were model-side (the structural-token refiner's cross-chain
  conditioning, the diffusion sampler's docking-mode settings, or a weight/checkpoint
  mismatch in the Ab-Ag-specific `opendde_abag.pt` routing); P11 (2026-07-14) exonerated all
  of them for 9dsg by reproducing the failure in the reference — see "P11 — reference-vs-
  device Ab-Ag parity (decisive)". Real template **search**
  is a separate, de-prioritized lift: the dummy-template embedder already runs at `nt=4`,
  and real templates are not the Ab-Ag lever (reference masks template pair features to
  same-chain), though porting OpenDDE's HMMER/Kalign search pipeline + a PDB template DB is
  still a multi-day data-pipeline lift with no reusable search stage in tt-bio. Also not
  yet: nucleic-acid/ligand structural tokens. (--fast/multi-card verification is done; see
  the Status section.)
  OpenDDE is deliberately not in the README `--model`
  table yet: the port reproduces the reference (9dsg 0.011 == 0.011, P11), but the
  opendde_abag checkpoint does not solve the 9dsg Ab-Ag target, so the Ab-Ag differentiator
  is measured and reference-parity-verified but not a 9dsg win.

## P11 — reference-vs-device Ab-Ag parity (decisive, 2026-07-14)

The one hole in the parity story was that the device's 9dsg Ab-Ag DockQ 0.011 / fnat 0 was
measured against the crystal structure, never against the reference OpenDDE's own 9dsg
output — so "port bug" and "9dsg is just hard for this preview checkpoint" could not be
told apart. P11 closes it by running the REFERENCE OpenDDE (CUDA) on the exact same input
and settings and computing its DockQ against the same ground truth, with the same tool.

**Reference run.** OpenDDE @ `a0d5134` (the port pin commit) on a rented vast.ai RTX 4090,
`opendde_abag.pt` (verified loaded: "Loading from .../opendde_abag.pt, strict: True",
655.79M params), the same 9dsg input as `examples/9dsg_abag.yaml` (antigen A 196 / Fab
heavy H 248 / Fab light L 212), MSA via the reference's own `opendde msa` stage (ColabFold
MMseqs2 API, unpaired+paired A3M, N_msa 14553), templates off (the reference masks template
pair features to same-chain regardless, so templates carry no cross-chain signal), 10
recycles / 200 diffusion steps, best-of-5, seed 101, bf16. (bf16 after fp32 OOM'd at 200
steps on 24 GB; the Ab-Ag outcome is fnat 0 at both precisions in the reference, so the
memory concession does not affect the verdict.) DockQ via `scripts/opendde_dockq.py`
(DockQ==2.1.3) vs `examples/ground_truth_structures/9dsg.cif` — the SAME tool and ground
truth the device leg used.

**9dsg — reference vs device (A-H = the paratope-epitope interface):**

| leg | best-of | A-H DockQ (range) | A-H fnat | H-L DockQ | H-L fnat |
|---|---:|---|---:|---:|---:|
| device (this port) | 5 | 0.011 (0.0110-0.0113) | 0 | 0.497 | 0.825 |
| reference (CUDA) | 5 | 0.011 (0.0107-0.0116) | 0 | 0.41-0.49 | 0.79-0.83 |

Indistinguishable. The reference places the antigen at random relative to the Fab paratope
(fnat 0 in all 5 samples) exactly as the device does, and assembles the Fab internally
(H-L ~0.48, fnat ~0.82) exactly as the device does.

**Confirmatory second target — 1ahw (a standard SAbDab/PDB Ab-Ag complex), reference only:**
because the reference also failed 9dsg, the protocol calls for one standard Ab-Ag target
the paper's regime should handle, to confirm the checkpoint is not globally broken. The
reference scores global DockQ 0.83-0.86 / fnat 0.87-1.0 across all three native interfaces
(best-of-3, same regime) — in the paper's good-Ab-Ag regime and above it. So the
opendde_abag checkpoint's Ab-Ag structural prior works on standard targets; 9dsg is
specifically hard for it.

**Verdict.** NOT a port bug. The device faithfully reproduces the reference on 9dsg (both
0.011 / fnat 0); the opendde_abag preview checkpoint does not solve 9dsg but does solve
standard Ab-Ag complexes (1ahw 0.86). The model-side suspects (structural-token refiner
cross-chain conditioning, diffusion docking-mode/sampler settings, opendde_abag.pt
routing/loading) are exonerated for 9dsg: the reference, with none of the port's wiring,
fails identically. The device 1ahw leg (the symmetric cross-check on the second target) was
not run here — it needs a Tenstorrent card and this was a vast.ai/CPU task with no card
lease; it is the recommended follow-up, but the reference failing 9dsg identically already
settles port-bug-vs-checkpoint. Pharma framing: we do NOT reproduce OpenDDE's Ab-Ag
accuracy on 9dsg-class targets; we DO match the reference on 9dsg. Full numbers + GPU $
in `~/.coworker/state/opendde-9dsg-reference-dockq.md`.

