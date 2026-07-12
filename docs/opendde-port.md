# OpenDDE port

Resume anchor for porting OpenDDE onto Tenstorrent inside tt-bio. OpenDDE's
compute graph is Protenix-v2's (already fully ported in `tt_bio/tenstorrent.py` +
`tt_bio/protenix.py`) plus exactly one novel module, so the port reuses that
entire ttnn stack and adds the one block. Status: identity + architecture +
redundancy **measured** (not assumed); the novel block (`StructuralTokenExpander`)
is **ported to ttnn and on-device parity-verified** (PCC ≥ 0.99999 vs the Phase-0
golden, qb2 card 0); the real `opendde.pt`/`opendde_abag.pt` checkpoints are **pulled**,
the **weight remap + expander→refiner pipeline assembly are done and on-device verified
with real weights**, and the `--model opendde`/`opendde-abag` CLI entry points are wired.
End-to-end co-folding remains blocked on the structural-token tokenizer port (+ two small
shared-stack deltas) — see "Remaining".

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

## Remaining (before a full residue→structure fold)

`OpenDDE.fold()` raises with this list until each lands:

1. **Structural-token tokenizer/featurizer** (`opendde/data/tokenizer.py` + `featurizer.py`,
   not ported): emits `structural_token_index`, `atom_to_structural_token_idx`,
   `atom_to_structural_tokatom_idx`, subtoken roles, parent/prev/next adjacency, and the
   structural frames + rep-atom masks that both the expander and the structural-axis
   diffusion/confidence consume. This is the one real data-pipeline chunk; the compute is
   already ported.
2. **`c_z`-parametric shared `Trunk`** — `protenix.Trunk` hardcodes `C_Z=256`; OpenDDE is
   `c_z=384` / 12 triangle heads. Parametrize (the primitives already read channel dims
   from weights; only the reshapes + head counts are hardcoded).
3. **Diffusion-conditioning `z_trunk` branch** — add the
   `diffusion_module.diffusion_conditioning.*_z_trunk` term OpenDDE has over Protenix-v2.

Then: `--fast` + multi-card `--devices` ride the existing predict scheduler (memory
`predict-multicard-already-exists` — no new fanout path), and one unified README section
(user-facing only; internals linked here).

CLI: `tt-bio predict --model opendde` / `--model opendde-abag` are **wired** as recognized
entry points (`tt_bio/main.py`); until (1)-(3) land they raise a clear "co-folding pending
the structural-token tokenizer" message rather than folding. Co-folding → `predict`, not
`gen` (no design mode in the release).

## Accuracy gate

- **Metric:** Ca-RMSD vs ground truth (`scripts/release_gate.py` method) for
  co-folding, plus **DockQ on antibody-antigen** complexes (the whole reason to
  add the model, and what the paper reports). No designability gate — that path
  does not exist in the release.
- **Stochasticity:** diffusion is seed-stochastic and the repo warns outputs are
  not reproducible across releases, so parity is per-target Ca-RMSD/DockQ within
  sample variance (as for Boltz-2 / Protenix-v2), not bit-exact.

## Status

- Identity, measured redundancy, architecture mapping, gate choice: **done.**
- Novel-block torch reference + golden captured (`scripts/opendde_structtoken_ref.py`).
- **ttnn `StructuralTokenExpander` ported and on-device parity-verified**
  (`opendde_structtoken_parity.py`, PCC ≥ 0.99999).
- **Real checkpoints pulled** — `opendde.pt` + `opendde_abag.pt` (2.6 GB each) on qb2 in
  the HF cache.
- **Weight remap + pipeline assembly done and on-device verified with real weights**
  (`route_opendde_weights`, `OpenDDE`, `scripts/opendde_assembly_verify.py`): full routing
  coverage, shared subtree Protenix-v2-key-identical, expander→refiner seam runs finite on
  card 0. Real measured results, not estimates.
- **CLI entry points wired** (`--model opendde` / `opendde-abag`), guarded pending the
  tokenizer.
- End-to-end co-folding + Ca-RMSD/DockQ: **not yet** — blocked on the three prerequisites
  above (chiefly the structural-token tokenizer port). No reference (CUDA) OpenDDE run is
  available here, so accuracy numbers are deferred, not fabricated.

**Next action:** port OpenDDE's structural-token tokenizer/featurizer (remaining item 1),
then make `Trunk` `c_z`-parametric (item 2) + add the diffusion `z_trunk` branch (item 3);
`OpenDDE.fold()` names exactly these three in its error.
