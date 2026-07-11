# OpenFold port — status & plan

Porting a classic **OpenFold** (AlphaFold2 reproduction) into tt-bio as a new
`--model openfold` predict target, per `port-bio-model-to-tenstorrent`.

## Model choice (researched 2026-07-11, not assumed)

**Pick: classic OpenFold — `aqlaboratory/openfold`, the AF2 monomer reproduction
(MSA + Evoformer + IPA structure module, pTM).**

Why this and not OpenFold3:

| Candidate | Family | Overlap with tt-bio | Verdict |
|---|---|---|---|
| **OpenFold3** (Oct-2025 preview) | AF3 diffusion (complexes, NA, ligands) | **redundant** with `protenix-v2` (already an AF3 reproduction) + `boltz2` | reject: duplicates an existing niche, and it is an explicit *research preview* not yet at AF3 parity (full retrain targeted early 2026) |
| **classic OpenFold** (AF2) | MSA + Evoformer + **IPA** structure module, deterministic | **none** — nothing in tt-bio has AF2 Evoformer+IPA | **pick**: fills a genuine gap, mature stable weights, still the reference open AF2 impl |

What it complementarily adds: tt-bio has `boltz2`/`protenix-v2` (AF3 diffusion,
MSA) and `esmfold2` (single-sequence, no MSA, no Evoformer). OpenFold uniquely
brings **classic AF2 behaviour**: full 48-block MSA Evoformer + a *deterministic*
IPA/frame structure module (not diffusion) — the canonical AF2 accuracy/behaviour
point real users still reach for, and architecturally distinct from every shipped
model. It has a pure-PyTorch inference path (Low-Memory Attention); the custom
CUDA kernels (DeepSpeed EvoformerAttention / FastFold) are optional — exactly what
a ttnn port needs.

Sources:
- Repo: https://github.com/aqlaboratory/openfold (AF2 reproduction; monomer/multimer/soloseq checkpoints via `scripts/download_openfold_params.sh`).
- OpenFold paper (retrained AF2, on-par accuracy): Nature Methods 2024, https://www.nature.com/articles/s41592-024-02272-z
- OpenFold3 preview (AF3-family, research preview): https://github.com/aqlaboratory/openfold-3 ; Nature news 2025-10 https://www.nature.com/articles/d41586-025-03546-y
- Benchmark landscape (OpenFold ≈ AF2 on monomers; AF3-family separate): https://academic.oup.com/bib/article/26/6/bbaf616/8351050

## Architecture (what must be ported)

`AlphaFold` (`_vendor/openfold/model/model.py`): InputEmbedder + RecyclingEmbedder
(+ optional TemplateEmbedder, ExtraMSAStack) → **EvoformerStack** (48 blocks:
MSA row-attn w/ pair bias, MSA col-attn, MSA transition, OuterProductMean,
TriangleMultiplication ×2, TriangleAttention ×2, pair transition) → **StructureModule**
(8× Invariant Point Attention + backbone frame update + sidechain torsions) →
AuxiliaryHeads (pLDDT, pTM/PAE, distogram). Recycling ×3-4.

## Reuse map (verified on device, not assumed)

**Pair-track heavy ops reuse directly** (`tt_bio/tenstorrent.py`, via
`protenix_weights` remaps) — these are the O(L²·c)/O(L³) hotspot:
`TriangleMultiplication`, `TriangleAttention`, `OuterProductMean`. All PCC-verified
(below). `remap_triangle_multiplication`/`remap_outer_product_mean` already target
OpenFold key names.

**NOT directly reusable (AF2 ≠ AF3 shape) — corrected after inspecting the code:**
- **Transitions.** AF2 `PairTransition`/`MSATransition` are plain **ReLU MLPs**
  (`LayerNorm → Linear(c→n·c) → ReLU → Linear(n·c→c)`); tt-bio `Transition` is a
  gated **SwiGLU** (AF3). Needs a small AF2 ReLU-MLP transition (net-new, tiny).
- **MSA track.** AF2 `MSARowAttentionWithPairBias` + `MSAColumnAttention` are **gated
  softmax attention** (reusing the `Attention` primitive + a pair-bias projection);
  tt-bio's `MSALayer`/`MSA` is the AF3 pair-weighted-averaging formulation — different
  op. AF2 MSA attention is net-new (can reuse sdpa + gating patterns from the
  verified `TriangleAttention`).

**Net-new (nothing in tt-bio has it):** AF2 **Invariant Point Attention (IPA)**
structure module + frame/quaternion updates (`utils/rigid_utils.py`). NOTE: the
Evoformer trunk dominates compute (ESMFold2 lesson: trunk ~67%, structure module
small), so IPA may stay a host reference with only heavy ops on device — decide by
profiling, per the playbook ("swap only the heavy ops").

Torch-side Linear/LayerNorm refs in `tt_bio/boltz2.py` + `tt_bio/reference.py`.

## Vendoring

Inference subset of `aqlaboratory/openfold` vendored to
`tt_bio/_vendor/openfold/` (model/, needed utils/, np/ minus relax, config). Training
code (Lightning, losses, optimizers, torchscript) not vendored. Imports rewritten to
`tt_bio._vendor.openfold.*`; the compiled CUDA attention-kernel import made lazy
(pure-PyTorch path). Data/MSA pipeline (`openfold/data/`) deferred to the end-to-end
phase. TODO before merge: license headers/NOTICE, `pyproject` deps, package-data.

## Parity status (per-module PCC > 0.98 gate; on-device, card 1, qb2)

| Component | Status | PCC | Note |
|---|---|---|---|
| Reference harness imports (vendored, CPU) | ✅ | — | lazy CUDA-kernel stub works |
| **TriangleMultiplication** (Outgoing+Incoming) | ✅ | **0.99999** | reuses shared ttnn block via `remap_triangle_multiplication`; AF2 biased linears **and** AF3 bias-free path both 0.99999 (`tests/test_openfold_triangle.py`) |
| **TriangleAttention** (Starting+Ending) | ✅ core | **0.99997** | reuses shared block; remap = strip `mha.` prefix (`tests/test_openfold_triangle_attn.py`). q/k/v bias-free; o/g gated bias = mechanical follow-up (same as tri-mul) for real weights |
| **OuterProductMean** | ✅ | **0.99999** | reuses shared block via `remap_outer_product_mean` (`tests/test_openfold_opm.py`). Note: parity needs normal-magnitude weights — `*0.1` underflows bf16 through the outer product (0.74), not a bug |
| **PairTransition / MSATransition** (ReLU MLP) | ✅ | **0.99999** | net-new `tt_bio.openfold.ReluTransition` (`tests/test_openfold_transition.py`); keys match reference directly, no remap |
| **MSA row attention + pair bias** | ✅ | **0.99998** | net-new `tt_bio.openfold.MSARowAttentionWithPairBias` (shared `_MSAGatedAttention` core) |
| **MSA column attention** | ✅ | **0.99997** | net-new `tt_bio.openfold.MSAColumnAttention` (same core, transposed, no bias) — `tests/test_openfold_msa.py` |
| Evoformer block (assembled) | ⬜ next | — | all block primitives above verified — compose + verify |
| **IPA structure module** | ⬜ | — | **net-new device code** |
| Heads (pLDDT/pTM/distogram) | ⬜ | — | keep on host (cheap), per playbook |
| End-to-end Cα-**RMSD** vs ground truth | ⬜ | — | release-gate: `examples/prot.yaml` (7ROA), Kabsch vs `examples/ground_truth_structures/prot.cif` |

### Resolved — biased linears (AF2 support added to shared block)

Classic AF2/OpenFold uses **biased** triangle linears (gating bias=1.0); the fused
shared block was written for the AF3-family (bias-free) and dropped them → full-bias
PCC was 0.670. Fixed by adding **state_dict-gated optional bias** to the shared block
(landed, small + additive):
- `WeightScope.__contains__` (tenstorrent.py) — was missing; `x in scope` silently
  broke via integer-index fallback.
- `TriangleMultiplication` — optional fused g_in/p_in bias + g_out/p_out linear bias,
  applied only when the bias keys are present.
- `remap_triangle_multiplication` — emits bias keys only when the source has them.

Result (on device): AF2 biased **0.99999**, AF3 bias-free path **0.99999** (byte-for-
byte the same gated-off code → protenix-v2 / Boltz-2 unaffected; recommend the
orchestrator re-run their release gate at merge as belt-and-suspenders). The other
reused blocks (TriangleAttention, OPM, MSA, transitions) likely need the same gated
bias — audit as each is verified.

## Next steps (resume here)

1. Assemble + verify one full Evoformer block (reused triangle mul/attn + OPM + `ReluTransition` + MSA row/col attention). Add gated o/g bias to TriangleAttention + the MSA `_MSAGatedAttention` core (same pattern as tri-mul) for AF2 real weights.
2. Build IPA structure module (net-new) + PCC-verify.
4. Vendor `openfold/data/` MSA pipeline; wire real weights (`openfold_weights.py`, protenix_weights style).
5. Wire CLI/worker (3 dispatch points: `main.py` Choice, `worker.py` load_model + predict_one; `release_gate.py` floor) + `--fast` + `--device_ids`.
6. End-to-end on device; Cα-RMSD vs ground truth; release_gate; unify README.

## Run recipe (qb2, card 1)

```
WT=/home/ttuser/.coworker/wt/tt-bio-openfold-port
export TT_MESH_GRAPH_DESC_PATH=/home/ttuser/tt-bio/env/lib/python3.12/site-packages/ttnn/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
TT_VISIBLE_DEVICES=1 PYTHONPATH=$WT /home/ttuser/tt-bio/env/bin/python tests/test_openfold_triangle.py
```
(`TT_MESH_GRAPH_DESC_PATH` is required standalone — qb2 misdetects its P150 as a
dual-chip P300; the predict/worker path sets this automatically.)
