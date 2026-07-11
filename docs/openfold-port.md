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

**Device/host split (design decision).** Only the O(L³) **Evoformer trunk** goes on
device (`EvoformerStack`, verified). The **input/recycling embedders, the IPA structure
module, and the heads stay as the vendored host reference** — the trunk dominates
compute (ESMFold2: trunk ~67% vs head ~2%; Protenix built a full ttnn confidence head
that never paid off and was deleted), and the playbook says swap only the heavy ops.
This avoids reimplementing AF2's **Invariant Point Attention** (frame/quaternion point
attention, `utils/rigid_utils.py`) in ttnn — large, low-value. To be re-confirmed by
an e2e profile (measured, not assumed) before final sign-off.

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
| **Evoformer block (full, assembled)** | ✅ | **m 0.99988 / z 0.99983** | all 9 sub-blocks composed in AF2 order on device (residuals, shapes, tri-att ending) — `tests/test_openfold_evoformer_block.py` |
| **EvoformerStack** (N blocks + s-proj) | ✅ | **m 0.99986 / z 0.99979 / s 0.99985** | real device-trunk module `tt_bio.openfold.EvoformerStack`; 2-block chain + `s=Linear(m[...,0])` — `tests/test_openfold_evoformer_stack.py` |
| **EvoformerStack from real ckpt tree** | ✅ | **m 0.99984 / z 0.99979 / s 0.99984** | `openfold_weights.evoformer_stack_subs` scopes a real reference `EvoformerStack` state_dict (`blocks.{i}.pair_stack.*`, `msa_att_col._msa_att.*`, `linear.*`) → device stack; validates the real weight-load path — `tests/test_openfold_stack_realtree.py` |
| **REAL weights** (finetuning_ptm_1.pt, block 0) | ⚠️ | **m 0.99816 / z 0.97427** | loader handles the real `core.*` ckpt layout; o/g bias ruled out; z-track below gate under OOD random input — needs real-input recheck (see Real-weight findings) |
| Structure module (IPA) + heads | host | — | **host reference by design** (see device/host split) — not device-ported |
| Heads (pLDDT/pTM/distogram) | ⬜ | — | keep on host (cheap), per playbook |
| End-to-end Cα-**RMSD** vs ground truth | ⬜ | — | release-gate: `examples/prot.yaml` (7ROA), Kabsch vs `examples/ground_truth_structures/prot.cif` |

### Real-weight findings (finetuning_ptm_1.pt, block 0 on device)

Downloaded the real released pTM checkpoint and ran block 0 with **real weights**
(`tests/test_openfold_realweights_block.py`). Findings:
- **Checkpoint layout differs from the vendored reference.** Released `finetuning_*.pt`
  put the pair-track/OPM/transition ops under `evoformer.blocks.{i}.core.*` (older
  `EvoformerBlockCore`); current vendored main uses `pair_stack.*` + direct. The loader
  (`openfold_weights`) now accepts **both** layouts. Triangle layout is non-fused
  (matches the remap). So real AF2 weights load into the device trunk.
- **o/g gate/output bias is negligible** at real magnitudes: full real weights
  m 0.99816 / z 0.97427 vs o/g-zeroed m 0.99846 / z 0.96906 — the currently-dropped
  o/g bias is NOT the parity driver (de-prioritizes that follow-up).
- **Pair (z) track parity is 0.974 — below the 0.98 gate — under a random OOD input.**
  m-track 0.998 is fine. The random `*0.5` input is out-of-distribution for trained
  weights (real embedder activations have a specific scale), the likely cause; the
  O(L³) pair-track bf16 precision is the other candidate. **Must re-check with real
  embedder inputs (needs the data pipeline) before trusting/curing this** — don't
  assume it's a bug or that it's fine.

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

1. Integration (ESMFold2 `_SPEC`/adapter style): in the vendored `AlphaFold.forward`, replace the reference `EvoformerStack` with the device `tt_bio.openfold.EvoformerStack` (weights via `openfold_weights.evoformer_stack_subs`, now validated against the real ckpt tree); keep embedders/structure-module/heads on host. Add gated o/g bias to TriangleAttention + the MSA `_MSAGatedAttention` core for AF2 real weights.
2. Download real AF2 weights — **no `aws` on qb2; use anonymous HTTPS** (verified reachable):
   `curl -O https://openfold.s3.amazonaws.com/openfold_params/finetuning_ptm_1.pt` (~375 MB,
   the canonical pTM monomer checkpoint). Then vendor `openfold/data/` + get MSA for 7ROA
   (precomputed alignments or the ColabFold server tt-bio already uses); run reference e2e on
   CPU for the accuracy baseline, then device-trunk e2e. Weights are small/fast — the real
   remaining effort is the data/MSA pipeline, not the download.
3. Wire CLI/worker (3 dispatch points: `main.py` `--model` Choice, `worker.py` load_model + predict_one; `release_gate.py` floor) + `--fast` + `--device_ids`.
4. End-to-end on device (device trunk + host structure module); Cα-RMSD vs ground truth; release_gate; unify README; confirm the device/host split by profiling.

## Run recipe (qb2, card 1)

```
WT=/home/ttuser/.coworker/wt/tt-bio-openfold-port
export TT_MESH_GRAPH_DESC_PATH=/home/ttuser/tt-bio/env/lib/python3.12/site-packages/ttnn/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
TT_VISIBLE_DEVICES=1 PYTHONPATH=$WT /home/ttuser/tt-bio/env/bin/python tests/test_openfold_triangle.py
```
(`TT_MESH_GRAPH_DESC_PATH` is required standalone — qb2 misdetects its P150 as a
dual-chip P300; the predict/worker path sets this automatically.)
