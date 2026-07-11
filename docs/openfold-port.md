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

## Reuse map (shared tt-bio primitives — do NOT duplicate)

Reusable as-is (`tt_bio/tenstorrent.py`): `TriangleMultiplication`,
`TriangleAttention`, `OuterProductMean`, `PairWeightedAveraging`, MSA `MSALayer`/`MSA`,
`Transition`, `AttentionPairBias`, recycling embedders. Torch-side refs/Linear/LayerNorm
in `tt_bio/boltz2.py` + `tt_bio/reference.py`. Weight remaps: `tt_bio/protenix_weights.py`
(its `remap_triangle_multiplication` already targets **OpenFold** key names).

**Net-new (nothing in tt-bio has it):** the AF2 **Invariant Point Attention (IPA)**
structure module and frame/quaternion updates (`utils/rigid_utils.py`). This is the
main new device code.

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
| **TriangleMultiplication** (Outgoing+Incoming) | ✅ | **0.99999** | reuses shared ttnn block via existing `remap_triangle_multiplication`, zero new device code (`tests/test_openfold_triangle.py`) |
| TriangleAttention | ⬜ next | — | shared block; expect direct reuse |
| OuterProductMean / MSA attn / transitions | ⬜ | — | shared blocks |
| Evoformer block (assembled) | ⬜ | — | |
| **IPA structure module** | ⬜ | — | **net-new device code** |
| Heads (pLDDT/pTM/distogram) | ⬜ | — | keep on host (cheap), per playbook |
| End-to-end Cα-**RMSD** vs ground truth | ⬜ | — | release-gate: `examples/prot.yaml` (7ROA), Kabsch vs `examples/ground_truth_structures/prot.cif` |

### Open finding — biased linears (blocks real-weight parity)

The shared fused `TriangleMultiplication` **drops** the input-projection, gate, and
output-linear biases (fine for protenix-v2: AF3 triangle linears are bias-free).
Classic AF2/OpenFold uses **biased** linears with gating bias=1.0. Measured on device:
core (biases zeroed) PCC **0.99999**, but full biased PCC only **0.670**. → before
real-weight parity, the fused block must add optional bias (present-in-state_dict
gated, so protenix/boltz2 stay bit-identical). Same audit needed for every reused
block (TriangleAttention, OPM, MSA, transitions). This is the first real integration
task, not a blocker to the reuse approach.

## Next steps (resume here)

1. Extend shared blocks with optional bias (state_dict-gated) + re-verify protenix/boltz2 unaffected; re-run triangle full-bias PCC → expect >0.98.
2. PCC-verify TriangleAttention, OuterProductMean, MSA row/col attn, transitions (all shared blocks + remaps).
3. Assemble + verify one full Evoformer block.
4. Build IPA structure module (net-new) + PCC-verify.
5. Vendor `openfold/data/` MSA pipeline; wire real weights (`openfold_weights.py`, protenix_weights style).
6. Wire CLI/worker (3 dispatch points: `main.py` Choice, `worker.py` load_model + predict_one; `release_gate.py` floor) + `--fast` + `--device_ids`.
7. End-to-end on device; Cα-RMSD vs ground truth; release_gate; unify README.

## Run recipe (qb2, card 1)

```
WT=/home/ttuser/.coworker/wt/tt-bio-openfold-port
export TT_MESH_GRAPH_DESC_PATH=/home/ttuser/tt-bio/env/lib/python3.12/site-packages/ttnn/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
TT_VISIBLE_DEVICES=1 PYTHONPATH=$WT /home/ttuser/tt-bio/env/bin/python tests/test_openfold_triangle.py
```
(`TT_MESH_GRAPH_DESC_PATH` is required standalone — qb2 misdetects its P150 as a
dual-chip P300; the predict/worker path sets this automatically.)
