# ABodyBuilder3 port

Resume anchor for porting ABodyBuilder3 (Exscientia, Apache-2.0) onto Tenstorrent
inside tt-bio. ABodyBuilder3 is a single, MSA-free, one-hot antibody Fv structure
module: 8 invariant-point-attention (IPA) update blocks + a single pLDDT head
(~7.2M params, fixed short length ~110-260 residues). First-pass target is the
one-hot `plddt-loss` checkpoint (NOT ABB3-LM/ProtT5 — stretch goal). ImmuneBuilder
(nanobody+TCR) and OpenMM relaxation are explicit follow-ons.

Status: the standard structure-module components + the IPA linear projections
are **ported to ttnn and on-device parity-verified** (PCC ~1.0 vs the reference
golden, real `plddt-loss` checkpoint, real 6yio H0-L0 inputs, card 0, bf16 weights
+ fp32 dest accumulation); a **hybrid end-to-end StructureModuleTT runs and is
parity-verified** (Cα-RMSD 0.016 Å, pLDDT PCC 0.99998 vs the PyTorch reference);
**CLI/predict integration is done** (`tt-bio predict --model abodybuilder3`,
lightweight dedicated path, writes a PDB). The IPA **attention** (scalar q.k AND
point) is a **documented ceiling**: it needs subtile head/point-dim reshapes
(head=12, head_dim=16, point coords=3, P_q/P_v=4/8) that ttnn stock ops cannot
express on device, so it runs host-side fp32 in the hybrid loop; a fully on-device
IPA needs a custom tt-metal point-attention kernel (kernel authoring is a separate
domain, deferred). See "The IPA attention ceiling".

## Identity (re-verified 2026-07-17)

- **What:** ABodyBuilder3 = Exscientia's antibody Fv structure predictor. An
  AF2-style structure module (8 IPA blocks + BackboneUpdate + AngleResnet +
  pLDDT head), MSA-free, one-hot input (aatype one-hot 21 + is_heavy 2 = c_s 23;
  relpos one-hot 2*64+1 + edge-chain 3 = c_z 132). Apache-2.0; vendored inference-only
  under `tt_bio/_vendor/abodybuilder3/` (training deps stripped, namespace rewritten).
- **Scale:** 7,223,437 params (~7.2M, single model). Confirmed from the checkpoint
  state_dict AND a vendored-StructureModule build (exact match).
- **Config** (`plddt-loss` ckpt, from upstream `params.yaml` + state_dict shapes):
  c_s=23, c_z=132, embed_dim=128, c_ipa=16, c_resnet=256, no_heads_ipa=12,
  no_qk_points=4, no_v_points=8, no_blocks=8, no_transition_layers=1,
  no_resnet_blocks=2, no_angles=7, trans_scale_factor=1 (NOT 10), epsilon=1e-7,
  inf=1e7, rotation_propagation=True, use_original_sm=True (AF2 bias + 2-layer
  angle-resnet blocks), use_plddt=True.
- **Max length:** not hard-capped (fully attention/convolutional on N). Natural
  antibody Fv ~110-260 residues; verified 229 (H=122/L=107) on 6yio H0-L0.
- **Weights:** PyTorch Lightning `.ckpt` on Zenodo record 11354577, converted to a
  plain `{state_dict}` `.pt` (Lightning baggage + `ml_collections` stripped) so the
  predict path needs no training deps. NOT on HF.
- **Input contract** (`string_to_input`): `single` (1,N,23), `pair` (1,N,N,132),
  `aatype` (N,), `residue_index` (heavy 0..H-1, light 500..500+L-1), `is_heavy` (N,).
  Output: `positions` (8,N,14,3 — one per block), `plddt` (1,N,50 logits), `frames`,
  `angles`, `states`. atom14->atom37 via `make_atom14_masks` + `atom14_to_atom37`.
- **Relaxation** (OpenMM `fix_pdb`) + ANARCI numbering stay host-side + OPTIONAL
  (mirrors Boltz-2). The harness writes PDB directly via `to_pdb` (no openmm dep).

## Critical finding (corrects the proposal's premise)

The proposal (NEXT_RESEARCH_9) assumed tt-bio's ESMFold2 port already has a
reusable AF2-style IPA primitive. **It does not.** tt-bio's `esmfold2` /
`esmfold2_hf` is a **diffusion** folder (DiffusionModule/DiffusionTransformer,
OpenFold3-family) — there is no `InvariantPointAttention` / AF2 structure module
anywhere in `tt_bio/` or `tt_bio/_vendor/`. ABodyBuilder3 *is* an AF2 structure
module, so **IPA must be ported from scratch**; the effort is MEDIUM (not LOW).
This is the long pole of the port.

## What is ported to ttnn + parity-verified

The standard (non-novel) structure-module components and the IPA linear
projections, PCC-gated vs the reference golden (real `plddt-loss` checkpoint, real
6yio H0-L0 inputs, card 0, bf16 weights + fp32 dest accumulation). Bar = PCC > 0.98
per component. **Measured numbers:**

| component | PCC vs golden |
|---|---|
| IPA LayerNorm (c=128) | 1.00000 |
| BackboneUpdate (Linear 128→6) | 1.00000 |
| StructureModuleTransition (1 layer + LN) | 0.99999 |
| AngleResnet (linears on device; size-2 normalize = host fp32 tail) | 1.00000 |
| pLDDT head (LN→2×Lin→50 bins) | 0.99999 |
| IPA linear projections: q, kv, qp, kvp, pair bias b | 1.00000 |

Module: `tt_bio/abodybuilder3.py`. Tests: `tests/test_abodybuilder3_ttnn_components.py`
(5 passed) + `tests/test_abodybuilder3_parity.py` (the IPA-projections and the
end-to-end hybrid tests; full ABB3 suite 9 passed, 1 skipped — the skipped one is
the fully-on-device IPA, behind the ceiling).

## The IPA attention ceiling (empirically confirmed, not a guess)

The IPA **attention** — both the scalar q.k AND the point-attention — is NOT
portable to ttnn stock ops on device. Both need subtile trailing-dim reshapes that
ttnn cannot express:

- **scalar q.k** needs a head reshape to `[1,N,12,16]` (head=12, head_dim=16, both
  subtile). `ttnn.reshape` re-tiles and scrambles the layout (the on-device q.k
  matmul PCC dropped to 0.05 vs the reference internals); `nlp_create_qkv_heads`
  scrambles non-32 head_dim (the documented ttnn-tile-alignment hazard).
- **point-attention** needs subtile point coords (3) / P_q,P_v (4,8)
  broadcast/sum/reshape and front-padding. `ttnn.pad` rejects front-padding of
  subtile dims on device (`TT_FATAL ... on device tile padding does not support
  front padding`); subtile broadcast/sum over the 3/4/8 dims are unsupported
  (`TT_FATAL ... Invalid subtile broadcast type`).

A full on-device IPA therefore needs a **custom tt-metal point-attention kernel**
(kernel authoring is a separate domain, out of scope for this stock-op port). The
attention math stays host-side fp32 in the structure-module loop; only the
projections run on device. Documented in the `IPALayer` docstring and the skipped
`test_abodybuilder3_ipa_on_device` test's skip message.

The bisect oracle is `scripts/abb3_ipa_internals.py` (runs the reference IPA
block 0 on the golden inputs and dumps every intermediate — q, k, v, q_pts, k_pts,
v_pts, b, a_scalar, pt_att, a, attn, o, o_pt, o_pt_norm, o_pair, cat, delta;
self-consistency delta PCC vs golden ipa_delta = 1.0) + `scripts/abb3_ipa_bisect.py`
(checks each on-device sub-step against it). Every on-device sub-step that IS
expressible (the projections, the scalar q.k once reshaped on host, the rigid-apply,
the point-attention, the aggregation) validates PCC 1.0; the wall is specifically
the on-device *reshape/broadcast/pad* over the subtile dims, not the math.

## Hybrid end-to-end StructureModuleTT (parity-verified)

`tt_bio/abodybuilder3.py` `StructureModuleTT` runs the full 8-block structure
module as a hybrid:

- **On device** (bf16, fp32 dest acc; PCC ~1.0 component-by-component): the input
  embeddings (`linear_in_node`, `linear_in_edge`), the IPA linear projections
  (q, kv, qp, kvp, pair bias b) + `linear_out`, the post-IPA LayerNorm, the
  Transition, BackboneUpdate, the AngleResnet linears, and the pLDDT head.
- **On host fp32** (the documented ceiling): the IPA rigid-apply + scalar/point
  attention + value aggregation, the quaternion backbone compose
  (`compose_q_update_vec`), and `torsion_angles_to_frames` +
  `frames_and_literature_positions_to_atom14_pos` (rigid compositions with
  residue-constant lookup tables — the same "cheap host code" boundary ESMFold2
  uses for its confidence head).

The host tail is the exact reference math (reused from the vendored
StructureModule), so end-to-end parity is governed only by the bf16 device
projections — which are PCC 1.0. **Measured on 6yio H0-L0 (N=229), card 0:**

| metric | hybrid vs PyTorch reference |
|---|---|
| Cα-RMSD (Kabsch) | 0.0164 Å |
| pLDDT PCC | 0.99998 (mean 93.92 vs ref 93.93) |

Gate: `tests/test_abodybuilder3_parity.py::test_abodybuilder3_hybrid_end_to_end`
(Cα-RMSD < 0.5 Å, pLDDT PCC > 0.98) — PASSES. Reproduce:
`scripts/abb3_predict.py` (runs the hybrid + the reference, prints Cα-RMSD/pLDDT,
writes a PDB).

## CLI/predict integration (done, verified end-to-end)

`--model abodybuilder3` is added to `tt_bio/main.py`'s `predict` command. It rides
a **lightweight dedicated path** (`_run_abodybuilder3_predict`), NOT the Boltz
scheduler (which is MSA/complex-oriented; ABodyBuilder3 is a simple single-Fv
model). It parses a paired heavy+light Fv from a FASTA (two records id'd H/L or
heavy/light) or YAML (two protein entries id heavy/light), runs
`predict_abodybuilder3`, and writes one PDB per input (pLDDT in B-factors).

Verified end-to-end on real weights, real device (card 0):
```
tt-bio predict 6yio.fasta --model abodybuilder3 --out_dir out/
# -> 6yio — H=122 L=107 residues, mean pLDDT 93.92, wrote PDB -> out/6yio.pdb
```

## Perf (warm single-card, 2026-07-17)

`scripts/abb3_perf.py` (2 warmup runs incl compile, then 5 warm iters, N=229, card 0):

| path | ms/structure | structures/s |
|---|---|---|
| hybrid (this port, warm) | 415.7 | 2.41 |
| reference (CPU fp32, N=229) | 225 | 4.44 |

The hybrid is **slower than the CPU reference** — by design at this stage: the
attention is host-side fp32 (the ceiling), and the loop does ~6 host↔device
push/readback transfers per block × 8 blocks, so readback/compile overhead
dominates the small per-block compute. This is an honest perf number, not a win:
the on-device pieces are correct and parity-verified, but the host attention tail
+ transfer overhead caps throughput below the CPU baseline until the custom
point-attention kernel lifts the ceiling (the perf-relevant piece). No
release-gate treatment yet.

## Remaining

- **Custom tt-metal point-attention kernel** to lift the IPA-attention ceiling — the
  only blocker for a fully-resident on-device IPA (and the perf-relevant piece).
  Until then the hybrid loop runs the attention host-side fp32 with the
  projections + standard components on device.
- **Resident recurrence**: keep the pair state `z` on device across the 8 blocks
  (mirror ESMFold2's resident-trunk pattern) once the attention is on device, to
  remove the per-block z readback.
- **OpenMM relaxation + ANARCI numbering**: host-side + OPTIONAL (mirrors
  Boltz-2); not wired into the predict path yet.
- **ImmuneBuilder (nanobody + TCR)**: explicit follow-on (same AF2 structure
  module, narrower input contract).
- **README fold-in**: ABodyBuilder3 is deliberately not in the README `--model`
  table yet — the port is parity-verified and the CLI works, but the hybrid is
  slower than CPU (the attention ceiling), so it is not a perf win until the
  custom kernel lands.

## Parity bar (the standard tt-bio gate, restated)

PCC vs reference on real weights, not synthetic inputs alone. No fabricated
numbers — the PCC values above are measured on card 0 against the golden, and the
Cα-RMSD/pLDDT end-to-end numbers are measured against the PyTorch reference on
the same real checkpoint + 6yio H0-L0 input. End-to-end parity (Cα-RMSD 0.016 Å)
is achieved within the bf16 projection precision.

## How to reproduce the parity numbers
```
source /home/moritz/tt-bio/env/bin/activate
cd /home/moritz/.coworker/wt/tt-bio-antibody-structure-port
TT_VISIBLE_DEVICES=0 \
ABB3_GOLDEN=/tmp/abb3_cache/abb3_golden.pkl \
TT_BIO_CACHE=/tmp/abb3_cache \
ABB3_CKPT=/tmp/abb3_zenodo/plddt-loss/best_second_stage.ckpt \
PYTHONPATH=. python -m pytest tests/test_abodybuilder3_parity.py \
    tests/test_abodybuilder3_ttnn_components.py -q
# Golden (re)capture: python scripts/abb3_golden.py --out /tmp/abb3_cache/abb3_golden.pkl
# End-to-end + PDB:    PYTHONPATH=. python scripts/abb3_predict.py
# Perf (warm):        PYTHONPATH=. python scripts/abb3_perf.py
```
