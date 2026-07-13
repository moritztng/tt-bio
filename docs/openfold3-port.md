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

- [x] **P0 reference harness**: `scripts/of3_golden.py` -- CPU venv (`/tmp/of3-venv`:
      torch + ml-collections + gemmi + biotite; rdkit/kalign only needed for the JSON data
      pipeline, not the trunk modules), loads real `of3-p2-155k.pt`, captures golden
      activations for PairFormerBlock-0 and the full 48-block stack to `~/of3_ref_out.pkl`.
      Inputs are deterministic seeded tensors of the config shapes (N=37) -- sufficient for
      component PCC (the device gets the identical tensor). Full JSON-to-feats real-input
      golden reuses the P1 vendor pipeline below.
- [x] **P1 vendor** host-side data pipeline (JSON query → feats dict, CCD/ligand,
      relpos, token bonds) into `tt_bio/_vendor/openfold3/` — inference-only, strip
      training/lightning/losses/optimizers. No runtime git-clone, no sys.path shims.
      Done on `wk/tt-bio-openfold3-port-p1` — see "P1 status" below.
- [x] **P2** `tt_bio/openfold3_weights.py`: remaps OF3 checkpoint keys onto the
      proven protenix-v2 primitive layout. OF3 is the same AF3 family, so each function
      renames OF3 keys to protenix key names and delegates to `protenix_weights` (zero
      duplicated remap logic). Verified three ways: (a) byte-lossless value conservation,
      16/16 (every target tensor equals the exact source tensor / correct concat); (b) it
      produces the exact 53-key-per-block tt-bio Pairformer layout; (c) on-device (card 0)
      PairFormerBlock-0 `s_pcc=0.99985`. Single-block pair-path `z_pcc=0.894` on adversarial
      random input -- a full-bf16 CPU run of the same reference block/input already falls to
      0.977, and the device adds the shared-primitive bf16-kernel error (identical to what
      protenix runs). Definitive stack z-gate (>0.97 on the settled distribution, protenix
      own gate) pending: blocked by the device-open-lock fd-leak (see status log), not by
      any remap defect.
- [x] **P3 PCC gate, smallest first** (tick 4 status -- see status log for detail):
      TriangleMultiplication/TriangleAttention/AttentionPairBias/one Pairformer block:
      DONE, `s_pcc=0.99985`. 48-block trunk: DONE but honestly OPEN -- real-distribution
      golden captured, `s_pcc=0.996` (remap solid), `z_pcc=0.649` (real, checkpoint-specific
      device-precision gap, not an artifact; `xfail`). MSA block: remap DONE
      (`remap_msa_block`/`remap_msa_module`), gate DONE but same open z-track gap
      (`m_pcc=0.99999`, `z_pcc=0.71-0.75`; `xfail`). Still not started: template, atom
      encoder, InputEmbedder itself (only used as a golden source so far, not ported to
      device), DiffusionConditioning, DiT block, atom decoder, confidence heads. Threshold
      PCC > 0.98 per module vs real-weight golden.
- [~] **P4 assemble** `OpenFold3` class (`load_from_checkpoint` + `fold`), EDM sampler.
      Trunk forward (`run_trunk`) assembled + PCC-gated this tick (P8 tick 13); the
      `OpenFold3` class shell, `fold()`, and the EDM sampler are still TODO (DiffusionModule
      internals ported by the parallel `tt-bio-openfold3-port-p8-dit` stream).
- [ ] **P5 integrate**: `--model openfold3` in CLI/worker/scheduler, `--fast` block-fp8,
      `--devices` fanout — consistent with predict precedent. ONE unified README --model
      table row (no parallel prose block; bio audience, no ttnn/driver detail).
- [ ] **P6 HARD GATE**: `examples/prot.yaml` end-to-end → parsed output → vs-ground-truth
      Kabsch RMSD via `scripts/release_gate.py`'s method. No fabricated numbers.

Closest existing model to diff against at every step: **protenix.py** (same v2 atom
transformer + EDM + confidence structure). Start remap from `protenix_weights.py`.

## Status log

- 2026-07-12 (tick 6 = P6, branch `wk/tt-bio-openfold3-port-p6` off main @519a2e4): **Assets
  rebuilt on qb2 (qb1 down) + MSA block-0 mechanism bisected (thread 3). No new components
  ported this tick -- the component-port thread (1) needs assembly work that can't be
  PCC-gated in one bounded turn; flagged precisely for P7 below.**
  1. **Assets were on qb1, which is down.** Rebuilt on qb2 from public sources: real ckpt
     `~/of3-weights/of3-p2-155k.pt` (curl from `s3://openfold3-data/...`, 2.29GB), ref clone
     `/tmp/of3-ref` (shallow `github.com/aqlaboratory/openfold-3`), CPU ref venv `/tmp/of3-venv`
     (`pip install openfold3==0.4.3` + torch 2.13). Regenerated `~/of3_ref_out.pkl` via
     `scripts/of3_real_golden.py` (reproduces the P5 numbers: input_embedder z_init std 17.64,
     pairformer_stack_real z_out std 29.17 / prefix47 std 139.5, msa_block0_real z_out std
     270.24). The existing pairformer + MSA device tests are unblocked again on qb2.
  2. **Key architectural finding for the port strategy:** OF3 is NOT a pure weight-remap onto
     `protenix.Trunk`. Protenix-v2 instantiates `Pairformer(48, 32, 8, 24, 16, ...)` with
     `Trunk.C_Z=256` and `no_heads_pair=8`; OF3 uses `c_z=128`, `no_heads_pair=4`
     (`tests/test_openfold3_pairformer.py` `_DIMS=(32,4,24,16)`). Same AF3 family, different
     hyperparameters. So the OF3 assembly must compose the shared `tenstorrent` primitives
     with OF3-specific dims (as the existing pairformer/MSA tests already do), reusing only
     the per-primitive `protenix_weights` remaps + `protenix.py`'s component *structure* as a
     reference -- NOT `protenix.Trunk`/`DiffusionModule`/`ConfidenceHead` instances verbatim.
     This is the "vendored-model + built-in flag" style (like `boltz2.py`), not a remap-onto-
     Protenix shortcut.
  3. **Thread 3 -- MSA block-0 z gap bisected (DECISIVE).** `scripts/of3_msa_bisect_cpu.py`
     runs the reference `MSAModuleStack` block-by-block AND block-0 sub-op-by-sub-op
     (opm_first=True order: z+=OPM; m+=PWA; m+=transition; z=pair_stack) on real ubiquitin,
     with bf16 controls. Result:
     - z std: init 17.64 -> OPM 18.17 -> PWA 18.17 -> transition 18.17 -> **pair_stack
       270.13**. The ENTIRE ~15x single-block amplification is in the pair_stack
       (tri_mul_in/out + tri_att_start/end + pair_transition). OPM/PWA/transition leave z
       at ~18. P5's "OPM ~15x amplification" hypothesis was close in magnitude but wrong on
       the sub-op: it is the pair_stack, not OPM, that amplifies.
     - CPU bf16 controls: full-bf16 stack z_pcc = **0.9998** every block; storage-only =
       1.0; block-0 pair_stack sub-op bf16-vs-fp32 z_pcc = 0.9998 (others 1.0). So CPU-bf16
       tracks this block essentially perfectly -- the device z_pcc=0.708 loss
       (tests/test_openfold3_msa.py) is a DEVICE-pair_stack-compute precision issue at large
       activation magnitude (z std 18->270 within one pair_stack call), NOT a bf16-only
       artifact and NOT the pairformer's final-block cancellation. Distinct mechanism, now
       localized to the device's pair_stack compute path -- confirms P5's "different root
       cause" call. Result + sub-op goldens saved to `~/of3_msa_bisect.pkl` for the device
       sub-op bisect (P7).
  **NEXT (P7):** (a) port the remaining components so `OpenFold3.fold()` runs -- build
  `tt_bio/openfold3.py` composing the shared `tenstorrent` primitives with OF3 dims
  (InputEmbedder device leg via the shared atom encoder, template embedder, atom enc/dec,
  DiffusionModule, confidence heads), PCC-gating each vs the regenerated real golden; the
  OF3 key subtrees are mapped (input_embedder.atom_attn_enc.atom_transformer 3-block AdaLN,
  template_embedder.template_pair_stack 2 pair-only blocks reuse `remap_msa_pair_stack`,
  diffusion_module.{diffusion_conditioning, atom_attn_enc, diffusion_transformer 24-block
  DiT, atom_attn_dec}, aux_heads.{pairformer_embedding, pae, pde, plddt, distogram,
  experimentally_resolved}); (b) `examples/prot.yaml`/ubiquitin end-to-end -> parsed
  structure -> vs-ground-truth Kabsch Cα-RMSD (the merge gate, replacing raw stack-z); (c)
  device sub-op bisect of the MSA pair_stack (which of tri_mul/tri_att/transition loses
  precision at the 18->270 magnitude jump), using `~/of3_msa_bisect.pkl`'s sub-op goldens.

- 2026-07-12 (tick 7 = P6 cont.): **Component-port reusability matrix scoped + input-embedder
  atom-encoder sub-outputs captured for the device PCC gate. No device component landed this
  tick -- writing + PCC-debugging a full OF3 device component in one bounded turn risks
  shipping unverified code (violates the PCC-gate-each-piece rule); the verified increment
  this tick is the golden extension + the reusability matrix that makes the device ports
  fast + safe next tick.**
  1. **Reusability matrix (verified against the OF3 config + checkpoint + protenix.py):**
     - **Reusable for OF3 with key remap (dims match exactly):** `protenix.DiffusionModule`
       (24 DiT blocks, 16 heads, head_dim=48, sigma_data=16, NQ=32/NK=128/PAD_LEFT=48 -- all
       identical to OF3's `diffusion_transformer`/`atom_attn_enc` config), its `AtomTransformer`
       (3 blocks, 4 heads, head_dim=32, c_atom=128, c_atom_pair=16), and the DiT
       `AdaLN`/`AttentionPairBias`/`ConditionedTransition` primitives. OF3 c_token=768 for the
       DiT (protenix same). Only the key *names* differ (OF3 `diffusion_module.atom_attn_enc`
       vs protenix `diffusion_module.atom_attention_encoder`; OF3 `conditioned_transition`
       vs protenix `conditioned_transition_block`); a `remap_for_protenix_diffusion` handles it.
     - **NOT reusable, needs OF3-specific device code:** (i) the **InputEmbedder atom
       featurization** -- OF3 `RefAtomFeatureEmbedder` uses separate per-feature linears
       (linear_ref_mask/element(119)/atom_chars(256)) + a p_lm path with linear_ref_offset/
       inv_sq_dists/valid_mask terms, vs protenix's `AtomFeaturization` combined W_f(cat) +
       W_d/W_invd/W_v (different feature dims: element 119 vs 128, different p_lm terms); the
       math is NOT equivalent. (ii) the **trunk** -- OF3 c_z=128/no_heads_pair=4 vs protenix
       C_Z=256/no_heads_pair=8 (already handled by the existing OF3 pairformer/MSA tests
       composing `tenstorrent.Pairformer` directly). (iii) the **confidence heads** -- OF3
       `aux_heads.{pairformer_embedding,pae,pde,plddt(1150,384),distogram,
       experimentally_resolved}` vs protenix `confidence_head.*` (different head layout).
     - Net: the DiffusionModule + atom-transformer/DiT are the biggest reusable chunk
       (~the heaviest compute); the InputEmbedder featurization, trunk glue, and confidence
       heads need OF3-specific code reusing the shared `tenstorrent` primitives
       (`AdaLN`, `AttentionPairBias`, `Transition`, `PairformerLayer`).
  2. **Golden extended** (`scripts/of3_real_golden.py`): now also captures
     `input_embedder_atom_enc_real` = the InputEmbedder's atom-encoder sub-outputs (ai
     (76,384) pre-s_inputs-concat, ql (601,128), cl (601,128), plm (19,32,128,16); N_atom=601,
     N_token=76) on real ubiquitin. This lets the device InputEmbedder leg be PCC-gated at
     sub-component granularity (atom encoder vs the linear_s/z_i/z_j/relpos/token_bonds glue)
     rather than only at the combined s_input/s/z -- the same incremental-gate discipline the
     pairformer/MSA ports used.
  **NEXT (P7, precise):** build `tt_bio/openfold3.py` in this order, PCC-gating each vs the
  golden: (1) InputEmbedder device leg (OF3-specific featurization + `AtomTransformer`-style
  3-block windowed attention reusing `tenstorrent.AdaLN`/`AttentionPairBias`, glue linears) --
  gate ai then s_inputs then s/z vs `input_embedder_atom_enc_real`/`input_embedder_real`;
  (2) Trunk assembly (reuse the existing `remap_pairformer_stack`/`remap_msa_module` +
  `tenstorrent.Pairformer`/MSA primitives, OF3 dims, + template embedder via
  `remap_msa_pair_stack` + `PairformerLayer`, + the linear_z/linear_s cycle glue) -- gate
  vs `pairformer_stack_prefix47`/`pairformer_stack_real`; (3) DiffusionModule (reuse
  `protenix.DiffusionModule` via `remap_for_protenix_diffusion`, OF3 c_z=128 pair-z) --
  capture a single-denoise-step golden in the venv, gate vs it; (4) ConfidenceHead
  (OF3-specific `aux_heads`) -- gate vs an `aux_heads` golden; (5) `OpenFold3.fold()` +
  EDM sampler + `examples/prot.yaml`/ubiquitin end-to-end -> Kabsch Ca-RMSD (the merge gate).

- 2026-07-12 (tick 5 = P5): **Bisected the 48-block stack. The z_pcc gap is NOT uniform
  and NOT a device-precision mystery: it is a single-block catastrophic cancellation in
  the FINAL block, and no bf16 implementation (device OR CPU) can clear a >0.97 raw-z gate
  against it. Tick 4's "device compounds a real gap beyond bf16" framing was half right;
  the missing piece is WHERE (last block) and WHY (cancellation), which reframes it from
  "open device bug" to "wrong metric for a cancelling final block."**
  1. **Captured the full per-block reference trajectory** (`scripts/of3_bisect_cpu.py`,
     scratch; drives the real featurized ubiquitin through the reference `InputEmbedder`
     then runs `PairFormerStack` block-by-block via its own `_prep_blocks`). Two things
     fell out immediately:
     - The z-track is **well-conditioned**: `z_init` std 17.6, final std **29.8**. The
       "~1.8e4 residual" tick 4 blamed for the z gap is the **S-track** magnitude (s_out
       std 18497) -- a red herring for z. The z stream never blows up.
     - z std **climbs 30→~226 over blocks 0-42, then the final block collapses it back to
       ~30** with a near-total cancelling update (`||dz||/||z_prev|| = 0.97`, vs ~0.03-0.07
       for interior blocks). That is a difference-of-large-numbers: two ~std-134 quantities
       subtract to a ~std-30 result, amplifying any rounding error ~5-10x (100x on the
       fully-cancelled components).
  2. **CPU bf16 controls (per block)**: a full-bf16 stack holds z_pcc ≥ 0.998 through
     block 39 and **only drops at the very end -- 0.9947 (block 46) → 0.9035 (block 47)**.
     A storage-only control (fp32 compute, z/s rounded to bf16 between blocks) barely moves
     (0.9941 at block 47) -- so it is bf16 *compute* in the cancelling block, not
     inter-block storage. **CPU-bf16 itself cannot pass >0.97**; this is not a device-only
     problem.
  3. **On-device bisect** (`scripts/of3_bisect_device.py`; tt-bio `Pairformer(48)`
     run block-by-block from the same real `z_init`): cumulative z_pcc **holds ≥ 0.975
     through block 46, then drops to 0.658 at block 47**; s_pcc stays 0.9965 throughout.
     An **isolated block-47 run fed a PERFECT fp32 reference input** still only reaches
     **z_pcc=0.922** (device) / 0.903 (CPU-bf16) -- the block's own bf16 compute in the
     cancellation regime is the floor. A **per-position LayerNorm** of the final z (how z
     is actually consumed downstream) does **not** recover it (0.667 device, from 0.658) --
     the cancellation destroys signal LN cannot restore, so "just gate the normalized z"
     is not a fix. Trying to run the stack fully fp32 on device is blocked anyway: the
     SDPA path in triangle/pair attention hard-requires bf16/bf8 inputs.
  4. **Fixability verdict: not fixable within the bf16 regime the port runs in, and it is
     not a bug.** The remap is byte-correct (block-0 s_pcc=0.99985) and the device tracks
     the reference to z_pcc ≥ 0.975 through 47 of 48 blocks -- the port is correct. The
     failure is an intrinsic bf16-conditioning limit of THIS checkpoint's cancelling final
     block; even a perfect trunk feeding a perfect fp32 input caps at 0.90-0.92 in bf16.
     The only thing that clears >0.97 is fp32 matmul *inputs* for the tail block, which the
     Tensix bf16-input matmul can't provide and SDPA forbids -- a large perf/precision
     change for a quantity that is renormalized downstream anyway.
  5. **Right acceptance gate going forward.** Added
     **`test_of3_pairformer_stack_prefix47_on_device`** (gates the 47-block prefix,
     s_pcc>0.98 ∧ z_pcc>0.97 -- **passes**, s_pcc=0.99637/z_pcc=0.97616) as the honest
     stack-correctness signal; the full-48 test stays `xfail` with the precise
     cancellation root cause (device z_pcc=0.66241). `scripts/of3_real_golden.py` now also
     captures `pairformer_stack_prefix47`. Suite: **2 passed, 1 xfailed** on card 1.
     The definitive full-model gate is end-to-end structure RMSD -- exactly how Protenix-v2
     and Boltz-2 (which reuse this same `tenstorrent.Pairformer` primitive) are validated,
     none of them gate raw stack-z. That is a P6 item, unblocked once InputEmbedder /
     diffusion / confidence are ported and `OpenFold3.fold()` can run.
  6. **MSA-block z gap is a DIFFERENT mechanism, not the same signature** (tick 4 asked).
     Pairformer *block 0* on real input tracks to z_pcc=0.998; the MSA *block 0* only gets
     0.708 -- so the MSA gap is not the pairformer's final-block cancellation reappearing.
     It is a single-block issue: the MSA block's `OuterProductMean` + pair_stack take z
     from std ~18 to ~270 (a ~15x single-block amplification) and the device loses
     precision in that amplification. Distinct root cause; owns its own P6 bisect (start
     at OPM output scale). Not resolved this tick -- flagged, not hand-waved.
  **NEXT (P6):** (a) port the remaining components (InputEmbedder device leg, template
  embedder, atom enc/dec, DiffusionModule, confidence heads) so `fold()` runs and the
  real end-to-end structure gate replaces raw stack-z; (b) separately bisect the MSA
  block-0 z amplification (OPM scale). Do NOT keep chasing the pairformer raw-z >0.97 gate
  -- it is provably unreachable in bf16 and is the wrong metric.

- 2026-07-12 (tick 4): **Real-distribution golden captured; stack gate re-run (honest
  result: still fails, for a DIFFERENT and more interesting reason than tick 3 thought);
  MSA-block remap landed + gated (also honest-fails).**
  1. `scripts/of3_real_golden.py` runs the real OF3 `InputEmbedderAllAtom` +
     `MSAModuleEmbedder` on a real featurized example (ubiquitin, via P1's
     `build_openfold3_features`), mirroring `protenix_ref_forward.py`'s real-weights +
     real-features method. Adds `input_embedder_real`/`pairformer_stack_real`/
     `msa_block0_real`/`msa_stack_real` to `~/of3_ref_out.pkl`.
  2. **Tick-3's root cause was wrong.** Re-running the pure-CPU reference 48-block stack
     on this REAL (s, z) still explodes to the SAME order of magnitude as the synthetic
     N(0,1) case (s_out std ~1.8e4, vs tick 3's ~3.7e4) -- with no device, no remap, no
     bf16 involved. Real input does NOT fix the magnitude. This falsifies "off-manifold
     synthetic input" as the cause: the 48-block `PairFormerStack` on this checkpoint
     genuinely produces an unnormalized, large-magnitude residual stream regardless of
     input distribution (plausible for a pre-LN stack with no final norm -- every
     downstream consumer LayerNorms before use, so nothing requires s/z to stay O(1)).
     Re-ran the on-device stack gate against the real golden:
     **s_pcc=0.996 (up from 0.906 -- confirms the remap is solid), z_pcc=0.649 (still
     fails, down from a differently-flawed 0.164).** A pure-CPU fp32-vs-bf16 control on
     the same real input gets z_pcc=0.903 -- bf16 alone already can't cleanly track this
     checkpoint's large residual stream to gate precision (>0.97), and the device
     compounds a further 0.90→0.65 drop on top of that (`fp32_dest_acc_en` is already on
     in the test's compute-kernel config). Test updated to gate on the real golden
     (`pairformer_stack_real`), block-0 gate unchanged (still passes, `s_pcc=0.99985`).
     Kept `xfail(strict=False)` with the honest reason (open device-precision gap, not an
     artifact) -- **do not loosen the PCC threshold**, this is a real open item.
  3. **MSA-block remap landed**: `tt_bio.openfold3_weights.remap_msa_block`/
     `remap_msa_module` -- pure key-rename + delegate to the proven
     `protenix_weights.{remap_outer_product_mean,remap_pair_weighted_averaging,
     remap_transition,remap_msa_pair_stack}`, same style as the pairformer remap.
     Confirmed OF3's `msa_module.opm_first=True` (checkpoint has no `msa_att_row`/
     `msa_transition` keys on the last of the 4 blocks, matching `skip_msa_update =
     last_block and opm_first`) -- the OPPOSITE of `tt_bio.tenstorrent.MSALayer`'s
     hardcoded opm-after-update order (which matches Boltz-2, not AF3-family
     Protenix-v2/OF3). So `tests/test_openfold3_msa.py` composes the raw primitives
     directly in OF3's order (mirrors `test_protenix_trunk_msa.py`'s existing workaround
     for the exact same mismatch) instead of instantiating `MSALayer` -- using `MSALayer`
     here would silently apply the wrong order. On-device: **m-track byte-correct
     (m_pcc=0.99999, block 0) -- proves the remap + ordering are right.** z-track fails
     the same way as the pairformer stack: block-0 z_pcc=0.708, full 4-block z_pcc=0.745,
     while a pure-CPU bf16 control on the identical input gets z_pcc=0.9998. Same
     qualitative finding as item 2: this checkpoint's activations are large enough
     (single pair_stack call here takes z from std ~18 to ~270, a ~15x jump) that the
     DEVICE loses real precision beyond generic bf16 rounding -- Protenix-v2's own real
     MSA-stack gate, called the identical way, passes >0.99. `xfail(strict=False)`, same
     honest-not-rationalized treatment.
  **Net effect**: real-distribution input is confirmed necessary and fixed the s-track
  (0.906→0.996 pairformer stack); it is NOT sufficient to pass the z-track PCC gates
  anywhere in the trunk (pairformer stack OR MSA block). The remaining z-track gap is a
  real, open, checkpoint-specific device-precision problem (large real activation
  magnitudes, device compounding beyond bf16-alone), not a golden-harness artifact and
  not a remap defect -- confirmed twice now (pairformer stack + MSA block) via matching
  m/s-track-correct-but-z-track-fails CPU-bf16-vs-device signatures. **NEXT TICK:**
  either invest in the device-precision gap directly (why does the device lose more
  than bf16 rounding predicts -- check intermediate accumulation dtype through
  TriangleMultiplication/TriangleAttention/Transition at large activation scale), or
  continue component coverage (template embedder, atom encoder/decoder, DiffusionModule,
  confidence heads) and revisit precision once the full trunk shape is known.

- 2026-07-12 (tick 3): **Blocked 48-block stack z-gate RUN + root-caused; MSA/template remap
  scoped.** First cleared a recurring host-wide device-open-lock deadlock: a wedged
  shared-checkout boltz2 parity run's multiprocessing spawn-child held
  `/tmp/tt-bio-device-open.lock` ~2.6h while its parent circularly waited on the esmfold2
  run's UMD `CHIP_IN_USE_2` mutex (which in turn waited on the global lock) -- killed the
  wedged tree by explicit PID, which let esmfold2 finish and freed everyone (recipe:
  memory `device-open-lock-fleet-deadlock`). Then ran the gate on card 0:
  **block-0 s_pcc=0.99985 (remap byte-correct, reproduced); 48-block stack s_pcc=0.906,
  z_pcc=0.164.** The low stack PCC is a GOLDEN-HARNESS artifact, NOT a port defect:
  `of3_golden.py` feeds synthetic N(0,1), which is off the learned manifold, so the
  reference trunk EXPLODES over 48 blocks (out s std 3.69e4 vs a real fold's ~1.85e2). At
  that magnitude bf16 alone collapses z -- a pure-CPU fp32-vs-bf16 control (NO device, NO
  remap) on the SAME input gives s_pcc=0.99993 / **z_pcc=0.718**; the device compounds it
  over 48 blocks near the bf16 precision ceiling. Protenix's PASSING stack gate uses REAL
  captured trunk I/O (s std 0.43, z std 32, stable out std ~185/31.6) for exactly this
  reason. Made the suite honest: block-0 now gates on `s_pcc>0.98` (the robust correctness
  signal; z on N(0,1) is a recorded bf16 artifact), stack test marked `xfail` with the
  documented reason, caveat added to `of3_golden.py`.
  **NEXT TICK (do first):** make the stack gate valid = capture a REAL-distribution golden
  by running the OF3 reference `InputEmbedderAllAtom`(+MSAModuleEmbedder) on a real example
  (P1 `build_openfold3_features` -> reference input_embedder -> real (s,z) into the stack),
  mirroring `protenix_ref_out.pkl`. That IS the P3 InputEmbedder item, so it unblocks the
  stack re-gate as a side effect; then proceed to MSA-block/template gates. MSA-block remap
  is already scoped: OF3 `MSAModuleBlock` = `MSAPairWeightedAveraging` (layer_norm_m/z,
  linear_z/v/g/o) + OPM (layer_norm/linear_1/2/out, protenix-identical names) + SwiGLU
  `msa_transition` + `pair_stack`, all delegating to `protenix_weights`
  (`remap_pair_weighted_averaging`/`remap_outer_product_mean`/`remap_transition`/
  `remap_msa_pair_stack`); tt-bio target `MSALayer` (scopes pair_weighted_averaging,
  msa_transition, outer_product_mean, pairformer_layer). Watch `opm_first` ordering vs
  MSALayer's fixed OPM-after-update order.

- 2026-07-12 (tick 2): **P0 + P2 done.** `scripts/of3_golden.py` captures PairFormerBlock-0
  + 48-block-stack golden from real weights (`~/of3_ref_out.pkl`); `tt_bio/openfold3_weights.py`
  remaps OF3 keys onto the protenix primitive layout (byte-lossless, delegates to
  `protenix_weights`); `tests/test_openfold3_pairformer.py` is the device PCC gate. Verified:
  remap byte-lossless (16/16 conservation) + on-device block-0 `s_pcc=0.99985`; a CPU-bf16 run
  of the same block gives `z_pcc=0.977`. The 48-block stack device gate was left queued but
  blocked by a fleet **device-open-lock fd-leak**: other-worker multiprocessing spawn /
  resource-tracker children INHERIT the `/tmp/tt-bio-device-open.lock` fd and hold it for the
  whole run, so `_device_init_lock` serialises away *every* card open host-wide (victims sit in
  `locks_lock_inode_wait`). Recover by killing the leaked-fd child by explicit pid; real fix =
  close-on-exec / close the fd before the multiprocessing fork. **NEXT TICK:** read
  `~/of3_stack_test.log` for the stack z-PCC once the lock frees, then P1 (vendor the host-side
  data pipeline) and continue P3 (MSA / template / atom-enc / DiT / confidence remaps + gates).
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

## P6 tick 8 -- InputEmbedder glue leg PCC-gated on device (s/z = 1.00000)

Landed the first device component of the OpenFold3.fold() assembly: the
InputEmbedderAllAtom *glue* leg (s_input -> s, z) in tt_bio/openfold3.py as
InputEmbedderGlue, reusing the shared tenstorrent.Module base + _lin. The five
weight-only linears (linear_s 449->384, linear_z_i/linear_z_j 449->128,
linear_relpos 139->128, linear_token_bonds 1->128) plus the outer-sum
z[i,j] = z_i[i] + z_j[j] + relpos_emb + token_bonds_emb. The outer sum runs on device
via two single-dim broadcast adds (zero-seeded, same path as protenix's pair-bias add).

Golden: extended scripts/of3_real_golden.py to capture the reference relpos
(OF3 relpos_complex, 139-dim) into input_embedder_real so the device glue is gated
against the *exact* reference relpos, not a re-computation. (OF3 relpos_complex is
verified identical to Protenix._generate_relp -- same 139-dim logic, r_max=32, s_max=2.)

Gate: tests/test_openfold3_input_embedder.py feeds golden s_input + relpos +
token_bonds -> device glue -> compares s, z to golden. Result on qb2 card 0
(HiFi4 + fp32 dest acc): **s_pcc=1.00000, z_pcc=1.00000** -- both >0.98. The glue linears +
outer-sum z are byte-correct on device. The atom-encoder -> s_input leg (gated vs
input_embedder_atom_enc_real) is the next increment.

qb2 device note: opening a single P150 chip requires TT_MESH_GRAPH_DESC_PATH to point
at ttnn's p150_mesh_graph_descriptor.textproto (known qb2 firmware quirk -- the board
misreads as a dual-chip P300, blocking ttnn.open_device with Custom fabric mesh graph

## P6 tick 8 -- InputEmbedder glue leg PCC-gated on device (s/z = 1.00000)

Landed the first device component of the `OpenFold3.fold()` assembly: the
`InputEmbedderAllAtom` *glue* leg (`s_input -> s, z`) in `tt_bio/openfold3.py` as
`InputEmbedderGlue`, reusing the shared `tenstorrent.Module` base + `_lin`. The five
weight-only linears (`linear_s` 449->384, `linear_z_i`/`linear_z_j` 449->128,
`linear_relpos` 139->128, `linear_token_bonds` 1->128) plus the outer-sum
`z[i,j] = z_i[i] + z_j[j] + relpos_emb + token_bonds_emb`. The outer sum runs on device
via two single-dim broadcast adds (zero-seeded, same path as protenix's pair-bias add).

Golden: extended `scripts/of3_real_golden.py` to capture the reference `relpos`
(OF3 `relpos_complex`, 139-dim) into `input_embedder_real` so the device glue is gated
against the *exact* reference relpos, not a re-computation. (OF3 `relpos_complex` is
verified identical to `Protenix._generate_relp` -- same 139-dim logic, r_max=32, s_max=2.)

Gate: `tests/test_openfold3_input_embedder.py` feeds golden `s_input` + `relpos` +
`token_bonds` -> device glue -> compares `s`, `z` to golden. Result on qb2 card 0
(HiFi4 + fp32 dest acc): **s_pcc=1.00000, z_pcc=1.00000** -- both >0.98. The glue linears +
outer-sum z are byte-correct on device. The atom-encoder -> `s_input` leg (gated vs
`input_embedder_atom_enc_real`) is the next increment.

qb2 device note: opening a single P150 chip requires `TT_MESH_GRAPH_DESC_PATH` to point
at ttnn's `p150_mesh_graph_descriptor.textproto` (known qb2 firmware quirk -- the board
misreads as a dual-chip P300, blocking `ttnn.open_device` with "Custom fabric mesh graph
descriptor path must be specified for CUSTOM cluster type"). Set in the parent process
before importing ttnn. Device tests run as:
`TT_VISIBLE_DEVICES=0 TT_BIO_LOGICAL_DEVICE_ID=0 TT_MESH_GRAPH_DESC_PATH=<p150.textproto> PYTHONPATH=<worktree> <env>/python -m pytest tests/test_openfold3_*.py -x -s`

## P6 tick 9 -- InputEmbedder atom-featurization leg PCC-gated on device (cl/plm = 1.00000/0.99999)

Landed the second device component of the `OpenFold3.fold()` assembly: the
`RefAtomFeatureEmbedder` (reference-conformer atom featurization) in `tt_bio/openfold3.py`,
reusing the shared `tenstorrent.Module`/`_lin`. Two legs:

- **Single leg -> `cl`** [N_atom, c_atom=128]: five weight-only linears over per-atom
  reference features (`linear_ref_pos` 3->128, `linear_ref_charge` arcsinh 1->128,
  `linear_ref_mask` 1->128, `linear_ref_element` 119->128, `linear_ref_atom_chars`
  256->128) summed on device.
- **Pair leg -> `plm`** [N_blk, N_q, N_k, c_atom_pair=16]: three weight-only linears over
  the precomputed block inputs (`linear_ref_offset` 3->16, `linear_inv_sq_dists` 1->16,
  `linear_valid_mask` 1->16), each gated by `vlm`: `plm = off*vlm + isd*vlm + vm*vlm`.

The block construction (`convert_single_rep_to_blocks` + `get_block_indices`) is
mask-derived gather that is non-trivial to replicate on device, so the golden now carries
the precomputed `dlm`/`vlm`/`inv_sq_dists` (`input_embedder_ref_atom_feat_real`) and the
device pair linears consume them directly -- isolating the device linear precision from
the blocking logic, the same discipline as the glue's golden relpos. All eight linears are
bias-free in the OF3 checkpoint.

Gate: `tests/test_openfold3_ref_atom_feat.py` feeds golden per-atom features + block inputs
-> device embedder -> compares `cl`, `plm` to golden. Result on qb2 card 0 (HiFi4 + fp32
dest acc): **cl_pcc=1.00000, plm_pcc=0.99999** -- both >0.98. The atom featurization is
byte-correct on device. The `AtomTransformer` (3-block windowed DiT, -> `ql`) and the
atom->token aggregation (`linear_q` + mean, -> `ai`) are the next increment; gated together
they close the InputEmbedder atom-encoder leg (`s_input = cat([ai, restype, profile,
deletion_mean])`).


## P7 tick 10 -- InputEmbedder AtomTransformer + atom->token aggregation PCC-gated on device (ql/ai = 1.00000/1.00000)

Landed the final device component of the InputEmbedder atom-encoder leg: the
`AtomTransformer` (OF3 `DiffusionTransformer(cross_attention_mode=True)`, 3-block
windowed sequence-local atom attention, n_query=32 / n_key=128 / 4 heads / head_dim=32)
plus the atom->token aggregation (`relu(linear_q) + mean -> ai`), in
`tt_bio/openfold3_atom_transformer.py`. This closes `InputEmbedder -> s_input` together
with the already-gated glue (`s_input -> s, z`) and RefAtomFeatureEmbedder (`-> cl, plm`)
legs.

The OF3 `atom_transformer` block topology is NOT a key-remap onto `protenix.AtomTransformer`
(whose block layout differs): OF3 uses `attention_pair_bias.layer_norm_a_q`/`layer_norm_a_k`
(double AdaLN conditioning), `linear_ada_out` (a `sigmoid(W(s))` output gate),
`mha.linear_g` (a query gate), and a `conditioned_transition` of
`SwiGLU(linear_a, linear_b) -> linear_out` with a `sigmoid(linear_g(s))` zero gate. So this
is a fresh device port, not a reuse. The shared `AdaLN` math is identical, so the nine
conditioning submodules (q/kv/transition x 3 blocks) reuse `tenstorrent.AdaLN` via a new
`remap_of3_adaln` (OF3 `layer_norm_s`/`linear_g`/`linear_s` -> tenstorrent
`s_norm`/`s_scale`/`s_bias`); the top-level `layer_norm_z` (pair LN, applied once to `plm`)
is shared across blocks.

The mask-derived block gather (OF3 `convert_single_rep_to_blocks`: centered key windows
with underflow/overflow shift) is precomputed on host (`key_block_idxs`, `invalid_mask`,
`mask_trunked`) and replayed on device via `ttnn.embedding` with the fixed gather indices.
The device re-blocks the evolving single `a` every block (the conditioning `s = cl` is
fixed, so `s_q`/`s_k` are built once), so the port is reusable as-is for the diffusion atom
encoder/decoder, which use the same class with `a != s` (noisy-position `ql` vs `cl`).

Golden: `scripts/of3_atom_transformer_golden.py` extends `~/of3_ref_out.pkl` with
`input_embedder_atom_transformer_real` -- the host-precomputed block-gather artifacts, the
`atom_to_token_mean` aggregation matrix, and the reference `ql`/`ai` -- so the device
AtomTransformer is gated against the exact reference block structure (the gather is
captured, not re-derived; same discipline as the RefAtomFeatureEmbedder `dlm`/`vlm`/
`inv_sq_dists`).

Gate: `tests/test_openfold3_atom_transformer.py` feeds golden `cl` (as both `a_init` and
`s`) + `plm` + the host block artifacts -> device AtomTransformer -> `ql`; then
`relu(linear_q(ql))` mean-aggregated to tokens -> `ai`. Result on qb2 card 0 (HiFi4 + fp32
dest acc): **ql_pcc = 1.00000, ai_pcc = 1.00000** -- both > 0.98. The atom-encoder leg is
byte-correct on device end-to-end (`cl + plm -> ql -> ai`), and `ai` concatenates with
`[restype, profile, deletion_mean]` to form `s_input` (the glue leg, already gated, consumes
it). The InputEmbedder -> `s_input` path is now fully device-validated.

Golden-pkl fix (incidental): re-dumping `~/of3_ref_out.pkl` had made it require
`ml_collections` (the `config` field carried a nested `ConfigDict` at
`config.linear_init_params`; `ConfigDict` is neither a `dict` nor `collections.abc.Mapping`
subclass, so a naive `dict()` strip missed it). The device env has no `ml_collections`, so
both this test and the existing `test_openfold3_ref_atom_feat.py` failed to unpickle. Fixed
by stripping ConfigDicts via a `.items()` duck-type and re-dumping; the new golden script
does the same strip defensively on every run. The `config` field is plain metadata (no test
reads it), so the strip is behavior-neutral.

**NEXT (P8):** (a) Trunk assembly -- OF3-dims Pairformer (c_z=128, no_heads_pair=4) + MSA
module + template embedder + cycle glue, composing the already-gated
`pairformer_stack`/`msa` bisect results; (b) DiffusionModule + confidence heads via key
remap (dims confirmed identical to protenix-v2's, the cheapest remaining leg); (c) wire
`OpenFold3.fold()` + EDM sampler end-to-end and run `examples/prot.yaml` (the canonical
ubiquitin target) for a real vs-ground-truth Kabsch Ca-RMSD -- the actual merge gate.


## P7 tick 11 -- MSAModuleEmbedder (s_input -> m) PCC-gated on device (m_pcc = 1.00000)

Landed the next trunk sub-component: the OF3 ``MSAModuleEmbedder`` (AF3 Algorithm 8
lines 1-4) in ``tt_bio/openfold3_msa_embedder.py``. It is two bias-free linears
(``linear_m`` 34->c_m=64 over ``cat([msa, has_deletion, deletion_value])``;
``linear_s_input`` c_s_input=449->c_m=64) and a broadcast add over the MSA-sequence dim:
``m = linear_m(msa_feat) + linear_s_input(s_input).unsqueeze(-3)``.

The MSA subsampling (stochastic, AF3 SI 2.2) is host-side. The original golden set
``torch.manual_seed(0)`` before the InputEmbedder (which consumes no random state), so
the MSA embedder's ``torch.randint`` is the first draw after seed 0; re-setting seed 0 in
the new golden reproduces the exact same subsample (verified, repro max_abs = 0.0).
``scripts/of3_msa_embedder_golden.py`` captures the post-subsample ``msa_feat`` via a
``linear_m`` ``register_forward_pre_hook`` and adds ``msa_module_embedder_real`` to
``~/of3_ref_out.pkl`` -- so the device embedder is gated against the exact reference
subsample, isolating the device linear precision from the subsample logic (same discipline
as the other OF3 golden legs).

Gate: ``tests/test_openfold3_msa_embedder.py`` feeds golden post-subsample ``msa_feat`` +
``s_input`` -> device embedder -> compares ``m`` to golden. Result on qb2 card 0 (HiFi4 +
fp32 dest acc): **m_pcc = 1.00000** -- > 0.98. This extends the trunk validation past the
InputEmbedder (``s_input -> m``), complementing the already-gated MSA stack
(``m, z -> z`` in ``tests/test_openfold3_msa.py``). The MSA module (embedder + stack) is
now device-validated at the sub-component level.

**Reusability-matrix correction (item 3 / DiffusionModule):** the matrix's "reuse
protenix.DiffusionModule via key remap" does NOT hold. OF3's diffusion atom enc/dec use
the OF3 ``AtomTransformer`` topology ported in tick 10 (``linear_ada_out`` output gate,
``mha.linear_g`` query gate, ``swiglu.linear_a/b``+``linear_out`` transition), not
protenix's ``AtomTransformer``; and the token-DiT ``conditioned_transition`` also differs
from protenix's. So the DiffusionModule is a multi-leg port (OF3 DiT block + diffusion
conditioning / NoisyPositionEmbedder + atom enc/dec wiring + EDM sampler + trunk
conditioning golden), not a mechanical key remap. Flagged for P8 scoping.

**NEXT (P8):** (a) template embedder (OF3 ``TemplatePairStack`` 2-block pair stack +
embedder linears -> ``z_template`` added to ``z``); (b) assemble the full trunk forward
(InputEmbedder -> 48-block Pairformer + 4-block MSA + template, xN cycles -> s_trunk,
z_trunk) and PCC-gate vs ``pairformer_stack_real`` / a new full-trunk golden; (c) the
DiffusionModule multi-leg port above; (d) wire ``OpenFold3.fold()`` + EDM sampler
end-to-end and run ``examples/prot.yaml`` for a real vs-ground-truth Kabsch Ca-RMSD -- the
actual merge gate.

## P8 tick 12 -- DiffusionConditioning (Algorithm 21) PCC-gated on device (si/zij = 1.00000/0.99999)

Landed the first device sub-leg of the OF3 ``DiffusionModule``: ``DiffusionConditioning``
(AF3 Algorithm 21) in ``tt_bio/openfold3_diffusion.py`` as ``OF3DiffusionConditioning``,
reusing the shared ``tenstorrent.Module``/``_lin``. It produces the conditioned single
``si`` [N, c_s=384] and pair ``zij`` [N, N, c_z=128] that drive the diffusion transformer,
from the trunk outputs plus a noise level ``t``:

  - pair leg: ``cat([zij_trunk, relpos])`` (267) -> weight-only ``LN_z`` -> ``linear_z``
    (267->128) -> 2x ``SwiGLUTransition`` (masked by the pair token mask);
  - single leg: ``cat([si_trunk, si_input])`` (833) -> weight-only ``LN_s`` -> ``linear_s``
    (833->384), plus the Fourier noise embedding ``n = fourier_emb(0.25*log(t/sigma_data))``
    -> weight-only ``LN_n`` -> ``linear_n`` (256->384), broadcast-added over the token dim,
    then 2x ``SwiGLUTransition`` (masked by the token mask).

The three top LNs are weight-only (``create_offset=False``); the four transition LNs carry
weight+bias; every linear is bias-free in the OF3 checkpoint. The ``SwiGLUTransition``
(``LN -> silu(linear_a)*linear_b -> linear_out``, masked) is a fresh small device class in
the same module -- the same SwiGLU math the trunk transition and the P7 AtomTransformer
conditioned transition use.

This is a fresh OF3 port, NOT a key-remap onto ``protenix.DiffusionConditioning``: OF3's
diffusion conditioning carries its own relpos bin concat (139-dim ``relpos_complex``) and
weight-only top LNs, and feeds the OF3 DiffusionTransformer (token-level DiT, ported next).
The dims (c_s=384, c_z=128, c_fourier_emb=256, relpos=139, sigma_data=16) match OF3's
config; the conditioning's ``max_relative_idx=32``/``max_relative_chain=2`` produce the same
139-dim relpos as the trunk InputEmbedder.

Golden: ``scripts/of3_diffusion_conditioning_golden.py`` reuses the already-captured
real-distribution trunk tensors from ``~/of3_ref_out.pkl`` as the conditioning inputs
(``si_input`` from ``input_embedder_real/out``; ``si_trunk``/``zij_trunk`` from
``pairformer_stack_real/out``, the 48-block Pairformer output -- a real trunk-scale
single/pair representation, exactly what the conditioning consumes in one recycle). ``t``
is a real noise level (``s_max=160`` from the AF3 noise schedule, the initial sampling
sigma), so the Fourier noise embedding is on-manifold. The reference relpos and the
post-Fourier ``n`` (256-dim) are captured via forward hooks (on ``layer_norm_z`` input and
``fourier_emb`` output), so the device port is gated against the exact reference artifacts
-- isolating the device linear/LN/SwiGLU precision from the relpos/Fourier host math, the
same discipline as the other OF3 golden legs. Adds key ``diffusion_conditioning_real``.

Gate: ``tests/test_openfold3_diffusion_conditioning.py`` feeds golden ``zij_trunk`` +
``relpos`` + ``si_trunk`` + ``si_input`` + ``n_emb`` + masks -> device conditioning ->
compares ``si``, ``zij`` to golden. Result on qb2 card 3 (HiFi4 + fp32 dest acc):
**si_pcc = 1.00000, zij_pcc = 0.99999** -- both > 0.98. The conditioning leg is
byte-correct on device.

**NEXT (P8, DiffusionModule remainder):** (a) the OF3 token-level DiT block
(``DiffusionTransformer`` / Algorithm 23, non-cross-attention path: ``AttentionPairBias``
with ``use_ada_layer_norm=True`` -- ``AdaLN`` + ``linear_ada_out`` output gate + per-block
``layer_norm_z``/``linear_z`` pair bias + ``mha`` query gate, then a
``ConditionedTransitionBlock`` SwiGLU zero-gated transition), a fresh port reusing the P7
``AdaLN``/SwiGLU primitives where the math matches -- PCC-gate vs a real DiT golden; (b) the
atom enc/dec inside ``DiffusionModule`` reuse the P7 ``OF3AtomTransformer`` directly (same
topology, already gated, with ``a != s`` -- noisy-position ``ql`` vs ``cl``); (c) wire
``DiffusionModule.forward`` (conditioning -> atom enc -> DiT -> atom dec -> EDM output
scaling) and PCC-gate ``xl_out`` vs a full-module golden; (d) ``SampleDiffusion``/EDM
sampler + ``OpenFold3.fold()`` end-to-end, then run ``examples/prot.yaml`` for a real
vs-ground-truth Kabsch Ca-RMSD -- the actual merge gate for the whole port.

## P8 tick 12 -- TemplatePairFeatureEmbedder PCC-gated on device (t_embed = 1.00000); pair-stack leg device-xfail

Landed the device port of the OF3 ``TemplateEmbedderAllAtom`` feature-processing leg
(``TemplatePairEmbedderAllAtom``) in ``tt_bio/openfold3_template.py``, PCC-gated on
device. The 2-block AF2 ``TemplatePairStack`` and the full embedder are wired and gated
as documented-xfail on a device limitation (below), the same way the MSA pair_stack is
handled.

**Leg 1 (gated): TemplatePairFeatureEmbedder.** Eight bias-free linears summed into
``a`` [N_templ, N, N, c_t=64] (``dgram_linear`` 39->64, ``pseudo_beta_mask_linear``
1->64, ``aatype_linear_1``/``_2`` 32->64, ``x``/``y``/``z_linear`` 1->64 over the unit
vector components, ``backbone_mask_linear`` 1->64), plus the shared
``z_bias = linear_z(layer_norm_z(z))`` [1, N, N, 64], and ``t_embed = z_bias + a``
broadcast over the template dim. All eight feature linears and ``linear_z`` are
bias-free in the OF3 checkpoint; ``layer_norm_z`` is affine (128). The mask-derived
feature products (multichain / pseudo-beta / backbone-frame pair masks) are precomputed
on host and captured in the golden, so the device linears are gated against the exact
reference masks, isolating the device linear precision from the mask logic (same
discipline as the RefAtomFeatureEmbedder ``dlm``/``vlm``/``inv_sq_dists`` and the
InputEmbedder glue's ``relpos``).

**Leg 2 (xfail): TemplatePairStack.** Two AF2 PairBlocks (tri_mul_out/in +
tri_att_start/end + swiglu ``pair_transition``, ``tri_mul_first=True``) + a final affine
stack ``layer_norm``. Structurally identical to the MSA module's ``pair_stack`` subtree,
so it reuses ``PairformerLayer(transform_s=False)`` via a new
``remap_template_pair_stack`` (delegates each block to ``pw.remap_msa_pair_stack``;
returns the per-block primitive dicts plus the stack final-LN weights). Templates do not
interact, so the stack runs per-template (the reference's per-template loop is
mathematically identical to a batched pass, but the device ``TriangleAttention`` reshape
assumes a singleton batch dim, so the loop is kept). Two compounding device limitations
make this leg documented-xfail:

  1. The same primitive set is already documented-xfail for the MSA pair_stack at c=128
     (``tests/test_openfold3_msa.py``: z_pcc~0.75 on a pure device run vs 0.9998 for a
     pure-CPU fp32-vs-bf16 control -- an OPEN device-precision gap on OF3's
     large-magnitude pair activations, not a remap bug). The template pair_stack runs at
     c_t=64 on the same OF3-magnitude regime, so the gap applies a fortiori.
  2. The template pair_stack additionally runs at ``c_hidden_tri_att=16`` /
     ``no_heads=4`` (head_dim=16, sub-tile), where the shared ``TriangleAttention`` path
     hits a ttnn ``Invalid subtile broadcast type`` in ``gate_and_project``'s
     ``multiply_``: ``nlp_create_qkv_heads``/``nlp_concat_heads`` at head_dim=16 yield an
     ``o_in`` of [76,76,128] vs ``g_in`` [76,76,64] (the MSA path at head_dim=32 is
     tile-aligned and runs clean). Localized via a multiply_ shape trace
     (``scripts/of3_template_embedder_golden.py``-style diagnostic): the last multiply
     before the throw is ``(76,76,128) x (76,76,64)``. Fixing it means rewiring the
     shared ``TriangleAttention`` primitive, which all of MSA/Boltz-2/Protenix reuse at
     head_dim=32 -- out of scope for this leg and a release-gated risk to the already-
     validated ports. Flagged for a dedicated device-numerics/kernel pass.

**Leg 3 (xfail): full TemplateEmbedderAllAtom.** ``linear_t(relu(mean_t(t_stack)))``
[1, N, N, c_z=128]. Inherits leg 2's pair-stack limitation; documented-xfail.

**Golden:** ``scripts/of3_template_embedder_golden.py`` extends ``~/of3_ref_out.pkl``
with ``template_embedder_real`` -- the cycle-0 trunk z (the embedder input; reproduced
exactly from the trunk's top-level ``layer_norm_z``/``linear_z`` weights as
``z_init + linear_z(layer_norm_z(zeros))``, a constant shift of ``z_init`` -- NOT
``z_init`` itself, since ``layer_norm(zeros)`` returns the affine bias, not zeros), the
per-template feature tensors (mask products precomputed on host), and the reference
``t_embed`` / ``t_stack`` / ``z_template`` captured via forward hooks on
``template_pair_embedder`` / ``template_pair_stack`` / ``template_embedder``. Verified
the per-template feature reconstruction is bit-exact against the reference ``t_embed``
(max abs 0.0) before building the device port.

**Gate:** ``tests/test_openfold3_template.py`` -- three sub-leg tests. Result on qb2
card 0 (HiFi4 + fp32 dest acc): **A (feature embedder) t_embed_pcc = 1.00000** (>0.98,
gated, passing); B (pair stack) and C (full embedder) documented-xfail on the device
limitation above. Session: ``1 passed, 2 xfailed``. The feature-processing leg of the
template embedder is byte-correct on device; the 2-block pair-stack leg is blocked by a
device kernel limitation (sub-tile head_dim=16) on top of the known OF3 pair-stack
precision gap, and is the next item for a device-numerics pass.

**Reuse note (corrects the P8 scoping hint):** the task hint suggested reusing
"pairformer-stack blocks" for the template pair stack, but OF3's ``TemplatePairStack`` is
the AF2 ``PairBlock`` (tri_mul + tri_att + pair_transition), NOT the Pairformer/
AttentionPairBias block. The correct reuse is the MSA pair_stack's primitive set
(``TriangleMultiplication``/``TriangleAttention``/``Transition`` via
``PairformerLayer(transform_s=False)``), which ``remap_template_pair_stack`` wires up.

**NEXT (P8 cont):** (a) device-numerics/kernel pass to unblock the template + MSA
pair_stack at OF3 pair magnitudes / sub-tile head_dim (shared ``TriangleAttention``
rewire -- release-gated); (b) assemble the full trunk forward (InputEmbedder -> 48-block
Pairformer + 4-block MSA + template, xN cycles -> s_trunk, z_trunk) and PCC-gate vs
``pairformer_stack_real`` / a new full-trunk golden; (c) the DiffusionModule multi-leg
port (OF3 DiT block + diffusion conditioning + atom enc/dec + EDM sampler); (d) wire
``OpenFold3.fold()`` + EDM sampler end-to-end and run ``examples/prot.yaml`` for a real
vs-ground-truth Kabsch Ca-RMSD -- the actual merge gate.

## P8 tick 13 -- Trunk assembly PCC-gated on device (cycle glue + assembled Pairformer path; template/MSA pair_stacks substituted)

Assembled the OF3 ``run_trunk`` (AF3 Algorithm 1 lines 1-14) on device in
``tt_bio/openfold3_trunk.py``, PCC-gated against a real golden of the reference trunk
forward. This wires InputEmbedder -> N-cycle(48-block Pairformer + 4-block MSA module +
template embedder) -> s_trunk, z_trunk from the already-gated components plus the
genuinely-new top-level cycle glue.

**The new device code: ``OF3TrunkGlue``** -- the top-level trunk cycle glue that the
reference ``run_trunk`` applies each cycle, separate trunk weights (NOT the
InputEmbedder's linears): affine ``layer_norm_z``/``layer_norm_s`` (eps=1e-5) + bias-free
``init="final"`` ``linear_z`` (128->128) / ``linear_s`` (384->384):

  z = z_init + linear_z(layer_norm_z(z_prev))
  s = s_init + linear_s(layer_norm_s(s_prev))

s/z start at zeros; ``s_init``/``z_init`` are the InputEmbedder constants (P7-gated
end-to-end). ``OF3Trunk`` composes ``OF3TrunkGlue`` + the 48-block OF3-dims ``Pairformer``
(c_z=128, no_heads_pair=4) via the existing ``remap_pairformer_stack``.

**Reference golden (``scripts/of3_trunk_golden.py``):** runs the real OF3
``run_trunk`` on featurized ubiquitin for ``num_cycles = num_recycles+1 = 4`` (the config
default), replicating the reference cycle body (z-glue -> template embedder -> MSA module
-> s-glue -> 48-block Pairformer) with the real top-level glue weights and per-cycle
MSA subsampling (``torch.manual_seed(0)``; each cycle's ``m`` is a fresh draw, cycle-0
``m`` matches the one stored under ``input_embedder_real["msa_out"]``). Captures per-cycle
``z_prev``/``z_after_zglue``/``z_after_template``/``m``/``z_after_msa``/``s_prev``/
``s_after_sglue`` and the final ``s_trunk``/``z_trunk`` into ``~/of3_ref_out.pkl`` key
``trunk_real``. z_trunk std 130.19 (well-conditioned); s_trunk std 16869 (the S-track
magnitude, a red herring for z -- P5).

**Two gates (``tests/test_openfold3_trunk.py``):**

1. ``test_of3_trunk_glue_on_device`` -- GATES the new cycle glue in isolation across all
   4 cycles: feed the golden per-cycle ``z_prev``/``s_prev`` -> device glue -> compare to
   the golden ``z_after_zglue``/``s_after_sglue``. Result on qb2 card 0 (HiFi4 + fp32 dest
   acc): **z_pcc = 1.00000, s_pcc = 1.00000** every cycle (min over cycles = 1.00000 /
   1.00000). The new glue code is byte-correct on device, gated tight (>0.98), with REAL
   non-zero inputs (cycle 0 is the zeros->constant shift; cycles 1-3 feed the real
   pairformer z/s at std ~113/~16000).

2. ``test_of3_trunk_assembly_on_device`` -- GATES the assembled trunk forward (cycle glue
   + 48-block Pairformer, s AND z tracks) on the real settled trunk distribution, WITH
   the template + MSA pair_stack z substituted from the golden (``z_after_msa`` per cycle)
   so the Pairformer receives the correct z each cycle. Result on qb2 card 0:
   **s_trunk_pcc = 0.99981, z_trunk_pcc = 0.99936** -- both > 0.98.

**Honest gated scope (NOT a fully-device-gated trunk):** the template pair_stack throws
on device (sub-tile head_dim=16 ttnn kernel bug, P8 tick 12) and the MSA pair_stack is
z-xfail (~0.75, P8 tick 4/6); both are substituted from the reference golden in the
assembled run, and both are documented-xfail in their own tests
(``tests/test_openfold3_template.py``, ``tests/test_openfold3_msa.py``). What IS gated
here is the device-runnable assembly path: the new cycle glue + the 48-block Pairformer
(s and z tracks) run end-to-end across all 4 cycles. The per-cycle z the Pairformer
receives is the golden z (the template+MSA pair_stacks that produce it are
device-xfail), so no fully-device-gated trunk PCC number is claimed. The remaining
blocker for a fully-device trunk is the template + MSA pair_stack device kernel gap, not
the Pairformer or the glue.

**Notable finding (refines P5):** the device Pairformer z-track gates cleanly on the
real cycle-3 trunk z (z_trunk_pcc=0.99936), unlike the cycle-0 (s_init, z_init)
single-pass case where the P5 final-block catastrophic cancellation caps z_pcc at
~0.66 (``test_of3_pairformer_stack_on_device`` xfail). The cancellation is a
cycle-0-input-specific artifact of the (s_init, z_init) distribution; the actual trunk's
final-cycle Pairformer z does not trigger it. So the real trunk z_trunk is
device-achievable on the Pairformer side, pending the template+MSA pair_stack kernel fix
-- the P5 "no bf16 impl can clear >0.97 raw-stack-z" verdict was specific to the
cycle-0 input, not a fundamental blocker for the settled trunk output.

**Device note (qb2):** the cycle-0 ``s``/``z`` are seeded as ``ttnn.zeros(...,
layout=TILE_LAYOUT)`` -- a ROW_MAJOR 4D zeros passed straight into ``ttnn.linear`` hits
the tile-size check (flattened M = 76*76 not %32); TILE_LAYOUT pads to 96, and the
N-dim padding does not affect per-channel LayerNorm. A transient card-0 NOC/sysmem
wedge during one run cleared with ``tt-smi -r 0`` (a pharma worker was concurrently on
card 2, untouched).

**NEXT (P8 cont):** (a) device-numerics/kernel pass to unblock the template + MSA
pair_stack at OF3 pair magnitudes / sub-tile head_dim (shared ``TriangleAttention``
rewire -- release-gated; the separate ``tt-bio-openfold3-accel-triangleattn-subtile``
stream is scouting the head_dim=16 fix); once that lands, drop the golden substitution in
``OF3Trunk.__call__`` and gate the fully-device trunk; (b) the DiffusionModule multi-leg
port (OF3 DiT block + diffusion conditioning [already gated] + atom enc/dec [P7
AtomTransformer reuse] + EDM sampler) -- parallel stream ``tt-bio-openfold3-port-p8-dit``
owns the DiT block, do not duplicate; (c) wire ``OpenFold3.fold()`` + EDM sampler
end-to-end and run ``examples/prot.yaml`` for a real vs-ground-truth Kabsch Ca-RMSD --
the actual merge gate for the whole port (no ``fold()`` claim without it).
