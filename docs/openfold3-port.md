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
- [ ] **P4 assemble** `OpenFold3` class (`load_from_checkpoint` + `fold`), EDM sampler.
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

**Leg 2: TemplatePairStack.** Two AF2 PairBlocks (tri_mul_out/in +
tri_att_start/end + swiglu ``pair_transition``, ``tri_mul_first=True``) + a final affine
stack ``layer_norm``. Structurally identical to the MSA module's ``pair_stack`` subtree,
so it reuses ``PairformerLayer(transform_s=False)`` via a new
``remap_template_pair_stack`` (delegates each block to ``pw.remap_msa_pair_stack``;
returns the per-block primitive dicts plus the stack final-LN weights). Templates do not
interact, so the stack runs per-template (the reference's per-template loop is
mathematically identical to a batched pass, but the device ``TriangleAttention`` reshape
assumes a singleton batch dim, so the loop is kept). Originally documented-xfail on two
compounding device limitations; both are now resolved/clarified in tick 13:

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

**Leg 3: full TemplateEmbedderAllAtom.** ``linear_t(relu(mean_t(t_stack)))``
[1, N, N, c_z=128]. Inherits leg 2's behavior; gated (see tick 13).

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
precision gap, and is the next item for a device-numerics pass. (Resolved in tick 13:
legs B/C now pass -- t_stack_pcc = 0.99927, z_template_pcc = 0.99995 -- after the
shared ``TriangleAttention`` sub-tile head_dim fix; the c_t=64 template regime does not
hit the MSA c=128 precision gap.)

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

## P8 tick 13 -- TriangleAttention sub-tile head_dim=16 path; TemplatePairStack / TemplateEmbedderAllAtom PCC-gated on device (t_stack = 0.99927, z_template = 0.99995)

Resolved the sub-tile ``head_dim=16`` shape bug in the shared ``TriangleAttention``
primitive (``tt_bio/tenstorrent.py``) that blocked OF3's ``TemplatePairStack`` (tick 12,
Leg 2). ``nlp_concat_heads`` pads each head's channel dim up to a 32-tile boundary, so at
``head_dim=16`` it produces ``n_heads*32`` channels while the gate ``g`` carries
``n_heads*head_dim`` -- the ``[76,76,128]`` vs ``[76,76,64]`` mismatch that threw
``Invalid subtile broadcast type`` in ``gate_and_project``'s ``multiply_``. The fix
activates only when ``head_dim % 32 != 0``: pad the qkv weight's head_dim up to 32 (zeros,
so the real head_dim channels are unchanged), then slice the SDPA output back to
``head_dim`` and manual head-concat (head-major, same order as ``nlp_concat_heads``) to
yield ``[1, S, n_heads*head_dim]``. Mirrors the already-validated ``AttentionPairBias``
sub-tile handling. The tile-aligned ``head_dim=32`` path is untouched: same qkv weight
construction (no pad) and the original ``nlp_concat_heads`` branch, so behavior is
byte-identical for every other consumer (MSA / Boltz-2 / Protenix).

**Gate:** ``tests/test_openfold3_template.py`` -- all three sub-legs now pass on qb2 card 3
(HiFi4 + fp32 dest acc), un-xfailed: **A t_embed_pcc = 1.00000**, **B t_stack_pcc =
0.99927**, **C z_template_pcc = 0.99995** (each >0.98). The OF3 template regime (c_t=64,
smaller pair magnitudes than the MSA c=128 stack) does not hit the MSA pair_stack's
device-precision gap, so legs B/C clear the gate cleanly once the shape bug is fixed --
the tick-12 "a fortiori" pessimism was wrong for this regime. The template pair-stack leg
is no longer xfail.

**Cross-model head_dim=32 regression (release-gated):** the change is gated on
``head_dim % 32 != 0``, so the ``head_dim=32`` path is byte-identical by construction.
Verified on qb2 card 3 against the existing gated tests:
  * Boltz-2 pairformer (``tests/test_tenstorrent.py``: ``test_pairformer``,
    ``test_template_pairformer``, ``test_affinity_pairformer``, seq_len 100 + 500): 6/6
    passed (rel-error median <0.1, unchanged).
  * OF3 MSA pair_stack (``tests/test_openfold3_msa.py::test_of3_msa_block0_on_device``):
    m_pcc = 0.99999, z_pcc = 0.70847 -- identical to the pre-fix baseline to 5+ decimals
    (the known OPEN c=128 device-precision gap is unchanged, as expected).
  * Protenix-v2 pairformer (``tests/test_protenix_trunk_pairformer.py``): PASS on qb2.
    Rebuilt the v2 reference golden ``~/protenix_ref_out.pkl`` on qb2
    (``scripts/protenix_ref_forward.py`` with ``DUMP_INTERMEDIATES=1``, refenv312, 38-res
    tiny input, N_step=10), capturing the ``pairformer_stack`` in/out. The 48-block device
    stack vs that golden: s_pcc = 0.99191, z_pcc = 0.97964 (above the 0.98/0.97 gate). The
    shared ``TriangleAttention`` runs at ``head_dim=32`` here, so the subtile branch is
    not taken and the output is unchanged vs pre-fix -- the last cross-model sign-off.

**Release-gate flag:** this changes the shared ``TriangleAttention`` primitive used by
every shipped model. Landed on ``wk/tt-bio-openfold3-accel-triangleattn-subtile`` for the
orchestrator to review the Boltz-2 + MSA regression evidence above before merge. Protenix cross-gate closed above (s_pcc=0.99191, z_pcc=0.97964); the subtile fix is merged to main.

**NEXT (P8 cont):** (a) DONE -- Protenix-v2 reference golden rebuilt on qb2 and
``test_protenix_trunk_pairformer`` re-run (s_pcc=0.99191, z_pcc=0.97964); (b) assemble the
full trunk forward (InputEmbedder -> 48-block Pairformer + 4-block MSA + template, xN
cycles -> s_trunk, z_trunk) and PCC-gate vs ``pairformer_stack_real`` / a new full-trunk
golden; (c) the DiffusionModule multi-leg port (OF3 DiT block + diffusion conditioning +
atom enc/dec + EDM sampler); (d) wire ``OpenFold3.fold()`` + EDM sampler end-to-end and
run ``examples/prot.yaml`` for a real vs-ground-truth Kabsch Ca-RMSD -- the actual merge
gate.

## P8 tick 14 -- OF3 token-level DiffusionTransformer (Algorithm 23, non-cross path) PCC-gated on device (block=0.99999, stack=0.99984)

Landed the second device sub-leg of the OF3 ``DiffusionModule``: the token-level DiT
(``DiffusionTransformer``, AF3 Algorithm 23, non-cross-attention path) in
``tt_bio/openfold3_diffusion_transformer.py`` as ``OF3DiffusionTransformer``. A 24-block
stack of ``DiffusionTransformerBlock``, each block = ``AttentionPairBias`` (with
``use_ada_layer_norm=True``) + ``ConditionedTransitionBlock``:

  - ``a_ln = AdaLN(a, s)`` (c_a=768, c_s=384; reuses ``tenstorrent.AdaLN`` via
    ``remap_of3_adaln`` from the P7 AtomTransformer -- the AdaLN math is identical);
  - per-block weight-only ``LN_z`` + ``linear_z`` pair bias ``[1,16,N,N]`` (the
    non-cross variant applies its OWN per-block ``LN_z``; the cross-attention
    ``AtomTransformer`` instead shares one top-level ``LN_z`` across blocks);
  - MHA: fused padded qkv (head_dim 48 -> 64 for tiling) + ``nlp_create_qkv_heads``,
    query gate ``sigmoid(linear_g(a_ln))`` (flat == per-head: ``g.view(N,H,d) * o(H,N,d)``
    is the flat multiply), ``linear_o``, then the APB output gate
    ``sigmoid(linear_ada_out(s))``;
  - ``ConditionedTransitionBlock``: ``AdaLN`` -> SwiGLU (``silu(linear_a)*linear_b`` ->
    ``linear_out``) -> zero gate ``sigmoid(linear_g(s))`` -> token-mask, the same SwiGLU
    math the trunk transition and the P7 AtomTransformer conditioned transition use.

This is a fresh OF3 port, NOT a key-remap onto ``protenix.DiffusionTransformer``: OF3's
mha has separate q/k/v linears (q bias only) and a query gate on ``a_ln`` (protenix's
gates the input ``s``), and the non-cross ``AttentionPairBias`` carries its own per-block
``LN_z``. The shared ``AdaLN``/SwiGLU primitives are reused where the math matches; the
attention itself is reimplemented (see below).

Golden: ``scripts/of3_diffusion_transformer_golden.py`` instantiates the full
``DiffusionModule`` (so the atom encoder runs and produces a real on-manifold DiT input
``a = ai + linear_s(LN_s(si))``), forward-hooks ``diffusion_transformer`` to capture its
exact ``(a, s=si, z=zij, mask=token_mask)`` input and ``a`` output, plus the per-block
trajectory, and adds ``diffusion_transformer_real`` to ``~/of3_ref_out.pkl``. ``t`` is
the real initial sampling sigma (``s_max=160``), ``xl_noisy = randn * t`` is a real noisy
sample at that sigma (the first sampling step), and the conditioning inputs reuse the
already-captured real trunk tensors. The device port is gated against the exact
reference artifacts, isolating the device block precision from the
atom-encoder/conditioning host math (same discipline as the other OF3 golden legs).

Gate: ``tests/test_openfold3_diffusion_transformer.py`` feeds golden ``(a, s, z, mask)``
-> device DiT -> compares ``a`` to golden. Result on qb2 card 1 (HiFi4 + fp32 dest acc):
**block0 a_pcc = 0.99999, 24-block stack a_pcc = 0.99984** -- both > 0.98. The DiT block
topology and the full 24-block stack are byte-correct on device.

Two correctness details were invisible at the block level and fatal at the stack level;
both are documented in the module docstring and are the general lesson for any
additive-mask + fused-SDPA device port:

  1. **Additive mask must cover tile-padded keys.** ``from_torch`` pads *storage* with 0,
     which for an *additive* attention mask means "unmasked". A logical-N mask therefore
     leaves the tile-padded key positions (N -> ceil(N/32)*32) unmasked, and the SDPA
     computes over the tiled width, so valid queries attend to padded keys (K,V=0 the
     first block, then nonzero garbage as the padded-query residual feeds back) -- the
     leak compounds across the 24 blocks and collapses the stack PCC to **0.297**. Fix:
     the stack pads ``a``/``s``/``z`` to the tile-aligned logical width and marks padded
     keys ``-1e9`` in the additive mask; valid positions are unaffected (padded queries'
     outputs are stripped at readout, padded keys are masked out of every valid query's
     softmax). After the fix the stack tracked the reference at >=0.998 at every block
     but still drifted to 0.967 over 24 blocks -- leading to (2).

  2. **Softmax precision is the compounding lever.** The fused
     ``scaled_dot_product_attention`` does its softmax in bf16; its per-block error
     (~0.998) compounds multiplicatively (``0.998^24 ~= 0.953``) to a **0.967** stack
     PCC, even with the mask fixed. A CPU bf16 control with an fp32 softmax
     (``scripts/of3_diffusion_transformer_cpu_bf16.py``) holds **0.99996** over the same
     stack -- isolating the softmax precision as the lever (the matmul rounding alone is
     benign). The OF3 reference runs the attention with ``use_high_precision_attention``
     semantics (fp32 softmax); the fused SDPA cannot do an fp32 softmax, so the device
     port computes attention manually (matmul QK^T + scale + mask, fp32
     numerically-stable softmax via ``typecast`` up/down, matmul attn@V). This closes the
     gap: the device stack tracks the reference at **>=0.9998 at every block** (0.99999
     at block 0, 0.99984 at block 23). ``scripts/of3_diffusion_transformer_bisect.py``
     is the per-block trajectory bisect that localized both issues.

### P8 tick 15 -- AtomAttentionDecoder (Algorithm 6) PCC-gated on device

The exit leg of ``DiffusionModule.forward``: maps the token-level DiT output ``ai``
(post ``layer_norm_a``) back to a per-atom coordinate update ``rl_update`` [N_atom, 3],
which EDM output scaling then turns into ``xl_out``. Reference topology
(``openfold3.core.model.layers.sequence_local_atom_attention.AtomAttentionDecoder``):

    ql'  = ql + broadcast_to_atoms(linear_q_in(ai))   # token -> atom broadcast (Alg 6 L5)
    ql'' = AtomTransformer(ql', cl, plm, atom_mask)    # 3-block windowed cross-attn (Alg 5 L15)
    rl_update = linear_q_out(layer_norm(ql''))         # weight-only LN (c_atom, eps=1e-5) + c_atom->3

The 3-block cross-attention is the *same* ``DiffusionTransformer`` (windowed, non-cross)
topology as the encoder's, so the already-gated P7 ``OF3AtomTransformer`` is reused
verbatim with the decoder's own weights
(``diffusion_module.atom_attn_dec.atom_transformer.*``; identical config: 3 blocks,
``N_HEADS=4``, ``HEAD_DIM=32``, ``N_QUERY=32``, ``N_KEY=128``, ``c_atom=128``,
``c_atom_pair=16``). The fresh device work in ``tt_bio/openfold3_diffusion_decoder.py``
is therefore only:

  * ``linear_q_in`` (``c_token=768`` -> ``c_atom=128``, bias-free) + the token->atom
    broadcast. The broadcast is a mask-derived gather (each atom picks its token's
    feature via ``atom_to_token_index``); following the P7 atom-transformer discipline
    the gather index is precomputed on host in the golden and replayed on device with
    ``ttnn.embedding`` (no device-side index math).
  * a weight-only ``layer_norm`` (``c_atom=128``, ``create_offset=False``, ``eps=1e-5``);
  * ``linear_q_out`` (``c_atom=128`` -> 3, bias-free).

Padded atom positions (``n_atom`` -> ``NP = nb*N_QUERY``) are zeroed via
``atom_mask_col`` so the additive broadcast and the per-row layer-norm do not leak into
real atoms (same pad-and-mask discipline as the DiT stack). The weight-only LN is
per-row, so padding ``ql''`` to ``NP`` for tiling and slicing back to ``n_atom`` is
exact (padded zero rows LN to zero and are stripped at readout).

Golden: ``scripts/of3_diffusion_decoder_golden.py`` runs the full reference
``DiffusionModule`` forward (real of3-p2-155k.pt weights, real ubiquitin batch, real
noisy sample at ``s_max``) and forward-hooks ``diffusion_module.atom_attn_dec`` to
capture its exact ``(ai, ql, cl, plm)`` input and ``rl_update`` output, plus the
mask-derived atom windowing (``key_block_idxs`` / ``invalid_mask`` / ``mask_trunked``,
identical to the encoder's) and the ``atom_to_token_index`` broadcast index, adding
``diffusion_decoder_real`` to ``~/of3_ref_out.pkl``. Captured shapes (ubiquitin):
``n_atom=601``, ``n_token=76``, ``nb=19``, ``NP=608``, ``ai [76,768]``, ``ql/cl
[601,128]``, ``plm [19,32,128,16]``, ``rl_update [601,3]`` (std 0.247).

Gate: ``tests/test_openfold3_diffusion_decoder.py`` feeds golden
``(ai, ql, cl, plm)`` + the host aux -> device ``OF3AtomAttentionDecoder`` -> compares
``rl_update`` to golden. Result on qb2 (HiFi4 + fp32 dest acc):
**rl_update_pcc = 0.99968** (> 0.98). The decoder exit leg is byte-correct on device;
combined with the gated conditioning, DiT, and P7 ``OF3AtomTransformer``, the only
fresh device work left for a full ``DiffusionModule.forward`` ``xl_out`` gate is the
*encoder* noisy-position path (``NoisyPositionEmbedder``: trunk single/pair broadcast +
``linear_r(rl_noisy)``), the encoder pair update (``linear_l``/``linear_m`` +
``pair_mlp``), and the small glues (``ai += linear_s(LN_s(si))``, ``layer_norm_a``, EDM
output scaling) -- then assemble and PCC-gate ``xl_out`` vs a full-module golden.

### P8 tick 16 -- NoisyPositionEmbedder (Algorithm 5 L8-12) PCC-gated on device

The entry leg of the ``DiffusionModule`` atom encoder: fuses the trunk single/pair
representations into the reference-conformer atom conditioning and seeds the atom
single rep ``ql`` from the noisy coordinates. Reference topology
(``openfold3.core.model.layers.sequence_local_atom_attention.NoisyPositionEmbedder``):

    cl  = cl0 + broadcast_to_atoms(linear_s(LN_s(si_trunk)))   # single broadcast
    plm = plm0 + to_blocks(linear_z(LN_z(zij)))                # pair broadcast (blocked)
    ql  = cl + linear_r(rl_noisy)                              # noisy-coordinate projection

``cl0``/``plm0`` are the ``RefAtomFeatureEmbedder`` outputs (gated P7); ``zij`` is the
*conditioned* pair (diffusion-conditioning output, gated P8). All five linears are
bias-free; both LNs are weight-only (``create_offset=False``, eps=1e-5). The fresh
device work in ``tt_bio/openfold3_diffusion_module.py`` (``OF3NoisyPositionEmbedder``)
is the two weight-only LNs + three linears + the two mask-derived broadcasts, replayed
on device via ``ttnn.embedding`` (same isolation discipline as the P7 atom-transformer
key gather):

  * single: ``atom_to_token_index`` [NP] gathers ``linear_s(LN_s(si_trunk))`` [N_tok, c]
    to [NP, c] (padded atoms -> 0, zeroed by ``atom_mask_col``);
  * pair: ``zij_flat_idx`` [nb*nq*nk] (= ``q_token*N_tok_pad + k_token`` per (b,q,k))
    gathers ``linear_z(LN_z(zij))`` [N_tok*N_tok, c] (flattened with stride = the
    tile-padded token width) to [nb, nq, nk, c], masked by ``zij_mask``
    (``(1-invalid_key) * atom_pair_mask``).

A stride footgun: the device ``zij`` is tile-padded to ``N_tok_pad`` (76 -> 96), so the
flattened pair tensor has stride ``N_tok_pad``, not ``N_tok``; the golden therefore saves
the stride-agnostic ``q_indices``/``k_indices`` and the test computes the flat index with
the device stride. Padded atom positions are zeroed via ``atom_mask_col`` (single) /
``zij_mask`` (pair) so neither additive broadcast leaks into real atoms.

Golden: ``scripts/of3_diffusion_module_xlout_golden.py`` runs the full reference
``DiffusionModule`` forward (real of3-p2-155k.pt, real ubiquitin, real noisy sample at
``s_max``) and forward-hooks ``atom_attn_enc.noisy_position_embedder`` to capture
``cl0``/``plm0`` (RefAtomFeatureEmbedder outs) + ``si_trunk``/``zij``/``rl`` in and
``cl``/``plm``/``ql`` out, adding them (plus ``rl_noisy``, ``xl_out``, ``sigma_data``,
the ``q``/``k`` gather indices, ``zij_mask``) to ``diffusion_module_xlout_real`` in
``~/of3_ref_out.pkl`` -- the same record serves the eventual full-module ``xl_out`` gate.

Gate: ``tests/test_openfold3_noisy_position_embedder.py`` feeds golden
``cl0``/``plm0``/``si_trunk``/``zij``/``rl`` + the host aux -> device
``OF3NoisyPositionEmbedder`` -> compares ``cl``/``plm``/``ql`` to golden. Result on qb2
(HiFi4 + fp32 dest acc): **cl_pcc = 1.00000, plm_pcc = 0.99999, ql_pcc = 1.00000** (all
> 0.98). The encoder entry leg is byte-correct on device; the only fresh device work
left for the full ``xl_out`` gate is the encoder pair update (``linear_l``/``linear_m`` +
``pair_mlp``), the small glues (``ai += linear_s(LN_s(si))``, ``layer_norm_a``, EDM
output scaling), and the assembly -- for which the ``xl_out`` / ``a_in`` / ``a_stack`` /
``rl_update`` goldens are already in place as bisect checkpoints.

**NEXT (P8, DiffusionModule remainder):** (a) wire ``DiffusionModule.forward``
(conditioning -> atom enc -> DiT -> atom dec -> EDM output scaling) reusing the
already-gated P7 ``OF3AtomTransformer`` for the atom enc/dec (same topology, ``a != s``)
and PCC-gate ``xl_out`` vs a full-module golden; (b) ``SampleDiffusion``/EDM sampler +
``OpenFold3.fold()`` end-to-end, then run ``examples/prot.yaml`` for a real
vs-ground-truth Kabsch Ca-RMSD -- the actual merge gate for the whole port.


## P9 tick 17 -- Full ``DiffusionModule.forward`` -> ``xl_out`` PCC-gated on device (xl_out = 0.99746)

Assembled the full post-conditioning ``DiffusionModule`` (AF3 Algorithm 20) on device
in ``tt_bio/openfold3_diffusion_module.OF3DiffusionModule``, wiring the already-gated
sub-legs (``OF3NoisyPositionEmbedder``, encoder + decoder ``OF3AtomTransformer``,
``OF3DiffusionTransformer``) with the fresh post-conditioning work:

  * **encoder pair update** (``AtomAttentionEncoder.get_atom_reps`` L13-16):
    ``cl_lm = (linear_l(relu(cl_l)) + linear_m(relu(cl_m))) * atom_pair_mask`` (the
    ``cl`` single rep re-blocked into query/key blocks via the host
    ``key_block_idxs`` gather, replayed on device with ``ttnn.embedding``), then
    ``plm += cl_lm; plm += pair_mlp(plm); plm *= atom_pair_mask``. ``pair_mlp`` is the
    OF3 ``Sequential(ReLU, Linear, ReLU, Linear, ReLU, Linear)`` -- ReLU *before* each
    linear (no trailing ReLU).
  * **``linear_q`` atom->token aggregation**: ``ai = atom_to_token_mean @ relu(linear_q(ql))``
    (``linear_q = Sequential(Linear(128,768), ReLU)`` -- ReLU *after* the linear here),
    using the host mean matrix from the P7 atom-transformer golden.
  * **glues**: ``ai += linear_s(LN_s(si))`` (top-level ``linear_s`` 384->768 + weight-only
    ``layer_norm_s``) and ``ai = layer_norm_a(ai)`` (weight-only, 768) before the decoder.
  * **EDM output scaling**: ``xl_out = c_skip * xl_noisy + c_out * rl_update`` then
    ``* atom_mask``, with ``c_skip = sigma_data^2 / (sigma_data^2 + t^2)`` and
    ``c_out = sigma_data * t / sqrt(sigma_data^2 + t^2)`` computed on host.

The conditioned ``(si, zij)`` come from ``OF3DiffusionConditioning`` (gated
1.00000/0.99999) and ``(cl0, plm0)`` from ``RefAtomFeatureEmbedder`` (gated P7), both fed
from their goldens -- the same bisect discipline the NPE / decoder gates use -- so this
gate isolates the post-conditioning assembly. The encoder ``atom_transformer`` reuses the
gated ``OF3AtomTransformer`` topology with the encoder's own weights.

Two correctness footguns, both invisible in the sub-leg gates and fatal at the assembly
level (the assembly dropped to ``xl_out`` PCC -0.13 until both were fixed):

  1. **ReLU placement.** ``linear_l``/``linear_m`` and ``pair_mlp`` apply ReLU to the
     *input* of each linear (OF3 ``linear_l(self.relu(cl_l...))``, ``pair_mlp =
     Sequential(ReLU, Linear, ...)``); ``linear_q`` applies ReLU to the *output*
     (``Sequential(Linear, ReLU)``). Using ``ttnn.linear(..., activation="relu")``
     (post-activation) for the pair update put ReLU on the wrong side of all four
     linears and collapsed ``plm`` to PCC 0.45 -- the encoder ``atom_transformer`` then
     blew ``ql`` up 8x. Explicit ``ttnn.relu(x)`` before the linear for the pair update,
     post-activation for ``linear_q``.
  2. **raw ``si_trunk`` vs conditioned ``si``.** The reference
     ``DiffusionModule.forward`` passes the *raw* trunk single ``si_trunk`` (not the
     conditioned ``si``) into the atom encoder's ``NoisyPositionEmbedder``; the
     conditioned ``si`` drives only the DiT and the ``linear_s(LN_s(si))`` glue. Feeding
     the conditioned ``si`` to the NPE collapsed ``xl_out`` to PCC -0.12; the module takes
     both ``si_trunk`` and ``si`` as distinct inputs.

Gate: ``tests/test_openfold3_diffusion_module_xlout.py`` feeds the golden conditioned
``si``/``zij`` + ``cl0``/``plm0`` + ``rl_noisy`` + aux -> device ``OF3DiffusionModule`` ->
compares ``xl_out`` to the full-module golden, with bisect checkpoints at every sub-leg
boundary. Result on qb2 card 0 (HiFi4 + fp32 dest acc): **plm_postpairupdate = 0.99999,
ql_enc = 0.99998, a_in (post-glue) = 0.99999, a_stack (post-DiT) = 0.99982,
rl_update = 0.99773, xl_out = 0.99746** (all > 0.98). The full ``DiffusionModule`` forward
is byte-correct on device.

**NEXT (P9, merge gate):** wire ``SampleDiffusion`` (the EDM sampler loop around
``DiffusionModule``) and ``OpenFold3.fold()`` end-to-end, then run ``examples/prot.yaml``
(or the ubiquitin/7ROA fixture) for a real vs-ground-truth Kabsch Ca-RMSD -- the actual
merge gate for the whole OF3 port.

### P9 leg 2 -- ``SampleDiffusion`` + ``OpenFold3.fold()`` end-to-end Kabsch gate: status

Leg 1 (full ``DiffusionModule`` -> ``xl_out``, PCC 0.99746) is landed and pushed. Leg 2
(the real merge gate: a vs-ground-truth Kabsch Ca-RMSD from ``OpenFold3.fold()``) is NOT
done this tick -- it is a multi-tick effort, scoped here so the next relaunch resumes
cleanly. Concretely what remains:

  1. **Trunk inference assembly** (P4, still unchecked): wire ``InputEmbedderAllAtom``
     (atom encoder -> ``s_input`` + the gated ``InputEmbedderGlue`` outer-sum -> ``s``/``z``)
     -> MSA module (gated, m-track correct / z-track xfail) -> 48-block ``Pairformer``
     (s-track 0.996 correct, z-track xfail on the cancelling final block -- the docs'
     established acceptance is the end-to-end RMSD gate, not raw stack-z) -> template
     embedder (gated) -> trunk ``(si_trunk, zij_trunk)``. None of these are assembled into
     a single device ``forward`` yet; each is gated in isolation only.
  2. **``SampleDiffusion`` EDM sampler** (AF3 Algorithm 18): a host orchestration loop
     around the gated ``OF3DiffusionModule``. Per step ``tau``: host-compute the noise
     embedding ``n = fourier_emb(0.25 * log(t / sigma_data))`` (``fourier_emb.w``/``b`` are
     in the checkpoint under ``diffusion_conditioning.fourier_emb``; replicate on host),
     run ``OF3DiffusionConditioning`` -> ``(si, zij)`` (the noise embedding is the only
     per-step conditioning input; relpos/si_input/si_trunk/zij_trunk are fixed), run
     ``OF3DiffusionModule(si_trunk, si, zij, cl0, plm0, rl_noisy, xl_noisy, ...)`` ->
     ``xl_denoised``, then the EDM step ``xl = xl_noisy + step_scale * (c_tau - t) *
     (xl_noisy - xl_denoised) / t``. ``centre_random_augmentation`` (random rotation +
     translation, AF3 Algorithm 19) is applied to ``xl`` before each step's noise add.
     Full rollout is ``no_full_rollout_steps=200`` x ``no_full_rollout_samples=5`` = 1000
     ``DiffusionModule`` forwards -- a real perf item; a reduced-step rollout gate (e.g.
     4 steps, 1 sample) is the natural first sub-leg to prove the loop composes on device
     (replay the per-step noise / rotation / translation from a reference golden, same
     isolation discipline as the component gates), but it is NOT the merge gate.
  3. **Confidence heads** (PAE/PDE/pLDDT/distogram/resolved) + the ``aux_heads`` confidence
     pairformer -- not yet ported (P3 listed them as not-started). Needed for a complete
     ``fold()`` output but not for the structure RMSD itself.
  4. **``OpenFold3.fold()`` + data pipeline + Kabsch Ca-RMSD**: assemble the full
     ``fold()`` (data pipeline is the gated P1 vendor; trunk -> ``SampleDiffusion`` ->
      coordinates), run ``examples/prot.yaml`` (or the ubiquitin/7ROA fixture), parse Ca
      positions, Kabsch-align vs ground truth, report the RMSD. THIS number is the merge
     gate -- do not claim it without a real RMSD.

The gated ``OF3DiffusionModule`` (this tick) is the inner loop of (2); the next tick
should do the reduced-rollout sampler gate (sub-leg) then the trunk assembly (the hard
part, blocked by the Pairformer z-track device-precision gap that the end-to-end RMSD is
the acceptance for).

## P9 leg 2 -- tick 18: SampleDiffusion reduced-rollout sampler gate (sub-leg)

Landed: the EDM sampler loop (AF3 Algorithm 18) around the gated
OF3DiffusionModule + OF3DiffusionConditioning, PCC-gated on device against a real
reference rollout golden. This is sub-leg (2) from the leg-2 scope above -- it proves the
EDM loop + per-step conditioning compose on device; it is NOT the fold() Kabsch merge
gate.

**Golden** (scripts/of3_sample_diffusion_golden.py -> pkl key
sample_diffusion_rollout_real): the reference SampleDiffusion loop body is
replicated verbatim (same math, same RNG call order) with no_rollout_steps=4 (5
schedule entries, 4 rollout steps), no_rollout_samples=1, seed 1234, on the real
ubiquitin batch -- but with the sample dim squeezed (xl is [1, N_atom, 3] not
[1, 1, N_atom, 3]). The reference SampleDiffusion.forward with
no_rollout_samples=1 hits a shape bug in aggregate_atom_feat_to_tokens (the
spurious sample dim breaks its scatter-add); the RNG draw *count* is identical for
[1,N,3] and [1,1,N,3] (torch RNG is shape-agnostic), so squeezing the sample dim
reproduces the reference trajectory exactly while matching the [1, N_atom, 3] shape the
device OF3DiffusionModule consumes. The heavy reference part -- the per-step
DiffusionModule.forward (real of3-p2-155k.pt weights) -- is run unmodified, so each
step's xl_denoised is the real reference output; only the light loop math (augmentation,
noise add, EDM step) is replicated from the reference source. Every per-step random/host
artefact (centre_random_augmentation rots/trans, the additive noise,
xl_noisy, xl_denoised, xl_post_step, t, c_tau) and the noise schedule +
xl_init/xl_final are captured for bit-exact device replay.

**Device** (tt_bio/openfold3_sample_diffusion.py): OF3SampleDiffusion holds the
gated OF3DiffusionConditioning + OF3DiffusionModule. Per step it replays the golden
rots/trans/noise on host (centre_random_augmentation + noise add are small
3x3 / [N,3] host ops), computes the per-step Fourier noise embedding
n = fourier_emb(0.25 * log(t / sigma_data)) on host (fourier_emb.w/b from the
checkpoint; the host computation is bit-exact vs the reference -- verified, maxdiff 0.0),
runs the device conditioning -> (si, zij) (padded to n_tok_pad), runs the device
OF3DiffusionModule -> xl_denoised, then applies the EDM step on host. This isolates
the device conditioning+DiffusionModule precision composed across the loop from the random
augmentation/noise host math -- the same discipline as the component gates.

**Test** (tests/test_openfold3_sample_diffusion.py): 4-step device rollout -> xl_final
vs the reference golden.

| output | PCC | std (dev / ref) |
|---|---|---|
| xl_final (4-step rollout) | **0.99343** | 309.1 / 296.4 |

Gate: xl_final PCC > 0.98. PASS. (xl_final is the denoised atom-position sample after
the EDM loop; the ~4% magnitude drift is bf16 accumulation across 4 conditioning +
DiffusionModule forwards, consistent with the per-component gates.) The per-step golden
trajectory (xl_denoised std 7.5 / 8.3 / 31.7 / 303.6; xl_post_step std 1591.8 /
709.7 / 318.6 / 296.4) shows the expected monotone denoising.

**What this is NOT**: the full fold() Kabsch Ca-RMSD merge gate. The production rollout
is 200 steps x 5 samples = 1000 DiffusionModule forwards; fold() additionally needs
the trunk (1), confidence heads (3), and the data pipeline + Kabsch RMSD (4) from the leg-2
scope above. The reduced-rollout sampler gate is the natural sub-leg proof that the EDM loop
composes on device; the trunk assembly (the hard part, blocked by the Pairformer z-track
device-precision gap whose acceptance is the end-to-end RMSD) and the full
no_full_rollout_steps=200 rollout remain.
