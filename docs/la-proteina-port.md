# La-Proteina

La-Proteina (NVIDIA, 2025) is a generative protein-design model that jointly
produces a sequence and a full all-atom structure (backbone + side chains). It
is a non-equivariant transformer denoiser trained with a partially-latent
flow-matching objective: the backbone C-alpha trace is modeled explicitly and
the sequence plus side-chain detail live in a per-residue 8-dimensional latent.
It supports unconditional all-atom generation, motif scaffolding, and long-chain
generation up to ~800 residues.

## License

- Code: Apache-2.0 (vendored under `tt_bio/la_proteina/_vendor/la-proteina-ref`).
- Weights: NVIDIA Open Model License (NOML). Models are commercially usable and
  derivative models may be created and distributed; NVIDIA claims no ownership
  of outputs. Compatible with a free public hosted service. See
  `tt_bio/la_proteina/NOTICE` for the required attribution.

## Status

Port in progress on branch `wk/tt-bio-la-proteina-port-p4` (not merged; subject
to the model-merge-approval-gate).

- Pass 1 cleared the license gate, confirmed the parameter count (~160M
  denoiser, ~130M each for the autoencoder encoder and decoder, ~420M total),
  and scoped the architecture.
- Pass 2 vendored the reference implementation, built a component-level PyTorch
  golden harness, and ported the denoiser's core sequence-side attention block
  to ttnn (PCC 0.9997 on fixed inputs in bf16).
- Pass 3 ported the rest of the denoiser trunk component-by-component against
  the unmodified vendored reference, same random-weight PCC bar (>= 0.999): the
  transition block, the timestep/noise conditioning pathway, the two output
  heads, a full single-block trunk layer, and the non-tri-mult pair update.
- Pass 4 ported the full multi-layer trunk forward and the remaining denoiser +
  autoencoder surfaces, all at the same bar (random weights, bf16, HiFi4 +
  fp32_dest_acc, both all-True and partial masks):
  - the FULL 14-layer denoiser trunk (160M config) + cond + both heads, and the
    160M_tri config (update_pair_repr=True, every_n=2, use_tri_mult=True). Error
    does not compound below 0.999 over 14 layers (local_latents 0.99996, ca
    0.99995 on both configs).
  - the tri-mult pair-representation update path (a self-contained direct ttnn
    port of openfold Algorithms 11/12, wired into PairReprUpdate), exercised
    both as a component and inside the full 160M_tri trunk.
  - the flow-matching Euler integration step (all four sampling modes: vf,
    vf_ss, sc, vf_ss_sc_sn) for both data modalities (bb_ca d=3, local_latents
    d=8), with the stochastic noise draw fed as an explicit shared input so the
    device port and the CPU reference see identical draws (PCC 0.999996-0.999998
    across 18 cases).
  - the autoencoder encoder (latent head, shared-eps z) and decoder (sequence
    logits + all-atom coordinate head with the abs_coors post-process), 12-layer
    trunks each (PCC 0.99996-0.99999).

Real-weight parity is still blocked on NGC checkpoint access (the NGC catalog
serves the `.ckpt` via a browser-auth file-browser, not a direct download), so
parity is on random/seeded weights for now. The full nsteps Euler sampler LOOP
around the denoiser is gated on the FeatureFactory/PairReprBuilder dataset
feature-pipeline port (the same gate that blocks the full end-to-end forward
with real input features); the per-step integrator math is ported and
parity-verified. No end-to-end real-weight parity is claimed yet. See
`~/.coworker/notes/tt-bio-la-proteina-port-p4.md` for the per-pass detail.
