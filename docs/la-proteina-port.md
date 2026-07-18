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

Port in progress on branch `wk/tt-bio-la-proteina-port-p3` (not merged; subject
to the model-merge-approval-gate).

- Pass 1 cleared the license gate, confirmed the parameter count (~160M
  denoiser, ~130M each for the autoencoder encoder and decoder, ~420M total),
  and scoped the architecture.
- Pass 2 vendored the reference implementation, built a component-level PyTorch
  golden harness, and ported the denoiser's core sequence-side attention block
  to ttnn (PCC 0.9997 on fixed inputs in bf16).
- Pass 3 ported the rest of the denoiser trunk component-by-component against
  the unmodified vendored reference, same random-weight PCC bar (>= 0.999): the
  transition block (TransitionADALN), the timestep/noise conditioning pathway
  (two SwiGLU transitions on the conditioning vector), the two output heads
  (C-alpha coordinate head and the per-residue 8-D latent head), a full
  single-block trunk layer (attention + transition, sequential, both residual),
  and the pair-representation update (non-tri-mult path). All clear the bar on
  both all-True and partial masks in bf16.

Real-weight parity is still blocked on NGC checkpoint access (the NGC catalog
serves the `.ckpt` via a browser-auth file-browser, not a direct download), so
parity is component-level on random/seeded weights for now. The full multi-layer
denoiser forward, the Euler sampler loop, the autoencoder, and the tri-mult
pair-update path are follow-on passes. No end-to-end parity is claimed yet.
