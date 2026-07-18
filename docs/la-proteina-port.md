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

Port in progress on branch `wk/tt-bio-la-proteina-port-p2` (not merged; subject
to the model-merge-approval-gate).

- Pass 1 cleared the license gate, confirmed the parameter count (~160M
  denoiser, ~130M each for the autoencoder encoder and decoder, ~420M total),
  and scoped the architecture.
- Pass 2 vendored the reference implementation, built a component-level PyTorch
  golden harness, and ported the denoiser's core sequence-side attention block
  to ttnn. The block (adaptive layer norm + pair-biased attention with QK-LN,
  pair bias, and gated output + adaptive output scale) matches the reference at
  PCC 0.9997 on fixed inputs in bf16, clearing the tt-bio parity bar.

The flow-matching denoiser transformer (full trunk: pair-representation update
with optional triangular multiplicative update, the transition block, and the
Euler sampler loop) and the autoencoder are follow-on passes. No numerical
parity has been claimed end-to-end yet.
