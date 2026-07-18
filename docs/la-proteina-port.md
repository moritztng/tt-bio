# La-Proteina

La-Proteina (NVIDIA, 2025) is a generative protein-design model that jointly
produces a sequence and a full all-atom structure (backbone + side chains). It
is a non-equivariant transformer denoiser trained with a partially-latent
flow-matching objective: the backbone C-alpha trace is modeled explicitly and
the sequence plus side-chain detail live in a per-residue 8-dimensional latent.
It supports unconditional all-atom generation, motif scaffolding, and long-chain
generation up to ~800 residues.

## License

- Code: Apache-2.0.
- Weights: NVIDIA Open Model License (NOML). Models are commercially usable and
  derivative models may be created and distributed; NVIDIA claims no ownership
  of outputs. Compatible with a free public hosted service. See
  `tt_bio/la_proteina/NOTICE` for the required attribution.

## Status

Port in progress on branch `wk/tt-bio-la-proteina-port` (not merged). Pass 1
cleared the license gate, confirmed the parameter count (~160M denoiser,
~130M each for the autoencoder encoder and decoder, ~420M total), and scoped
the architecture. The flow-matching denoiser transformer is the first port
target because it reuses tt-bio's existing pair-biased-attention trunk,
triangular-multiplicative-update primitive, and Euler sampler loop. The
autoencoder is a follow-on pass. No numerical parity has been claimed yet.

See `~/.coworker/notes/tt-bio-la-proteina-port-p1.md` for the full pass-1
findings and pass-2 plan.
