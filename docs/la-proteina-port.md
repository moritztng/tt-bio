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

Port in progress on branch `wk/tt-bio-la-proteina-port-p6` (not merged; subject
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

- Pass 5 ported the dataset feature pipeline and wired the full sampler loop
  end-to-end (random weights, bf16 trunk + factory, fp32 Euler score math,
  HiFi4 + fp32_dest_acc, both all-True and partial masks):
  - the `FeatureFactory` / `PairReprBuilder` feature pipeline that rebuilds
    `seqs` / `c` / `pair_rep` from `x_t` / `t` (and self-conditioning `x_sc`)
    at each sampler step -- the piece Pass 4 bypassed by injecting at the
    post-builder interface. Verified in isolation against the vendored
    `FeatureFactory` / `PairReprBuilder`: seq factories at 0.99999, pair builder
    at 0.996 in bf16 (LayerNorm-over-256 precision) and 1.0000 in fp32 (no bug).
  - the full nsteps Euler sampler LOOP around the denoiser (denoiser trunk +
    Euler integrator + builders, with shared stochastic draws via a resettable
    `torch.Generator`), parity vs the reference loop output on final coordinates
    across 3 seeds x nsteps {3,4,5,6} x both data modes (PCC 0.99986-0.99999).

- Pass 6 did the performance work on the random-weight sampler loop (real
  weights still blocked):
  - Shipped a device-resident cache for the deterministic features that depend
    only on the sequence length (not on `x_t` / `t` / `x_sc`): `rel_seq_sep`,
    `optional_ca_pair_dist`, and the optional zeros seq features are built once
    and reused every step instead of recomputed on host and re-shipped
    host->device each step. Bit-identical, parity-verified (same PCCs as pass 5).
    Wall-clock: 22.0 -> 20.8 ms/step (5.5%); nsteps=5 110 -> 104 ms on card 0.
  - Investigated ttnn trace capture for the per-step trunk. The N=64 trunk is
    host-dispatch-bound (1.44x in isolation: 12.82 -> 8.91 ms), but the trace
    breaks in-loop because the eager Euler step running between trace replays
    corrupts the trace's intermediate buffer pool -- the same ttnn trace +
    interleaved-eager buffer-aliasing issue that led Boltz-2 to drop trace. Not
    shipped; a future pass could land it by making the Euler device-resident so
    the whole denoiser+Euler traces as one unit.

Real-weight parity is still blocked on NGC checkpoint access (the NGC catalog
serves the `.ckpt` via a browser-auth file-browser, not a direct download), so
parity is on random/seeded weights for now. The full sampler loop is wired,
parity-verified, and (pass 6) measurably faster on random weights; what remains
is real-weight parity (NGC-blocked) and the in-loop trunk trace. See
`~/.coworker/notes/tt-bio-la-proteina-port-p6.md` for the per-pass detail.
