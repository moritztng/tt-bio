# Interformer port — tt-bio

Interformer (Tencent AI4S, Apache-2.0) is a graph-transformer for protein–ligand
docking: it predicts an interaction-aware energy function per protein–ligand atom
pair (used by a Monte-Carlo sampler to generate poses) and, in a second head, a
pose-sensitive affinity + pose score. This is the fast pose-prediction /
screening layer between tt-bio's sequence-only affinity pre-filter and full
Boltz-2 co-folding — distinct from Boltz-2's per-complex co-folding.

License: code **and** the full released weight set are Apache-2.0 (repo
`LICENSE`; Zenodo record 10828798, `license.id = apache2.0`, covers
`checkpoints.zip`). Confirmed before any port effort.

## The hybrid split (the key design decision)

Interformer is a **hybrid host + device** port. The transformer trunk is dense
and TT-friendly; the pose-generation machinery is irregular and stays on host.

**On device (this port — `tt_bio/interformer.py`):** the interaction-aware
(edge-biased) transformer encoder and the affinity readout. Pure dense math —
matmul, LayerNorm, softmax, FFN — and the accuracy-relevant trunk.
- `ComplexEncoder.rel_pos_3d_proj` — the RBF→hidden Linear projection.
- `EncoderLayer` stack: `intra` × `n_layers` (per-molecule) + `inter` ×
  `n_layers//2` (cross-molecule). Each layer is pre-LN self-attention with
  **edge-biased logits** `logit_ij = Σ_d q_i · k_j · e_ij` (a 3-way q·k·e
  interaction, not standard additive-bias attention), a node FFN (ReLU), and an
  edge FFN that updates the pair features.
- Affinity readout: `final_ln` on the virtual-node token + a PReLU FFN → scalar
  pIC50. (Pose-selection head is the same shape — deferred, see below.)

**On host (NOT ported — irregular glue):**
- Graph construction: RDKit / OpenBabel atom typing (29 GNINA atom types),
  Reduce protonation, 10 Å pocket extraction, CCD-ligand handling.
- Distance matrix `D`, the RBF expansion (cutoff polynomial × Gaussian basis),
  the learned atom / edge **embeddings** (gather ops), padding, and the
  attention-bias / pair masks.
- The MDN / `VinaScoreHead` (energy mode): per-pair Gaussian heads
  (`mean`/`sigma`/`Weight`) over pair embeddings, the VdW / hydrophobic / H-bond
  pair-type lookup tables, and the `shelve` energy-file output consumed by the
  sampler.
- The Monte-Carlo / differential-evolution docking sampler and energy
  minimization (`docking/reconstruct_ligands`, compiled C++).

The device forward takes host-prepared `node_feats`, `intra`/`inter edge_feats`,
and `intra`/`inter attn_bias`, and returns the affinity scalar — the same
interface the reference exposes for parity.

## Current parity state (measured on qb1 card 1, bf16, HiFi4)

Reference: a from-scratch PyTorch reimplementation
(`tests/interformer_reference.py`, weight-compatible with the source — no
pytorch_lightning / torchmetrics / obabel dependency, neither of which is in the
dev env). The on-device port is built from the reference's state_dict (identical
weights); host glue (embeddings, RBF, masks) is computed by the reference and fed
**identically** to both sides, so the comparison isolates the on-device dense
math. Random weights, `b=1`, `n=64`, hidden=128, 8 heads, 6+3 layers.

| Component | Metric | Value |
|---|---|---|
| `rel_pos_3d_proj` (RBF→hidden Linear) | PCC | 1.0000 |
| EncoderLayer — node output | PCC | 0.99998 |
| EncoderLayer — edge output | PCC | 1.0000 |
| Full backbone (intra×6 + inter×3) — inter_node | PCC | 0.99986 |
| Full backbone — affinity (scalar) | absdiff | 0.0092 |
| Affinity readout in isolation (scalar) | absdiff | 0.0014 |

Gate: PCC ≥ 0.999 for the dense tensors; the affinity scalar is checked by
absdiff (PCC is undefined for a single element). **All pass.** Numbers are
measured, not estimated. Run: `TT_VISIBLE_DEVICES=1 PYTHONPATH=. python3
tests/test_interformer.py`.

## What is deferred (honest)

- **Real-weight parity.** Random-weight parity clears the gate above; loading
  the released Zenodo affinity checkpoint (1.1 GB `checkpoints.zip`) into both
  the reference and the port for real-weight PCC is the next step. The port's
  submodule names mirror the source, so a strict `load_state_dict` should load
  directly once the checkpoint's exact hyperparameters (n_layers / hidden_dim /
  edge_feat_size) are read from its hparams.
- **End-to-end on one real complex (pose RMSD + affinity).** The stock
  Interformer inference pipeline needs OpenBabel + Reduce + PLIP +
  pytorch_lightning + the compiled C++ docking sampler — none are in the qb1 dev
  env. Reproducing a single PoseBusters pose is therefore blocked on environment
  setup (a host-side, non-port task), not on the device port. The on-device
  trunk is verified against the reference forward; wiring it into the host
  MDN/docking pipeline is the integration step that follows.
- **`--fast` (block-fp8) path, load/predict timing, multi-card fanout** —
  perf characterization, after real-weight parity and end-to-end wiring.
- **Pose-selection head + energy/MDN head on device** — the energy head's
  pair-type gather / shelve output is irregular and intentionally host; the
  pose-selection head is a small FFN that could be ported trivially if needed.

## Generalizable lesson

Interformer's attention is `logit_ij = Σ_d q_i k_j e_ij` — a **multiplicative**
per-pair edge bias, not the additive bias `ttnn.transformer.scaled_dot_product_attention`
takes. It cannot use SDPA; it needs a manual `qk_e = q·k·e` broadcast-multiply
then reduce over `d`, then softmax + `@v`. The 3-way intermediate
`[b·h, N, N, d_head]` is the memory cost (d_head=16 here, so small). Name:
"Interformer edge-biased attention is multiplicative (q·k·e), not SDPA-able —
manual broadcast-multiply + reduce + softmax + matmul."
