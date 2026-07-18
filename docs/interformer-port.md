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

## Parity state (measured on qb1 card 0, bf16, HiFi4)

Reference: a from-scratch PyTorch reimplementation
(`tests/interformer_reference.py`, weight-compatible with the source — no
pytorch_lightning / torchmetrics / obabel dependency for parity). The on-device
port is built from the **same** state_dict; host glue (embeddings, RBF, masks)
is computed by the reference and fed **identically** to both sides, so the
comparison isolates the on-device dense math.

**Real-weight parity** — both sides loaded from the released Zenodo affinity
checkpoint (`v0.2_affinity_model/model0`, exact hparams read from the
checkpoint: hidden=128, 8 heads, 6+3 layers, node_feat_size=1). The reference
loads the real state_dict strict (0 missing / 0 unexpected). `b=1`, `n=64`.

| Component | Metric | Value | Gate |
|---|---|---|---|
| `rel_pos_3d_proj` (RBF→hidden Linear) | PCC | 1.00006 | ≥0.999 pass |
| EncoderLayer — node output | PCC | 0.99989 | ≥0.999 pass |
| EncoderLayer — edge output | PCC | 1.00003 | ≥0.999 pass |
| Full backbone (intra×6 + inter×3) — inter_node | PCC | 0.99990 | ≥0.999 pass |
| Full backbone — affinity (scalar) | absdiff | 0.01726 | <0.05 pass |
| Affinity readout in isolation (scalar) | absdiff | 0.00813 | <0.05 pass |

Gate: PCC ≥ 0.999 for the dense tensors; the affinity scalar is checked by
absdiff (PCC is undefined for a single element). **All pass with real weights.**
Numbers are measured, not estimated. Run:
`TT_VISIBLE_DEVICES=0 PYTHONPATH=.:tests python3 tests/test_interformer_realweights.py`
(the pass-1 random-weight test, `tests/test_interformer.py`, also still passes).

## End-to-end on a real complex (affinity)

The on-device port runs the full released data pipeline on the 2qbr demo complex
(pocket + native ligand, 258 graph atoms after PLIP featurization) and produces
a real affinity prediction:

| Output | Value |
|---|---|
| Reference affinity (real weights, fp32) | 6.7666 |
| Port affinity (real weights, bf16 HiFi4) | 6.75 |
| Experimental pIC50 label (demo_dock.csv) | 6.33 |
| inter_node PCC (ref vs port) | 0.99986 |
| affinity absdiff (ref vs port) | 0.0166 |

The port matches the reference on a real complex (PCC 0.99986, absdiff 0.0166),
and both land within ~0.42 pIC50 of the experimental label — inside the model's
single-complex error (PDBbind core-set RMSE ≈ 1.0–1.3 pIC50). The native pose is
scored directly, so this does not exercise the docking sampler.

## What is deferred (honest)

- **Pose-RMSD end-to-end** (the full PoseBusters pipeline). Reproducing a docked
  pose needs the energy model (`v0.2_energy_model`, in `checkpoints.zip`) to
  predict per-pair Gaussian energy functions, then the compiled C++ docking
  sampler (`docking/reconstruct_ligands`, built via `python setup.py install`)
  to generate poses, then the affinity model to score them. The C++ sampler
  build is the hard blocker — not yet attempted. The affinity e2e above sidesteps
  it by scoring the native pose.
- **`--fast` (block-fp8) path, load/predict timing, multi-card fanout** — perf
  characterization, after pose-RMSD e2e.
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
