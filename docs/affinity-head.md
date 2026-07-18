# Standalone affinity head (PLAPT-style) — pass 1

Predicts protein-ligand binding affinity (Kd/IC50 as neg_log10_affinity_M) from
**sequence + SMILES only** — no structure, no folding. A frozen protein PLM
(ProtBERT) and a small ligand encoder (ChemBERTa-zinc-base-v1) feed a light
fusion MLP that emits a normalized affinity, rescaled to pKd-like units.

Reference: [PLAPT](https://github.com/Bindwell/PLAPT) (Bindwell, MIT) — ProtBERT
+ ChemBERTa pooler outputs concatenated, fed to a small branching MLP.

## Status (pass 1)

This is a **pass-1 port**: the two portable components are on Tenstorrent with
verified per-component PCC parity vs a from-scratch PyTorch reference. The
protein tower and end-to-end wiring are deferred to pass 2.

| Component | Port | Parity (PCC) | Weights |
|---|---|---|---|
| ChemBERTa ligand encoder (6-layer RoBERTa + pooler, ~43M) | ttnn, done | embeddings 0.99999, layer 0.99999, pooler 0.99994, full 0.99986 (real weights) | `seyonec/ChemBERTa-zinc-base-v1` (HF, MIT) |
| Fusion MLP head | ttnn, done | 0.9997 (real ONNX weights) | PLAPT `affinity_predictor.onnx` (MIT), vendored in `tt_bio/_vendor/plapt/head_weights.npz` |
| ProtBERT protein tower | host-side torch only (not ported) | n/a | `Rostlab/prot_bert` (HF) |
| End-to-end + SMILES tokenization + benchmark | deferred | n/a | — |

Run the parity tests on a TT card:

```bash
TT_VISIBLE_DEVICES=1 python tests/test_affinity.py
```

## Protein tower reuse decision

PLAPT's fusion head is **ProtBERT-specific**: its protein branch takes a
1024-d pooler input (`ProtLinear_Weights` is `[512, 1024]`) and was trained on
ProtBERT pooler semantics. tt-bio's existing ESMC port produces 960-d
embeddings (ESMC-300M) — dimensionally incompatible, and a different PLM with
different pooling. So ESMC cannot drop-in substitute ProtBERT for the PLAPT
head; reusing ESMC would require retraining the head on ESMC embeddings (a
training task, out of scope for an inference port).

For pass 1 the protein tower is kept host-side (frozen ProtBERT, CPU torch) so
the port stays parity-faithful to the PLAPT reference. Porting ProtBERT to ttnn
(BERT-large, ~420M, standard post-LN transformer — same pattern as the ESMC
encoder) and/or retraining the head on ESMC embeddings are pass-2+ research
directions, flagged here honestly rather than forced.

## Architecture (from the PLAPT source + ONNX graph)

- Protein: `Rostlab/prot_bert` (BERT-large, hidden 1024), `pooler_output`
  (tanh(dense(CLS))). Sequence preprocessed: space-separated, U/Z/O/B → X,
  `max_length=3200`.
- Ligand: `seyonec/ChemBERTa-zinc-base-v1` (RoBERTa, 6 layers, hidden 768, 12
  heads, intermediate 3072, gelu, LN eps 1e-5, vocab 767, max_pos 514),
  `pooler_output`. `max_length=278`.
- Fusion: concat `[prot_pooler(1024) || mol_pooler(768)]` → branch A
  `Linear(1024→512)+ReLU`, branch B `Linear(768→512)+ReLU` → concat 1024 →
  `BatchNorm` → `Linear(1024→512)+ReLU` → `Linear(512→64)+ReLU` →
  `Linear(64→1)`.
- Output: `neg_log10_affinity_M = out * 1.5614094578916633 + 6.51286529169358`.

## License

PLAPT code + the affinity-predictor weights: MIT (Bindwell 2024), vendored under
`tt_bio/_vendor/plapt/` (see `LICENSE-PLAPT`, `NOTICE`). ChemBERTa tokenizer +
config: MIT (Seyonec); weights fetched at runtime from HuggingFace.
