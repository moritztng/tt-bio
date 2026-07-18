# Sequence-based affinity prediction (PLAPT-style)

Predicts protein-ligand binding affinity (pKd) from **sequence + SMILES only** —
no structure, no folding. A frozen protein language model (ProtBERT) and a small
ligand encoder (ChemBERTa-zinc-base-v1) feed a light fusion MLP that emits a
normalized affinity, rescaled to `neg_log10_affinity_M` (pKd).

Reference: [PLAPT](https://github.com/Bindwell/PLAPT) (Bindwell, MIT) — ProtBERT
+ ChemBERTa pooler outputs concatenated, fed to a small branching MLP.

## Run it

```bash
# single pair
tt-bio affinity --protein MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG \
                --smiles "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F"

# batch (CSV with protein,smiles columns)
tt-bio affinity --pairs pairs.csv --out_dir ./affinity
```

Output (`--out_dir`, default `./affinity`):

```
affinity.json   # one row per pair: protein, smiles, neg_log10_affinity_M, affinity_uM
```

`neg_log10_affinity_M` is the pKd (higher = tighter binder); `affinity_uM` is
the same value in micromolar. Weights for ProtBERT and ChemBERTa are fetched
from HuggingFace on first use and cached; the fusion-head weights and both
tokenizers are vendored under `tt_bio/_vendor/plapt/`.

## Inputs

- **Protein**: a raw amino-acid string. `U/Z/O/B` are mapped to `X` (ProtBERT's
  unknown residue), residues are space-separated, `[CLS] … [SEP]` added, and the
  sequence is truncated to 3200 residues (ProtBERT's max length).
- **SMILES**: a ligand SMILES string. Tokenized with the vendored ChemBERTa
  RoBERTa BPE (pure-python, no `transformers` runtime dependency), `[s] … [/s]`
  added, truncated to 278 tokens.

## Accuracy

On the PLAPT CSAR-HiQ_36 held-out benchmark (36 protein-ligand pairs with
experimental pKd), the on-device pipeline scores:

| Metric | Value |
|---|---|
| Pearson r | 0.724 |
| RMSE | 1.365 pKd |
| MAE | 1.165 pKd |

This is PLAPT's inherent sequence-only accuracy — the on-device port matches
the original PLAPT pipeline (HF ProtBERT + HF ChemBERTa + the ONNX fusion head)
to PCC 0.9986 / MAE 0.078 pKd on the same pairs, so no accuracy is lost in the
port. Reproduce with `tests/test_affinity.py::test_e2e_affinity` (parity) and
the CSAR runner noted in the pass-2 notes (accuracy).

## Parity vs the PLAPT reference

Per-component PCC (real weights, on-device vs a from-scratch PyTorch reference
in `tests/affinity_reference.py`):

| Component | PCC |
|---|---|
| ProtBERT embeddings | 0.99999 |
| ProtBERT layer 0 | 0.99996 |
| ProtBERT pooler | 0.99999 |
| ProtBERT full (pooler) | 0.99999 |
| ChemBERTa embeddings | 0.99999 |
| ChemBERTa pooler | 0.99994 |
| ChemBERTa full (real weights) | 0.99986 |
| Fusion head (real ONNX weights) | 0.9997 |
| SMILES tokenizer | bit-exact vs `RobertaTokenizer` |

End-to-end pKd on held-out pairs is within ~0.18 of the fp32 PLAPT reference;
the residual is bf16 inference noise in the ChemBERTa pooler (the fusion head
itself contributes <0.02 pKd). Run the parity suite on a TT card:

```bash
TT_VISIBLE_DEVICES=0 python tests/test_affinity.py
```

## License

PLAPT code + the affinity-predictor weights: MIT (Bindwell 2024), vendored under
`tt_bio/_vendor/plapt/` (see `LICENSE-PLAPT`, `NOTICE`). ChemBERTa tokenizer +
config: MIT (Seyonec); ProtBERT and ChemBERTa weights are fetched at runtime
from HuggingFace (`Rostlab/prot_bert`, `seyonec/ChemBERTa-zinc-base-v1`).
