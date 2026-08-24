# X-Cell

X-Cell predicts the transcriptional response to a CRISPRi gene knockdown: given a set of control
single cells and a target gene symbol, it predicts the perturbed transcriptome. It is a set-level
diffusion language model over genes, not a structure model, and it is the first transcriptomics
model in tt-bio.

**There are no public trained weights.** tt-bio ships the architecture and its measured device
performance. `tt-bio perturb` says so and exits unless you pass `--architecture-only`.

## Status

| | |
|---|---|
| Architecture | complete, PCC > 0.998 per component against our torch reference |
| Performance | measured on a p150, 15.6 TFLOP/s at G'=4000 |
| Trained weights | **do not exist publicly** |
| Accuracy | **unmeasurable, and no claim is made** |
| Licence | CC BY-NC-SA 4.0. Non-commercial, so not on the hosted platform |

As of 2026-08-24 the [HuggingFace repo](https://huggingface.co/Xaira-Therapeutics/X-Cell) holds
three files and no checkpoint, and upstream's `from_pretrained` and `predict` both raise
`NotImplementedError`. `scripts/xcell_watch.py` checks whether that has changed and exits non-zero
the day it does.

## Usage

```bash
# Reports that upstream has not released weights, and exits.
tt-bio perturb control_cells.h5ad --perturbation BRCA1

# Runs the real network on the card with random weights, to measure the shape.
# The output is correctly shaped and biologically meaningless.
tt-bio perturb control_cells.h5ad --perturbation BRCA1 --architecture-only
```

Input is an `.h5ad` of control cells (expression in `.X`, gene symbols in `.var_names`) or an
`.npz` holding `expression` `[cells, genes]` and `gene_names`. Reading `.h5ad` needs
`pip install anndata`, which tt-bio does not install by default.

Expression is normalised to log1p CP10k unless you pass `--pre_normalized`. The knockdown target
must be present in the dataset: X-Cell forces it into the gene subsample, so a target the matrix
does not carry cannot be represented.

`--n_genes` is the lever that matters for cost. Attention over genes is about 72% of the
arithmetic at the published 4000, so cost grows faster than linearly in this.

## What the model is

55M parameters, 12 layers, hidden dim 512, 8 heads, initialised from scGPT. Cross-attention at
layers 2, 5, 8 and 11 conditions each gene on six prior-knowledge tokens for the perturbation
(ESM-2, STRING, GenePT, DepMap, JUMP-Cell Painting, and the gene's own embedding). Inference runs
4 coarse-to-fine refinement steps, revealing 25%, 50%, 75% and 100% of genes in turn.

Two of the six priors are worth knowing about if you plan to supply your own: they are per-gene
vectors that upstream pre-computed and has not published, and X-Cell's ESM-2 prior is
5120-dimensional, which is ESM-2 15B rather than the 650M tt-bio pins elsewhere.

## Performance

Measured on one Blackhole p150, warm, random weights. Architecture-only: these are real costs and
carry no accuracy meaning.

| Genes G' | Cells per call | Time | Cells/s | TFLOP/s |
|---:|---:|---:|---:|---:|
| 512 | 8 | 38.9 ms | 205.7 | 6.5 |
| 2048 | 8 | 146.6 ms | 54.6 | 11.1 |
| 4000 | 8 | 301.5 ms | 26.5 | 15.6 |
| 4000 | 32 | 1452.3 ms | 22.0 | 13.0 |

One `predict` at the published defaults (64 cells, batch 8, 4 steps) is 1203 TFLOP at G'=4000 and
runs in about 93 s on one card.

The model is shape-limited rather than implementation-limited. At hidden dim 512 every projection
is a 512x512 matmul, which caps at 32 TFLOP/s on this card against 194 TFLOP/s for a large square
one, and at long gene contexts the model is attention-bound where the measured SDPA roof is 15.5
TFLOP/s. It is not dispatch-bound: throughput is linear in the cell count from 8 cells upward.

Reproduce with `scripts/xcell_perf.py` and `scripts/xcell_roof.py`.

## Why there is a torch reference in the tree

`tt_bio/xcell_reference.py` is a transcription of the preprint's Appendix A, and it is the PCC
target the ttnn port is scored against, because there is no upstream implementation to score
against. It is our own code, written from the paper's description, and it does not derive from
anything Xaira has published. Verify it with `scripts/xcell_parity.py`.

The reconstruction has one strong external check: the paper's larger variant, X-Cell-Ultra, is
published at 4.87B parameters and this reference builds it at 4.860B, 0.2% off. That agreement is
only reachable if the block shape, the feed-forward width, the layer count, the tied head and all
11 cross-attention blocks are simultaneously right.

Three widths the paper never states are config fields rather than guesses: the prior-MLP hidden
width, the two decoder hidden widths, and the norm epsilon. A real checkpoint resolves all three
by inspection.

## The gene axis is deliberately not padded

tt-bio pads every model's token axis to a multiple of 32, because a ragged tile tail can reach an
attention kernel as unmasked key columns. X-Cell's gene axis is the exception, and it is measured
rather than assumed:

- ttnn's attention kernel masks its own ragged tail when the caller supplies no additive bias.
  Relative error against a torch reference is 0.068 at 98 tokens and 0.062 at 450, against 0.028
  at 32 and 0.040 at 64: the same floor ragged as aligned. End-to-end model PCC is 0.9993 at a
  ragged 65 genes against 0.9988 at an aligned 32.
- Padding would force a mask, and the kernel refuses the cheap key-only broadcast form, so the
  mask would be a 32 MB tensor per call at 4000 genes.
- Padding without masking is genuinely wrong: 3.1x the reference error.

The axis that *is* always ragged is the six-token prior context, since 6 is never a multiple of
32. That one is padded to a full tile and masked on every call, with tile padding and an absent
prior source taking the same path.

The cell-set axis carries no reduction at all. X-Cell is set-level in its training objective and
per-cell in its inference graph, so cells never mix inside a forward pass.

See `tt_bio/token_axis.py` for the row and its measurements.

## Licence

X-Cell is CC BY-NC-SA 4.0. Non-commercial, so it is not offered on the hosted platform, and its
weights must not be redistributed in any converted form: fetch from HuggingFace and convert
locally. tt-bio's own implementation is separate work and carries tt-bio's licence.
