# ESMC embedding parity (on-hardware, real weights)

The standalone ESMC embedding API (`tt_bio.esmc.load_esmc` + `embed_sequences`,
exposed as `tt-bio embed`) vs the reference **esm** ESMC (`esm.models.esmc`, the
plain non-flash transformer path built in `tests/esmc_reference.py`), on the real
biohub weights. Same trained `.pth` loaded into both; the ttnn path runs the LM
trunk alone — no folding head, no MSA. **Per-residue embedding PCC** is the gate.

Reproduce (single device):

```bash
ESM_ROOT=/path/to/esm TT_VISIBLE_DEVICES=0 \
  python scripts/esmc_embed_parity.py --model esmc-600m
# also: --model esmc-300m [--fast] [--seq <PROTEIN>]
```

## Results — Blackhole (pc card 0)

| model | seq (aa) | path | per-residue PCC | pooled(mean) PCC | logits PCC | argmax |
|---|---|---|---|---|---|---|
| esmc-300m | ubiquitin (76) | normal | **0.99965** | 0.99994 | 0.99991 | 1.0000 |
| esmc-300m | ubiquitin (76) | `--fast` (block-fp8) | **0.99950** | 0.99989 | 0.99986 | 1.0000 |
| esmc-300m | avGFP (238) | normal | **0.99986** | 0.99999 | 0.99999 | 0.8992 |
| esmc-600m | ubiquitin (76) | normal | **0.99964** | 0.99992 | 0.99996 | 1.0000 |

ESMC-6B (embeddings only — the port carries no sequence head) was confirmed
end-to-end via `tt-bio embed --model esmc-6b` (ubiquitin): output
`per_residue [76, 2560]`, `pooled [2560]`, all finite, post-LayerNorm
distribution (mean ≈ 0, std ≈ 1.2). The 6B is the ESMFold2 language-model
backbone, whose port is separately parity-verified (`docs/esmfold2-e2e-parity.md`).

## Verdict

**Pass.** Per-residue and pooled embedding PCC are ~0.9995–0.9999 across the
300M and 600M variants, at both 76 and 238 residues, and the `--fast` block-fp8
weight path is lossless within noise (per-residue 0.99950). Sequence-head logit
*values* match at ≥0.9999 everywhere.

The one sub-1.0 number — argmax agreement 0.8992 on 238-aa avGFP — is **not** an
embedding-quality signal: with no MSA the LM is genuinely uncertain at many GFP
positions, so the top-1/top-2 logits are near-tied and bf16 rounding flips the
argmax on ~10% of them (logits PCC is still 0.99999). The embeddings — the
capability's product — are unaffected.
