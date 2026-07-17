# SaProt structure-aware embedding parity (on-hardware, real weights)

SaProt (westlake-repl) is an ESM-2 masked-LM encoder over a fused structure-aware
vocabulary: 20 amino acids x 21 Foldseek 3Di states plus 5 special tokens (446 total). The
tt-bio port (`tt_bio.saprot`, exposed as `tt-bio saprot`) is a from-scratch ttnn implementation
of that ESM-2 encoder; the reference is the canonical HuggingFace `EsmForMaskedLM`
(westlake-repl/SaProt_*_AF2). Same trained checkpoint loaded into both; the ttnn path runs the
encoder trunk plus the MLM head. **Per-residue embedding PCC** and **MLM-logits PCC** vs the
reference are the gate.

Reproduce (single device):

```bash
FOLDSEEK_BIN=/path/to/foldseek TT_VISIBLE_DEVICES=0 \
  python scripts/saprot_parity.py --model saprot-650m
```

## Results — Blackhole (qb1 card 0)

| model | seq (aa) | per-residue emb PCC | MLM logits PCC | argmax |
|---|---|---|---|---|
| saprot-650m | ubiquitin (76) | **0.999638** | 0.999927 | — |

Input is ubiquitin paired with a deterministic 3Di string (the 3Di content does not affect
parity — both paths see identical tokens). Length is bucketed to 128 (a multiple of 64) so
the fused rotary-embedding kernel runs on-device; padded positions are masked out of
attention and zeroed in the embedding, so the 76 real residues are identical to a
no-padding forward.

## Verdict

**Pass.** Per-residue embedding PCC 0.9996 and MLM-logits PCC 0.9999 for saprot-650m, in line
with the ESMC port's 0.9995–0.9999 band. The structure-aware embeddings — the capability's
product — match the reference to within bf16 noise.

## What is deferred

- **saprot-35m**: ESM-2 35M has `head_dim = 24`, which is neither tile-aligned (32) nor
  aligned with the fused on-device `rotary_embedding` kernel (which needs `head_dim % 64 == 0`).
  The elementwise rotate-half fallback also breaks on a tile-padded head (it splits the
  padded 32, not the real 24). A correct port needs host-side RoPE or head_dim padding to
  64 — deferred. The 650M (`head_dim = 64`) is unaffected and is the recommended variant.
- **saprot-1.3b**: same ESM-2 family, `head_dim = 64`; not yet parity-run in this cut
  (weights are public; the port loads it via the same path — left for a follow-up).
- **Multi-card fanout**: the embed-style data-parallel fanout transfers verbatim (SaProt
  embeddings are row-independent); not wired into the `saprot` CLI in this cut.

## Warm throughput (single card)

`saprot-650m`, batch 32 x bucketed length 128, warm: **~33k tokens/s** (123 ms/forward) on
one Blackhole card.
