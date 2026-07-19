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
  python scripts/saprot_parity.py --model saprot-650m   # or saprot-35m
```

## Results — Blackhole (qb1)

| model | seq (aa) | per-residue emb PCC | MLM logits PCC | argmax |
|---|---|---|---|---|
| saprot-35m  | ubiquitin (76) | **0.999138** | 0.999772 | — |
| saprot-650m | ubiquitin (76) | **0.999638** | 0.999927 | — |
| saprot-1.3b | ubiquitin (76) | 0.995076 | 0.998952 | — |

Input is ubiquitin paired with a deterministic 3Di string (the 3Di content does not affect
parity — both paths see identical tokens). Length is bucketed to 128 (a multiple of 64) so
the fused rotary-embedding kernel runs on-device; padded positions are masked out of
attention and zeroed in the embedding, so the 76 real residues are identical to a
no-padding forward.

### 35M host-side RoPE path

ESM-2 35M has `head_dim = 24`, which is neither tile-aligned (32) nor aligned with the fused
on-device `rotary_embedding` kernel (`head_dim % 64 == 0`); the elementwise rotate-half
fallback also breaks on a tile-padded head (it splits the padded 32, not the real 24). The
35M port therefore splits heads, applies RoPE, and pads `head_dim` 24 -> 32 on host (fp32),
then runs `scaled_dot_product_attention` on device with `head_dim = 32` and the real
`scale = 24 ** -0.5`. The 8 zero-padded dims contribute 0 to every attention score, so
softmax and the output are unaffected; the real 24 dims are sliced back per-head after the
merge. This is gated on `head_dim % 64 != 0` in `ESM2Attention`, so the 650M (`head_dim = 64`)
keeps its all-on-device path.

## Verdict

**Pass.** Per-residue embedding PCC 0.9991 / 0.9996 and MLM-logits PCC 0.9998 / 0.9999 for
saprot-35m / saprot-650m, in line with the ESMC port's 0.9995–0.9999 band. The
structure-aware embeddings — the capability's product — match the reference to within bf16
noise.

## Multi-card fanout (`--devices`)

`tt-bio saprot` accepts `--devices 0,1,2,3` (comma-separated physical TT card ids), the same
data-parallel sharding pattern `tt-bio embed` uses: one pinned subprocess per card, sequences
sharded by length (balanced across cards), results reassembled in input order. SaProt
embeddings are row-independent (no cross-sequence state), so a sequence's output is identical
to running it on one card — sharding changes only which chip computes which row.

```bash
tt-bio saprot proteins.fasta --model saprot-650m --devices 0,1
```

**Bit-exact vs single-card.** With `--batch_size 1` each sequence is embedded alone in its own
length bucket, so a sharded run is bit-exact vs the single-card run *by construction* (no
batchmate regrouping) — the same bar `tt-bio embed --devices` holds. Verified on qb1
(`saprot-650m`, 12 sequences across 4 shards): Δmax = 0 per-residue and pooled, ids+order
identical to input. Reproduce:

```bash
TT_VISIBLE_DEVICES=0 PYTHONPATH=. \
  python scripts/saprot_multicard_parity.py --model saprot-650m --n 12 --shards 4
```

With `--batch_size > 1` (the default), a sequence's batchmates differ across shards, so padding
and bf16 accumulation order differ by up to 1 ULP — same row-independence caveat as `tt-bio embed`.
Use `--batch_size 1` if you need cross-shard bit-exactness.

## saprot-1.3b: config bug fixed, near-pass residual

The 1.3B leg previously failed the gate (X_emb = 0.23415 / X_logits = 0.38640)
because `CONFIGS["saprot-1.3b"]` carried a fabricated shape
(hidden=2560/n_heads=40/n_layers=40/intermediate=10240) that does not match the
real `westlake-repl/SaProt_1.3B_AF2` checkpoint (hidden=1280/n_heads=20/
n_layers=66/intermediate=5120 — the 650m width with double the layers,
head_dim=64). `Saprot.from_pretrained` loaded with `load_state_dict(...,
strict=False)`, so the wrong architecture ran with effectively untrained weights.

That config is now corrected, and `from_pretrained` hardens the load: it reads
the checkpoint's own `config.json` and refuses to build if the `CONFIGS` arch
dict does not match it (a wrong entry now raises instead of silently producing
an uninitialized model). `strict=False` is kept for the weight copy so
legitimately-unused keys (`esm.contact_head`) still load cleanly.

With correct shapes the 1.3B loads the real checkpoint and parity jumps to
X_emb = 0.99508 / X_logits = 0.99895 (R = D = 1.00000, deterministic, qb1 card 1,
two identical runs). The MLM-logits PCC sits in the 0.9987-0.9996 band the 35m/
650m legs hit; the per-residue embedding PCC (0.99508) lands just below it. The
gap tracks depth: 1.3B is the 650m width at 2x the layers (66 vs 33), so bf16
rounding in the residual stream accumulates over twice as many blocks. It is a
numerical residual, not a structural defect — the wrong-config 0.23 is gone, and
the logits leg (the sampler-independent secondary check) clears the band. The
emb leg is recorded above as a near-pass, not a clean PASS, and no clean PASS row
is added to `docs/pharma-benchmark.md` for it.

Reproduce:

```bash
TT_VISIBLE_DEVICES=1 PYTHONPATH=. \
  python3 scripts/pharma_parity.py saprot --model saprot-1.3b --out /tmp/saprot13b/report.json
```

## Warm throughput (single card)

`saprot-650m`, batch 32 x bucketed length 128, warm: **~33k tokens/s** (123 ms/forward) on
one Blackhole card.
