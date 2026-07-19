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

## Results — Blackhole (qb1 card 0)

| model | seq (aa) | per-residue emb PCC | MLM logits PCC | argmax |
|---|---|---|---|---|
| saprot-35m  | ubiquitin (76) | **0.999138** | 0.999772 | — |
| saprot-650m | ubiquitin (76) | **0.999638** | 0.999927 | — |

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

## What is deferred

- **saprot-1.3b**: parity-run in this cut and FAILS the gate. The public
  `westlake-repl/SaProt_1.3B_AF2` checkpoint downloads cleanly and the HF
  `EsmForMaskedLM` reference loads normally, but the device-vs-reference PCC is
  X_emb = 0.23415 / X_logits = 0.38640 (R = D = 1.00000), far below the
  0.9987-0.9996 band the 35m/650m legs hit. Root cause is a port config bug: the
  real 1.3B is the 650m width (hidden=1280, n_heads=20, head_dim=64,
  intermediate=5120) with double the layers (66 vs 33), but
  `CONFIGS["saprot-1.3b"]` is set to hidden=2560/n_heads=40/n_layers=40/
  intermediate=10240. `Saprot.from_pretrained` loads with
  `load_state_dict(..., strict=False)`, so the shape-mismatched weights are
  silently dropped and the device runs uninitialized. head_dim=64 is tile-aligned,
  so no 35m-style host-RoPE path is needed. The fix is a one-line config change
  (release-gated, could change accuracy) and is out of scope for this verification
  pass; tracked as a follow-up.

## Warm throughput (single card)

`saprot-650m`, batch 32 x bucketed length 128, warm: **~33k tokens/s** (123 ms/forward) on
one Blackhole card.
