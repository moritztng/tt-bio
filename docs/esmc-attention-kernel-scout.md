# Kernel scout: fused RoPE for ESMC attention (win, release-gated)

## Result

ESMC's `Attention` (`tt_bio/esmc.py`) is the dominant per-block cost — about 57%
of a full warm forward at esmc-300m, well above the FFN. Its single largest
sub-op is not a matmul or SDPA but **RoPE**: `apply_rotary` is a six-op
rotate-half pile (chunk, neg, concat, two multiplies, add), applied to both q
and k, so twelve dispatches per attention. Replacing it with the fused
`ttnn.experimental.rotary_embedding` kernel (one dispatch per tensor) speeds up
the whole forward by **1.10-1.12x on esmc-300m/600m** and 1.03-1.06x on
esmc-6b, and is accuracy-neutral against the reference esm ESMC.

Shipped behind a tile-alignment gate in `esmc._rope`: the fused kernel needs a
seq length that is a multiple of 32, which the bucketed LM path always is
(`BUCKET` is 64; the 6B backbone pads to it internally), so `tt-bio embed`'s
batched path and every 6B forward take the fast path. Arbitrary single-sequence
lengths fall back to the exact `apply_rotary`. This changes device numerics
(one bf16 ULP in RoPE) so it is **release-gated** — kept on the branch, not
merged, pending the standard on-hardware parity gate.

## Method

Hardware: qb2, physical card 1 (Blackhole P150; the board is P300-misdetected so
runs need `TT_MESH_GRAPH_DESC_PATH=…/p150_mesh_graph_descriptor.textproto`).
Real biohub checkpoints (esmc-300m / 600m / 6b). Every timed path is warm and
ends with a device synchronization; transfers and comparisons are outside timed
regions. `scripts/esmc_attention_profile.py` (share + decomposition) and
`scripts/esmc_rotary_fusion_ab.py` (fusion A/B + reference parity) reproduce
everything below; both reuse the shipped `tt_bio.esmc` load path.

Two share methods, deliberately:
* **Skip-based (real pipelined wall-clock).** Replace attention (or the FFN)
  with one cheap eltwise of the right shape and take the delta vs the full
  forward. Free of per-op-sync distortion — this is the honest top-line share.
* **Per-op sync (decomposition).** Synchronize around every sub-op, as in
  `docs/boltz2-protenix-kernel-scout.md`. This inflates dispatch-bound ops (a
  pile of small eltwise pays full dispatch+sync each), so it over-weights
  attention and, within attention, RoPE — but it is the right lens for *where*
  the dispatch cost sits.

## Attention's share (esmc-300m, skip-based)

| L (tokens) | full forward | attention | FFN |
|---:|---:|---:|---:|
| 78  | 13.77 ms | 7.88 ms (**57.2%**) | 4.92 ms (35.7%) |
| 386 | 18.00 ms | 10.28 ms (**57.1%**) | 5.79 ms (32.2%) |

Attention is the majority of the forward, and its lead over the FFN is larger
than TriangleAttention's share of the folding trunk. Per-op-sync attributes
70-72% of the block stack to attention — an upper bound, consistent with
attention being dispatch-bound (14 ms clean vs 27 ms with per-op sync).

## Attention decomposition (esmc-300m, per-op sync, % of attention)

| sub-op | L=78 | L=386 | ops |
|---|---:|---:|---|
| **RoPE (apply_rotary ×2)** | **28.6%** | **28.6%** | 12 eltwise |
| qkv projection | 12.2% | 10.0% | 1 matmul |
| q/k LayerNorm (×2) | 12.1% | 11.6% | 2 |
| head split | 9.8% | 9.6% | 1 |
| output projection | 9.2% | 6.7% | 1 matmul |
| input LayerNorm | 7.2% | 6.0% | 1 |
| SDPA | 6.2% | 10.7% | 1 |
| chunk (q/k/v) | 5.5% | 6.3% | 1 |
| concat re-pack | 4.6% | 5.6% | 1 |
| head merge | 4.5% | 4.9% | 1 |

RoPE dominates precisely because it is the only sub-op that is a *pile* of small
elementwise dispatches rather than one kernel. SDPA, by contrast, is cheap here
(short sequences, 15-40 heads) — the opposite of TriangleAttention, where SDPA
led. QKV packing was **not** pursued: it already is one projection, and the
`nlp_create_qkv_heads` split it feeds is cheap (~10%); the TriangleAttention
QKV-pack attempt regressed for the analogous reason
(`docs/boltz2-protenix-kernel-scout.md`).

## The fusion

`ttnn.experimental.rotary_embedding(x, cos, sin)` implements the same rotate-half
NeoX convention as `apply_rotary` and consumes the existing `rope_tables`
cos/sin `[1,1,L,d]` unchanged (ESMC head_dim is 64 for all three sizes, so no
head-dim padding). Isolated, it collapses the twelve-op q+k RoPE to two
dispatches:

| model (head_dim 64) | apply_rotary ×2 | fused ×2 | speedup |
|---|---:|---:|---:|
| esmc-300m, L=96  | 0.118 ms | 0.012 ms | 10.1x |
| esmc-600m, L=96  | 0.120 ms | 0.012 ms | 9.6x |
| esmc-6b,   L=96  | 0.123 ms | 0.026 ms | 4.7x |
| esmc-300m, L=416 | 0.119 ms | 0.036 ms | 3.3x |

## End-to-end win (shipped `_rope`, fused vs manual)

| model | tokens=96 | tokens=416 |
|---|---:|---:|
| esmc-300m | **1.118x** | **1.106x** |
| esmc-600m | 1.104x | 1.101x |
| esmc-6b   | 1.026x | 1.058x |

The win is **largest on the small models**, not the 6B. RoPE is a fixed-cost
dispatch pile whose relative weight is highest when the surrounding matmuls are
small; at 6B the 2560-wide, 80-layer matmuls dominate the block. This is the
opposite of the SwiGLU scout's round-trip logic (`docs/esmc-swiglu-fusion-scout.md`),
where the benefit scaled *with* model size — a different lever with a different
shape, which is why this scout found a win where that one found none. 300m/600m
are the embed workhorses (`tt-bio embed`, JapanFold embeddings), so the win lands
on the hot path.

## Accuracy (release-gate metric)

Per-residue embedding PCC vs the reference esm ESMC (esmc-300m, tile-aligned L):

| L (tokens) | manual | fused |
|---:|---:|---:|
| 96  | 0.999615 | 0.999609 |
| 416 | 0.999737 | 0.999800 |

Fused tracks manual within bf16 noise (neutral, and at L=416 marginally better),
both far above the 0.99 gate. The batched bucketed path (`esmc_batch_parity.py`,
which exercises the fused path since it pads to 64) passes with worst
per-residue PCC 0.999600. Isolated max-abs of fused vs manual RoPE is 3.1e-2 —
exactly one bf16 ULP at that magnitude, i.e. a single rounding difference.

## Verdict

Genuine win, held on the branch. `esmc._rope` uses the fused kernel for
tile-aligned L (the entire bucketed + 6B path) and the exact `apply_rotary`
fallback otherwise. Merge after the standard parity gate confirms no accuracy
regression on the shipped embed path.
