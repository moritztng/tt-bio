# Kernel scout: fused RoPE for ESMC attention (win, release-gated)

## Result

ESMC's `Attention` (`tt_bio/esmc.py`) is the dominant per-block cost and its
single largest sub-op is **RoPE**. `apply_rotary` is a six-op rotate-half pile
(chunk, neg, concat, two multiplies, add), applied to both q and k, so twelve
tiny elementwise dispatches per attention. At esmc-300m/600m the forward is
host-dispatch-bound, so those ~360 dispatches (30 layers x 12) dominate wall
clock rather than any matmul. Replacing them with the fused
`ttnn.experimental.rotary_embedding` kernel (one dispatch per tensor) speeds up
the whole warm forward by **1.46-1.67x on esmc-300m/600m** and 1.13-1.16x on
esmc-6b, accuracy-neutral against the reference esm ESMC.

Shipped behind a tile-alignment gate in `esmc._rope`: the fused kernel needs a
seq length that is a multiple of 32, which the bucketed LM path always is
(`BUCKET` is 64; the 6B backbone pads to it internally), so `tt-bio embed`'s
batched path and every 6B forward take the fast path. Arbitrary single-sequence
lengths fall back to the exact `apply_rotary`. This changes device numerics
(one bf16 ULP in RoPE) so it is **release-gated**, kept on the branch, not
merged, pending the standard on-hardware parity gate.

The forward runs eagerly (no ttnn trace anywhere in the ESMC/embed path), so the
eager, host-dispatch-bound regime measured here is exactly production. That is
why the win is large: fusing removes dispatch count, and dispatch is the
bottleneck at the embed model sizes.

## Method

Hardware: qb2, physical card 0 (Blackhole P150; the board is P300-misdetected so
runs need `TT_MESH_GRAPH_DESC_PATH=.../p150_mesh_graph_descriptor.textproto`).
Real biohub checkpoints (esmc-300m / 600m / 6b). Every timed path is warm
(2 warmup forwards, median of 5-7) and ends with a device synchronization;
transfers and comparisons are outside timed regions. `scripts/esmc_attention_profile.py`
(share + decomposition) and `scripts/esmc_rotary_fusion_ab.py` (fusion A/B +
reference parity) reproduce everything below; both reuse the shipped
`tt_bio.esmc` load path.

Two share methods, deliberately:
* **Skip-based (real pipelined wall-clock).** Replace attention (or the FFN)
  with one cheap eltwise of the right shape and take the delta vs the full
  forward. Free of per-op-sync distortion, so this is the honest top-line share.
* **Per-op sync (decomposition).** Synchronize around every sub-op, as in
  `docs/boltz2-protenix-kernel-scout.md`. This inflates dispatch-bound ops (a
  pile of small eltwise pays full dispatch+sync each), so it over-weights RoPE
  in absolute terms, but it is the right lens for *where* the dispatch cost sits.
  The A/B (fused vs manual) is itself the honest skip-based measurement of
  RoPE's true pipelined cost.

## Attention's share (esmc-300m, skip-based)

With the shipped fused RoPE, attention drops well below the manual baseline:

| L (tokens) | full forward | attention | FFN |
|---:|---:|---:|---:|
| 96  | 14.71 ms | 5.21 ms (35.4%) | 3.45 ms (23.5%) |
| 416 | 16.17 ms | 6.00 ms (37.1%) | 3.94 ms (24.4%) |

With the manual RoPE the same attention is ~16.3 ms (63% of a 25.8 ms forward):
RoPE alone is ~40% of the whole manual forward. Per-op sync attributes 71.9% of
the block stack to attention, an upper bound consistent with attention being
dispatch-bound.

## Attention decomposition (esmc-300m, per-op sync, manual RoPE, % of attention)

| sub-op | L=96 | L=416 | ops |
|---|---:|---:|---|
| **RoPE (apply_rotary x2)** | **51.2%** | **46.3%** | 12 eltwise |
| q/k LayerNorm (x2) | 7.5% | 8.1% | 2 |
| qkv projection | 7.4% | 6.8% | 1 matmul |
| chunk (q/k/v) | 6.7% | 7.7% | 1 |
| head split | 5.9% | 6.2% | 1 |
| output projection | 5.8% | 4.7% | 1 matmul |
| input LayerNorm | 4.5% | 4.8% | 1 |
| concat re-pack | 4.3% | 5.4% | 1 |
| SDPA | 4.1% | 7.0% | 1 |
| head merge | ~2.6% | ~3.0% | 1 |

RoPE is about half of attention because it is the only sub-op that is a *pile*
of small elementwise dispatches rather than one kernel. SDPA, by contrast, is
cheap here (short sequences, 15-40 heads), the opposite of TriangleAttention
where SDPA led. QKV packing was **not** pursued: it already is one projection,
the `nlp_create_qkv_heads` split it feeds is cheap (~6%), and the analogous
TriangleAttention QKV-pack attempt regressed (`docs/boltz2-protenix-kernel-scout.md`).

## The fusion

`ttnn.experimental.rotary_embedding(x, cos, sin)` implements the same rotate-half
NeoX convention as `apply_rotary` and consumes the existing `rope_tables`
cos/sin `[1,1,L,d]` unchanged (ESMC head_dim is 64 for all three sizes, so no
head-dim padding). It collapses the twelve-op q+k RoPE to two dispatches per
layer, removing ~340 dispatches at 300m/600m.

## End-to-end win (shipped `_rope`, fused vs manual, warm, eager)

| model | tokens=96 | tokens=416 |
|---|---:|---:|
| esmc-300m | **1.65x** (25.8 -> 15.5 ms) | **1.67x** (28.1 -> 16.9 ms) |
| esmc-600m | **1.66x** (30.6 -> 18.4 ms) | 1.46x (34.0 -> 23.3 ms) |
| esmc-6b   | 1.16x (149.0 -> 127.9 ms) | 1.13x (290.6 -> 257.1 ms) |

The win is **largest on the small models**, not the 6B. RoPE is a fixed-count
dispatch pile whose relative weight is highest when the surrounding matmuls are
small; at 6B the 2560-wide, 80-layer matmuls hide most of the dispatch. This is
the opposite of the SwiGLU scout's round-trip logic
(`docs/esmc-swiglu-fusion-scout.md`), where the benefit scaled *with* model size,
a different lever with a different shape, which is why this scout found a win
where that one found none. 300m/600m are the embed workhorses (`tt-bio embed`,
JapanFold embeddings), so the win lands on the hot path.

## Accuracy (release-gate metric)

Per-residue embedding PCC vs the reference esm ESMC, and fused vs manual:

| model | L=96 fused-vs-manual | L=96 ref (manual / fused) | L=416 fused-vs-manual | L=416 ref (manual / fused) |
|---|---:|---:|---:|---:|
| esmc-300m | 0.999620 | 0.999646 / 0.999683 | 0.999780 | 0.999737 / 0.999800 |
| esmc-600m | 0.999681 | 0.999712 / 0.999701 | 0.999816 | 0.999698 / 0.999732 |
| esmc-6b   | 0.999853 | (ref not run) | 0.999792 | (ref not run) |

Fused tracks manual within bf16 noise (neutral, and often marginally better),
all far above the 0.99 gate. The fused-vs-manual delta is one bf16 ULP in RoPE,
a single rounding difference. 6B uses the identical fused kernel (head_dim 64),
so its reference parity follows from 300m/600m plus the bit-level fused-vs-manual
agreement.

## Verdict

Genuine, large win on the embed hot path, held on the branch. `esmc._rope` uses
the fused kernel for tile-aligned L (the entire bucketed + 6B path) and the exact
`apply_rotary` fallback otherwise. Merge after the standard parity gate confirms
no accuracy regression on the shipped embed path.

**Update (2026-07-13):** the on-hardware parity gate has now been run and passes
(300m/600m per-residue PCC 0.99961/0.99964 vs reference esm, floor 0.99); see
`docs/esmc-rope-fusion-release-gate.md` for the evidence and the re-measured
speedup on current main.
