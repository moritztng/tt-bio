# Kernel scout: fused SwiGLU for ESMC (no-win)

## Result

`SwiGLUFFN` (shared by ESMFold2's pair transition and ESMC's own transformer FFN,
`tt_bio/esmc.py`) already supports a release-gated FC1+SwiGLU epilogue fusion via
`ttnn.experimental.minimal_matmul(..., fuse_swiglu=True)`. It is enabled at ESMFold2's
pair-transition call site (1.18x/1.14x at N=512/1024, see `docs/kernel-scout-next.md`) but
was never wired through at ESMC's own call site (`Block.__init__`) — a second, cheap
consumer of an already-proven fusion. Turning it on and measuring honestly: **no
measurable throughput change**, at any model size or sequence length tested. Not landed.

## Why: intermediate size, not op cost, drives the win

The fusion's benefit is removing the wide FC1-output tensor's DRAM round trip before the
SiLU/multiply epilogue. In ESMFold2's pair transition that intermediate is `[B,L,L,c]` —
quadratic in L, measured at 1 GiB (N=512) to 4 GiB (N=1024) per block. In ESMC's FFN the
same tensor is `[B,L,d]` — linear in L, and at ESMC's `d_model` (960-2560) and realistic
sequence lengths, several orders of magnitude smaller. The round-trip removed is real but
too small relative to fixed per-op dispatch cost to show up.

## Measurements

Hardware: pc, physical card 0 (Blackhole P150a). Real checkpoints (biohub esmc-300m /
esmc-600m / esmc-6b). Single-process A/B: `fuse_swiglu=True` wired at the ESMC `Block`
call site vs the shipped default (`False`), same process family, same warm program
cache, 20-iteration median (reset static cache between calls to keep each call an
independent warm forward).

Single-sequence warm forward latency (median of 20, batch=1):

| model | L (residues) | baseline | fused | delta |
|---|---:|---:|---:|---:|
| esmc-300m | 76 | 15.17 ms | 15.30 ms | +0.9% (noise) |
| esmc-300m | 2048 | 59.70 ms | 59.87 ms | +0.3% (noise) |
| esmc-600m | 76 | 20.35 ms | 20.38 ms | +0.1% (noise) |
| esmc-6b | 76 | 130.35 ms | 130.00 ms | -0.3% (noise) |

Batched path (`scripts/esmc_batch_parity.py --model esmc-300m --n 32 --batch-size 8`,
mixed lengths 42-119, warm bucket cache): 261.6 seq/s baseline vs 262.9 seq/s fused —
also flat.

All deltas are within min/max run-to-run jitter (~1-2%) for both baseline and fused —
there is no direction consistent with a real effect, at short (76), long (2048), or
batched-mixed-length input, or across the 300M/600M/6B size range including the model
(6B) with the longest per-layer critical path.

## Accuracy

Per-residue embedding PCC vs the reference `esm` ESMC (`scripts/esmc_embed_parity.py`,
ubiquitin, esmc-300m): fused 0.99963 vs the documented baseline band 0.99965
(`docs/esmc-embeddings-parity.md`) — same band, and the small shift itself confirms the
fused path really is engaging (not silently falling back to the unfused op).

## Verdict

Genuine no-win, not shipped. `Block.__init__` keeps calling `SwiGLUFFN(...)` without
`fuse_swiglu=True`. The flag, feature-detection, and fallback machinery already exist and
are untouched — enabling this for ESMC later is a one-line, zero-risk change if a future
larger-`d_model` variant or different shape ever makes the round-trip large enough to
matter; today's ESMC sizes and sequence lengths do not.
