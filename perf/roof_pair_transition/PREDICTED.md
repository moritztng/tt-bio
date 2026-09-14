# PREDICTED — roof-pair-transition, written before the first device run

Committed before any device command in this row. The deliverable is an honest denominator for
`Transition|1x512x512x128`, so the denominator is pre-registered here.

## What the budget says (qb2 card 2, p300c, tip `f072ae02f`)

    Transition|1x512x512x128   280 calls  12.8689 ms/call  103.113 GFLOP  340.017 MB  258 ops
                               303.26 FLOP/byte, COMPUTE-bound against a 247.1 balance
                               8.013 TFLOP/s achieved = 7.6 % of a measured 104.93 TFLOP/s
                               3.328 s above roof, 2.251 s at the 17.340 s cell

Two siblings from the same class, same code, different shape, same session:

    Transition|1x512x384    1.812 GFLOP  0.1955 ms   9.271 TFLOP/s   8 ops/call   380.08 F/B
    Transition|1x512x768    3.624 GFLOP  0.1329 ms  27.271 TFLOP/s   8 ops/call   380.08 F/B

## What the unit actually is

`tt_bio/tenstorrent.py:7763` `Transition.__call__`, 4-D branch, eager path. At 512 aa H=W=512 and
c=128, so `W=512 > TRANSITION_H_CHUNK_BIG_MAX_W=384` and the row chunk is
`TRANSITION_H_CHUNK_SIZE = 16`: **32 row blocks**, each

    ttnn.chunk slice -> layer_norm (L1) -> linear fc1 + fused silu (L1) -> linear fc2 (L1)
                     -> multiply_ (L1) -> linear fc3 (DRAM)        then one ttnn.concat

The capture agrees: 32 SliceDeviceOperation, 32 layer_norm, 96 linear, 32 multiply_, 1 concat.
Per block the matmuls are `[8192,128]x[128,512]` twice and `[8192,512]x[512,128]` once, i.e.
**K = 4 tiles** on two of the three. 3.222 GFLOP per block, 402 us per block measured.

## The prediction

**Outcome 3, leaning 2.** I do not expect ~2 s to be sitting here.

1. **The 303.26 FLOP/byte is an aggregate that hides the chain's own working set.** The byte
   counter charges DRAM buffers. This chain's two widest intermediates, `x_1` and `x_2`
   (`[1,16,512,512]` bf16, 8.39 MB each per block), are `L1_MEMORY_CONFIG` and are therefore
   free in that column. Priced per stage the chain is not compute-side anywhere: fc1 moves
   2.10 MB in and 8.39 MB out for 1.074 GFLOP (102 FLOP/byte), fc3 the mirror, `multiply_` and
   `layer_norm` are pure streaming. The one class in the fold whose intensity clears the machine
   balance clears it only because its own intermediates were not charged.
2. **The row blocking costs a full DRAM round trip of the pair tensor.** 512x512x128 bf16 is
   67.1 MB. Read once, written once by the 32 slices, read back, written once by fc3, read and
   written again by the concat: ~335 MB, which is the 340.017 MB the counter reports. So roughly
   **200 MB of the 340 MB is the blocking, not the mathematics.**

Registered numbers, same session, same card, dense-cube roof measured alongside so the ratio is
internally consistent:

    R_cube    dense bf16 4096-cube                         90 - 110 TFLOP/s  (calibration)
    R_mm3     the three real-shape matmuls, back to back,
              best config, no LN/silu/multiply                   30 TFLOP/s  (+/- 10)
    R_chain   the whole SwiGLU chain, best config I can find     14 TFLOP/s  (+/- 5)
    R_ship    the shipped op reproduced                           8 TFLOP/s  (reproduce 8.013)

So I expect the shape-honest statement to be **"this unit runs at 50-70 % of what its shape and
its chain structure allow, not 7.6 %"**, and the recoverable time at the cell to be
**0.6 - 1.2 s, not 2.251 s**. If R_chain comes out above 40 TFLOP/s I am wrong and outcome 1 is
live; if the shipped op is within 1.1x of R_chain I am wrong the other way and outcome 2 holds.

## Consequence for the campaign if this lands

`roof-budget`'s 6.934 s floor is set by bandwidth and is unaffected by this row. What IS affected
is the per-row deficit column for the three compute-side rows: pricing them against a dense-cube
rate charges them 3.328 + 0.047 + 0.039 s of "waste" that their own shape cannot deliver. The same
error sits under every compute-side row in the table, which is why this is worth more than a lever.

## Named mechanism I will test if a gap survives

In order: (a) the chunk/concat DRAM round trip — does `h=32`, or no chunking at all, or writing
fc3 to L1 and concatenating once, delete it; (b) the K=4-tile matmul's own efficiency, measured
standalone against the same session's cube; (c) FPU/SFPU head-of-line blocking across the fused
silu — `TT_BIO_UNFUSED_SILU` already exists as an A/B toggle at `tenstorrent.py:244`.
Kind: (a) and (c) are **placement** (k = 1.161 +/- 0.047), (b) would be **arithmetic**
(k = 0.623 +/- 0.090) if it changed the blocking algebra.
