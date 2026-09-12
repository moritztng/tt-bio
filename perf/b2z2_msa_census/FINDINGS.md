# `b2z2-msa-layer-census` — the MSA track, per program and per sub-unit (WH, whglx card 1)

Every number here is Wormhole, one card, one commit (`5360418c`), 512 tokens, 1024 padded MSA rows,
full protocol. Fractions transfer, seconds do not. Artifacts `perf/b2z2_msa_census/`, branch
`wk/b2z2-msa-layer-census`, state doc `state/b2z2-msa-layer-census.md`.

**1. THE CONTESTED FIT IS SETTLED AND IT WAS 1.2 % HIGH.** `139.47 ms + 0.0995 ms/padded row`
predicts 241.4 ms at the shipped 1024 bucket; the layer measures **238.5071 ms** replayed and
**238.626 ms** inside a real fold, 0.05 % apart. The fixture carries **1024 padded rows for 35 real
alignment rows** (3.42 % real), and the track is **3.8337 s of a 41.4333 s fold = 9.25 %**, 16
`MSALayer` calls. Price MSA levers against those two numbers.

**2. THE MSA TRACK IS THE TRUNK'S SHAPE, NOT THE SAMPLER'S.** Of the math thread's 202.5782 ms of
residency: **67.0 % input wait / 15.2 % output wait / 17.8 % compute**, against the block's
62.0/10.8/27.1 and the step's 55.8/12.1/32.1. It is the most input-stalled block the wave has
measured, and its movement-free multiplier is **3.3110x** against the trunk's 2.4390x and the step's
1.6428x. The shape of the wait, same reader and same three models:

| model | MSALayer | Pairformer block | diffusion step |
|---|---|---|---|
| tile arrivals only | **-0.7138** | -0.5696 | +0.0295 |
| DRAM + L1 bytes | 0.4088 | 0.5487 | 0.6375 |
| DRAM + L1 + per-program | 0.4216 | 0.5585 | 0.8638 |

Three-term decomposition of the 135.769 ms of input wait: **bytes 74.8 %, per-program 8.7 %**
(18.27 us over 647 compute programs of mean 341.5 us). The sampler's 62.9 % per-program does not
generalise off the sampler; **a megakernel here is aimed at 11.8 ms of 135.8**. Spearman: DRAM bytes
+0.623, bytes +0.618, tile arrivals +0.551, programs +0.148.

**3. THE TILE-ARRIVAL MODEL IS NOW REFUTED ON ALL THREE BLOCKS**, same sign every time: -0.5696
trunk, +0.0295 step, -0.7138 MSA. Bytes rank first on all three. Anyone still pricing in tile passes
is pricing the worst of the three models.

**4. THE ARMED SPAN IS USABLE ON THIS BLOCK.** Profiler cost **1.0035x** (239.3492 armed against
238.5071 bare), against 3.81x on the step. The 3.81x is a property of 1066 programs of mean 20 us,
not of the build; 782 programs of mean 341.5 us leave the host fully ahead.

**5. THE SUB-UNITS ARE FOUR DIFFERENT REGIMES AND ONE OF THEM WAS NEVER OPENED.**

| sub-unit | programs | kernel ms | share | mean us | in-wait % of its TRISC1 | MB |
|---|---|---|---|---|---|---|
| `pairformer_layer` | 203 | 78.381 | 33.3 % | 365.2 | 60.9 % | 4548 |
| `pair_weighted_averaging` | 104 | 73.987 | 31.4 % | 710.9 | **86.9 %** | 3498 |
| `outer_product_mean` | 15 | 61.401 | 26.1 % | 3552.0 | 62.8 % | 2586 |
| `msa_transition` | 321 | 21.518 | 9.1 % | 61.0 | 31.8 % | 1013 |

**11.6 GB per call, 186 GB per fold in the MSA track alone.** `pairformer_layer` is pair-track cost
another row already counts — do not add. `pair_weighted_averaging` is the largest MSA-specific term
and the most input-stalled thing in the wave.

**6. THE LEVER: EIGHT PER-HEAD TOKEN-WEIGHT PROJECTIONS ARE ONE PROJECTION. 1.03672x / 1.03718x ON
THE LAYER, BIT-EXACT.** `PairWeightedAveraging.proj_z` is `[c_z, n_heads]` — one COLUMN per head —
and eight columns live inside the same 32-wide tile, so the eight matmuls had byte-for-byte the same
padded output shape and each re-read the whole 67.1 MB normed pair tensor to fill a column of a tile
it then discarded 31/32 of: **8.436 ms and 536.9 MB of reads**, plus eight permutes, eight softmaxes
and fourteen weight-slice retiles. Two independent paired A/Bs agree to 0.04 % against A/A floors of
1.00014x and 1.00035x; **8.456 ms/call saved against 8.4 ms the census predicted before the build**.
In a real fold the track goes **3.8337 -> 3.7054 s**, 0.1283 s. `torch.equal` True on both outputs,
max |delta| exactly 0, negative control correctly False. `TT_BIO_PWA_BATCH_HEAD_WEIGHTS`, default
on, guarded on `n_heads <= 32` (a shape property). **0.31 % of a WH fold is under the 1.143 % fold
A/A floor, so no fold ratio is claimed.**

**7. THE CAPTURE REPORTS `1x1x32x32` FOR A CLASS OF IN-PLACE `BinaryNg` ADDS.** 14 programs per
MSALayer, **11.594 ms, 4.9 % of kernel time**, contributing exactly zero bytes and zero tile
arrivals to every model built off `ops_perf`. Their real traffic is ~200 MB each, ~2.8 GB, ~19 % of
the layer's total. So the byte fraction in §2 is a FLOOR and correcting it moves the answer further
toward bytes. **`census_tiles.py` and `stall_split.py` both inherit this**, so the trunk's 66.1 % and
the step's 23.9 % carry the same blind spot.

**8. AN ARM THAT SETS AN ENVIRONMENT VARIABLE A MODULE-SCOPE `env_flag` ALREADY READ MEASURES THE
BASE AGAINST ITSELF.** This pass produced a full byte-identical parity pass and a 1.00039x "result"
that way, inside its own A/A floor, before the override was made an attribute set. Both readings
were consistent and both were of nothing. `msa_probe.py` now refuses an arm that sets nothing.

**9. NAMED, PRICED, NOT BUILT.** (a) `PairWeightedAveraging`'s per-head `proj_o` accumulation: 8
matmuls writing 536.9 MB plus 8 in-place adds of ~200 MB, ~1.6 GB; contracting once puts the head
sum in the matmul's fp32 accumulator, which is more accurate and **not** bit-exact, so it needs the
298 aa control and the 512 aa per-pseudo-domain score. Largest remaining MSA-specific byte term.
(b) `OuterProductMean`'s layout chain: Untilize 5.380 + Tilize 4.886 + Transpose 5.445 + two
Permutes 3.525 = **19.24 ms, 8.2 % of the call** — a fused kernel, as `b2z2-msa-track-attack`
concluded. (c) `msa_transition` runs 321 programs for 21.5 ms because `TRANSITION_H_CHUNK_SIZE = 16`
is a pair-track constant (W=1024, c=128) applied at c=64 where the per-core L1 term allows ~30 rows:
64 row blocks instead of ~35, ~2.4 ms, and the wall is documented non-monotonic in this height so it
is a sweep. `TT_BIO_TRANSITION_H_CHUNK` is the hook and `msa_probe.py --mode ab` has the arms.

**10. FOR THE RELEASE GATE.** `tenstorrent.PairWeightedAveraging` is shared with **protenix-v2**
(`protenix.py:2499`) and **openfold3** (`openfold3_msa_embedder.py:79`), both 8 heads. The
bit-exactness argument is structural and the guard is a shape property, but only Boltz-2 was run.
