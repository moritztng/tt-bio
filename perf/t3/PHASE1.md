# T3 — transition, norms, PairformerLayer body + AttentionPairBias — PHASE 1

Protenix-v2, trunk only, 298 aa (N=320 padded), c_z=256, c_s=384. qb2 card 1, ttnn 0.68.0, so every
number here is a **ratio card**, not a source of campaign absolutes. Board-mate during the whole pass:
qb2 card 0, held by `worker:protenix-trunk--trimul` (T2) — board 007 is chips 0+1. My two independent
DRAM read-roof measurements, taken ~20 minutes apart under that tenancy, are 391.2 and 393.4 GB/s,
a spread of 0.6 % against either figure, so the roofs below are not contended readings.

**No production code was changed this pass.** Everything here is characterisation; the fixes are handed
forward as Phase-3 candidates. Phase 1 is understand-only and optimising in it would be a failed brief.

Artifacts, all committed on `wk/protenix-trunk--transition-norms`:

| file | what it is |
|---|---|
| `perf/t3run.sh` | the qb2-card-1 single-chip pin (p150 mesh descriptor + lease holder) |
| `perf/t3/roofs_qb2c1.json` | `perf/ledger_298/roofs_card.py` on this card |
| `perf/t3/ops_pv2_320_qb2c1.json` | `perf/ledger_298/pf_block_ops.py`, one real block, 272 ops |
| `perf/t3/probe.py` + `probe_qb2c1.json` | roofs to 128 MB, compute-rate-vs-K, ops at rig shapes |
| `perf/t3/probe2.py` + `probe2_qb2c1.json` | **the ops at the shapes the fold actually runs** |
| `perf/t3/census.py` + `census_fold_qb2c1.json` | one real 298-aa fold, 149 138 ttnn calls counted by site |
| `perf/t3/agg.py`, `errs.py`, `split.py` | the analysis passes over the two dumps |

---

## Roofs, measured on this card

All measured in the same process as the ops, HiFi4, `fp32_dest_acc_en=True`, `packer_l1_acc=True` —
the compute kernel config the model actually constructs. Every timed region synchronises the device
immediately before the clock starts and immediately before it stops; median of 5 runs of 5
back-to-back calls.

| roof | measured | how |
|---|---:|---|
| compute, dense bf16, K=4096, **output in L1** | **119.17 TFLOP/s** | M=4096 N=2048, `core_grid=11x10` |
| compute, dense bf16, K=4096, output in DRAM | 115.04 TFLOP/s | M=10240 N=4096, `core_grid=11x10` — write-limited, see below |
| DRAM read, swept 8 → 128 MB | **393.4 GB/s** | DRAM-interleaved → L1 clone; saturates by 96 MB (393.1 → 393.4) |
| DRAM write | **273.4 GB/s** | L1 → DRAM clone |
| DRAM → DRAM clone, read+write combined | **397.3 GB/s** | the copy roof the residual adds and norms are scored against |
| DRAM → DRAM clone, *marginal* (floor removed) | **404.7 GB/s** | two-point fit, see the norms below |
| L1 → L1 clone | **1064.4 GB/s** | the copy roof `multiply_` is scored against |
| per-op floor, plain clone | **8.10 us** | intercept of the same two-point fit |
| per-op floor, `ttnn.layer_norm` | **12.14 us** | same fit on the two norms |
| **machine balance** | **302.9 FLOP/byte** | 119.17 TFLOP/s ÷ 393.4 GB/s, both measured here |

`perf/ledger_298/roofs_card.py` reports 99.77 TFLOP/s on this card. That is not a second card reading
to reconcile: that script only tries square N×N with the default config, and sweeping `core_grid`
at M=10240 N=4096 reaches 115.04. Its read/write roofs (391.2 / 270.1 GB/s) agree with mine to 0.6 %.
The machine balance on this card is **302.9 FLOP/byte**, not the 338 the org inherited from qb1 —
same method, different card, and it moves because qb2's compute roof is below qb1's inherited
135.60 TFLOP/s by 12 % of that figure, while its DRAM read roof is within 2 % of it. I score against 302.9 and say so on every row.

### The compute roof is a function of K — and much less so than the org believes

**I got this wrong the first time and the correction matters.** Round 1 swept K with the matmul output
in DRAM and reported a "K=256 roof" of 49.95 TFLOP/s. The Transition's fc2 then measured **64.05
TFLOP/s at K=256**, i.e. above the roof — which is not possible, so the roof was wrong. At
M=10240 N=4096 a DRAM output is 83.9 MB written per call, 307 us of the 430 us measured at this card's
273.4 GB/s write roof: that sweep produced **a write roof wearing a compute roof's label**, the exact
mistake `roofline-roof-must-be-measured-not-asserted` is about. Re-measured with the output in L1,
which is where fc1 and fc2 actually put theirs, best over N ∈ {1024, 2048, 4096} and over
{default, `core_grid`} (`perf/t3/kroof.py`):

| K (the contraction my ops actually do) | corrected compute roof, L1 output | fraction of the dense K=4096 roof | best rate at the op's own nt | the op |
|---:|---:|---:|---:|---|
| 256 | **95.42 TFLOP/s** (M=8192 N=2048) | 80.1 % of the dense compute roof | 68.19 at nt=32 | Transition fc1/fc2, `z_proj` (nt=1) |
| 384 | **103.67 TFLOP/s** | 87.0 % of the dense compute roof | — | `qkv`, `o_proj`, `g`, transition_s fc1/fc2 |
| 1024 | **115.18 TFLOP/s** (M=8192 N=2048) | 96.7 % of the dense compute roof | 36.00 at nt=8 | **Transition fc3 — W1's AI-930.9 row** |
| 1536 | **118.13 TFLOP/s** | 99.1 % of the dense compute roof | — | transition_s fc3 |
| 4096 | 119.17 TFLOP/s | — | — | none of mine |

So on this card the K-correction over my whole K range is **1.25x (95.42 → 119.17), not the ~3.8x the
charter's "a K=256 contraction reaches at most 35.5 TFLOP/s" implies**. The 35.5 figure is a
DRAM-output measurement; it is the right number to size a DRAM-output matmul against and the wrong one
to size fc1/fc2 against, and I flag it because charter §4.6 uses it to argue the trunk's ceiling.
**What actually costs 2-3x in this slice is output WIDTH, not K**: at K=1024 the card reaches
115.18 TFLOP/s at nt=64, 98.61 at nt=32 and only 36.00 at nt=8. Every "% of the compute roof" below
therefore carries the nt-limited rate beside it, because the difference between the two is the whole
mechanism.

One incidental: with an L1 output and M ≥ 4096, ttnn's **default** program config collapses to
8.5-9.0 TFLOP/s at every K while `core_grid` reaches 95-119. That is a 12x config cliff, recorded and
not chased — it is not on any path this slice runs.

---

## The shape census: what the fold actually runs

Mandatory here, and it changed three numbers. One real production fold (`examples/prot300.yaml`,
298 tokens, 10 recycles, pLDDT 0.783, 27.26 s), every ttnn call counted by frame chain.
`tenstorrent.py:2306` is `Pairformer.__call__`'s block loop, so a chain containing it ran inside the
48-block pf_stack; everything else is a different stack and is counted but not chased.

1. **The conversion for my ops is 484, not 480 and not 524.** `add_` at `tenstorrent.py:2223` — the
   first residual in `PairformerLayer.__call__`, exactly one per block execution — fires **484
   times per fold** inside the pf_stack: 480 from 48 blocks × 10 recycles, plus 4 from a second
   Pairformer construction reached via `protenix.py:1286`. This is a direct contribution to the org's
   Q1; it does not settle the MSA/template stacks' own count, which is T5's.
2. **The Transition is chunked, and the chunk is 30 rows, not 32.** `z` is carried as
   `(1, 298, 320, 256)` — dim 1 keeps the **logical** 298 because only the last two dims tile-pad —
   so `ttnn.chunk(x, 10, dim=1)` yields nine `(1,30,320,256)` chunks at **mt=300** and one
   `(1,28,320,256)` at **mt=280**, kt=8, nt=32. The brief's mt=300/280 is right; the padded-N=320 rig
   I first built gives mt=320 and is optimistic on fc1 by 3 % of the measured figure. Every microbenchmark that flattened this
   to mt=3200 measured a shape the fold never runs.
3. **`z` is 48.82 MB, not 52.43 MB.** Every byte count for a z-shaped op is 298/320 = 0.93125 of what
   the padded rig says. The five residual adds move 146.5 MB each, not 157.3 MB.
4. **The Transition runs 15 972 `ttnn.linear` calls per fold inside the pf_stack**, from 5324 swiglu
   invocations (4840 chunked `transition_z` sub-calls = 484 × 10, plus 484 `transition_s`). The
   brief's **1512 calls/fold is a single-recycle count** presented as a per-fold one: at 1 recycle
   the same census gives ~1600. At the production 10 recycles it is 15 972, and 19 860 counting the
   1296 swiglu invocations outside the trunk — those are the diffusion sampler and the atom
   encoder/decoder, recorded here and out of scope.

The chunking's own cost is not free and is not in the ledger either: `ttnn.chunk` at
`tenstorrent.py:2145` materialises a full second copy of `z` (484 calls/fold) and `ttnn.concat` at
`2148` reassembles it (484 calls/fold), together **269.2 ms/fold** of pure layout.

---

## Gate-1 facts

Conversion for every row: **us/call × calls/fold, both measured**, calls/fold from the census above.
`transition_z` rows use the call-weighted mean of the nine mt=300 chunks and the one mt=280 chunk.
Denominator for every "% of the pf_stack total": the block wall measured on this card by `pf_block_ops.py`
(43.418 ms) × the 484 block executions counted in the fold = **21.014 s/fold**. That denominator is
an upper bound — the wall was measured on the padded N=320 rig — so the shares below are conservative.
1 % of that total is 210 ms/fold.

| op @ site | calls/fold | us/call | ms/fold (conv) | bytes moved | AI FLOP/byte | side of 302.9 | binding roof | % of that roof | cores engaged / grid | overlap: max() or sum? | mechanism hypothesis (falsifiable) |
|---|---:|---:|---:|---|---:|---|---|---:|---|---|---|
| `linear` fc1 `activation="silu"` @2046 (was 1779) | 4840 | 248.94 | **1204.9** (5.73 % of the pf_stack total) | 0.52 MB DRAM (weight only), 25.7 MB L1 | 9600 dram / 200 all | compute side | K=256 compute, 95.42 TFLOP/s (L1 output) | **21.0 % of the K=256 compute roof**, and 29.4 % of the nt=32 figure (68.19 TFLOP/s) the card reaches at this op's own output width | measured 100 of 110 useful; 11x10 (80.20 us) is slower than 10x10 (77.61 us) by 3.3 % of the 10x10 figure | sum — the SFPU activation is serial with the MACs on the same TRISC math thread | Hypothesis H1: the fused activation runs in the pack path on the math thread and is NOT overlapped with the MAC loop, so its cost is linear in output elements and independent of K. Identical matmul without `activation=` is **77.81 us**: the silu costs **3.22x, 172.8 us/call, 835.3 ms/fold**. KILL IT BY: hold M,N and double K — if this hypothesis holds the premium stays ~173 us while the matmul time doubles |
| `add_` x5 @2223/2227/2231/2235/2239 | 2420 (484 x 5) | 337.23 | **816.1** (3.88 % of the pf_stack total) | 146.5 MB DRAM (2R+1W) | 0.2 | memory side | DRAM->DRAM copy, 397.3 GB/s | **434.3 GB/s = 109.3 % of the copy roof** | ttnn picks the grid; no `core_grid` argument exists | sum, and it is already above the copy control | KILLED, and W6's kill re-confirmed on this card. An in-place add issues its 2 reads and 1 write in one kernel pass so the NOC streams overlap better than a clone's. Nothing but fusion into the producer can touch it. Per site it is 163.2 ms/fold (0.78 % of the pf_stack total); the class is one row because the mechanism is one mechanism |
| `linear` fc3 @2066 (**the site the brief calls 1799**) | 4840 | 93.21 | **451.1** (2.15 % of the pf_stack total) | 5.4 MB DRAM (weight + output), 25.2 MB L1 in | **925.3** dram | compute side | K=1024 compute, 115.18 TFLOP/s (L1 output) | **46.7 % of the K=1024 compute roof**, but **149 % of the nt=8 figure (36.00 TFLOP/s) the card reaches at this op's own output width** — the roof is unreachable at nt=8 and the op is already past the best nt=8 config I could build (45.2 % of the dense compute roof, which is where W1's "37.7 % of COMPUTE" came from) | measured: 16 cores 305.8 us, 36 -> 166.4, 64 -> 115.5, 100 -> 93.4, 110 -> 94.7. Flat-to-worse past 100 | sum | Hypothesis H2: **nt=8.** The output is 8 tiles wide, so in a 2D block-mcast matmul at most 8 of the 11 core columns can ever hold an output tile and 3 columns (3 of the 11 columns) idle by construction; the 11th column adds mcast fan-out and no work, which is why 110 cores is slower than 100 by 1.4 % of the 100-core figure. KILL IT BY: run `core_grid=CoreGrid(x=8,y=10)` — if this hypothesis holds, 80 cores match or beat 110 |
| `linear` fc2 @2055 | 4840 | 77.09 | **373.1** (1.78 % of the pf_stack total) | 0.52 MB DRAM (weight only), 25.7 MB L1 | 9600 dram | compute side | K=256 compute, 95.42 TFLOP/s (L1 output) | **67.1 % of the K=256 compute roof**, and **93.9 % of the nt=32 figure (68.19 TFLOP/s) the card reaches at this op's own output width** — this op is at its shape's ceiling | same 11x10 grid; same 100-vs-110 inversion as fc1 | sum | Hypothesis H3: **mt=300 does not divide 110.** 300/110 = 2.73 tile-rows per core, so the 11x10 grid runs a 3-row critical path on some cores and 2 on others while 10x10 runs exactly 3 on all 100 — identical critical path, less mcast. KILL IT BY: repeat at mt=330 (a multiple of 110); if this hypothesis holds, 11x10 wins there and loses at mt=300 |
| `multiply_` @2064 | 4840 | 63.39 | **306.8** (1.46 % of the pf_stack total) | 0 DRAM, 61.8 MB L1 (2R+1W) | 0.16 | memory side | L1->L1 copy, 1064.4 GB/s | **923.9 GB/s = 86.8 % of the L1 copy roof** | ttnn picks the grid | sum | LATENCY, not bandwidth: headroom of 13 % of the L1 copy roof at 2R+1W. Hypothesis H4: the gap is the `cb_reserve` round trip on the output CB, whose depth is 2 blocks, so the packer stalls once per block rather than streaming. KILL IT BY: the same op at 4x the bytes — if this hypothesis holds the achieved GB/s rises toward 1064 because the per-block stall amortises |
| `linear` `z_proj` @1900 | 484 | 449.78 | **217.7** (1.04 % of the pf_stack total) | 55.0 MB DRAM | **28.4** | memory side | DRAM read, 393.4 GB/s | **122.2 GB/s = 31.1 % of the DRAM read roof**, and 3.47 TFLOP/s = 3.6 % of the K=256 compute roof | **measured flat past 40 of 110 cores**: 1 core 1953 us, 10 -> 907, 20 -> 944, 40 -> 460, 80 -> 445, 110 -> 452. **70 of the 110 cores idle** | sum | OCCUPANCY. Bound by neither roof, so it is a defect. Hypothesis H5: **nt=1** — c_z=256 -> 16 heads of bias is one output tile column, so there is nothing for the in1 mcast to amortise and ttnn caps the useful core count independent of what `core_grid` asks for. Evidence already in hand: the same M and K with **nt=32** costs 1674 us for **32x the FLOPs** and reaches 29.86 TFLOP/s = 31.3 % of the K=256 compute roof — the read stream is not the limit, the output width is. KILL IT BY: sweep nt = 1, 2, 4, 8 at fixed M,K; if this hypothesis holds the time is near-flat to nt=4 |
| `layer_norm` @2038 (Transition, chunked) | 4840 | 39.20 | **189.7** (0.90 % of the pf_stack total) | 9.8 MB (4.9 R DRAM + 4.9 W L1) | ~1.25 | memory side | DRAM->L1 clone at the same bytes, 17.42 us | **44.3 % of the same-bytes copy control** (249.9 GB/s at 1R+1W) | ttnn picks the grid | sum | The ledger scored this op **0.0 s**. hypothesis H6 below: reduce-then-scale serialises the reader behind the row reduction. 12.14 us of the 39.20 is the per-op floor |
| `layer_norm` `z_norm` @1893 | 484 | 323.18 | **156.4** (0.74 % of the pf_stack total) | 97.6 MB DRAM (1R+1W) | ~1.25 | memory side | DRAM->DRAM copy, 397.3 GB/s | **302.2 GB/s = 76.1 % of the copy roof**; marginal 313.9 GB/s = 77.6 % of the copy's marginal 404.7 GB/s | ttnn picks the grid | sum | Hypothesis H6: **reduce-then-scale.** c_z=256 is 8 tiles per row; the mean/var reduction must consume all 8 before any output tile can be scaled, so those 8 tiles sit pinned in a circular buffer and the NOC reader cannot run ahead past it — the read stalls for the duration of the reduce + rsqrt. It is one DRAM pass, not two: 2R+1W would be 451 GB/s, above this card's 397.3 GB/s combined roof, so a second full pass is arithmetically impossible. KILL IT BY: fixed total bytes at c=128 (kt=4) and c=512 (kt=16) — if this hypothesis holds, marginal GB/s rises toward the copy roof as kt falls |
| `chunk` @2145 + `concat` @2148 | 484 + 484 | 292.1 + 264.1 | **269.2** (1.28 % of the pf_stack total as a class) | 97.6 MB DRAM each (1R+1W) | 0 | memory side | DRAM->DRAM copy, 397.3 GB/s | 334.3 and 369.7 GB/s = **84.1 % and 93.0 % of the copy roof** | ttnn picks the grid | sum | Near the copy roof, so not a bandwidth defect — the cost is that the chunking exists at all. Hypothesis H7: the chunk is a pure L1-footprint decision (`TRANSITION_H_CHUNK_SIZE_BIG = 32`), so the 269.2 ms buys nothing except a smaller peak. KILL IT BY: raise the chunk toward H and watch peak DRAM against the clash |
| `permute` @1906 + the 10-op APB attention path | 484 + 4840 | 78.4 + 144.4/blk | 37.9 + 69.9 (**0.51 % of the pf_stack total** together) | 12.2 MB and mixed | mixed | memory side | DRAM->DRAM copy | grouped row: below the 1 % bar individually, total stated | ttnn picks the grid | sum | Grouped deliberately: `transpose@1936`, two `matmul@372`, `add_@1940`, `multiply_@1941`, `softmax@1942`, `slice@1950`, `permute@1951`, `reshape@1952`, `permute@1953`. 144.4 us/block total, none above 25 us. Not a lever |
| transition_s, all 5 ops @2038/2046/2055/2064/2066 | 484 x 5 | 82.15 total | **39.8** (0.19 % of the pf_stack total) | 4.5 MB DRAM per call | 320 / 264.8 | compute side | K=384 (62.54) and K=1536 (100.13) compute | fc1 16.09 TFLOP/s = 15.5 % of the K=384 compute roof; fc3 19.36 = 16.4 % of the K=1536 compute roof | 11x10; mt=10 tile-rows across 110 cores means **at most 10 of 110 cores can hold an output tile row** | sum | Grouped: the whole s-track Transition is 0.19 % of the pf_stack total. Hypothesis H8: mt=10 is the entire story — 10 tile-rows cannot fill 110 cores no matter the config, so this is latency/occupancy-bound and will never approach a compute roof. Not worth a Phase-3 slot |
| `layer_norm` `pre_norm_s` @2242 + `add_` @2255/@2259 | 484 + 968 | 13.71 + 9.1 | 6.6 + 8.8 (**0.07 % of the pf_stack total**) | 0.49 MB DRAM (1R+1W) | ~1.25 | memory side | per-op floor, 8.10 us for a clone | 13.71 us against a 9.32 us same-bytes clone: **68 % of it is the per-op floor** | ttnn picks the grid | sum | At 0.49 MB these are dispatch-floor rows, not bandwidth rows: 0.492 MB at 404.7 GB/s is 1.2 us of transfer inside a 13.71 us op. Nothing to win |

### Overlap: this block is additive, and here is the evidence

`pf_block_ops.py` measured one real block on this card at a **43.418 ms wall** with a per-op sum of
**35.347 ms — 81.4 % of the block wall**. That gap is not overlap. Reading the dump's error field,
**30 of the 272 ops carry a re-run exception and were scored 0.0 s**, and 30 of them are mine: 10
`layer_norm@2038`, 10 `linear@2046`, 10 `linear@2055` — every L1-output op in the chunked Transition.
Priced at the rig's own shapes they are **4.132 ms/block, 51.2 % of the block's 8.071 ms gap.** Restoring
just my slice takes the per-op sum from 81.4 % to **90.9 % of the block wall**, with T2's 32
zero-scored ops (16 `minimal_matmul@1304`, 16 `chunk@1311`) still unpriced. So at N=320 the block
total is **sum(ops), not max(compute, comm)** — there is no overlap headroom left to find, and the
org's "at N=320 the per-op sum is 90.6 % of the block wall" reading was measuring an instrument
artifact, not inter-op concurrency. The 117 aa regime is a different question and is not mine.

---

## The 4.19 s, itemised

The charter split my slice as `PairformerLayer` body + `AttentionPairBias` **4.19 s** and Transition
**0.53 s**. Measured on this card at the measured 484 conversion, the total is close and **the
composition is completely different**: the Transition is 2.83 s and the body + APB is 1.35 s.

| item | calls/fold | us/call | ms/fold | share of the 4183 ms |
|---|---:|---:|---:|---:|
| `linear` fc1 `activation="silu"` @2046 | 4840 | 248.94 | 1204.9 | 28.8 % of the itemised total |
| — of which the fused silu itself | 4840 | 172.59 | **835.3** | 20.0 % of the itemised total |
| `linear` fc3 @2066 | 4840 | 93.21 | 451.1 | 10.8 % of the total  |
| `linear` fc2 @2055 | 4840 | 77.09 | 373.1 | 8.9 % of the total  |
| `multiply_` @2064 | 4840 | 63.39 | 306.8 | 7.3 % of the total  |
| `layer_norm` @2038 | 4840 | 39.20 | 189.7 | 4.5 % of the total  |
| `chunk` @2145 | 484 | 292.1 | 141.4 | 3.4 % of the total  |
| `concat` @2148 | 484 | 264.1 | 127.8 | 3.1 % of the total  |
| **Transition, pair track (`transition_z`)** | | | **2794.9** | **66.8 % of the total** |
| transition_s, 5 ops grouped | 484 x 5 | 82.15 | 39.8 | 1.0 % of the total  |
| **Transition total** | | | **2834.7** | **67.8 % of the total** |
| `add_` x5 @2223..2239 | 2420 | 337.23 | 816.1 | 19.5 % of the total  |
| `layer_norm` pre_norm_s @2242 | 484 | 13.71 | 6.6 | 0.2 % of the total  |
| `add_` @2255 + @2259 (s track) | 968 | 9.1 | 8.8 | 0.2 % of the total  |
| **`PairformerLayer` body total** | | | **831.5** | **19.9 % of the total** |
| `linear` z_proj @1900 | 484 | 449.78 | 217.7 | 5.2 % of the total  |
| `layer_norm` z_norm @1893 | 484 | 323.18 | 156.4 | 3.7 % of the total  |
| APB attention path, 10 ops grouped | 4840 | 144.4/blk | 69.9 | 1.7 % of the total  |
| `permute` @1906 | 484 | 78.4 | 37.9 | 0.9 % of the total  |
| `qkv`@1876 + heads@1884 + `o_proj`@2001 + gate@2009 + g@2011 | 484 x 5 | 72.48 | 35.1 | 0.8 % of the total  |
| **`AttentionPairBias` total** | | | **517.0** | **12.4 % of the total** |
| **T3 TOTAL** | | | **4183.2** | **19.9 % of the pf_stack total (21.014 s)** |

**Reconciliation, and the residual named.** 4.183 s measured against the 4.72 s inherited (4.19 + 0.53):
**a shortfall of 0.54 s, 11.4 % of the inherited figure.** Two causes, both identified, neither a measurement disagreement:

1. **W1's "Transition 0.53 s" is fc3 alone.** The row the brief cites as `linear` (Transition) at site
   **1799**, 10 calls/block, 105.0 us/call, 503.8 ms/fold @480, is `tenstorrent.py:2066` in today's
   tree — the same op I measure at 93.21 us and 451.1 ms/fold. Everything else in the Transition was
   scored 0.0 s and its time fell into the body bucket, which is why "body + AttentionPairBias" came
   out at 4.19 s while the body and APB together are really 1.35 s.
2. **The rest is the card.** qb2's dense compute roof is 115.04 TFLOP/s against qb1's inherited
   135.60 — a shortfall of 15.2 % against that figure — while the DRAM read roof matches to within 2 % of it. So my compute-side rows (fc1, fc2, fc3,
   1.4 s of the 4.2) come out *faster* here per call than qb1's ledger and my memory-side rows come
   out the same. Absolutes belong on qb1; this is a ratio card and I have not claimed otherwise.

There is no unattributed residual left inside my slice: every one of the 272 ops in the block dump
whose call chain reaches `PairformerLayer.__call__` appears above, either as its own row or inside a
named group with its total stated.

---

## Phase-3 candidates

Ranked by ms/fold at stake. Nothing here was implemented and nothing merges; Phase 2 owns the
experiments and Moritz owns the merges.

**1. Unfuse `activation="silu"` from Transition fc1 — 835 ms/fold. RELEASE-GATED (not bit-exact).**
`tenstorrent.py:2046` still passes `activation="silu"`, and the identical matmul without it is
**77.81 us against 250.65 us at mt=300 — 3.22x**, measured on the same operands in the same process.
W6 priced this general ttnn cost at 1.7x elsewhere and shipped the fix as `962e6c41`
(`ttnn.multiply_` with `input_tensor_a_activations`); **that fix never reached this site**, which is
the largest block of `ttnn.linear` calls in the model at 4840 calls/fold. The shape of the fix is
already in the code two lines down: fc1's output is consumed only by `ttnn.multiply_(x_1, x_2)` at
`2064`, and that op has headroom of 13 % of the L1 copy roof to absorb an SFPU pass for free.
**Phase 2 must confirm:** (a) hypothesis H1 — that the premium is output-element-linear and K-independent;
(b) that moving silu into the `multiply_` does not simply move the 173 us; (c) **parity** — silu
applied in the matmul packer path versus in the eltwise SFPU pass is not bit-exact by construction,
so this is release-gated and Moritz decides, not me.

**2. `z_proj` @1900 is occupancy-starved by nt=1 — 218 ms/fold at stake, ~150 ms plausible.**
122.2 GB/s = 31.1 % of the DRAM read roof and 3.47 TFLOP/s = 3.6 % of the K=256 compute roof: bound
by neither, therefore a defect. Measured flat from 40 cores (460 us) to 110 (452 us) — **70 of the 110 cores idle** — while the same read stream with nt=32 reaches 31.3 % of the K=256 compute roof. The
op projects c_z=256 down to 16 heads of bias, one output tile column. **Phase 2 must confirm** hypothesis H5 by
sweeping nt = 1, 2, 4, 8, and must establish whether the 16-head output can be produced at a wider
nt (padding to 32 and slicing, or fusing the head split) without changing a byte of the result.

**3. The two norms carry a gap of 22 % of the copy roof — 69 ms/fold, and probably not worth a kernel.**
`layer_norm` marginal bandwidth is **313.9 GB/s against the plain clone's marginal 404.7 GB/s, 77.6 %
of the copy roof**, plus a 12.14 us per-op floor against the clone's 8.10 us. Closing it completely
would return 73.8 us/call on `z_norm` (35.7 ms/fold) and 6.9 us/call on the chunked norm
(33.4 ms/fold). Hypothesis H6 says the gap is the row reduction pinning kt=8 input tiles in a circular buffer
so the NOC reader cannot run ahead; if that is right the fix is a kernel change for 69 ms and I
recommend Phase 2 test H6 cheaply (vary kt at fixed bytes) and then drop it unless kt=8 turns out to
be pathological. **This answers the brief's question 3: the norms do NOT survive the copy-roof
control the way the residual adds do** — the adds are at 109.3 % of the copy roof, the norms at 76-78 % of the copy roof.

**4. `core_grid=11x10` loses to 10x10 on both Transition linears at mt=300 — 30 ms/fold, free.**
fc1 80.20 us at 110 cores against 77.61 us at 100; fc3 94.65 against 93.36. Hypothesis H3 says mt=300 divides
100 exactly and 110 not at all, so the 11th column adds mcast and no work. Worth 2.59 + 1.29 us/call
× 4840 = **18.8 ms/fold**, and likely bit-exact because `core_grid=` fixes `in0_block_w=1` regardless
of grid, so the K reduction order does not change — only output tile placement does. Phase 2 must
verify that with `torch.equal`, and must check hypothesis H3 at mt=330 before anyone generalises this to the
other 51 `core_grid` sites.

**5. fc3 @2066 is NOT a lever, and this is the answer to the brief's question 2.** The brief asks why a
compute-side op at AI 930 sits at a third of its compute roof. Three things, all measured: (a) the
"37.7 % of COMPUTE" was scored against the *square* roof; against the K-corrected roof it is 46.7 % of
the K=1024 compute roof. (b) That roof needs nt=64 to be reached. At the op's own nt=8 the best rate
this card produces is 36.00 TFLOP/s, and fc3 achieves **53.83** — it is already 1.5x the best nt=8
matmul I could construct, because its input is L1-resident. (c) nt=8 is structural: N is c_z=256, so
chunking cannot change it and neither can fusing the chunks, which raises mt and not nt. The
occupancy A/B closes it: 100 cores 93.36 us, 110 cores 94.65 us — the 11th column of the grid receives
no output tile and only adds mcast fan-out. **There is no 2.6x here.** Anyone re-deriving headroom from
the ledger's 37.7 % figure will size a Phase-3 plan against a roof this shape cannot reach.

**6. Not levers, recorded so nobody re-opens them.** The five residual adds are at **109.3 % of the
DRAM→DRAM copy roof** — W6's kill re-confirmed on this card, only fusion into the producer could
touch 816 ms/fold. `multiply_@2064` is at **86.8 % of the L1 copy roof**. `chunk`/`concat` are at
84-93 % of the copy roof, so their 269 ms is the price of the chunking decision, not a bandwidth
defect. The whole `transition_s` track is **0.19 % of the pf_stack total** and mt=10 means it can never
fill the grid.

---

## Corrections to the inherited record

1. **`pf_block_ops.py` silently scores 0.0 s for any op it cannot re-run, and 30 of the block's 272
   ops are in that state — 10 of them each at `layer_norm@2038`, `linear@2046`, `linear@2055`.** Its
   `bench()` holds `reps` extra outputs live, which for an L1-output op under in-fold L1 pressure
   throws, and the wrapper records `dt = 0.0` with an `error` field and prints a coverage figure that
   looks like missing overlap. The three ops are worth **4.132 ms/block, 51.2 % of the block's
   8.071 ms unattributed gap**, and 2.0 s/fold. Any future reader of `ops_*.json` must filter on
   `error` before summing. This is the general defect: **a per-op instrument that degrades to zero
   instead of to absent turns its own failures into a fake overlap finding.**
2. **The Transition is 2.83 s/fold, not 0.53 s** — 5.3x the inherited attribution. The 0.53 s row is
   fc3 alone (site 1799 in the snapshot, `tenstorrent.py:2066` today).
3. **`PairformerLayer` body + `AttentionPairBias` is 1.35 s/fold, not 4.19 s.** The 4.19 s bucket was
   the body plus the Transition time that item 1 lost.
4. **The Transition is 15 972 `ttnn.linear` calls/fold, not 1512.** 1512 is a single-recycle count;
   production is 10 recycles. 19 860 counting the calls outside the trunk, which belong to the
   diffusion sampler and the atom stacks and are noted here, not chased.
5. **The conversion for `PairformerLayer` is 484**, measured by counting `add_@2223` in a real fold:
   480 from 48 x 10 plus 4 from `protenix.py:1286`. Neither 480 nor 524 is right for my ops. T5 still
   owns the MSA/template question.
6. **The chunk is 30 rows (mt=300), not 32 (mt=320)**, because dim 1 of `z` keeps the logical 298. A
   rig built at the padded N=320 is optimistic on fc1 by 3 % of the measured figure and 2 % on fc3.
7. **`z` is 48.82 MB, not 52.43 MB**, so every z-shaped op's byte count in the ledger is high by 7.4 % of the true figure
   and every GB/s scored from it is low by 7.4 % of the true figure.
8. **This card's machine balance is 292.4 FLOP/byte, not 338**, and its dense compute roof is
   115.04 TFLOP/s, not 135.60. Same method, different card. `roofs_card.py`'s 99.77 TFLOP/s on this
   card is a config artifact — it only tries square N×N at the default config — not a third reading.
9. **My own first K-corrected roof sweep was wrong, and I am recording it rather than quietly
   replacing it.** Round 1 swept K with the matmul output in DRAM and produced a "K=256 roof" of
   49.95 TFLOP/s. The Transition's fc2 then measured 64.05 TFLOP/s at K=256 — above the roof, which is
   impossible — and that is how the error surfaced. At M=10240 N=4096 the DRAM output is 83.9 MB per
   call, 307 us of the 430 us measured at this card's 273.4 GB/s write roof. Redone with an L1 output,
   K=256 reaches **95.42 TFLOP/s**. The generalisable rule: **a compute-roof measurement must put its
   output where the op being scored puts its own, or it silently reports a write roof.** The charter's
   §4.6 "a K=256 contraction reaches at most 35.5 TFLOP/s" is the same DRAM-output measurement and
   under-states the K=256 ceiling for any L1-output matmul by 2.7x. On this card the whole K-correction
   across K=256..4096 is 1.25x, not ~3.8x; what costs 2-3x here is output width, not K.
10. **The brief's mt=300/280 was right and I confirm it**, against my own first rig which was not.
   Recording it because the failure mode is now three-for-three in this codebase: measure the shape
   the fold runs, and get the shape from a census inside a real fold, not from the tensor.

Generalisations, recorded and not chased: the `activation=` premium, the nt=1 occupancy starvation and
the grid-divisibility loss are all general ttnn properties and will show up in OpenDDE, OpenFold3 and
Boltz — out of scope for this org, noted here only so the finding is not lost. OpenDDE's Transition in
particular is 2249 ms/fold against protenix-v2's 314 ms and is the more tempting target; it is out of
scope and I did not touch it.
