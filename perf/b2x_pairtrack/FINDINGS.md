# The 3.21x outlier was two sub-units spliced together. Measured apart, the pair-track Transition is 3.47x, it is neither its bytes nor its cores, and the one lever it hands over is 1.0195x bit-exact.

`ws:b2x-pairtrack-deficit`, R3 of `ws:boltz2-2x-orchestrator`. One card: **qb2 physical card 0**,
one Blackhole processor of a p300c, 11x10 grid, ttnn 0.68.0, commit `d4561d80`. Every device
figure is a ttnn trace replay, so the host pays 0.9-3.6 us per replay and nothing here is a
dispatch measurement. No model code changed.

## The short version

| question | answer |
|---|---|
| is the 8.705 ms/call the pair track? | **No.** It is the MSA track. The two 4D `Transition` inputs at 512 aa have the same volume -- pair `1x512x512x128` and MSA `1x1024x512x64` are both 33.55 M elements -- so the volume-based grab that produced the figure selected the MSA one. Reproduced here at **8.7046 ms**, 0.005 % from the published 8.705. |
| what is the pair track, then? | **7.9455 ms/call**, 280 calls/fold, **2.225 s**. The MSA track is 8.7046 ms x 16 calls = 0.139 s. |
| so where did 3.21x come from? | MSA-track device time over pair-track bytes and op count. Measured self-consistently: pair **3.47x** its model, MSA **2.28x**. |
| DISCRIMINATOR | **MIXED**, and BYTES is falsified. A buffer-address recount of the isolated pair call gives **474.235 MB** -- the same number the campaign used -- which is 1.066 ms of 7.9455 at 445.0 GB/s. Both ownership rules agree exactly. |
| CORE-UTIL | **64 of 110.** The body's fc1 saturates at 64 cores: 30.425 us at 8x8 against 28.237 us at 11x10. 1 core to 110 is 32.7x, 29.7 % parallel efficiency. |
| DEFICIT-SECONDS | **1.661 s** of the campaign's 6.3 s, fully apportioned into four named mechanisms below. |
| REACHABLE | **1.0195x, measured end-to-end and bit-exact** (23.955 -> 23.496 s, four paired deltas all negative, every fold `4f3995a69be5d610`), ~1.032x with the gated silu unfuse on top. Not 1.5x: the two 4D Transitions are 2.364 s of the 23.710 s fold, so deleting them **entirely** is 1.11x. |

## 1. The two tracks, each against its own bytes and its own op count

`pairtrack.py` intercepts `Transition.__call__` during one fold, clones the argument of a settled
call of each 4D shape, then graph-captures and trace-replays that exact call. Census from the same
fold: 960 `Transition` calls, of which **280 are `1x512x512x128`** (the pair track: one per
`PairformerLayer`, 256 trunk + 16 MSA-nested + 8 confidence) and **16 are `1x1024x512x64`** (the
MSA track). The rest are the 3D single-track and diffusion shapes, which take a different branch.

| | pair track | MSA track |
|---|---|---|
| z | 1x512x512x128, hidden 512 | 1x1024x512x64, hidden 256 |
| row block, chunks | 16, 32 | 16, 64 |
| **device ms/call** | **7.9455** (spread 0.23 %) | **8.7046** (spread 0.05 %) |
| calls/fold, s/fold | 280, **2.225** | 16, **0.139** |
| real DRAM MB, buffer-address dedupe | **474.235** | **606.2** |
| device programs | **193** | **387** |
| bytes at 445.0 GB/s | 1.066 ms | 1.362 ms |
| ops at 6.36 us | 1.228 ms | 2.461 ms |
| model | 2.293 ms | 3.823 ms |
| **measured / model** | **3.465x** | **2.277x** |

The published row had 8.705 ms against 474.2 MB and 258 ops. The bytes were the pair track's and
the recount lands on them to 0.007 %; the device time was the MSA track's. 3.21x was a ratio
between two different sub-units. Correcting it does not remove the outlier, it reassigns it: the
pair track is worse than the published figure made it look, 3.47x.

## 2. BYTES is falsified, and the deficit is inside the ops

Recounting on buffer address is exactly the correction that inverted the campaign's planning
decision elsewhere, so it was the first thing to check. It changes nothing here: 474.235 MB under
the range ownership rule and 474.235 MB under the stack rule, against the 474.2 MB already on
record. At the measured 445.0 GB/s roof that is **13.4 % of the call**. The two fattest ops are
the row-blocking machinery, not the math: `ttnn.chunk` 201.33 MB and `ttnn.concat` 201.33 MB of
the 474.235.

The counter cannot see a matmul that re-streams an operand over K, so the isolated-op leg is the
independent check. Each op of the body was rebuilt at the shipped chunk shape with the module's
own weights and memory configs and replayed **alone**:

| op, pair track, h=16 | us/call | x32 chunks, ms | share | limiter, measured |
|---|---|---|---|---|
| `ttnn.layer_norm` -> L1 | 23.608 | 0.755 | 9.4 % | 88.8 GB/s, 20 % of roof |
| `ttnn.linear` fc1, `activation="silu"` | **114.503** | **3.664** | **45.5 %** | 9.38 TFLOP/s |
| `ttnn.linear` fc2, bare | 27.041 | 0.865 | 10.8 % | 39.71 TFLOP/s |
| `ttnn.multiply_`, L1 only | 26.690 | 0.854 | 10.6 % | L1 |
| `ttnn.linear` fc3 -> DRAM | 35.990 | 1.152 | 14.3 % | 29.83 TFLOP/s |
| `ttnn.chunk`, once per call | 397.6 | 0.398 | 4.9 % | 337.6 GB/s, **75.9 % of roof** |
| `ttnn.concat`, once per call | 356.3 | 0.356 | 4.4 % | 376.7 GB/s, **84.6 % of roof** |
| **isolated sum** | | **8.045** | | |
| **shipped call, replayed** | | **7.9455** | | |

The sum closes on the independently measured call to **1.25 %**, so the deficit is in the
individual ops and not in the sequence, the allocator or the dispatch order. Same check on the
MSA track closes to 1.8 %. And `chunk` and `concat` are at 76-85 % of the bandwidth roof, so the
one part of this sub-unit that is bandwidth-bound is already nearly done.

## 3. The four mechanisms, with numbers

The pair track's deficit over its model is 7.9455 - 2.293 = **5.6525 ms/call**.

**(a) The fused silu, 1.5224 ms/call, 26.9 % of the deficit.** `activation="silu"` in the matmul
costs 114.503 us where the identical bare matmul costs 26.538 and a standalone `ttnn.silu` on the
same L1 tensor costs 40.781. Unfusing saves 47.184 us/chunk, 1.510 ms/call by the op ladder and
**1.5224 ms/call** by replaying the whole call in both arms -- two instruments, 0.8 % apart. On
the grid sweep the penalty is a clean 4.10x at every grid size (11x10: 115.884 us fused against
28.237 bare), which is why it is not a parallelisation effect.

This is **not new**. It is root-caused: ttnn computes silu's sigmoid with the accurate Cody-Waite
exp whatever the caller asks for, `TT_BIO_UNFUSED_SILU=1` gates the unfuse, and it is **not
bit-exact**, hence off by default (commit `1c70fc39`, `state/protenix-trunk--z-silu-lowering-fix.md`,
which also carries the two-line header fix and its upstream PR). What is new is its price on
Boltz-2 at 512 aa, which no artifact had: **0.451 s/fold**, 1.0194x.

**(b) The row block is too small, 1.3710 ms/call, 24.3 % of the deficit.** Forcing the row-block
height with the shipped `TT_BIO_TRANSITION_H_CHUNK` knob, same bytes, same FLOPs, half the ops,
twice the per-core work:

| row block | pair track ms | MSA track ms |
|---|---|---|
| 8 | 9.6460 | 12.2068 |
| **16, shipped** | **7.9450** | **8.7078** |
| 32 | **6.5745** | 6.7606 |
| 64 | L1 clash | **6.3787** |
| 128 | L1 clash | L1 clash |

16 is `TRANSITION_H_CHUNK_SIZE`, a constant tuned at the reference W=1024, c=128. At the pair
track's W=512 the live L1 per core is 171,585 B of the 1,572,864 available, and 32 lands at
343,170 B and works; 64 clashes with the matmul's static circular buffers. The MSA track's
narrower channel reaches 64. The reason Blackhole never finds this itself is structural: the
per-core L1 derivation in `Transition.__call__` (`_l1_rows_at`, the `SMALL_GRID_TRANSITION_ELEMS`
raise, the cap) is **all guarded on `_IS_SMALL_GRID`**, which is False here, so on Blackhole the
row block is whatever the constant says and is never checked against the part's own L1.

Why it pays is (c): at h=32 the same fc1 matmul runs at 47.98 TFLOP/s against 40.46 at h=16, and
`layer_norm` goes 88.8 -> 120.1 GB/s, `fc3` 61.9 -> 82.2 GB/s.

**(c) CORE-UTIL: the shipped chunk starves the grid, and the last 46 cores are free.** The body's
fc1 at the shipped shape is M=256 tiles, N=16 tiles, K=4 tiles. Measured `core_grid` sweep, bare
matmul:

| cores | 1 | 4 | 16 | 36 | 44 | 64 | 80 | 88 | 100 | 110 |
|---|---|---|---|---|---|---|---|---|---|---|
| us | 922.6 | 303.9 | 80.9 | 46.2 | 46.2 | 30.4 | 30.5 | 28.7 | 27.8 | 28.2 |

1 core to 110 is **32.7x, 29.7 % parallel efficiency**, and everything past **64 cores** buys
1.08x. So 46 of the 110 cores contribute nothing at the shipped row-block height. N=16 tiles
cannot occupy 11 grid columns, which is the grid-height-hole shape of this defect class, and the
fix for it is the bigger row block, not a different grid.

**(d) The bare matmuls are compute-bound below their own achievable rate, ~0.48 ms/call.** At
h=32 the three bare matmuls are 142.0 us/chunk, 2.273 ms/call, at 40.8-48.1 TFLOP/s. The best any
matmul of this family reaches on this card with this fidelity (HiFi4 by default) is **55.221
TFLOP/s**, measured at equal FLOPs by widening K:

| K | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|
| TFLOP/s at 1.074 GFLOP | 37.95 | 53.75 | **55.22** | 51.64 | 45.76 |

The body's K is the channel, 128, and it costs **1.46x** against K=512 at the same FLOPs. That is
a property of the model's channel width, not of tt-bio's code, and nothing short of fusing the two
`fc` matmuls into one K=128 x N=1024 call touches it.

Residual after (a) and (b) together: 7.9455 -> 5.5134 ms/call, **2.4321 ms/call, 43.0 % of the
deficit removed**, leaving 3.2204 ms still above the 2.293 ms model. (c) and (d) are what is left,
plus 0.823 ms/call of L1-only `multiply_` and 0.559 ms of `layer_norm` at 27 % of the DRAM roof.

## 4. DEFICIT-SECONDS and REACHABLE

**DEFICIT-SECONDS: 1.661 s** of the campaign's 6.3 s block deficit sits in the two 4D
Transitions -- 280 x 5.6525 ms = 1.583 s pair, 16 x 4.882 ms = 0.078 s MSA -- and every second of
it is now attributed to (a), (b), (c) or (d) above.

**REACHABLE.** The row block was taken end-to-end rather than left as a projection.
`hchunk_ab.py`, `tt_baseline.build_fold` driving the production `predict_one`, arms paired and
interleaved in one process with the within-pair order alternating and a discarded cold fold per
arm, 4 pairs:

| arm | warm folds s | median s | within-arm spread |
|---|---|---|---|
| `TT_BIO_TRANSITION_H_CHUNK=32` | 23.541, 23.462, 23.529, 23.444 | **23.496** | 0.415 % |
| `=16`, what production derives | 23.894, 23.948, 23.961, 23.995 | **23.955** | 0.420 % |

**Paired deltas -0.353, -0.486, -0.431, -0.551 s, median -0.459 s, all four negative and the
smallest 3.5x the within-arm spread. 1.0195x.** The per-call projection said 0.415 s, so it was
9.6 % low -- the fold gains slightly more than the two tracks' device deltas alone predict.

| arm | s/fold | fold | x | status |
|---|---|---|---|---|
| row block 16 -> 32 | **0.459** | **23.496** | **1.0195** | MEASURED, bit-exact |
| unfused silu, not bit-exact | 0.451 | 23.259 | 1.0194 | projected, release-gated |
| both (row block 32 + unfuse) | 0.728 | 22.982 | **1.0317** | projected |
| the 4D Transitions deleted entirely | 2.364 | 21.346 | 1.107 | the ceiling of this sub-unit |

The last row is the one that decides the round: **this sub-unit cannot produce 1.5x.** Even free,
it is 1.11x. The 3.21x label made it look like the place where a multiple was hiding; measured
apart from the MSA track it is 2.2 s of a 23.7 s fold, and the honest lever it hands the next task
is 1.018x bit-exact-candidate plus a 1.019x arm that is already gated.

## 5. What the next task should do with this

1. **Land the row block from the part's own L1, not from a constant.** The derivation already
   exists in `Transition.__call__`; it is fenced behind `_IS_SMALL_GRID`. Unfencing it -- take the
   largest row block whose live per-core L1 plus the matmul's circular buffers fit -- is a UNIFIED
   change that reaches every model through the one class, which is also why it is broad enough to
   need a release gate and a size ladder, not a self-merge. Pair track tops out at 32, MSA at 64,
   and the wall is documented non-monotonic in this height, so the arms have to be measured and
   not derived.
2. **The fold number is taken; only the code change is owed.** 1.0195x, bit-exact, above. What is
   left is landing the row block as a derivation instead of a forced environment variable, which
   is item 1.
3. **`tt_baseline.measure`'s `--ab-env` path cannot run a boltz-2 A/B at all**, which is a harness
   bug worth fixing centrally. It asserts `cold_metrics.get("msa")`, but the boltz-2 branch of
   `_WorkerState.predict_one` builds its metrics in `tt_bio.main.write_result`, which never sets
   an `msa` key -- `worker.py` sets it itself on the esmfold2, opendde, protenix, rf3 and
   openfold3 paths and the boltz-2 path was missed. So the assert is a **false negative for
   boltz-2 specifically**, not a fold that ran without an MSA. `hchunk_ab.py` works around it by
   proving the MSA with the thing that actually matters: arm `16` reproduces `4f3995a69be5d610`,
   the published digest for this protocol with its MSA, which a single-sequence fold cannot. The
   central fix is one line in the boltz-2 metrics dict, and it belongs to whoever next touches
   `worker.py`.
4. **Do not re-attack the bytes of this sub-unit.** `chunk` and `concat` are at 76-85 % of the
   measured roof and carry 85 % of its DRAM traffic.

## PARITY

**The row-block arm is bit-exact, measured.** All **10 folds** of the A/B -- 2 cold and 8 warm,
both arms -- wrote

    4f3995a69be5d610  cdk2x2_512.cif

byte-identical to `b2x-baseline-attrib`'s current-main reference. So `TT_BIO_TRANSITION_H_CHUNK=32`
changes no output byte, which is what row-local blocking predicts, and no `cdk2x2_298` control is
owed because a bit-identical arm has nothing to control against. The baseline arm reproducing the
same digest is also the positive proof that these folds ran the published protocol **with** their
MSA, which is the check `tt_baseline.measure` tries and cannot make on this model.

No model code changed on this branch. Every measurement leg either replays unmodified shipped
`Transition.__call__` with its own cloned argument, or rebuilds one of its ops in isolation with
the module's own weights. The `TT_BIO_UNFUSED_SILU` and `TT_BIO_TRANSITION_H_CHUNK` arms are set
on the module global and in the environment for the duration of a replay and restored after; both
are shipped screen hooks, both default off/unset, and neither is committed as a default here.

The fold this pass grabbed from ran in 25.028 s against the campaign's 23.710 s, because a sibling
worker was folding on card 1 of the same box (loadavg 4.74 at start) and this pass did not hold
benchlock for the grab. That wall is not used for anything. It cannot reach the device figures:
the MSA-track floor came out at **8.7046 ms against the published 8.705 ms, 0.005 %**, on a loaded
box, which is the instrument's own control.

## Files

* `pairtrack.py` -> `pairtrack_512_qb2c0.json`, `run2.log`. Both tracks, six legs each.
* `v1_msatrack_512_qb2c0.json`, `v1_msatrack.log`. The first pass, before the tracks were told
  apart: kept because reproducing the published 8.705 ms from a volume-based grab is the evidence
  for section 1.
* `hchunk_ab.py` -> `hchunk_ab_512_qb2c0.json`, `hchunk_ab.log`. The fold-level paired A/B,
  4 pairs, per-fold CIF digest and loadavg.

Each is `TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:b2x-pairtrack-deficit
python3 perf/b2x_pairtrack/<script>.py --out <json>` from the repo root with `PYTHONPATH` at the
checkout.
