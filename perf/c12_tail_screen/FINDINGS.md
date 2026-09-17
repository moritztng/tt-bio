# The 1.5672 s nobody had looked at: fourteen classes, itemised, priced, and five of them closed

`ws:c12-tail-classes-screen`, 2026-09-18, pc, `card=cpu`, no device and no lease. Every number
below comes from artifacts already committed on `origin/wk/c12-profiled-fold` at
`a63d6d8e3749843c1c58814cbabf87cd19a3bc17`, read through `git show` rather than copied onto this
branch. No production code changed.

## VERDICT

**GO on one item (L1, 0.3138 s, mechanism already built and shipped elsewhere), BLOCKED on one
(L6, needs a card), STOP on the other twelve classes.** The block is 1.5673 s and 11.86 % of the
13.2090 s device term. Every second of it is now itemised: 0.3138 s has a mechanism with a shipped
precedent (L1), 0.2451 s more has a named mechanism that needs a kernel build (L2 net + L4),
0.4335 s is blocked on a device measurement worth 0.077 s (L6), 0.2075 s is closed against a named
wall (L3 + L5), and the remaining 0.3441 s is signatures of 0.0406 s and below with no mechanism
between them.

## Reproduce

```
python3 perf/c12_tail_screen/screen.py --out perf/c12_tail_screen/screen.json
python3 perf/c12_tail_screen/leads.py
```

`screen.py` re-derives the fence window for each of the six composed units and refuses to print a
table unless every unit's device ms/call and programs/call reproduce `runs/composed.json` to
5e-5 ms. All six do, and the whole table sums to 13.2090 s against composed's 13.2090 s. Those two
controls are the reason anything below can be trusted: if the window this screen finds were not the
window `reduce.py` found, every second here would be wrong.

## THE TWELVE, broken out -- the aggregate is retired

"twelve smaller 0.8330 s" has appeared as one number in every C12 document. It is twelve items and
the largest is 0.1917 s:

| device op | s/fold | programs | Mcycles | DRAM GB | roof | % of roof | AI FLOP/B |
|---|---|---|---|---|---|---|---|
| SDPAOperation | 0.4335 | 6000 | 585.2 | 109.86 | 2R+1W | 58.2 | 50.1 |
| NlpCreateHeadsDeviceOperation | 0.3007 | 6264 | 405.9 | 64.06 | 1R+1W | 54.2 | 0 |
| SliceDeviceOperation | 0.1917 | 15856 | 258.8 | 63.36 | 1R+1W | 84.1 | 0 |
| ReshapeViewDeviceOperation | 0.1571 | 5082 | 212.0 | 26.24 | 1R+1W | 42.5 | 0 |
| PadDeviceOperation | 0.0864 | 3600 | 116.6 | 20.14 | 1R+1W | 59.3 | 0 |
| PermuteDeviceOperation | 0.0823 | 872 | 111.0 | 10.70 | 1R+1W | 33.1 | 0 |
| ConcatDeviceOperation | 0.0748 | 1481 | 101.0 | 29.20 | 1R+1W | 99.3 | 0 |
| UntilizeDeviceOperation | 0.0590 | 1232 | 79.7 | 19.78 | 1R+1W | 85.2 | 0 |
| EmbeddingsDeviceOperation | 0.0571 | 18 | 77.1 | 1.23 | 1R+1W | 5.5 | 0 |
| TilizeDeviceOperation | 0.0536 | 1232 | 72.3 | 20.08 | 1R+1W | 95.3 | 0 |
| SoftmaxDeviceOperation | 0.0323 | 280 | 43.6 | 4.43 | 1R+1W | 34.9 | 1.3 |
| CopyDeviceOperation | 0.0243 | 5328 | 32.8 | 5.78 | 1R+1W | 60.5 | 0 |
| NLPConcatHeadsDeviceOperation | 0.0131 | 1200 | 17.7 | 2.75 | 1R+1W | 53.5 | 0 |
| UnaryNgDeviceOperation | 0.0015 | 202 | 2.0 | -- | -- | UNPRICED | -- |
| **total** | **1.5673** | **48647** | **2115.8** | | | | |

A 0.8330 s bucket that was one 0.3 s op would have had a different verdict from twelve 0.07 s ops.
It is neither: it is one 0.19 s op, three between 0.05 and 0.09 s, and eight under 0.06 s. Nothing
in the twelve is worth a row on its own size.

**Traffic or arithmetic, decided rather than assumed.** Eleven of the thirteen priced classes do
**zero** arithmetic -- they are layout moves -- so the question resolves by inspection and needs no
balance number. The two that compute both come out far below the measured 260.9 FLOP/byte machine
balance: SDPA at 50.1 FLOP per charged DRAM byte, Softmax at 1.3. SDPA's 5.506 TFLOP over the fold
is 0.0484 s at the 113.68 TFLOP/s HiFi2 roof against 0.4335 s measured, so arithmetic is 11.2 % of
it. **The whole block is traffic-bound.**

## Two byte-model corrections, because a rate above the roof is a broken model

1. **Charging every recorded operand put Slice at 1336.2 GB/s**, 3.4x this part's DRAM roof, because
   it charged the whole `[1,512,512,128]` input for a slice that only reads the 47 rows it returns.
   Each op is now charged from its own read and write set, named in `screen.py:TRAFFIC`, and the
   table prints charged bytes as a percentage of all recorded operand bytes so nothing is dropped
   silently.
2. **SDPA's K and V are read once per q chunk**, and the chunk count comes from the `q_chunk_size`
   in each row's own executed `program_config`. Charging each operand once understated the token
   site by 1.50x and put it at 37.4 % of roof when it is at 56.0 %. This is the "a resource gap is
   not headroom" trap arriving in its exact predicted form, and the executed program config is what
   closed it.

**The roofs are arms, not ceilings.** Two signatures come out ABOVE the 393.31 GB/s `bw_clone` roof
-- Concat at 103.3 % and the MSA Tilize at 112.7 % -- and the Tilize's 442.8 GB/s is 1.6 % above
`bw_add8192`'s 435.73 GB/s as well. So the true 1R+1W DRAM peak on this part is at least
443 GB/s, and any signature this screen reports above ~95 % should be read as "at the roof", not as
a measurement of slack.

## The leads, grouped by the mechanism that would move them

The fourteen op codes are not the unit a lever acts on. The diffusion transformer's head plumbing
is spread across four classes; the pair Transition's chunking across two. `leads.py` regroups them.

| lead | kind | s/fold | Mcycles | programs | Transpose alongside (claimed by nobody) |
|---|---|---|---|---|---|
| L6 diffusion SDPA | blocked | 0.4335 | 585.2 | 6000 | 0 |
| L1 head-major qkv / concat-heads | **delete** | **0.3138** | **423.6** | 7464 | 0 |
| L3 pair Transition 11-way chunk | stop | 0.1504 | 203.0 | 3360 | 0 |
| L2 diffusion head-48 merge | delete, with a debit | 0.1354 | 182.8 | 9600 | 0.0547 |
| L4 OuterProductMean layout round-trip | named, not opened | 0.1330 | 179.5 | 48 | 0.0455 |
| L5 Embeddings gather | stop | 0.0571 | 77.1 | 18 | 0 |
| grouped | | 1.2231 | 1651.2 | | |
| ungrouped, all signatures under 0.05 s each | | 0.3441 | 464.5 | | |

### L1 -- GO. 0.3138 s, and the lever is already written, shipped and default-on next door

`tt_bio/triatt_qkv.py` deletes `nlp_create_qkv_heads` and `nlp_concat_heads` for triangle attention
by having the qkv matmul write q, k and v itself: output tile *(i, n)* of the matmul already **is**
tile *(batch, head, row)* of q, k or v, so only the destination address changes and no element moves
inside a tile. `TRIATT_HEAD_MAJOR_QKV` and `TRIATT_HEAD_MAJOR_TAIL` are both `True` in the shipped
tree, `torch.equal` at six sizes, and the fold A/B at 512 aa reads TriAtt body
19,719.8 -> 16,716.5 ms, 1.1797x.

**The precondition is head_dim being a whole number of tiles in PADDED form, and every signature in
L1 satisfies it.** The token transformer's packed qkv is `[1,1,512,3072]` = 3 x 16 x 64, so each
head is exactly two tiles; the atom blocks and the trunk remainder are head_dim 32, one tile. The
diffusion side simply never got the lever:

| site | s/fold | programs | cores | % of roof | shape |
|---|---|---|---|---|---|
| DiT token qkv split | 0.2047 | 4800 | 16 | 37.5 | `[1,1,512,3072]` -> 3 x `[1,16,512,64]` |
| atom-block qkv split | 0.0902 | 1200 | 110 | 93.1 | `[140,1,128,128]`+`[140,1,128,256]` -> 3 x `[140,4,128,32]` |
| trunk remainder | 0.0058 | 264 | 16 | 36.3 | `[1,1,512,1536]` -> 3 x `[1,16,512,32]` |
| atom-block head merge | 0.0131 | 1200 | 110 | 53.5 | `[140,4,32,32]` -> `[140,1,32,128]` |

The mechanism is **deleted work**, not occupancy and not a program count: the bytes are never
moved. That matters here because the token split's 16-of-110 cores looks like an occupancy story and
is not the one being told. For the record the occupancy reading is at least self-consistent -- the
same op code reaches 93.1 % of roof on the atom signature at 110 cores while the token signature
sits at 37.5 % on 16, and per core the 16-core arm is already sustaining 9.2 GB/s against the
3.3 GB/s the 110-core arm needs, which names per-core NOC bandwidth as what binds it -- but that is
a rate argument worth ~0.12 s and it is not the one L1 rests on. Deletion recovers the measured
seconds whether the op ran at 37 % of its roof or at 99 %.

### L2 -- fold into L1's row as a second arm, not its own row. 0.1121 s net

The token transformer's head_dim is **48** (768 / 16), so the merge back to `[1,1,512,768]` has to
drop 16 pad channels per head and elements *do* move inside tiles. `nlp_concat_heads` cannot be used
at all -- it would pad each head to 64 and produce 1024 -- and L1's tile-id re-point does not
transcribe. Today the merge is Slice -> Transpose -> ReshapeView -> Transpose, and the executed graph
prices one DiT attention site like this:

```
Matmul qkv   [512,768]->[512,3072]   39,847 ns   Matmul class
NlpCreateHeads -> 3 x [1,16,512,64]  42,664 ns   16 cores      <- L1
SDPA                                 76,475 ns                 <- L6
Slice        [1,16,512,64] log 48     6,304 ns                 <- L2
Transpose    -> [1,16,48,512]         6,562 ns   Transpose class, not claimed
ReshapeView  -> [1,1,768,512]        21,809 ns   21.4 % of roof, the worst rate in the tail  <- L2
Transpose    -> [1,1,512,768]         4,851 ns   Transpose class, not claimed
Matmul out   [512,768]->[512,768]    14,919 ns   Matmul class
```

**The layout plumbing around one SDPA costs 82,190 ns against the attention's own 76,475 ns.** The
out projection could absorb the merge as a 16-way batched matmul accumulating over heads, with
zeroed weight rows on the pad lanes, which moves no element -- but it grows that matmul's K from
768 to 1024, +33 % FLOP on a measured 0.0699 s, so 0.0233 s comes straight back. Net upper bound
0.1354 - 0.0233 = **0.1121 s**, and 0.0547 s of Transpose alongside it that this screen does not
claim.

### L3 -- STOP, and it settles the chunk-height dispute from the fold's own dispatch

`tenstorrent.py:8483` runs `ttnn.chunk(x, -(-H // transition_h_chunk_size), dim=1)`. At H=512 the
executed chunk height is **47**: the graph shows 10 slices of `[1,47,512,128]` and one of
`[1,42,512,128]` per Transition, 3080 slices and 280 concats per fold. So 47, from the dispatch,
not 16 and not 32.

**And it does not matter, because the traffic is byte-neutral in the chunk count.** Every element is
read exactly once and written exactly once whatever the height, so chunk-height tuning cannot move
one byte of the 0.1504 s. Both ops are already at 92.8 % and 103.3 % of their mix-matched roof.
Deleting the traffic needs a zero-copy dim-1 slice and a write-into-preallocated concat; ttnn offers
neither, and that is a tt-metal feature request, not a tt-bio change.

### L4 -- named, not opened. 0.1330 s in this screen's classes

`OuterProductMean` at 512 aa writes a `[16384,16384]` product -- 268.4 M bf16 elements, 536.9 MB --
and its consumer wants `[524288,512]`. That reshape is not expressible as a tile relabelling, so
ttnn falls back to untilize -> row-major reshape -> tilize, and a transpose follows. Four passes over
the same 536.9 MB. Per MSALayer call:

```
Untilize    [1,1,16384,16384]     2.841 ms   96.4 % of roof
ReshapeView -> [1,1,524288,512]   3.054 ms   89.3 %
Tilize      [1,512,1024,512]      2.428 ms  112.7 %
Transpose   -> [1,512,512,1024]   2.842 ms   Transpose class, not claimed
```

Every one of the three in this screen's classes is at or past the roof, so there is no rate to win,
only deletion -- and deletion means a custom outer-product kernel that writes the consumer's axis
order. 16 calls per fold, 0.1330 s, and `trimul-kernel-build-run` already measured that a custom
matmul loses to `ttnn.matmul` at production sizes, so the case rests entirely on the four deleted
passes rather than on the product itself. Worth a row after L1; not worth one before it.

### L5 -- STOP on a known hardware floor

21.5 GB/s, **5.5 % of the copy roof**, the worst rate in the tail. It is not a defect of ours:
ttnn gather and scatter on Blackhole are per-element-rate limited at ~10-14 cycles per element, and
a bfp8 arm at half the bytes came out 0.1 % apart, which is the proof it is not bandwidth. No index
dtype, layout, `sub_core_grids` or preallocation moves it. 0.0571 s over 18 programs.

### L6 -- BLOCKED on a card, and the prize is 0.077 s

The biggest single item in the tail and the one with no mechanism CPU evidence can name. The
q_chunk is **not** mis-set: the executed program config reads q_chunk 128 / k_chunk 256 at the token
site and 32 / 128 at the atom site, which is exactly what `tenstorrent.py:1167 _grid_q_chunk`
specifies for work=16 on a 110-core grid, and that rule carries its own measured sweep in which 128
is the best rung (95.60 us against 140.40 at chunk 32 and 182.70 at 512). The only mechanism left is
the `[1,16,512,512]` bias, 44.5 % of the token site's charged traffic, in bfp8: 0.792x the bytes and
so ~0.077 s (104 Mcycles) if the rate holds. That is a dtype change on an attention bias and it owes
an Angstrom reading against the seed-scatter floor, which needs a device.

## What this does to the campaign's arithmetic

If every priced class reached its own mix-matched roof the block would read 0.9329 s instead of
1.5657 s, so **0.6328 s / 854.3 Mcycles is the existence bound** on the whole tail. That is a bound,
not a prediction: nothing on this part has ever reached its roof, and the best of generic_op's six
sites reaches 79.39 %. The buildable part of it is L1's 0.3138 s, and L1 is a deletion, so the roof
does not enter its sizing at all.

## Two method notes worth keeping

1. **Dedupe an executed-graph attribute read on the SHAPE, never on a prefix of the attribute
   string.** Both SDPA signatures share their first 80 characters of `ATTRIBUTES`, so a dedupe keyed
   on that prefix showed only the atom site's `q_chunk_size=32` and this screen nearly reported the
   token site as running the worst rung of its own measured sweep. Re-keying on `INPUT_0`'s shape
   showed 128 / 256, which is correct.
2. **`UnaryNgDeviceOperation` is reported and not priced.** Three of its 202 programs carry a
   `[1,512,512,128]` bf16 operand (67.1 MB) and finish in 2.2-3.0 us on 110 cores, 70x faster than
   the DRAM roof allows for that volume, while the same op on the same shape inside
   `PairAssemblyDevice` takes 485 us (276 GB/s, plausible). Either the shape or the duration is
   wrong on those rows. It is 0.0015 s of 13.2090 s so it changes nothing, but a class whose rate is
   physically impossible does not get a roof percentage printed next to it.

## Clock

**1350 MHz** for every figure above. The device seconds were measured by `c12-profiled-fold` with
AICLK forced through tt-kmd's ARC queue and sampled inside every profiled region, min = max = 1350
on every sample with zero read errors; the roofs come from
`perf/c12_genop_rate/runs/roofs_qb2c3_r40.json`, qb2 card 3, 110 cores, same clock. This row ran no
fold and measured no new second: it is a CPU-only re-reading of committed artifacts, and the only
numbers it originates are arithmetic on them.
