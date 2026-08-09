# W7 — block-level SRAM strategy for one Pairformer block at 298 aa

**Host:** qb2 (tt-quietbox2), card 2, ttnn **0.68.0**, grid 11x10 = 110 cores, 1 532 416 B of
unreserved L1 per core = **168.6 MB aggregate**. Not the production pin. Every absolute ms below is
a qb2 number and only the ratios travel. Branch `wk/perfwar-sram-blocking-strategy`, scripts in
`perf/sram_strategy/`.

**Shape:** protenix-v2 trunk PairformerLayer, layer-0 weights, 298 aa padded to **N=320**, c_z=256.
Block wall, warm, synced both sides, median of 11: **45.28 ms**.

**Roofs, re-measured on this card** (`block_working_set.py --mode roofs`, `chain_ab.py --arm roof`):

| roof | value | method |
|---|---|---|
| DRAM read | 403.2 GB/s | 67.1 MB DRAM->L1 clone |
| DRAM write | 268.3 GB/s | 67.1 MB L1->DRAM clone |
| DRAM mixed (read+write aggregate) | 443.7 GB/s | DRAM->DRAM add, 3 x 67.1 MB |
| dense bf16 matmul, HiFi4 + fp32_dest_acc + packer_l1_acc | **113.79 TFLOP/s** | 4096^3, DRAM result |

---

## 1. The headline: at 298 aa the block is bound by neither roof, so L1 residency is not the lever it looks like

Instrumenting every ttnn call the block makes (`block_working_set.py --mode ledger`, 302 ops,
`analyze_ledger.py` for the tables) gives the block's whole DRAM traffic:

**4263.8 MB read + 2876.0 MB write = 7.14 GB per block.**

At the measured roofs that costs `max(4263.8/403.2, 2876.0/268.3, 7139.8/443.7) = 16.09 ms`.
The block takes 45.28 ms, so **DRAM is 35.5 % utilised**. The arithmetic is 560.6 GFLOP/block
(W8's standalone census at this exact shape), which at 113.79 TFLOP/s is 4.93 ms = **10.9 % of the
compute roof**.

So the block sits at 35.5 % of DRAM and 10.9 % of compute. Removing DRAM bytes does not move the
wall unless the specific op holding the critical path is itself memory-bound. That is the single
most important thing this leg has to hand to W4 and W5: **"keep it in SRAM" is not free money at
298 aa, it is worth exactly as much as the memory-bound ops it removes, and that has to be
measured per op, not assumed from the byte count.**

Measured value of the placement lever where it does apply, below.

## 2. The working set, with lifetimes

Distinct device buffers the block touches: **125, 1175.2 MB in total** (1053.7 DRAM, 121.5 L1).
Live-set high water from the lifetime model: **DRAM 532.5 MB** (peak at `nlp_create_qkv_heads` in
tri_att_start), **L1 72.1 MB** (peak at the trimul's triangle matmul).

Largest buffers, `first`/`last` are op indices in the 302-op trace, `span` is the lifetime:

| MB | buf | shape | first | last | span | touches | phases |
|---|---|---|---|---|---|---|---|
| 157.29 | DRAM | 320x320x768 | 193 | 196 | 3 | 4 | tri_att_start (fused qkv) |
| 157.29 | DRAM | 320x320x768 | 210 | 213 | 3 | 4 | tri_att_end (fused qkv) |
| 52.43 | DRAM | 1x320x320x256 | 0 | 280 | **280** | **40** | every phase (the pair rep) |
| 52.43 | DRAM | 1x320x320x256 | 0 | 279 | 279 | 20 | every phase |
| 52.43 | DRAM | 1x320x320x256 | 88 | 221 | 133 | 34 | trimul, tri_att, residual |
| 52.43 | DRAM | 1x320x320x256 | 89 | 215 | 126 | 10 | trimul, tri_att_end |
| 52.43 | DRAM | 320x8x320x32 (x3) | 196 | 214 | 18 | 4 | q / k / v |
| 45.88 | DRAM | 1x320x320x224 | 77 | 182 | 105 | 4 | trimul running concat |
| 26.21 | **L1** | 1x320x320x128 | 1 | 173 | 172 | 32 | trimul |
| 20.97 | **L1** | 1x32x320x1024 (x2) | 224 | 272 | 48 | 40 | transition_z |
| 6.55 | **L1** | 1x320x320x32 (x7) | 2 | 182 | ~176 | 32-96 | trimul chunks |

Traffic by phase (`floor_ms` = those bytes at this card's roofs):

| phase | ops | read MB | write MB | L1 out MB | floor ms |
|---|---|---|---|---|---|
| tri_att_end | 17 | 847.7 | 742.2 | 0.0 | 3.583 |
| trimul_start | 93 | 918.3 | 498.1 | 838.9 | 3.192 |
| trimul_end | 93 | 918.3 | 498.1 | 838.9 | 3.192 |
| tri_att_start | 15 | 742.9 | 637.3 | 0.0 | 3.111 |
| residual adds | 8 | 525.5 | 262.9 | 0.0 | 1.777 |
| transition_z | 52 | 173.0 | 157.3 | 681.6 | 0.744 |
| attention_pair_bias | 19 | 134.3 | 79.9 | 0.0 | 0.483 |
| transition_s | 5 | 3.8 | 0.2 | 3.2 | 0.009 |

**Re-reads dominate.** Counting every DRAM buffer read by more than one op, **3210.1 MB of the
4263.8 MB read (75.3 %) is a re-read**, and three pair-shaped 52.43 MB buffers read 28, 15 and 11
times account for 2831 MB of it on their own.

## 3. The L1 budget is 18.6 MB, not 168.6 MB — this is the constraint everything else obeys

Every earlier attempt to pin ~52 MB in L1 died with the same throw:

```
Statically allocated circular buffers in program N clash with L1 buffers on core range
[(0,0)-(10,9)]. L1 buffer allocated at 1148928 and static circular buffer region ends at 1159680
```

so the binding constraint is not the 1.53 MB/core of unreserved L1, it is what the block's own
kernels leave underneath their circular buffers. `l1_headroom.py` measures it directly: pin a dummy
L1-interleaved tensor of X MB, run the workload, bisect X.

| workload | aggregate L1 still free | per core |
|---|---|---|
| **whole block** | **18.6 MB** | 169 KB of 1532 |
| trimul_start | 18.6 MB | 169 KB |
| transition_z | 31.7 MB | 288 KB |
| tri_att_start | 44.8 MB | 408 KB |
| attention_pair_bias | 135.6 MB | 1233 KB |

The pair representation at N=320, c_z=256 is **52.43 MB = 2.8x the block-wide budget**. There is no
"pin the pair tensor" strategy at 298 aa. Everything has to be a slice.

**Largest pair-shaped row block that is L1-resident** (`R x 320 x 256` bf16 = 0.164 MB per row):
**96 rows (15.7 MB)** if it must survive the whole block, **256 rows (41.9 MB)** if it only has to
live inside tri_att. W5's row-blocking leg should size against those two numbers.

Independent confirmation from the chain A/B: accumulating the ten 5.24 MB transition_z outputs in L1
instead of DRAM throws the same CB clash on the **tenth** chunk, i.e. ~47 MB fit and 52 MB did not.

## 4. Classification

**(a) Pin for the whole block.** Only the weights qualify: 9.50 MB per block-execution against an
18.6 MB budget. **But it is worthless.** Weight re-reads are 15.7 MB of the block's 4263.8 MB of
read traffic (0.37 %) = 0.04 ms at the read roof. Anyone proposing weight pinning as a lever at this
size is wrong, and this is the cheapest thing on the list to test.

**(b) Chunk so a slice is resident.** The pair representation and everything derived from it. This
is where all the traffic is (2831 MB of re-reads on three buffers) and it is the only class with a
real prize. Production already does this in two places, and the trimul already runs a channel-chunked
loop with L1-resident chunks (the 26.21 MB and 6.55 MB L1 rows in the working-set table).

**(c) Genuinely DRAM-resident.** The two 157.29 MB fused qkv projections and the three 52.43 MB
per-head q/k/v in tri_att — 314.6 MB live at once inside tri_att_start, against 44.8 MB of headroom. These cannot be made
resident whole; they can only be made *smaller* by row-blocking (b) or removed by fusing the
projection into the attention (W4/W5/W6's lane, not a placement decision).

## 5. Measured: what placement is actually worth, on a real chain

`transition_z` is the clean testbed. It is 14.2 % of the block on this card (6.43 of 45.28 ms), purely row-local
(layer_norm -> fc1(silu) -> fc2 -> multiply -> fc3), and its row-block width is free to choose.
`chain_ab.py`, one arm per process (a variant that overflows L1 fragments the allocator and poisons
every later arm in the same process — that is how the first W7 A/B lost four of five legs).

| arm | placement of the carried tensors | ms | vs production | bit-exact |
|---|---|---|---|---|
| `module` | production: L1-interleaved, DRAM output, h=32 | **6.430** | 1.000 | ref |
| `module_h16` | same, row block 16 | 6.407 | 1.004 | **yes** |
| `module_h64` | same, row block 64 | 6.409 | 1.003 | **yes** |
| `dram` | every carried tensor DRAM-interleaved | 9.267 | **0.694** | no (0.22 % rmsd, pcc 0.999945) |
| `shardb` | **block-sharded 8x8, carried across the whole chain** | 6.602 | **0.974** | no (0.40 % rmsd, pcc 0.999915) |
| `l1` | as production + L1 output accumulation | FAIL | CB clash at chunk 10 | |
| `mm` | `ttnn.linear` -> `minimal_matmul` | FAIL | CB clash | |
| `shard` | height-sharded from the top | FAIL | `layer_norm`: "Height sharded inputs are not supported" | |

Four things fall out, all of them things another leg would otherwise have to rediscover.

**L1-vs-DRAM on a carried chain is worth 1.44x, and production already banks it.** 9.267 -> 6.430 ms.
That is the honest size of the placement lever on a chain that is already well chunked.

**Sharding is not a lever. KILLED.** A block-sharded config carried across the entire chain is
**0.974x** — slightly *slower* than plain L1-interleaved, on 64 cores instead of 110. Height sharding
does not even start: ttnn 0.68's `layer_norm` rejects `HEIGHT_SHARDED` input outright, and a sharded
matmul additionally refuses `ttnn.linear(activation=...)` ("this should be placed in the program
config's fused_activation field") and constrains `out_block_w == per_core_N or out_block_h == 1`.
The brief flagged this as "cheap to test and potentially large". It is cheap and it is nothing.
W4 and W5 should not spend time on sharded memory configs; the win is L1-vs-DRAM, not
sharded-vs-interleaved.

**Row-block width is arithmetic-inert here, unlike the trimul's channel chunk.** h=16, 32 and 64 are
all `torch.equal` and all within 0.4 % of each other. So transition_z's chunk size is a free
parameter for a residency strategy — and there is nothing to win by changing it, which is a useful
negative for anyone tuning `TRANSITION_H_CHUNK_SIZE_BIG`.

**A placement change is NOT automatically bit-exact, and this is a trap.** Both non-bit-exact arms
changed only *where* tensors live — but a different memory config makes ttnn select a different
matmul program config, which changes the accumulation order. `dram` is 0.22 % rmsd, `shardb` 0.40 %,
on an output whose std is 14.86. Any leg that plans to justify a placement change with "it only
moves bytes, so it must be bit-exact" needs to check, not assert.

## 6. Why transition_z has no headroom left, and what that says about the rest

Same card, the exact shapes the chain runs (`chain_ab.py --arm roof`):

| matmul | shape | `ttnn.linear(core_grid=11x10)` | `minimal_matmul` |
|---|---|---|---|
| fc1/fc2 | [10240,256] @ [256,1024] | **73.26 TFLOP/s** (64.4 % of dense roof) | 54.76 (0.75x) |
| fc3 | [10240,1024] @ [1024,256] | **80.63 TFLOP/s** (70.9 %) | 45.04 (0.56x) |

The three matmuls therefore cost 0.733 + 0.733 + 0.666 = **2.13 ms of the phase's 6.43 ms (33 %)**,
already at 64-71 % of the dense roof, while the phase's DRAM traffic floor is 0.744 ms (12 %).
Two thirds of `transition_z` is layer_norm, the gate multiply, the slicing and the final concat.

**Cross-leg correction, please read before inheriting W6's rule.** W6 measured
`minimal_matmul` **1.66x faster** than `ttnn.linear(core_grid=...)` at [102400,256]@[256,256]. At the
two `transition_z` shapes it is **0.75x and 0.56x — slower**. The swap is shape-dependent, not a
general rule. Check your own shape.

## 7. Recompute vs re-read at this machine balance

Machine balance on this card is 113.79e12 / 403.2e9 = **282 FLOP/byte** on the read roof (not the
231 the warroom quotes; W1 made the same correction on qb1 and got 338). Recompute beats re-read
when the recompute costs less than the bytes it saves.

Going through the re-read census, **every candidate in this block is strictly dominated, and the
reason is structural, not marginal**: the block's re-read buffers are all normalisation or projection
outputs whose recompute recipe starts by reading an operand of *exactly the same shape*. The trimul's
`x_norm_in` is the clearest case — it is 52.43 MB read once per channel chunk, and recomputing it
costs a 52.43 MB read of `z` plus the layer_norm arithmetic. Identical read bytes, strictly more
compute. There is no intermediate in the Pairformer block that is small to produce and large to
store; the pair track's shapes are all `N x N x c`, so producer and product are the same size.

This is an analytic kill from the measured per-op byte table, not a timed A/B — stated plainly so
nobody quotes it as a measurement. It is falsifiable: if someone finds an intermediate whose
recompute inputs are materially smaller than the intermediate itself, the conclusion changes.

## 8. Recommended assignment, and the falsifiable predictions

**The assignment.**

1. **Weights: leave them in DRAM.** 9.50 MB would fit the 18.6 MB budget and would buy 0.04 ms. The
   budget is better spent on (2).
2. **Pair representation: never resident whole, always row-blocked.** Cap the live pair-shaped
   residency at **96 rows (15.7 MB)** for anything that must survive the block, **256 rows
   (41.9 MB)** for anything confined to tri_att.
3. **Carried intermediates inside a chunk: L1-interleaved, DRAM output.** That is what production
   already does in `transition_z` and it is worth 1.44x over the DRAM-interleaved alternative. Do
   not accumulate the per-chunk *outputs* in L1: ten 5.24 MB outputs already overflow.
4. **Do not shard.** Block-sharded carried across a chain is 0.974x, height-sharded does not compile.
5. **The qkv round trip in tri_att is the one place left where placement and traffic coincide**
   (314.6 MB live at once, 157.29 MB written then re-read then re-written as 3 x 52.43 MB by the head split). It is W5's and
   W6's, and it is the correct place to spend the 44.8 MB of tri_att-phase headroom.

**Predicted saving.** Placement alone — moving bytes between DRAM and L1 with no op removed and no
kernel written — is bounded by the memory-bound ops it can delete. On the one chain where I measured
it end to end, the whole lever was 2.84 ms on a 6.43 ms phase, and production had already taken it.
I am **not** forecasting a block-level ms from placement, because at 35.5 % DRAM utilisation the byte
count is not the critical path and a floor-delta number would be a fabricated saving. The honest
prediction is the three falsifiable claims below.

**P1 (sharding).** Any leg that carries a sharded memory config across a chain of trunk ops at 298 aa
will measure **<= 1.00x** against the same chain L1-interleaved. Falsified by any sharded chain that
beats L1-interleaved by more than measurement noise.

**P2 (the L1 budget).** Any residency scheme whose live aggregate L1 exceeds **18.6 MB** block-wide
(**44.8 MB** if confined to tri_att, **31.7 MB** to transition_z) will fail with the
`Statically allocated circular buffers ... clash with L1 buffers` throw, not degrade gracefully.
Falsified by any scheme that runs above those numbers on this grid. Corollary W5 can test directly:
a pair-shaped row block of 96 rows fits and 128 rows (21.0 MB) does not.

**P3 (weights).** Pinning the block's 9.50 MB of weights in L1 saves **< 0.1 ms** of the 45.28 ms
block. Falsified by any measured weight-pinning win above that.

**P4 (bit-exactness).** A placement-only change that alters the memory config of a matmul operand or
result will **not** be bit-exact, because ttnn re-selects the program config. Falsified by a
`torch.equal` pass on such a change.

## 9. Reproduce

```
cd ~/.coworker/wt/perfwar-sram-blocking-strategy
export TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_HOLDER=worker:perfwar-sram-blocking-strategy
export TT_MESH_GRAPH_DESC_PATH=/home/ttuser/tt-bio/env/lib/python3.12/site-packages/ttnn/\
tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
PY=~/tt-bio/env/bin/python3

$PY perf/sram_strategy/block_working_set.py --mode roofs  --n 320 --out perf/sram_strategy/roofs_qb2_card2.json
$PY perf/sram_strategy/block_working_set.py --mode bench  --n 320 --out perf/sram_strategy/block_bench_n320.json
$PY perf/sram_strategy/block_working_set.py --mode ledger --n 320 --out perf/sram_strategy/ledger_n320.json
   python3 perf/sram_strategy/analyze_ledger.py perf/sram_strategy/ledger_n320.json   # tables, no device
$PY perf/sram_strategy/l1_headroom.py --n 320 --out perf/sram_strategy/headroom_n320.json
for a in roof module module_h16 module_h64 dram l1 mm shard; do
  $PY perf/sram_strategy/chain_ab.py --arm $a --n 320 --out perf/sram_strategy/chain/$a.json \
      --dump perf/sram_strategy/chain/out_$a.pt; done
$PY perf/sram_strategy/chain_ab.py --arm shardb --grid 8x8 --n 320 --out perf/sram_strategy/chain/shardb_8x8.json
```

No production code is changed on this branch beyond three benchmark-only gates in
`tt_bio/tenstorrent.py` (`_TRIMUL_CHUNK_OVERRIDE`, `_TRIMUL_L1_NORM`, `_TRIMUL_ONE_CONCAT`), all
defaulting to the current behaviour. Nothing here is proposed for merge.
