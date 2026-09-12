# The PairformerLayer block in tile passes — Boltz-2, 512 aa, c_z = 128, bf16

One `PairformerLayer` with `transform_s=True`, the shape the Boltz-2 trunk issues 280 times a fold.
A tile is 32x32 = 1024 datums = 2048 B in bf16. A **tile pass** is one tile crossing an **op
boundary**: written by the op that produced it, read by every op that consumes it. That is the
traffic a fused kernel deletes, and it is the only traffic a fusion can delete.

Derived by static reading of the executed branch of `tt_bio/tenstorrent.py` at `b82a5f03`, with
every default flag resolved to the value the fold actually runs (`_FAST_MODE=False`,
`_TRIMUL_MASK_AFTER_MOVE=True`, `_TRIMUL_INPROJ_ROWBLOCK=False`, `_UNFUSED_SILU=False`,
`_FP32_SOFTMAX=False`, `fused_tail` declines at (4,4), tri-att takes the fused bf16 SDPA,
AttentionPairBias takes the UNFUSED attention on purpose). Confirmed on device where it is cheap
to confirm: `hidden_z = [128, 512]`, `hidden_s = [384, 1536]`, `z = [1,512,512,128]`,
`s = [1,512,384]`, grid 13x10 on pc's p150a, 11x10 on the cell.

## The block

| stage | ttnn ops | tile reads | tile writes | **tile passes** | bytes |
|---|---|---|---|---|---|
| `tri_mul_out` (ending=False) | 14 | 524,656 | 491,520 | **1,016,176** | 2.081 GB |
| `tri_mul_in` (ending=True) | 15 | 524,912 | 491,776 | **1,016,688** | 2.082 GB |
| `tri_att_start` | 12 | 369,772 | 272,384 | **642,156** | 1.315 GB |
| `tri_att_end` | 14 | 435,308 | 337,920 | **773,228** | 1.584 GB |
| `transition_z` | 193 | 563,200 | 524,288 | **1,087,488** | 2.227 GB |
| `pre_norm_s` + `AttentionPairBias` | 21 | 103,172 | 69,248 | **172,420** | 0.353 GB |
| `transition_s` | 6 | 3,648 | 3,144 | **6,792** | 0.014 GB |
| 5 residual `add_` on z | 5 | 327,680 | 163,840 | **491,520** | 1.007 GB |
| 2 residual `add_` on s | 2 | 768 | 384 | **1,152** | 0.002 GB |
| **block** | **282** | **2,853,116** | **2,354,504** | **5,207,628** | **10.665 GB** |

## The number that decides the megakernel

The cell's block is **36.344 ms** (qb2 card 1, p300c, 11x10 = 110 cores, trace replay). Wave 1
measured its math thread blocked **18.3366 ms on input tiles** at **71.3 ns a tile**, i.e. about
**257,000 tile arrivals per core**, about **28.3 M tiles device-wide**.

This ledger says the whole block only moves **5.21 M tiles across op boundaries**.

**So 5.4x more tiles reach the math thread than ever cross an op boundary. About 81 % of what the
block waits for is a matmul re-streaming an operand out of L1 inside a single op** — the triangle
multiplication's own matmul reads 524,288 tiles at the compute boundary against 65,536 at the op
boundary, an 8x amplification at `per_core_M = per_core_N = 2`, and the block issues 113
matmul/linear calls.

Priced at 71.3 ns/tile over 110 cores, deleting **every** op boundary in the block is worth
**3.38 ms of 36.344 ms = 1.10x**. That is the ceiling of the fusion route, and it lands on top of
wave 1's independently measured 1.09x for the trimul chain — two different derivations, one
answer. A megakernel is worth roughly a tenth, not a half.

The corollary is where the seconds actually are: **matmul operand re-streaming, not op boundaries.**

## Fusion budget, by site, largest first

| adjacency | dead intermediate | passes deleted |
|---|---|---|
| `transition_z` `fc1`,`fc2` -> `multiply_` | x_1, x_2 (4,096 t/chunk x 32) | **524,288** |
| `transition_z` `multiply_` -> `fc3` | product | **262,144** |
| trimul in-proj -> the two gated moves | `gp_in_fused`, 131,072 t | **262,144** (built as `TRIMUL_INPROJ_ROWBLOCK`, measured +0.907 ms, OFF) |
| tri-att qkv projection -> SDPA | q, k, v | **196,608** |
| `transition_z` `layer_norm` -> `fc1`,`fc2` | x_norm | **98,304** |
| tri-att `layer_norm` -> its three projections | normed x, read 3x | **98,304** |
| tri-att gate `multiply_` + `out_proj` | o, g | **163,840** |
| APB `layer_norm` z -> bias projection -> permute | z_norm, zb | **81,920** |
| `transition_z` `chunk` -> LN, swiglu -> `concat` | 32 row blocks, twice | **131,072** |
| trimul out-proj pair -> gated `multiply_` | p_out, g_out | **131,072** (this is `fused_tail`; it declines at (4,4) and was measured a loss) |
| trimul matmul -> move-back -> `norm_out` | x_chunk | **131,072** |
| trimul gated move -> `transpose` (both ends) | b_chunk' / a_chunk | **131,072** |

## The maximal subgraph that can stay resident

**`Transition` — `layer_norm` -> `fc1`+silu -> `fc2` -> `multiply` -> `fc3` — is the whole module,
and it fits.** Per core on a 110-core grid, for one 16-row chunk: x and x_norm 8 tiles each, the
two hidden halves and their product 32 tiles each, output 8 tiles = 120 tiles = 245 KB live, plus
one resident copy of `fc1`, `fc2` (64 tiles each) and `fc3` (64 tiles) = 393 KB. **638 KB inside
Wormhole's 1,499,136 B**, and further inside Blackhole's. Nothing else in the block is resident at
this size: the trimul's fused in-projection is a single 256 MB intermediate, and triangle attention
contracts over the full 512-token axis.

`transition_z` is also the largest single stage in the ledger (1,087,488 passes, 20.9 % of the
block) and the one wave 1 recommended after it declined the trimul
(`state/b2z-pairformer-megakernel.md`: *"move the megakernel budget to Transition ... simpler chain,
unowned"*).

## Counting conventions, so this table can be checked

* A `ttnn.reshape` / `unsqueeze` / `squeeze` is a metadata view and costs 0. A `ttnn.slice` and a
  `ttnn.concat` are copies and are counted.
* An in-place op (`multiply_`, `add_`) reads its destination and rewrites it.
* Tile counts include tile padding, which is not a rounding detail here: triangle attention's bias
  projection writes `[512,512,4]` as 8,192 tiles of which **7/8 is padding**, and the APB's
  `[512,512,16]` write is half padding.
* The column is op-boundary traffic. It does NOT include an operand re-streamed from L1 inside one
  op, which is the larger term — see above. Both are lower bounds on what the unpacker moves.
