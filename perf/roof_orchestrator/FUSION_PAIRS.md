# Which programs are worth fusing, ranked by the bytes that vanish

`perf/b2z2_byte_floor/CENSUS.md` established that **58.8 % of one Pairformer block's DRAM traffic is
single-use intermediates** — allocations written by one program and read by exactly one other — and
then stopped, because deleting them is program fusion and the Pairformer megakernel lost 2.9 %. That
verdict is about fusing *everything*. It says nothing about which individual pairs are worth fusing,
and nothing in the archive ranks them. This does.

`fusion_pairs.py`, no device. Input is the committed buffer-keyed trace
`perf/b2z2_byte_floor/out/trace_512_wh_c10.json.gz` — one `PairformerLayer`, 512 tokens, Wormhole
card 10, 417 programs, every operand keyed by device allocation.

## Calibration

The rules are not re-derived. `split_io`, `NON_OPS` and `VIEWS` are imported out of
`perf/b2z2_byte_floor/census.py` with its trailing `main()` stripped, so there is one copy of the
arity / gated / in-place rule in the repo and this table cannot drift from the census it extends.

Two known answers, both hit:

| check | census | here |
|---|---|---|
| block DRAM traffic | 7.417 GB | **7416.9 MB** |
| single-use set, under the census's own owner-based definition | 2181.0 MB each way | **2187.6 MB**, 0.3 % apart |

The ranking below uses a **stricter** definition than the census's: exactly one write *event* and
exactly one read *event*, not one write owner and one read owner. That is 75 allocations and
**1650.7 MB each way**, because it excludes buffers a single owner reads more than once. The strict
set is the right one for a producer→consumer pair table; the looser one is the right one for the
census's redundancy question. Both are reported so the two numbers are never confused.

## Feasibility is a separate column, and it is the honest half

A byte count says what fusion would save. It does not say the fusion is available. Each pair is
classified by whether the consumer can take the producer's tiles in the order the producer makes
them. `ttnn.generic_op` is four different kernels in tt-bio, so it is resolved by the file that
issues the call — the same key `census.py:GENERIC_ARITY` uses.

- **EPILOGUE** — consumer is elementwise over the same shape. Its math runs in DST before the
  producer's tile is ever packed out.
- **PROLOGUE** — producer is elementwise, consumer is a matmul or attention reading it as an
  operand. The consumer's reader does the producer's math on the tile it just loaded.
- **ATTN-QKV** — a q/k/v projection feeding SDPA. Part free, part recompute; see below.
- **REORDER** — a transpose, channel move, concat or axis-changing reduction sits on one end. The
  tile orders do not match and fusion needs a materialised intermediate anyway. **Not available.**

## The table

```
  block DRAM traffic (this trace)         7416.9 MB
  single-use intermediates               75 allocations, 3301.4 MB of read+write
    fusable as epilogue/prologue           967.8 MB   = 13.05 % of the block
    qkv projection into attention          805.3 MB   = 10.86 % of the block, part free part recompute
    blocked by tile reorder               1523.3 MB   = 20.54 % of the block
    unclassified                             5.0 MB
```

| removable MB | n | class | producer → consumer |
|---|---|---|---|
| **805.3** | 6 | ATTN-QKV | `generic_minimal_matmul` (fused qkv in-projection) → `sdpa` (fused SDPA) |
| 268.4 | 2 | REORDER | `reblock_permute_gated` → `transpose` |
| 268.4 | 2 | REORDER | `matmul` → `reblock_permute_back` |
| 268.4 | 2 | REORDER | `reblock_permute_back` → `layer_norm` |
| **268.4** | 2 | PROLOGUE | `layer_norm` → `_pair_proj_linear` (trimul out-projection) |
| **268.4** | 2 | EPILOGUE | `_pair_proj_linear` → `multiply_` (trimul output gate) |
| **268.4** | 2 | EPILOGUE | `generic_minimal_matmul` → `gate_and_project` `multiply_` |
| 134.2 | 1 | REORDER | `transpose` → `matmul` |
| 134.2 | 1 | REORDER | `reblock_permute_gated` → `matmul` |
| 134.2 | 1 | REORDER | `generic_minimal_matmul` → `_pair_transpose_impl` |
| 134.2 | 32 | REORDER | `swiglu` `linear` → `concat` |
| 134.2 | 1 | REORDER | `concat` → `add_` |
| **134.2** | 1 | PROLOGUE | `_l1_layer_norm` → `_narrow_proj_linear` |
| 33.6 | 1 | REORDER | `_narrow_proj_linear` → `permute` |
| 16.8 | 1 | PROLOGUE | `softmax` → `batched_matmul` |
| 8.4 | 2 | PROLOGUE | `add` → `sdpa` |

Full output and the per-allocation rows are in `fusion_pairs.json`.

## The four targets, named

**1. The qkv projection into the SDPA — 805.3 MB, 10.86 % of the block.** Six allocations: q, k and
v for each of the two triangle attentions, each 67.1 MB, written to DRAM by the fused in-projection
and read straight back by the fused SDPA. Both ends are already `generic_op`, so this is one custom
kernel calling another, not a ttnn composition. **It is not uniformly free.** Q streams once and its
projection folds into the SDPA's reader for nothing. K and V are re-read once per query block, so
folding their projections in trades DRAM traffic for recomputed arithmetic — the classic
recompute-vs-store trade, and the fold has arithmetic to spare (206.706 TFLOP is 13.9 % of the
85.96 TFLOP/s roof, `RECONCILIATION.md`). Q alone is 268.4 MB, 3.62 % of the block, and is the part
to build first.

**2. The trimul output chain — 536.8 MB in two links, 7.24 % of the block.** `layer_norm` →
`_pair_proj_linear` → `multiply_` is three programs with two 67.1 MB round trips between them, and
both links classify fusable: a prologue then an epilogue. One program instead of three deletes both.
This is the same shape as the precedent the campaign already banked — reading the layer-normed pair
tensor once instead of twice was 1.086x and bit-exact.

**3. `generic_minimal_matmul` → `gate_and_project` `multiply_` — 268.4 MB, 3.62 %.** An elementwise
multiply immediately after a matmul that is *already* a custom fused kernel. Structurally the
cheapest item on the list: it adds a multiply to an existing kernel's pack stage and needs no new
descriptor.

**4. `_l1_layer_norm` → `_narrow_proj_linear` — 134.2 MB, 1.81 %.** Same prologue shape as 2.

## The negative result, which is worth as much

**1523.3 MB — 20.5 % of the block — is single-use and cannot be fused.** Every one of those pairs
has a transpose, a channel move, a concat or an axis-changing reduction on one end, so the producer
does not emit tiles in the order the consumer wants them. `reblock_permute_gated`,
`reblock_permute_back` and `_transform_chunk_gated` carry 939.5 MB of it between them. That traffic
is only reachable by removing the reorder, not by fusing across it — which is exactly the lever
`k10-p1-trimul-critpath` already attacked from the other side, finding the reorder's reader
bank-serialised and worth 1.6652x on the op.

## What this does NOT say

**Bytes are not seconds.** The block runs at 36.7 % of the 429.9 GB/s stream roof and
`k10-instrument` measured its compute cluster blocked on input 57.9 % with the reader on DRAM only
22.7 %, so this block is not bandwidth-bound and removing a byte does not return its time at the
roof rate. Everything here is a byte prize and it is stated as one. A device A/B prices the seconds
and that is the next step, not an inference from this table.

**This is a Wormhole trace.** The Blackhole census of the same block is 8.049 GB against this
trace's 7.417 GB, 1039 MB of the difference on the read side (537 MB of it the `reblock_permute_gated`
over-read). The percentages above move on Blackhole and the table should be re-taken there before
anything is sized off it.

**The trace predates the tip.** Its `owner` line numbers no longer resolve against `f072ae02f`,
which is why the table is keyed on call-chain function names instead. Levers have landed since that
change which tensors are L1-resident, and an L1-resident intermediate is not in this table at all.
