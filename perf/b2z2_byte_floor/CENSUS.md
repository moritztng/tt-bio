# The Pairformer block's 8.0493 GB, by site, with an owner and a redundancy column

One `PairformerLayer` at 512 tokens, traced operand by operand and keyed by device ALLOCATION, not
by tensor id and not by shape. WH card 10, `perf/b2z2_byte_floor/trace_block.py`, analysed by
`census.py`. The Blackhole capture the campaign's 8.0493 GB comes from is the comparison, not the
input: the two agree op for op (41 layer norms, 113 matmuls, 16 `generic_op`s in both).

|  | this trace (WH) | BH census |
|---|---|---|
| DRAM read | 3,672.0 MB | 4,710.8 MB |
| DRAM write | 3,744.9 MB | 3,338.5 MB |
| DRAM total | **7.417 GB** | 8.049 GB |
| L1 read / write | 943.5 / 1,347.3 MB | 1,582.6 / 1,515.1 MB |

The read side is 1,039 MB below the BH census and 537 MB of that is the
`reblock_permute_gated` over-read `b2z2-byte-axis-reopened` already found and this trace applies
(the kernel's reader walks two of its input's four channel slices). The write side is 406 MB above,
because a buffer-keyed ledger sees the fused qkv kernel's three output allocations where a
shape-keyed one sees one `OUTPUT_0`. Neither gap is the subject here; both are stated so the
denominators are not silently mixed.

## What a byte read twice is worth

**416.3 MB — 5.17 % of the campaign's 8.0493 GB, 5.61 % of this trace's own 7.417 GB.**

Ranked by removable bytes. "Removable" means two programs read one allocation with no write in
between, so one read would have served both.

| rank | site | removable | % of 8.0493 GB | shareable? |
|---|---|---|---|---|
| 1 | **triangle attention's normed pair tensor, read three times** — `_pair_proj_linear` (bias, 128->32), `qkv_heads` (128->384), `gate_proj` (128->128) each read the whole 67.1 MB `x_norm`. x2 attentions | **268.4 MB** | 3.33 % | yes, if the three projections share one pass over `x` |
| 2 | **triangle multiplication's normed pair tensor, read twice** — the fused four-way input projection (a, b, gate_a, gate_b) and the output gate `g_out` both read the whole 67.1 MB `x_norm_in`. x2 trimuls | **134.2 MB** | 1.67 % | yes, `g_out` is a fifth slice of a weight that is already a concatenation |
| 3 | transition `swiglu` weights, re-read once per row chunk | 12.4 MB | 0.15 % | only by unchunking; 0.1 MB weights x 32 chunks |
| 4 | four small pair-mask and bias tensors shared across sub-units | 1.5 MB | 0.02 % | already shared |

Everything else in the block is read once. **The byte axis's redundancy surface is 5.2 %, and 96.7 %
of it is one defect shape: a narrow projection that re-reads a pair tensor a wide projection has
just finished reading.** That is the same defect `PairWeightedAveraging.proj_z` had one level up
(`wk/b2z2-msa-layer-census@dfb8b9e08`, 536.9 MB of reads for one tensor's worth), and it is why this
row went looking.

## The other 94.8 %, and what it is

**58.8 % of the block's traffic is single-use DRAM intermediates**: 63 allocations written by one
program and read by exactly one other, 2,181.0 MB each way. Those bytes are not redundant. They are
materialised because two programs are two programs, and deleting them is program fusion, not
read-sharing. The campaign has already priced that lever and it did not pay: the Pairformer
megakernel lost 2.9 %.

Largest of them, so the next row does not have to re-derive the list:

| n | shape | each | what it is |
|---|---|---|---|
| 2 | `1x512x512x512` | 268.4 MB | the trimul's fused four-way input projection |
| 8 | `1x512x512x128` | 67.1 MB | normed pair tensors and residual updates |
| 8 | `512x4x512x32` | 67.1 MB | q, k, v and the attention output, head-major |
| 6 | `1x128x512x512` | 67.1 MB | the trimul's channel-moved operands and its product |
| 32 | `1x16x512x128` | 2.1 MB | the transition's row chunks |

## Ownership

Every removable byte in the table is owned by two call sites in `tt_bio/tenstorrent.py`:
`_pair_proj_linear` (line 3848) and `mm_generic.generic_minimal_matmul` (line 359), reading the
tensor a `ttnn.layer_norm` on the line above just wrote. No other pair of call sites in the block
shares a read of anything larger than 0.5 MB.
