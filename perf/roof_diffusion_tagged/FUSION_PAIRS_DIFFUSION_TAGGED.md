# The diffusion ranking, with the model functions in it

`perf/roof_orchestrator/FUSION_PAIRS_DIFFUSION.md` ranked the diffusion path's single-use DRAM
intermediates from a raw `ttnn.graph` node list, so its rows are named `linear -> multiply_` and
stop. This is the same ranking from a buffer-keyed, call-site-tagged trace, so every row names the
line of `tt_bio/tenstorrent.py` that issued it.

Instrument: `perf/b2z2_byte_floor/trace_block.py --layer difftx`, the tracer that already does this
for the Pairformer, extended rather than replaced. Ranking: `perf/roof_orchestrator/fusion_pairs.py`
unchanged (blob `fff4b3e1a0f8255b44b39d91baa40656b69e3f2f`, from `wk/roof-orchestrator` 6fa7fd28e),
called with the trace path.

**Nothing here is a performance number.** The tracer wraps all 583 `ttnn` operations and walks the
Python stack once per recorded call. Bytes only.

## The capture is stale by four and a half hours, and that is the headline

`graph_512_all.json.gz` was committed at `65eac7c9a`, 2026-09-11 15:04 UTC. `fc7fed56f`, 2026-09-11
19:43 UTC, flipped two defaults:

    BOLTZ2_TOKEN_DIT_SDPA    False -> True     token DiT runs the fused SDPA
    TT_BIO_ATOM_AXIS_BUCKET  False -> True     atom window count buckets to a multiple of 448

So the published table's **#1 row, `softmax -> matmul` at 80.53 GB/fold, no longer exists on
`main`**. The unfused chain that allocated the 8.39 MB probability matrix is gone; the layer now
issues one `ttnn.transformer.scaled_dot_product_attention` and that allocation is never made. And
the atom layer's shape moved from `1x224x32x128` to `1x140x32x128`, which is 37.5 % fewer windows.

Two arms were therefore captured. CONTROL runs with both flags forced off, which is the capture's
own configuration and reproduces its shapes exactly. DEFAULT runs today's `main`.

## INSTRUMENT-CONTROL: the fusable prize agrees to the MB, in both shapes

| token DiT `1x512x768` | graph capture | tagged CONTROL | tagged DEFAULT |
|---|---|---|---|
| single-use allocations | 18 | 22 | 20 |
| round trip | 48.8 MB | 56.6 MB | 37.7 MB |
| **fusable (epilogue/prologue)** | **38.8 MB** | **38.8 MB** | 22.0 MB |
| blocked by tile reorder | 10.0 MB | 17.8 MB | 15.7 MB |

| atom transformer | graph capture `1x224x32x128` | tagged CONTROL `1x224x32x128` | tagged DEFAULT `1x140x32x128` |
|---|---|---|---|
| single-use allocations | 16 | 24 | 24 |
| round trip | 77.1 MB | 113.0 MB | 78.5 MB |
| **fusable (epilogue/prologue)** | **36.7 MB** | **36.7 MB** | 22.9 MB |
| blocked by tile reorder | 36.7 MB | 68.3 MB | 49.0 MB |

**The fusable column, which is what the table exists to rank, agrees exactly: 38.8 vs 38.8 MB and
36.7 vs 36.7 MB.** At the fold that is 230.27 GB/fold against the published 230.3, a 0.01 %
disagreement between two instruments that share no counting code. The tolerance stated before the
run was 20 %.

The disagreement is entirely in the REORDER column, and it is one-directional: the trace counts
more. `fusion_pairs_diff.py` drops a pair with a name from
`{reshape, unsqueeze, squeeze, deallocate, Tensor.__getitem__, reallocate}` on either end. Two of
those six allocate on this path. `Tensor.__getitem__` is `ttnn.slice`, which is not a view (memory
`tt-bio-ttnn-slice-not-a-view-and-allocation-order-sensitivity`), and the `reshape` at 7654 merges
the head and channel axes of a tiled tensor, which is a copy. The four pairs that rule removes from
the token DiT are, in the trace, four distinct device addresses each written once by one program and
read once by another:

    matmul @7647 -> slice @7652     2.1 MB    o = o[:, :, :, :self.head_dim]
    slice @7652  -> permute @7653   2.1 MB
    permute @7653 -> reshape @7654  2.1 MB
    reshape @7654 -> permute @7655  1.6 MB

The ledger already drops a view whose output buffer IS one of its inputs, so anything left has a
fresh allocation and a real write. **Where the two instruments disagree, the trace is the one
holding the device address.** The published REORDER figure is a floor, not the number.

The atom layer's larger gap has a second cause on top of that one: the capture ran with
`_atom_shift_gather` OFF, so its gather is `matmul -> permute` (14.68 MB); the trace ran with it on,
so the gather is `slice -> concat -> pad` (19.1 MB across 6 allocations). Different code, both
REORDER, and the fusable column is unaffected.

### One defect found and fixed in the shared tracer

`install()` walked a hand-written module list (`ttnn`, `ttnn.experimental`, one level of
`ttnn.operations`) and so never wrapped `ttnn.transformer`, which is where
`scaled_dot_product_attention` lives. Any trace of a stack that reaches the fused SDPA recorded its
q/k/v as written-and-never-read and dropped the op's own 8.39 MB mask read entirely. It now walks
the tree: 569 -> 583 operations wrapped. **The committed Pairformer traces in
`perf/b2z2_byte_floor/out/` were taken with the old list**, so any of their rows that touch a fused
SDPA are short by that op's operands. Re-taking them is not this row's scope; it is flagged.

## TABLE

Four rankings, `perf/roof_diffusion_tagged/out/{control,default}/pairs_difftx_*.json`, printed by
the command in `RUN.md`. Fold columns use the call counts of record from
`perf/bioir_roofline/flops_bytes_512.json`: token DiT 4800/fold, atom transformer 1200/fold.

### Token DiT, DEFAULT arm, 4800 calls/fold

| removable MB | GB/fold | n | class | producer -> consumer, by line |
|---|---|---|---|---|
| 7.9 | 37.8 | 3 | EPILOGUE | `linear` -> `multiply_` (8981/8993, 8995/9002, 9014/9021) |
| 6.3 | 30.2 | 3 | REORDER | `nlp_create_qkv_heads` -> `scaled_dot_product_attention` |
| 4.7 | 22.6 | 3 | EPILOGUE | `linear` -> `multiply` (7714/7722, 7724/9103, 9091/9103) |
| 3.1 | 15.1 | 2 | EPILOGUE | `linear @s_terms` -> `multiply_` (8898/8940, AdaLN scale) |
| 3.1 | 15.1 | 2 | EPILOGUE | `linear @s_terms` -> `add_` (8906/8941, AdaLN bias) |
| 2.1 | 10.1 | 1 | REORDER | `scaled_dot_product_attention` -> `slice` (7652) |
| 2.1 | 10.1 | 1 | REORDER | `slice` -> `permute` (7653) |
| 2.1 | 10.1 | 1 | REORDER | `permute` -> `reshape` (7654) |
| 1.6 | 7.6 | 1 | REORDER | `reshape` -> `permute` (7655) |
| 1.6 | 7.6 | 1 | REORDER | `permute` -> `multiply` (7655/7722) |
| 1.6 | 7.6 | 1 | PROLOGUE | `multiply` -> `linear` (7722/7724) |
| 1.6 | 7.6 | 1 | EPILOGUE | `multiply` -> `add` (9103/9110) |

### Atom transformer, DEFAULT arm, 1200 calls/fold

| removable MB | GB/fold | n | class | producer -> consumer, by line |
|---|---|---|---|---|
| 11.5 | 13.8 | 3 | EPILOGUE | `linear` -> `multiply_` (8981/8993, 8995/9002, 9014/9021) |
| 9.2 | 11.0 | 1 | REORDER | `pad @_atom_shift_gather` (800) -> `linear` (7683) |
| 9.2 | 11.0 | 1 | REORDER | `reshape` -> `scaled_dot_product_attention` |
| 8.5 | 10.1 | 4 | REORDER | `slice` -> `concat @_atom_shift_gather` (798) |
| 8.5 | 10.1 | 1 | REORDER | `concat` -> `pad @_atom_shift_gather` (798/800) |
| 6.9 | 8.3 | 3 | EPILOGUE | `linear` -> `multiply` (7714/7722, 7724/9103, 9091/9103) |
| 4.6 | 5.5 | 2 | REORDER | `to_memory_config @s_terms` (8917) -> `multiply_` (8940) |
| 4.6 | 5.5 | 2 | REORDER | `to_memory_config @s_terms` (8918) -> `add_` (8941) |
| 2.4 | 2.8 | 1 | UNKNOWN | `pad` -> `to_layout @_atom_shift_gather` (794/796) |
| 2.3 | 2.8 | 1 | REORDER | `linear` (7675) -> `pad` (7695) |
| 2.3 | 2.8 | 1 | REORDER | `slice` -> `scaled_dot_product_attention` |
| 2.3 | 2.8 | 1 | PROLOGUE | `multiply` -> `linear` (7722/7724) |
| 2.3 | 2.8 | 1 | EPILOGUE | `multiply` -> `add` (9103/9110) |
| 2.1 | 5.1 | 2 | UNKNOWN | `slice` -> `to_layout` (791/792), `to_layout` -> `pad` (792/794) |

### The whole diffusion path

```
                                  published    CONTROL     DEFAULT
  single-use DRAM round trip      322.1       407.33      275.45  GB/fold
    fusable                       230.3       230.27      133.22  GB/fold
    blocked by tile reorder        91.9       167.51      134.32  GB/fold
    unclassified                     -          9.56        7.90  GB/fold
```

**133.22 GB/fold is the fusable prize that is actually still there**, 3.9 % of the fold's 3.405 TB,
down from 230.3. The 97.1 GB/fold difference is not a lever anyone still has to build: 80.5 of it is
the token DiT's probability matrix, deleted by the SDPA that shipped as a default the same evening
the capture was taken, and the rest is the atom-axis bucket shrinking the atom layer's windows.

## SITES

### 1. `softmax -> matmul`, 80.53 GB/fold: already deleted, and the SDPA question answered

**The site.** `tt_bio/tenstorrent.py:7636-7647`, the `else` branch of `AttentionPairBias.__call__`,
reached from `DiffusionTransformerLayer.__call__:9086` via `self.attn_pair_bias(b, z)`:

    kt = ttnn.transpose(k, -2, -1)                      # 7636
    logits = batched_matmul(q, kt)                      # 7637
    logits = ttnn.add_(logits, z)                       # 7640
    logits = ttnn.multiply_(logits, head_dim**-0.5)     # 7641
    probs  = ttnn.softmax(logits, dim=-1)               # 7644
    o      = batched_matmul(probs, v)                   # 7647

The control-arm trace names the allocation exactly: `softmax @tenstorrent.py:7644 -> matmul
@tenstorrent.py:2432:batched_matmul`, shape `1x16x512x512`, 8.389 MB written and 8.389 MB read back,
16.777 MB removable, which is the published 16.78 to three figures. 512 x 512 x 16 in bf16, the
probability matrix for all sixteen heads, as the published table inferred.

**Which attention runs today: `ttnn.transformer.scaled_dot_product_attention`.** Not a matmul pair.
`tenstorrent.py:7583` and `7614-7626` take the `token_dit_sdpa` branch whenever `_B2_TOKEN_DIT_SDPA` is set
(default True since fc7fed56f), `z` is not None and no `seq_mask` is passed, which is every Boltz-2
token DiT call. The default-arm trace shows one program consuming q, k, v at 1.049 MB each plus the
8.389 MB bias and writing 1.049 MB out. **No probability matrix is allocated.** 80.53 GB/fold is
already off the ledger; it is not a build queue item.

**Is `tt_bio/sdpa_generic.py` reachable for this shape? Yes, and it is worth nothing on this row.**
Its `plan()` accepts the token DiT geometry (B=1, 16 heads, Sq=Sk=512, DH=64) at 19 of the 20
(q_chunk, k_chunk) pairs that divide 512, with `use_padded_mask` false everywhere, splitting
(batch 1, heads 16, q-chunks 8) onto 128 of the p150a's 130 cores; at q_chunk=k_chunk=64 its CBs are
114.0 KB of the 1536 KB per core. But `sdpa_generic` is a transcription of the same
`sdpa_program_factory.cpp` the native op already runs, bit-exact and same-speed by construction, so
routing the token DiT through it deletes no byte. What it buys elsewhere is K2's ability to change
the work split and front the mask in a CB, and neither removes a DRAM round trip here, because a
circular buffer does not survive a program launch.

**What is actually left at this site is the mask read.** The bias is `1x16x512x512`, 8.389 MB, cut
once per fold by `AtomDiffusion._hoist_layer_bias` (`tenstorrent.py:10747`) and therefore constant across all 200 sampling
steps, and the fused SDPA reads all of it on every one of the 4800 calls: **40.3 GB/fold from a
tensor that never changes**, 10.2 % of the layer's per-call DRAM read. That is the largest single
remaining DRAM item in the token DiT path and it is a residency question, not a fusion one.

### 2. The gating multiply after a projection: 8 distinct call sites, 112.6 GB/fold

The published table called this eleven sites across `linear -> multiply_` and its variants, and
measured the family at 125.8 GB/fold. The control arm measures 125.83. On today's default it is
**112.62 GB/fold** (90.60 token DiT + 22.02 atom), the difference being the atom-axis bucket.

There are **8 distinct source lines**, not eleven; the larger count came from counting op-code rows
times their multiplicity. Every one is a `ttnn.linear` whose output is consumed by exactly one
binary elementwise op over the same shape:

| # | producer | consumer | role | token DiT | atom |
|---|---|---|---|---|---|
| 1 | `linear` 8981 `a_swish` | `multiply_` 8993, SILU on operand A | SwiGLU gate | 3.146 MB | 4.588 MB |
| 2 | `linear` 8995 `a_to_b` | `multiply_` 9002 | transition a_to_b gate | 3.146 MB | 4.588 MB |
| 3 | `linear` 9014 `b_to_a` | `multiply_` 9021, SIGMOID on operand A | transition output gate | 1.573 MB | 2.294 MB |
| 4 | `linear` 8898 `s_scale` | `multiply_` 8940, SIGMOID on operand B | **AdaLN scale** | 1.573 MB x2 | memoised |
| 5 | `linear` 8906 `s_bias` | `add_` 8941 | **AdaLN bias** | 1.573 MB x2 | memoised |
| 6 | `linear` 9091 `s_o` (activation="sigmoid") | `multiply` 9103 | DiT output gate, s side | 1.573 MB | 2.294 MB |
| 7 | `linear` 7724 attn out projection | `multiply` 9103 | DiT output gate, b side | 1.573 MB | 2.294 MB |
| 8 | `linear` 7714 `g` | `multiply` 7722, SIGMOID on operand B | attention output gate | 1.573 MB | 2.294 MB |

**Two of the eight are the AdaLN pair (4 and 5), six are gates.** Sites 4 and 5 fire TWICE per token
DiT layer call, because the layer runs `AdaLN` once itself and once inside
`ConditionedTransitionBlock`; that is the published `n=5` for `linear -> multiply_` and `n=2` for
`linear -> add_`. At the atom level they do not appear at all: `_B2_ADALN_S_MEMO` caches `s_scale`
and `s_bias` for the whole rollout (`tenstorrent.py:8874-8924`), so what the atom trace shows in
their place is `to_memory_config @s_terms -> multiply_/add_` (8917/8918), 11.0 GB/fold of REORDER, which is the
memo's own DRAM parking cost.

**Do they share one producer kernel? Six of the eight do, exactly.** Sites 1, 2, 3, 6, 7 and 8 are
all `ttnn.linear(..., compute_kernel_config=self.compute_kernel_config, core_grid=CORE_GRID_MAIN)`
with bf16 in and out and a DRAM output. Sites 4 and 5 differ in one argument: they pass an explicit
`memory_config` and no `core_grid`, because a core grid there was measured to cost accuracy
(`tenstorrent.py:8903`). So the producer side is one kernel configuration with one two-line
exception.

The consumer side is more uniform still. All eight are a binary elementwise op over two full-size
operands of identical shape, with an optional unary activation applied to one of them
(SILU, SIGMOID, or none). `ttnn.linear` already fuses a UNARY epilogue and site 6 uses it
(`activation="sigmoid"`). **What is missing is a matmul epilogue that takes a SECOND full-size
operand**: multiply-by-a-tensor rather than multiply-by-a-function. One such primitive serves all
eight sites in both layer shapes, which is why this row is worth more than any single one of them.

It is the same defect as the Pairformer's `roof-fuse-gate-epilogue`, at a different site, with a
200x rollout multiplier on it.

## PREDICTED vs MEASURED

Written in `PREDICTION.md` before the capture ran.

- **"The `softmax -> matmul` row is ABSENT from a default-arm trace."** Correct, and for the
  predicted reason.
- **"The control arm agrees with the published per-layer figures within 20 %."** Correct on the
  fusable column and better than predicted: exact, 38.8/38.8 and 36.7/36.7 MB. Not correct on the
  round-trip total, which is 16 % out in the token DiT and 47 % out in the atom layer.
- **"If they disagree, the trace reads LOWER."** Wrong, and it is the useful miss. The trace reads
  HIGHER, in REORDER only, because `itemize` excludes aliasing ops by NAME and several of those
  names allocate on this path. The reasoning that produced the wrong sign, that the trace can only
  miss an allocation the wrapper never saw, was right in form and then found a real instance: the
  wrapper was missing `ttnn.transformer` entirely.
- **"DEFAULT lands at 235-250 GB/fold."** Missed. Measured 275.45 GB/fold of round trip, 10 % above
  the band, because the band was computed by subtracting 80.53 from the published 322.1 and the
  trace's own control total is 407.33, not 322.1. Direction correct, magnitude anchored on the wrong
  instrument's total. The fusable prize, the number the band should have been stated against, is
  133.22 GB/fold.
