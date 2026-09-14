# The same ranking for the diffusion path, where 200 steps multiply every byte

`FUSION_PAIRS.md` ranked one Pairformer block. The diffusion path is the fold's other half — 1.320
TB of the 3.405 TB at the 23.841 s baseline — and 200 sampling steps mean a byte inside one layer
is paid 4800 times. This is the same ranking for the two diffusion layer shapes.

`fusion_pairs_diff.py`, no device. Input is the committed capture
`perf/b2x_difflayer/graph_512_all.json.gz`.

## The rules, and one new one

`itemize()` is imported from `perf/b2x_difflayer/itemize.py` — the counter
`b2x-diffusion-layer-bytes` built and `perf/b2x-baseline-attrib/roof_table.py` already reads rather
than vendoring a copy. `classify()` is imported from `fusion_pairs.py`, so the two tables cannot
disagree about what EPILOGUE means.

The new rule, and it moved the answer by 24 %: **`itemize` records a buffer against a view op that
aliases rather than moves**, so a `reshape` or a `Tensor.__getitem__` appears as a producer. Counting
one as a DRAM round trip invents traffic that never happened, which is the same class of error as
the tensor-id byte count that once inflated this fold 1.6x. Pairs with a view on either end are
dropped, using `real_traffic.py`'s own `NO_TRAFFIC` set. Before the rule the diffusion path read
425.9 GB/fold of single-use round trip; after it, **322.1 GB/fold**.

Two limits of this capture, stated because they bound what the table may claim. It is **4 sampling
steps, not 200** — per-call figures are per-call and the fold column multiplies by the call counts
in `perf/bioir_roofline/flops_bytes_512.json`, which is the same 512 aa / 3 recycle / 200 step
configuration. And it carries **no tt_bio call-site tags**, so pairs are named by op code only; the
Pairformer table resolves to model functions and this one cannot.

## Token DiT layer — 4800 calls/fold

```
  DRAM allocated in the call                81.0 MB
  single-use intermediates               18 allocations, 48.8 MB of read+write
    fusable as epilogue/prologue            38.8 MB    x 4800 calls =   186.2 GB/fold
    blocked by tile reorder                 10.0 MB    x 4800 calls =    47.8 GB/fold
    unclassified                             0.0 MB
```

| removable MB | GB/fold | n | class | producer → consumer |
|---|---|---|---|---|
| **16.78** | **80.53** | 1 | PROLOGUE | `softmax` → `matmul` |
| **11.01** | **52.85** | 5 | EPILOGUE | `linear` → `multiply_` |
| 4.72 | 22.65 | 3 | EPILOGUE | `linear` → `multiply` |
| 4.19 | 20.13 | 2 | REORDER | `nlp_create_qkv_heads` → `matmul` |
| 3.15 | 15.10 | 2 | EPILOGUE | `linear` → `add_` |
| 2.10 | 10.07 | 1 | REORDER | `nlp_create_qkv_heads` → `transpose` |
| 2.10 | 10.07 | 1 | REORDER | `transpose` → `matmul` |
| 1.57 | 7.55 | 1 | REORDER | `permute` → `multiply` |
| 1.57 | 7.55 | 1 | PROLOGUE | `multiply` → `linear` |
| 1.57 | 7.55 | 1 | EPILOGUE | `multiply` → `add` |

## Atom transformer layer — 1200 calls/fold

```
  DRAM allocated in the call               136.9 MB
  single-use intermediates               16 allocations, 77.1 MB of read+write
    fusable as epilogue/prologue            36.7 MB    x 1200 calls =    44.0 GB/fold
    blocked by tile reorder                 36.7 MB    x 1200 calls =    44.0 GB/fold
```

| removable MB | GB/fold | n | class | producer → consumer |
|---|---|---|---|---|
| **18.35** | **22.02** | 3 | EPILOGUE | `linear` → `multiply_` |
| 14.68 | 17.62 | 1 | REORDER | `matmul` → `permute` |
| **11.01** | **13.21** | 3 | EPILOGUE | `linear` → `multiply` |
| 7.34 | 8.81 | 2 | REORDER | `to_memory_config` → `multiply_` |
| 7.34 | 8.81 | 2 | REORDER | `to_memory_config` → `add_` |
| 3.67 | 4.40 | 1 | REORDER | `permute` → `matmul` |
| 3.67 | 4.40 | 1 | REORDER | `linear` → `pad` |
| 3.67 | 4.40 | 1 | PROLOGUE | `multiply` → `linear` |
| 3.67 | 4.40 | 1 | EPILOGUE | `multiply` → `add` |

## Two targets, and the first one is the largest found anywhere

**1. The attention probability matrix round-trips to DRAM — 80.53 GB/fold.** `softmax` → `matmul`
in the token DiT, one allocation of 8.39 MB per layer call, paid 4800 times. 8.39 MB of bf16 is
4.19 M elements, which is exactly 512 x 512 x 16 — the attention matrix for all sixteen heads. It is
written to DRAM by the softmax and read straight back by the PV matmul.

**Keeping `softmax(QK^T)` resident between the two matmuls is the defining move of flash attention**
and it is the single largest fusable item this campaign has found, in either half of the fold. The
Pairformer's triangle attention already has a fused SDPA (`tt_bio/sdpa_generic.py`); the token DiT's
attention does not, and the same kernel family may reach it.

**2. The gating multiply after a projection — 74.87 GB/fold across both layer shapes.** `linear` →
`multiply_` is 52.85 GB/fold in the token DiT over five sites and 22.02 GB/fold in the atom layer
over three, plus 35.86 GB/fold more in the `linear → multiply` and `linear → add_` variants. That is
the AdaLN/gating shape: a projection writes a tensor to DRAM and the very next op multiplies it by a
scale. **It is the same defect as the Pairformer's `roof-fuse-gate-epilogue` row** (268.4 MB there),
at a different site and with a 200x multiplier on it, and it is eleven sites rather than one — so
whatever mechanism makes an epilogue multiply cheap in one place makes it cheap in all of them.

## The whole diffusion path

```
  single-use DRAM round trip               322.1 GB/fold
    fusable                                230.3 GB/fold
    blocked by tile reorder                 91.9 GB/fold
```

**230.3 GB/fold is 6.8 % of the fold's 3.405 TB**, from the diffusion path's single-use
intermediates alone, and two mechanisms carry 155.4 GB of it.

## What this does NOT say

**Bytes are not seconds.** The diffusion step runs at 38.1 % of the 429.9 GB/s stream roof and
9.3 % of the 85.96 TFLOP/s compute roof — bound by neither. Removing a byte does not return its time
at the roof rate, and `k10-p1-diffusion-measure` found the step 94.34 % in-kernel, so this is not
dispatch either. Everything here is a byte prize and is stated as one.

**The baseline is stale.** 3.405 TB and its 1.320 TB diffusion share were counted at the 23.841 s
fold. The cell is 17.340 s. `roof-budget` owes the re-take.

**The capture has no call-site tags**, so no row above names a model function. A tagged re-capture
of the two diffusion layer shapes, the way `perf/b2z2_byte_floor/trace_block.py` tags the
Pairformer, is what turns this ranking into a build queue.
