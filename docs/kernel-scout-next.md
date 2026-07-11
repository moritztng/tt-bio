# Kernel scout: pair-transition SwiGLU fusion

## Result

The next production bottleneck after triangle multiplication is the ESMFold2 pair
transition. It occupies 25-28% of the real 48-block trunk. A true FC1+SwiGLU epilogue
fusion is fast (1.18x trunk speedup at N=512 and 1.14x at N=1024), but it does not clear
the parity gate: full-trunk PCC is 0.99984-0.99986 rather than 1.0. It is therefore
**release-gated and not promoted**. No shipped primitive or default changes.

The exact existing `ttnn.swiglu` composite was also tested. It remains bit-exact but
produces no speedup because it still materializes the FC1 output and expands to the same
split/SILU/multiply operations.

## Profile first: real 48-block trunk

Hardware: qb2 physical card 0, one Blackhole P150 chip; bf16; real ESMFold2 checkpoint
weights; deterministic dense pair input; 48 blocks. Every timed trunk is warm and ends
with a device synchronize. Component measurements add synchronization around each of the
48 calls; the unchanged output is PCC 1.0 / max-abs 0.0 against the normal execution.

ESMFold2's production folding trunk contains 96 triangle multiplications and 48 pair
transitions. It does **not** contain TriangleAttention or OuterProductMean, so those cannot
honestly be profiled through this trunk.

| N | trunk | trimul out | trimul in | pair transition | transition share |
|---:|---:|---:|---:|---:|---:|
| 512 | 5.0165 s | 1.7403 s | 1.7406 s | **1.4120 s** | **28.1%** |
| 1024 | 20.3027 s | 7.3961 s | 7.3952 s | **4.9956 s** | **24.6%** |

The transition is LN -> FC1 `[256,2048]` -> split -> `silu(gate)*up` -> FC2
`[1024,256]`. Its logical FC1 activation is 1 GiB per block at N=512 and 4 GiB at
N=1024. Row chunking limits peak allocation but not total DRAM traffic. This is a fusion
target, not a reason to replace either matmul.

## Attempt 1: post-FC1 `ttnn.swiglu`

The FC1 halves were reordered so `ttnn.swiglu` has exactly the existing
`silu(first_half)*second_half` semantics.

| N | current trunk | `ttnn.swiglu` trunk | speedup | PCC | max abs |
|---:|---:|---:|---:|---:|---:|
| 512 | 5.0150 s | 5.0195 s | **0.9991x** | **1.0** | **0.0** |

This is a semantic no-op and a performance no-op. In current tt-metal, `ttnn.swiglu` is
a composite split + SiLU + multiply; it does not fuse with FC1 or avoid the wide tensor's
DRAM round trip.

## Attempt 2: FC1 matmul epilogue fusion

An isolated tt-metal build at `1769bd090998f160771b4aace89c463cd28d6c01`
(2026-07-11 main) provides `ttnn.experimental.minimal_matmul(...,
fuse_swiglu=True)`. The benchmark tile-interleaves each real checkpoint FC1 weight as
`[gate_t0, up_t0, ...]`. The matmul consumes both projections together, evaluates
`silu(gate)*up` before packing, and writes only the half-width gated tensor for FC2.
This collapses the adjacent phase and removes the dominant intermediate write/read; it
does not alter FC2.

Fair A/B: baseline and fused paths run in one process on the same new tt-metal build,
with identical weights/input and a warm program cache.

| N | baseline 48-block trunk | fused trunk | speedup | full-trunk PCC | max abs | finite |
|---:|---:|---:|---:|---:|---:|:---:|
| 512 | 5.0110 s | 4.2296 s | **1.1847x** | **0.9998358** | 328 | yes |
| 1024 | 20.3022 s | 17.8046 s | **1.1403x** | **0.9998565** | 338 | yes |

The performance result is real and end-to-end over all 48 production blocks, not a
standalone kernel microbenchmark. The accuracy result is also real: the fused op uses
`minimal_matmul` rather than the existing `ttnn.linear` schedule, changing bf16
accumulation. Upstream's own single-op test accepts PCC >0.9999; recurrence through 48
blocks amplifies that floor below 0.9999. No structure RMSD or release-gate result was
fabricated, and this path is not enabled.

## Ceiling and next condition

Pair transition has a 1.39x/N=512 and 1.33x/N=1024 absolute trunk Amdahl ceiling even if
it becomes free. The measured fused path captures a useful part of it, so this is not a
compute-throughput dead end. It is an **accuracy/schedule ceiling**:

- dispatch-only cleanup is exhausted (`ttnn.swiglu`: 1.00x, PCC 1.0);
- eliminating the FC1 intermediate is worth 14-18% on the trunk;
- the available fused implementation changes matmul accumulation and fails the required
  full-trunk parity bar.

A promotable implementation must add the binary SwiGLU epilogue to the same matmul
program/config selected by `ttnn.linear`, or prove end-to-end structure parity through the
release gate. Until then, the measured speedup stays release-gated.

Reproduce the baseline profile with installed ttnn 0.68:

```bash
PYTHONPATH=. TT_VISIBLE_DEVICES=0 python3 scripts/kernel_scout_next_bench.py --sizes 512 1024
```

For `--swiglu-ab`, put the isolated tt-metal main package/build first in `PYTHONPATH` and
`LD_LIBRARY_PATH`, then run one size per process. The harness reports stable two-pass PCC,
max-abs, finiteness, synchronized component times, and full-trunk speedup.
