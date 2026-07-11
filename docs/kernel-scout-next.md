# Kernel scout: pair-transition SwiGLU fusion

## Result

The next production bottleneck after triangle multiplication is the ESMFold2 pair
transition. It occupies 25-28% of the real 48-block trunk. FC1+SwiGLU epilogue fusion
speeds up the trunk by 1.18x at N=512 and 1.14x at N=1024. The fused path passes the
ESMFold2 structure release gate and is enabled when the installed ttnn build exposes it.
Older builds keep the existing `ttnn.linear` path.

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

The performance result is end-to-end over all 48 production blocks, not a standalone
kernel microbenchmark. Full-trunk PCC alone remained below the promotion target, so the
source of that difference and the structure output were checked separately.

## Precision diagnosis

`minimal_matmul` completes the reduction before its fused output stage. With the model's
fp32 destination accumulation, that stage reads the fp32 intermediate directly. The
existing path first packs FC1 to bf16, then runs SiLU and multiply as separate operations.
The missing bf16 boundary is the main numerical difference.

A real-weight A/B on the first pair transition separated the matmul and epilogue effects:

| comparison | PCC | max abs |
|---|---:|---:|
| `ttnn.linear` vs unfused `minimal_matmul` | 0.9999994691 | 0.0625 |
| same minimal matmul, separate vs fused SwiGLU | 0.9999927063 | 0.5 |
| full transition output, existing vs fused | 0.9999961901 | 4.0 |

The matmul schedule contributes a smaller difference. SiLU is not running in a reduced
accumulator; fusion instead removes the bf16 pack/unpack point before it.

## Structure release gate

`scripts/release_gate.py --model esmfold2` was run on both paths with the production
200-step, five-sample protocol and seed 0:

| path | RMSD | TM-score | gate |
|---|---:|---:|:---:|
| existing | 2.7581 A | 0.7871 | pass |
| fused | 2.5923 A | 0.7978 | pass |

Both clear the 4.0 A / 0.65 release floor. This one-target gate establishes no accuracy
regression; it does not establish that fusion improves accuracy.

## Resolution

Pair transition has a 1.39x/N=512 and 1.33x/N=1024 absolute trunk Amdahl ceiling even if
it becomes free. The fused path captures a useful part of it. ESMFold2 now selects it when
`minimal_matmul` advertises `fuse_swiglu`; otherwise it preserves the existing linear,
split, SiLU, and multiply path. This requires no dependency bump and changes nothing on
ttnn 0.68.

Reproduce the baseline profile with installed ttnn 0.68:

```bash
PYTHONPATH=. TT_VISIBLE_DEVICES=0 python3 scripts/kernel_scout_next_bench.py --sizes 512 1024
```

For `--swiglu-ab`, put the isolated tt-metal main package/build first in `PYTHONPATH` and
`LD_LIBRARY_PATH`, then run one size per process. The harness reports stable two-pass PCC,
max-abs, finiteness, synchronized component times, and full-trunk speedup.
