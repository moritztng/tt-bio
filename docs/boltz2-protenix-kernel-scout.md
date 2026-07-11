# Boltz-2 and Protenix kernel scout

## Result

No production change is justified by this pass.

* FC2 plus its residual add has a 1.011x trunk ceiling at N=512 and 1.009x at
  N=1024. The matmul remains necessary, so a fused epilogue could only remove
  the residual dispatch.
* TriangleAttention is substantial, at 27-30% of an actual-input trunk cycle
  and 32-38% of its 48-block Pairformer, but it is not dominated by one
  avoidable channel move. The tested
  dispatch-collapse and transpose substitutions produced no speedup.
* OuterProductMean transpose decomposition speeds up the four OPM calls by
  1.073x at N=512 and 1.034x at N=1024. Its full-trunk upper bound is only
  1.002x and 1.001x, respectively, and the N=1024 output is not bit-exact.

The runtime code is unchanged. No release gate was needed.

## Method

Measurements used `pc` physical card 0, one Blackhole P150a, ttnn 0.68, real
ESMFold2 or Protenix-v2 checkpoint weights, and production dimensions. Every
timed path was warm and ended with a device synchronization. Transfers and
comparisons were outside timed regions.

Actual trunk inputs came from the first 512 residues of `examples/615.yaml` and
the first 1024 residues of `examples/1303.yaml`, processed by the model feature
builder and input atom encoder. The timed recycling cycle included four MSA
blocks and all 48 Pairformer blocks. These runs used their query-row MSA. OPM
was measured separately with all four production blocks at MSA depth 2048.

The A/B runs used deterministic dense activations at the same production
shapes. This isolates device execution while preserving module depth, real
weights, precision, and dimensions.

Instrumentation-only Pairformer runs were bit-exact at both sizes: PCC 1.0,
max-abs 0.0, and finite outputs.

## FC2 and residual

This measures the current ESMFold2 trunk after the merged FC1+SwiGLU fusion.
The synchronized component times cover all 48 blocks.

| N | 48-block trunk | FC2 | residual add | residual-free ceiling |
|---:|---:|---:|---:|---:|
| 512 | 4.8424 s | 0.2104 s (4.35%) | 0.0518 s (1.07%) | **1.0108x** |
| 1024 | 19.9294 s | 0.8192 s (4.11%) | 0.1804 s (0.91%) | **1.0091x** |

At N=1024, adding synchronization around every FC2 and residual leaves the
full output unchanged (PCC 1.0, max-abs 0.0). The N=512 random-state endpoint
became non-finite in both unmodified runs after 48 residual blocks, so it is
used for timing only and not presented as a parity result.

There is no current ttnn matmul epilogue for an arbitrary residual tensor.
Even a custom implementation that made the add free would stay below a 1.1%
trunk win. This is too small to justify a new accuracy-sensitive kernel path.

## TriangleAttention

The actual-input cycle confirms the isolated Pairformer profile:

| N | complete trunk cycle | TriangleAttention | share | OPM, query MSA |
|---:|---:|---:|---:|---:|
| 512 | 7.9485 s | 2.1089 s | **26.5%** | 0.0833 s |
| 1024 | 36.5762 s | 10.8332 s | **29.6%** | 0.4064 s |

The N=1024 synchronized profile is bit-exact against the normal cycle. At
N=512, `s` is bit-exact and `z` is PCC 0.999893. This is a synchronization-order
effect in the unchanged path, not an optimization comparison.

| N | 48-block Pairformer | start attention | end attention | total share |
|---:|---:|---:|---:|---:|
| 512 | 6.5052 s | 0.8597 s | 1.2417 s | **32.3%** |
| 1024 | 28.9539 s | 4.5154 s | 6.3749 s | **37.6%** |

The stack contains 96 TriangleAttention calls. Each run issues 96 SDPA calls,
192 QKV/gate projections, 192 output/bias linears, 192 permutes, and the head
split/concat operations around SDPA.

| synchronized phase, 96 calls | N=512 | N=1024 |
|---|---:|---:|
| SDPA | 0.7080 s | 5.1910 s |
| all permutes | 0.4114 s | 1.9315 s |
| minimal matmuls + linears | 0.6191 s | 2.1240 s |
| head split + concat | 0.2865 s | 1.0907 s |
| complete isolated attention | 2.3168 s | 11.1346 s |

Permutes account for 18-20% of TriangleAttention, not most of it. Making every
permute free would still cap the full Pairformer win at about 1.07x.

Three bit-exact A/B attempts were rejected:

| attempt | N=512 | N=1024 | parity |
|---|---:|---:|---:|
| ending-node permute to transpose | 1.0002x | 1.0001x | PCC 1.0, max-abs 0.0 |
| bias rotation to two transposes | 0.9720x | 0.9131x | PCC 1.0, max-abs 0.0 |
| pack QKV and gate into one projection | 0.6073x | 0.5801x | PCC 1.0, max-abs 0.0 |

The packed projection removes one matmul dispatch, but extracting QKV and gate
from the wider result costs more than the saved launch. A larger fused kernel
would need to absorb head creation, SDPA, gating, and projection. The profile
does not show a small adjacent-op fusion comparable to FC1+SwiGLU.

## OuterProductMean

The four OPM calls are mostly data movement at N=512. At N=1024, the outer
product matmul grows to the largest single phase.

| synchronized phase, four calls | N=512 | N=1024 |
|---|---:|---:|
| permutes | 0.0712 s | 0.1665 s |
| outer-product matmul | 0.0370 s | 0.3756 s |
| three linears | 0.0282 s | 0.0908 s |
| layout conversions | 0.0261 s | 0.0948 s |

Replacing OPM's three permutations with equivalent transpose sequences was
measured with three warm repetitions:

| N | current four OPM calls | transpose decomposition | OPM speedup |
|---:|---:|---:|---:|
| 512 | 0.18844 s | 0.17557 s | **1.0733x** |
| 1024 | 0.83523 s | 0.80801 s | **1.0337x** |

The first block is bit-exact at both sizes. The fourth block is PCC
0.99999999999999 / max-abs 0.0625 at N=512 and PCC 0.9999999973 / max-abs
64.125 at N=1024. The latter exceeds the unmodified repeat floor
(PCC 0.99999999999996 / max-abs 0.25).

Even ignoring the parity difference, inserting the production-depth OPM result
into the measured actual-input cycle gives optimistic full-trunk speedups of
1.0016x at N=512 and 1.0007x at N=1024. The omitted MSA-depth-dependent work
lowers those numbers further. The local OPM win is therefore not promoted.

## Reproduce

```bash
PYTHONPATH=. TT_VISIBLE_DEVICES=0 python3 scripts/kernel_scout_next_bench.py \
  --sizes 512 1024 --fc2-profile
PYTHONPATH=. TT_VISIBLE_DEVICES=0 python3 scripts/boltz2_protenix_kernel_scout.py \
  pairformer --sizes 512 1024
PYTHONPATH=. TT_VISIBLE_DEVICES=0 python3 scripts/boltz2_protenix_kernel_scout.py \
  real-trunk --sizes 512 1024
PYTHONPATH=. TT_VISIBLE_DEVICES=0 python3 scripts/boltz2_protenix_kernel_scout.py \
  triatt-projection --sizes 512 1024
PYTHONPATH=. TT_VISIBLE_DEVICES=0 python3 scripts/boltz2_protenix_kernel_scout.py \
  opm --sizes 512 1024 --msa-depth 2048 --repeats 3
```
