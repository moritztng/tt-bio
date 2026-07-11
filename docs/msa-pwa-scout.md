# MSA PairWeightedAveraging scout

## Result

Batching the eight heads is not justified. PairWeightedAveraging (PWA) is only
0.47% of a real Protenix-v2 trunk cycle at N=512 and 0.30% at N=1024. Removing
PWA entirely would improve the trunk by at most 1.0047x and 1.0030x,
respectively.

Replaying all three production PWA calls as one ttnn trace removes their Python
and API dispatch. It improves PWA itself by 1.0034x at N=512 and 1.0031x at
N=1024. The many per-head calls are already hidden behind device execution.

The runtime path is unchanged. No accuracy release gate was needed.

## Real-trunk profile

The measurement used physical card 0 on `pc`, real Protenix-v2 checkpoint
weights, and actual protein features from the first 512 residues of
`examples/615.yaml` and the first 1024 residues of `examples/1303.yaml`. The
query-row MSA depth was 1, matching the prior shared-trunk scout. Every timed
region was warm and ended with a device synchronization.

Protenix-v2 has eight heads of width 8 and PWA in the first three of its four
MSA blocks. The final block has no MSA stack.

| N | four-block MSA | three PWA calls | MSA share | complete trunk cycle | trunk share | PWA-free trunk ceiling |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.6592 s | 0.03730 s | 5.66% | 7.9491 s | 0.469% | **1.0047x** |
| 1024 | 2.9194 s | 0.11074 s | 3.79% | 36.4513 s | 0.304% | **1.0030x** |

The synchronized profile is bit-exact against the normal cycle at both sizes:
PCC 1.0 and max-abs 0.0 for both trunk outputs.

## Dispatch ceiling

The trace records the existing per-head implementation without changing its
math. It uses deterministic dense activations at the same production shapes
and real weights, with transfers outside the timed region. Its replay is the
device-execution floor after eliminating all Python loop and individual
operation dispatch.

| N | current PWA, median of 7 | traced floor, median of 7 | local speedup | full-trunk speedup |
|---:|---:|---:|---:|---:|
| 512 | 37.166 ms | 37.040 ms | **1.0034x** | **1.000016x** |
| 1024 | 110.685 ms | 110.338 ms | **1.0031x** | **1.000010x** |

Trace replay is bit-exact for the first and last PWA blocks at both sizes: PCC
1.0 and max-abs 0.0.

Per-operation barriers show where the device time goes. The values below are
attribution measurements, not additive wall time because each call has an
extra synchronization.

| synchronized phase, three PWA calls | calls | N=512 | N=1024 |
|---|---:|---:|---:|
| linear | 96 | 36.918 ms | 99.130 ms |
| permute | 72 | 5.373 ms | 9.042 ms |
| matmul | 24 | 1.448 ms | 2.027 ms |
| softmax | 24 | 1.518 ms | 2.108 ms |

The linears dominate. Matmul and softmax are small, and removing dispatch alone
does not move the measured wall time. A batched multi-head rewrite could not
beat the PWA-free trunk ceiling, even if all of its device work became free.

## Conclusion

No production change is warranted. The remaining shared-trunk scout list is
now exhausted: triangle multiplication, Transition, TriangleAttention,
OuterProductMean, and PairWeightedAveraging have all been measured. Further
Boltz-2 or Protenix-v2 acceleration must target a different model stage or a
deeper compute kernel, not another dispatch collapse in these primitives.

## Reproduce

```bash
PYTHONPATH=. TT_VISIBLE_DEVICES=0 python3 scripts/boltz2_protenix_kernel_scout.py \
  real-trunk --sizes 512 1024
PYTHONPATH=. TT_VISIBLE_DEVICES=0 python3 scripts/boltz2_protenix_kernel_scout.py \
  pwa --sizes 512 1024 --msa-depth 1 --repeats 7
```
