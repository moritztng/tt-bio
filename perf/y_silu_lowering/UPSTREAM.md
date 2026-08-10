# tt-metal issue draft — fused SILU cannot request its own accuracy mode

**Reported from** ttnn 0.67.4, Blackhole p150a (`tt-quietbox` card 0), compute grid 13x10.

## Summary

`calculate_silu` picks its exp and reciprocal implementations from `is_fp32_dest_acc_en`, and drops
the `APPROXIMATION_MODE` parameter its wrapper is handed. So a matmul that needs
`fp32_dest_acc_en=True` for its accumulation is forced onto the accurate sigmoid for its fused
activation, even when the packer writes bf16 and the extra accuracy cannot survive the write. The
cost is 2.30x and the accuracy it buys, measured at bf16 output, is 0.12 %.

## Where

`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_silu.h`

```cpp
template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_silu() { ... x * _sfpu_sigmoid_<is_fp32_dest_acc_en>(x) ... }

template <bool APPROXIMATION_MODE>
inline void silu_init() {
    // calculate_silu uses the non-approx sigmoid path via _sfpu_sigmoid_, so we must use non-approx sigmoid_init
    sigmoid_init<false>();
}
```

`calculate_silu` has no `APPROXIMATION_MODE` template parameter, while
`llk_math_eltwise_unary_sfpu_silu<APPROXIMATE, is_fp32_dest_acc_en, ITERATIONS>` accepts one and
`compute_kernel_api.h` passes `APPROX` into it. `calculate_sigmoid<APPROXIMATION_MODE,
is_fp32_dest_acc_en>` in `ckernel_sfpu_sigmoid.h` already does the right thing and routes to
`calculate_sigmoid_appx`.

`_sfpu_sigmoid_<true>` runs `_sfpu_exp_fp32_accurate_` (Cody-Waite range reduction, three predicated
special-case blocks, round-to-nearest-int32, two-step extended-precision reduction, polynomial) plus
`_sfpu_reciprocal_<2>`. `_sfpu_sigmoid_<false>` runs `_sfpu_exp_21f_bf16_` plus
`_sfpu_reciprocal_<1>`.

## Reproducer

`[1, 30, 298, 256] x [256, 1024]` bf16, both operands L1, HiFi4, `packer_l1_acc=True`, `core_grid`
13x10; k=10 calls per timed region, median of 5 reps, three alternated passes, device synchronised on
both sides. Penalty is measured against the identical bare matmul in the same pass.

| activation fused into `ttnn.linear` | `fp32_dest_acc_en` | penalty over bare, us/call |
|---|---|---:|
| **silu** | **True** | **171.481** |
| **silu** | **False** | **74.685** |
| gelu | True | 138.701 |
| gelu | False | 139.907 |
| standalone `ttnn.silu` over the same tensor | n/a (op takes no compute_kernel_config) | 84.393 |

gelu does not branch on `DST_ACCUM_MODE` and moves 0.9 % of its own penalty. silu moves 2.30x. At
matched lowering the fused silu is **0.885x** the standalone, so fusion itself is healthy.

## The accuracy the 2.30x buys

Same input values, `[1, 1, 320, 1024]`, scored against a torch fp32 reference of those values:

| | max abs dev | relative RMSD |
|---|---:|---:|
| accurate lowering, fp32 output | 9.54e-7 | 4.90e-8 |
| accurate lowering, rounded to bf16 | 0.0156245 | 1.5637e-3 |
| 21f lowering, bf16 output | 0.0156245 | 1.5656e-3 |

## Asked for

Let `calculate_silu` honour `APPROXIMATION_MODE` and route to `_sfpu_exp_21f_bf16_` +
`_sfpu_reciprocal_<1>` when it is set, independently of `is_fp32_dest_acc_en`, and make `silu_init`
follow. Then a matmul can keep `fp32_dest_acc_en=True` for accumulation and still ask for the
bf16-grade activation, which is the correct request whenever the output is bf16.

Worth ~503.8 ms per Protenix-v2 fold at 298 aa on this card, and ~1487 ms at 512 aa by extrapolation
from the measured per-element cost.

## Shape dependence, for completeness

The penalty is per output tile and roughly constant across shapes that keep the MATH RISC busy
(0.01786, 0.01706 and 0.01840 us per padded output tile at three different widths and batches). At
small M (Mt = 10) it drops to 0.00365 us/tile because the matmul there is bound by multicast and
dispatch rather than by MATH, so the extra SFPU work fits in that shadow.
