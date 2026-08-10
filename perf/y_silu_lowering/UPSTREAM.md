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

Worth **503.8 ms per Protenix-v2 fold at 298 aa** on this card and **1381.2 ms at 512 aa**, both
measured at the shape production runs. It grows with the sizes real targets have.

## Shape dependence, for completeness

The penalty is per padded output tile and it saturates. Its only variable is the tile count: a 3-D
shape and a 4-D one at the same tile count cost the same per tile and both reach the same asymptote,
so input rank is not the trigger. Two sweeps on the same card, `fp32_dest_acc_en=True`, interleaved
below by tile count so the families read against each other:

| in0 shape, against a `[K, N]` weight | rank | padded out tiles | us per tile |
|---|---|---:|---:|
| `[1, 1, 298, 256] x [256, 1024]` | 4-D | 320 | 0.000, straddles zero |
| `[1, 298, 384] x [384, 1536]` | 3-D | 480 | 0.0026 |
| `[1, 2, 298, 256]` | 4-D | 640 | 0.0023 |
| `[1, 596, 384]` | 3-D | 912 | 0.0064 |
| `[1, 4, 298, 256]` | 4-D | 1280 | 0.0139 |
| `[1, 1192, 384]` | 3-D | 1824 | 0.0164 |
| `[1, 8, 298, 256]` | 4-D | 2560 | 0.0136 |
| `[1, 2384, 384]` | 3-D | 3600 | 0.0180 |
| `[1, 16, 298, 256]` | 4-D | 5120 | 0.0184 |
| `[1, 4768, 384]` | 3-D | 7152 | 0.0165 |
| `[1, 16, 512, 256]` | 4-D | 8192 | 0.0193 |
| `[1, 30, 298, 256]` | 4-D | 9600 | 0.0179 |

Below about 700 tiles the fused activation is free, because the matmul's wall there is program launch
plus the in0/in1 multicast and the MATH RISC is blocked in `cb_wait_front` for most of it, so the extra
SFPU instructions land in dead cycles. Above about 3000 tiles the MATH RISC is the critical path and the
penalty sits at the SFPU issue rate. The residual spread among the saturated points, 0.0138 to 0.0194,
tracks the auto-chosen subblock and per-core geometry rather than the lowering: eight explicit
`MatmulMultiCoreReuseMultiCastProgramConfig`s at one fixed shape span 174.5 to 210.4 us for the same
penalty, which is the same width.
