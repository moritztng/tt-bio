# y-silu-lowering — predictions, committed before the device was opened

Written after reading the ttnn 0.67.4 wheel source and BEFORE any measurement this pass. Each
prediction names the outcome that makes it wrong.

The source read (quoted in the state doc): `silu_tile(idst)` expands to
`llk_math_eltwise_unary_sfpu_silu<APPROX, DST_ACCUM_MODE>`, and `calculate_silu` calls
`_sfpu_sigmoid_<is_fp32_dest_acc_en>`, which selects `_sfpu_exp_fp32_accurate_` + a 2-iteration
Newton reciprocal when `DST_ACCUM_MODE=1` and `_sfpu_exp_21f_bf16_` + a 1-iteration reciprocal when
it is 0. Production sets `fp32_dest_acc_en=True` (protenix.py:1609). So the prediction is that
y-silu arm B ran a DIFFERENT, CHEAPER silu than arm A, and the "2x ttnn defect" is not a defect.

- **P1 — `ttnn.silu(y)` with no `compute_kernel_config` takes `DST_ACCUM_MODE=0`.** WRONG if the
  default already carries fp32 dest accumulate.
- **P2 — `ttnn.silu(y, compute_kernel_config=<HiFi4, fp32_dest_acc_en=True>)` costs 150-200 us/call
  at `[1, 30, 298, 1024]`, i.e. 1.8-2.4x the ~84 us the default costs.** WRONG if it lands under
  120 us/call or over 230.
- **P3 — that figure lands within 25 us/call of the fused penalty `A - D` (y-silu: 174.0 us/call).**
  WRONG if the gap exceeds 40 us/call, which would leave a real residual defect to explain.
- **P4 — with both arms on the same silu, the unfuse win collapses from +96.2 us/call to under
  25 us/call, i.e. under 130 ms/fold at x524.** WRONG if `A - (D + S_fp32acc)` stays above
  60 us/call.
- **P5 — accuracy: against a torch fp32 reference on the same bf16 input, the fp32-dest-acc silu has
  a max abs deviation at least 4x smaller than the default silu.** WRONG if within 1.5x, which would
  mean the extra instructions buy nothing and the sequence really is wasteful.
- **P6 — `gelu` does not branch on `DST_ACCUM_MODE`**, so fused gelu and standalone gelu run the same
  code and the 1.04x y-silu measured needs no fusion explanation at all. WRONG if
  `ttnn.gelu(y, fp32_dest_acc_en=True)` differs from the default by more than 15 %.
- **P7 — the shape dependence is NOT a compute-config difference**: every `Transition` in a protenix
  fold shares one `compute_kernel_config` (HiFi4 / fp32_dest_acc_en=True / packer_l1_acc=True), so
  the c_s=384 sites run the same accurate silu and their 1.7 us fused cost is an overlap/floor
  effect, not a cheaper lowering. WRONG if any site is constructed with a different config.
- **P8 — the L1 copy floor this session lands in 33-45 us/call at 18.31 MB**, the band the two
  y-silu sessions bracket (35.171 and 40.573). WRONG outside it, which would say the host load moved
  the instrument and no cross-arm ratio from this session is safe.

## Round 2 — committed after round 1, before the round-2 device run

Round 1 killed the planned route: `ttnn.silu` accepts no `compute_kernel_config` on 0.67.4, so a
standalone silu cannot be pushed onto the accurate lowering. The matmul can be pushed the other way.

- **P9 — `ttnn.linear(activation="silu")` with `fp32_dest_acc_en=False` has a fused-silu penalty of
  70-100 us/call**, against 171.1 at `True`, because DST_ACCUM_MODE=0 selects the identical
  `_sfpu_exp_21f_bf16_` + 1-iteration reciprocal that `ttnn.silu` runs. WRONG outside 60-115 us.
- **P10 — the fused GELU penalty moves by less than 15 % between the two settings**, because gelu
  does not branch on DST_ACCUM_MODE. WRONG if it moves more than 25 %.
- **P11 — standalone `ttnn.silu` on an fp32 input is at least 1.6x the bf16 one in time and closer
  to a torch fp32 reference.** WRONG under 1.3x or if it is not closer.
- **P12 — the bare matmul at `fp32_dest_acc_en=False` is 5-25 % faster than at `True` and less
  accurate against the fp32 reference.** WRONG if it is slower, or if it is not less accurate.
