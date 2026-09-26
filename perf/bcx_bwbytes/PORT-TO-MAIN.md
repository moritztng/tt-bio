# The one graft, written against main's actual text

Everything else on `wk/bcx-bwbytes` duplicates main (see `DISPOSITION.md`). This is the single
change main does not have, written so it can be applied rather than re-derived.

## What it does

`moreh_softmax_backward` refuses fp32 —
`ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/device/moreh_softmax_backward_device_operation.cpp:79-84`
admits `BFLOAT16` and `BFLOAT8_B` only. That refusal is why `of3t-softbw` returned NO-GO on Route
A and why main ships `SOFTMAX_BW_FUSED = False`. Narrowing y and the cotangent once, before the
route is chosen, produces operands the guard admits — and halves every pass of the composed route
at the same time, which is why it is worth taking even if the fused branch stays off.

## The graft, onto `tt_bio/autograd.py`

Add beside the other policy flags:

```python
#: Operand dtype for the softmax backward. "keep" runs it in whatever the forward left, which for
#: the fp32 softmax island is fp32; "bf16" narrows y and the cotangent ONCE and runs the rest at
#: half the bytes, taking the summation precision from fp32 accumulation in DST instead of from
#: fp32 tensors in DRAM.
#:
#: It is also the only thing that makes `SOFTMAX_BW_FUSED` executable. `moreh_softmax_backward`
#: admits BFLOAT16 and BFLOAT8_B and refuses fp32, so on the shipped dtype the fused branch cannot
#: run at all -- `of3t-softbw`'s NO-GO on Route A is a NO-GO at the operand dtype, not a property
#: of the op.
#:
#: A PRECISION lever: graded against float64 as a STACK, never summed with other arms' readings.
SOFTMAX_BW_DTYPE = os.environ.get("TT_BIO_SOFTMAX_BW_DTYPE", "keep")

SOFTMAX_BW_DTYPE_STATS = collections.Counter()
```

and at the top of `softmax_bw`, ahead of its `ax = _last_axis(y, dim) if SOFTMAX_BW_FUSED else None`:

```python
    if SOFTMAX_BW_DTYPE == "bf16":
        if y.dtype != ttnn.bfloat16:
            y = ttnn.typecast(y, ttnn.bfloat16)
            SOFTMAX_BW_DTYPE_STATS["narrowed_y"] += 1
        if g.dtype != ttnn.bfloat16:
            g = ttnn.typecast(g, ttnn.bfloat16)
            SOFTMAX_BW_DTYPE_STATS["narrowed_g"] += 1
    SOFTMAX_BW_DTYPE_STATS[f"dx:{str(y.dtype).split('.')[-1]}"] += 1
```

That is the whole change. It needs no new call sites: main already routes all three callers
through `softmax_bw`.

## Two things that come with it and must not be dropped

1. **`softmax_bw_inner`'s numerator reduce needs the compute kernel config.** On main, line 119's
   `ttnn.sum(ttnn.multiply(g, y), dim=dim, keepdim=True)` carries none while the denominator's
   does. On fp32 operands that is invisible; on bf16 operands it is the whole argument, because the
   bf16 arm's entire claim is that the precision comes from the accumulator. Pass
   `compute_kernel_config=cfg` on both.
2. **`perf/land_standing/renorm_reach.py` must stay green.** Its CALLERS and SITES checks are both
   transitive on this branch, with controls proving a forward-reachable reader is still caught.
   If main's copy is the older literal one, the transitive version is on this branch and should go
   with the graft, not be reverted into.

## How to grade it

`perf/bcx_bwbytes/softmax_bw_probe.py`, six arms at n=224 and n=256, every one against a float64
reference built as the RENORMED expression so an arm that drops the correction is graded against
what dropping it costs. Its fp32 `moreh` arms are EXPECTED refusals; record them as error strings,
never as timings. Then `bytes.py arms --arms base,smbf16 --f64` for the block VJP, and
`round_ab.py --rounds 28 --precision --moreh` for the round, where 28 is what the wall needs at
this effect size and 3 is what `device_evoformer_s` needs.
