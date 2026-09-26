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

## The upgrade path, and the hook for it already exists

As written the lever pays two narrowing casts per call — one for y, one for the cotangent — and
the reach map's 3.161 GB is already NET of them. Both are avoidable in principle, because the
forward has already made the bf16 copy this lever reconstructs: `tenstorrent._fp32_softmax_tail`
computes the softmax in fp32 and its next statement is
`attn_bf = ttnn.typecast(attn, ttnn.bfloat16)`. The backward narrows a tensor whose narrowed form
existed a line later in the forward and was thrown away.

Main's `_v_softmax` already carries the mechanism that would fix it. It holds y indirectly:

```python
    box = [y0]
    def bw(g):
        y = box[0]                      # through the box, so `free` may evict y to DRAM
    ...
    out.box = box
```

The box is a redirection slot the Tensor machinery can rewrite — it exists so `free` can move y to
DRAM without the closure noticing. Writing the bf16 copy into `box[0]` instead would give the
backward bf16 y at zero cost AND let the fp32 `attn` be released one statement earlier, halving
what the softmax node retains (268 MB -> 134 MB per call at n=256).

It is deliberately NOT part of the graft above. It is cross-node plumbing: the narrowing happens
in a different taped call (`_identity_grad`'s typecast) from the one that owns the box, so wiring
it means one node reaching into another's lifetime, and a wrong guess there frees a buffer the
backward still reads. Worth roughly +1.5 GB on a 3.16 GB lever — real, not transformative — and it
should be measured with the simple version first so there is something to difference against.

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

## Preconditions for the 1.161x run, each checked against its source

The wheel route is worth 1.161x of the 1.250x ceiling and needs no kernel. Six things must hold
for it to even execute, and all six are verified card-free — so a run that fails does so for a
REAL reason (wrong on Blackhole, or slow), not because a precondition was never checked.

1. **All three callers reach the seam.** Main's `softmax`, `triangle_attention` and
   `taped_ttnn._v_softmax` all call `autograd.softmax_bw`. Read on main.
2. **The axis index is positive.** `moreh_softmax_backward` takes `dim` as a `uint32_t`
   (`moreh_softmax_backward.hpp`), so `-1` does not name the last axis to it. Main's `_last_axis`
   returns `rank - 1` when `dim` names the last axis and `None` otherwise, falling back to the
   composed path — "the right answer slowly rather than the wrong one quickly".
3. **The dtype is admitted.** `moreh_softmax_backward_device_operation.cpp:79-84` requires
   `BFLOAT16` or `BFLOAT8_B` for BOTH `output_tensor` and `output_grad_tensor`. `SOFTMAX_BW_DTYPE
   ="bf16"` narrows both. This is the one thing main does not have and the whole reason the flag
   ships off.
4. **The layout is TILE.** Same guard, and the score tensor is tiled throughout the trunk.
5. **The shape is tile-aligned at BC2's size.** 224 = 7 x 32, and the tensor is
   `[224, 4, 224, 224]`, so both reduced and non-reduced dims are whole tiles.
6. **It is not chunked at that size.** `_fp32_softmax_attention`'s budget is
   `_FP32_SOFTMAX_BLOCK_BYTES = 8 << 30` (`tenstorrent.py:3073`) per fp32 score copy, and the
   tensor is **179.8 MB** — two orders under it — so the backward sees one whole tensor rather
   than ragged chunks whose boundaries might not tile-align.

What is NOT checked and cannot be without a card: whether the kernel computes the right thing on
Blackhole, and whether it is faster. Its sibling `moreh_layer_norm_backward` is wrong there
(dx 2.741e+06 rel L2 in bf16, upstream #12349), which is exactly why the probe grades every arm
against float64 rather than against the composed path.

## How to grade it

`perf/bcx_bwbytes/softmax_bw_probe.py`, six arms at n=224 and n=256, every one against a float64
reference built as the RENORMED expression so an arm that drops the correction is graded against
what dropping it costs. Its fp32 `moreh` arms are EXPECTED refusals; record them as error strings,
never as timings. Then `bytes.py arms --arms base,smbf16 --f64` for the block VJP, and
`round_ab.py --rounds 28 --precision --moreh` for the round, where 28 is what the wall needs at
this effect size and 3 is what `device_evoformer_s` needs.
