# What of this branch should land, and what must not

Checked 2026-09-26 against `origin/main` at the symbol level, not the commit level — this branch
is based on `wk/bcx-tmplseam`, which forked before a large amount of main's backward work, and it
is 228 commits behind.

## FOUR of the five levers here are already on main, under other names

| this branch | main | main's default |
|---|---|---|
| `autograd._pairwise_sum0` + `LEADING_SUM_TREE_ROWS` | `autograd._tree_sum` + `TREE_REDUCE` | **True, live** |
| `reblock_permute.permute_via_reblock` + `taped_ttnn.PERMUTE_BW_REBLOCK` | `taped_ttnn._permute_back` + `REBLOCK_PERMUTE_BW` | **True, live** |
| `autograd.softmax_bw_dx` + `SOFTMAX_BW_ROUTE="moreh"` | `autograd.softmax_bw` + `SOFTMAX_BW_FUSED` | **False, built but OFF** |
| `autograd.FANIN_WIDEN_INCOMING=False` | `autograd.FANIN_MIXED=True` | **False, built but OFF** |

Not merely similar. Main's `softmax_bw` carries the SAME renorm identity this branch derived —
`moreh_softmax_backward(y/s, g) * s` — and main's `_permute_back` is the same two-move gate over
`reblock_permute_back` / `reblock_permute` with the same fallback. Independently re-derived, twice.

And main's versions are BETTER in the places they differ:

* `_tree_sum` gates on `dtype == float32` and on the axis position (`ax < len(gs) - 2`), where
  this branch gates on a row count. It also handles `_unshard(g)`, which this branch does not.
* main's `_sum_leading` routes fp32 through the tree with a TILE-AWARE scheme — reshape to
  `[n/32, 32, c]`, tree, relayout to row-major, tree again — reaching the rows inside a tile that
  a plain leading-axis tree cannot. This branch never addressed that case.

**So merging this branch as-is would REGRESS main.** It would replace `_tree_sum` and the
tile-aware `_sum_leading` with simpler versions, and `_permute_back` and `softmax_bw` with
duplicates. Do not merge it.

The fan-in one is the sharpest of the four. Main's `FANIN_MIXED` is the same
`ttnn.add(self._grad, grad, dtype=ttnn.float32)`, and it is graded BETTER than this branch graded
it: main's comment quotes **8.2e-5 relative L2 from float64 at two contributions and 5.5e-4 at
sixteen**, against 1e-10 and 1e-9 widened, from `perf/bcx_reduce/probe.json` — measured at the
fan-in COUNT that matters rather than on a single op, which is the axis this branch spent a pass
correcting itself about.

## ONE lever here is genuinely new, and it is the one that matters

`autograd.SOFTMAX_BW_DTYPE="bf16"` — the softmax backward with y and the cotangent narrowed once,
taking the summation precision from fp32 accumulation in DST rather than from fp32 tensors in
DRAM. Nothing on main does this.

On its own it is worth 2.144 GB of a 28.919 GB Evoformer block backward at n=224, about 7.4 % of
the backward's bytes and roughly 1.038x on a round — modest. **Its real value is that it is the
only thing that makes main's own fused route executable.** `SOFTMAX_BW_FUSED` ships off because
`moreh_softmax_backward` refuses fp32; the guard admits BFLOAT16 and BFLOAT8_B; this lever
produces exactly those operands. So the disposition is one graft, onto main's `softmax_bw`, ahead
of its `ax = _last_axis(...)` line — not a merge of this branch over it.

## What that leaves the campaign

1. **`SOFTMAX_BW_FUSED` is built on main and ships OFF — and it is off for a measured reason.**
   `of3t-softbw` returned **NO-GO on Route A**: `ttnn.moreh_softmax_backward` REFUSES fp32, and
   the shipped score tensor is fp32, so the fused branch cannot execute at all. Its saving is
   0 s, not a small one. (An earlier version of this file called turning the flag on "the cheapest
   unclaimed win, one card and no new code". That was wrong and is corrected here.)

   **But the refusal is conditional on the operand dtype, and this branch changes exactly that.**
   The guard, read from source at
   `ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/device/moreh_softmax_backward_device_operation.cpp:79-84`:

       TT_FATAL(output_tensor.dtype() == DataType::BFLOAT16 || output_tensor.dtype() == DataType::BFLOAT8_B, ...)
       TT_FATAL(output_grad_tensor.dtype() == DataType::BFLOAT16 || output_grad_tensor.dtype() == DataType::BFLOAT8_B, ...)

   bfloat16 and bfloat8_b are ADMITTED; fp32 is what is refused. `SOFTMAX_BW_DTYPE="bf16"`
   narrows y and the cotangent to exactly that dtype before the backward runs. So the precision
   lever this branch built for a BYTE reason is also the thing that makes Route A executable, and
   of3t-softbw's NO-GO is a NO-GO at the shipped dtype rather than a property of the op.

   Two rows each held half of this: `of3t-softbw` established the refusal and sized Route B; this
   one built the narrowing without knowing the refusal existed. Unverified so far — the guard
   admits the dtype, which is not the same as the kernel being correct on Blackhole or faster.
   `perf/bcx_bwbytes/softmax_bw_probe.py` already carries the `moreh_bf16` and `moreh_rn_bf16`
   arms and grades every one against float64; its fp32 arms are EXPECTED refusals, and recording
   a refusal as a reading is what of3t-softbw warns against.
2. The two precision levers above, graded as a STACK against float64.
3. The real kernels — softmax backward 10 passes -> 3, layer-norm backward 30 -> 4 — which take
   the ceiling to 1.265x on a round. `perf/bcx_bwbytes/FUSED_SOFTMAX_BW.md` has the tt-lang
   feasibility.

## What this does to the reach map — checked, not asserted

An earlier version of this file said the reach map was "unaffected by any of this". That was an
assertion. `reach.py --staleness` measures it, and the honest figure is **1.4 %**:

    1.073 GB  permutes         `_permute_back` swaps one kernel for another over the SAME
                               operands, so the byte count does not move at all
    0.539 GB  leading-axis sums `_tree_sum` replaces one `ttnn.sum` with a tree of adds, which
                               DOES move the census — upward, while real DRAM traffic stays flat
                               or falls, because the census sees the tree's depth-0 adds but not
                               `ttnn.sum`'s nested permute

So of 39.394 GB, only 0.539 GB would read differently on main's tree, and in the direction that
makes the census look worse rather than better. The headline survives intact: 82.5 % of the
softmax-backward bucket is multiply/subtract/typecast on the score tensor, which main has not
touched, and the n-cubed exponent comes from that tensor's shape rather than from any reduction.

A first cut of this check said 7.2 %. It counted `softmax_bw_inner`'s reduces, which are LAST-dim,
where `_reduce_to` gates the tree on `ax < len(gs) - 2`. Reading the gate is what separated a
caveat from a non-issue.

What IS affected is any sentence calling the softmax backward "unfused" — on main it has a fused
route, switched off, and off for a dtype reason this branch can lift.
