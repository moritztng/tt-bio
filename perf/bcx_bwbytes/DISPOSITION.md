# What of this branch should land, and what must not

Checked 2026-09-26 against `origin/main` at the symbol level, not the commit level — this branch
is based on `wk/bcx-tmplseam`, which forked before a large amount of main's backward work, and it
is 228 commits behind.

## Three of the five levers here are already on main, under other names

| this branch | main | main's default |
|---|---|---|
| `autograd._pairwise_sum0` + `LEADING_SUM_TREE_ROWS` | `autograd._tree_sum` + `TREE_REDUCE` | **True, live** |
| `reblock_permute.permute_via_reblock` + `taped_ttnn.PERMUTE_BW_REBLOCK` | `taped_ttnn._permute_back` + `REBLOCK_PERMUTE_BW` | **True, live** |
| `autograd.softmax_bw_dx` + `SOFTMAX_BW_ROUTE="moreh"` | `autograd.softmax_bw` + `SOFTMAX_BW_FUSED` | **False, built but OFF** |

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

## Two levers here are genuinely new

* `autograd.SOFTMAX_BW_DTYPE="bf16"` — the softmax backward on bf16 operands, precision from fp32
  accumulation in DST rather than fp32 tensors in DRAM.
* `autograd.FANIN_WIDEN_INCOMING=False` — a bf16 contribution into a mixed-dtype `ttnn.add`, fp32
  accumulator kept.

These are the two the reach map prices at **13.4 % of the backward's bytes, 1.071x on the round**,
and nothing on main does either. They should be ported onto main's names — `softmax_bw` rather
than `softmax_bw_dx` — not merged as a replacement for them.

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

The reach map itself is unaffected by any of this: it is a census of DRAM bytes by issuing site,
and the three duplicated levers move device milliseconds and layout kernels rather than the
buckets it measures. What IS affected is any sentence calling the softmax backward "unfused" —
on main it has a fused route, switched off.
