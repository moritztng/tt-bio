# of3t-tapeattn — pre-registration

Committed BEFORE the first device arm. Nothing below was written after a reading.

## The object

`of3t-shapekey` (5b75991d0) left D191 as a **per-block multiplicative factor of about 2.9x on the
single track's cotangent** between padded width 64 and padded width 384, measured as the growth of
every leaf in an `attn_pair_bias` sub-block together (block 44 median 2.9456, block 4 median 2.9272,
block 0 median 1.8463, block 47 median 1.0789) rather than as one leaf misbehaving. Its route census
also showed `_fp32_softmax_attention` takes 384 of 384 calls from `TriangleAttention` and 0 from
`AttentionPairBias` at both widths, so the single track's attention does not run through the
`tenstorrent.py` helper the earlier briefs charged. It runs on the tape.

## Where I expect it to run, and what the census has to show for that to be true

Read off the source, NOT yet counted (D196: a route read in source is not a mechanism). The
pairformer's `AttentionPairBias` unfused branch in `tt_bio/tenstorrent.py:8670-8681` is
`transpose` -> `batched_matmul(q, k^T)` -> `add_(logits, z)` -> `multiply_(logits, scale)` ->
`ttnn.softmax` -> `batched_matmul(probs, v)`. Under an open tape each of those reaches a verb in
`tt_bio/taped_ttnn.py`: `_v_transpose`, `_v_matmul`, `_binary` for `add_` and `multiply_`,
`_v_softmax`, `_v_matmul`. The census in deliverable 2 counts firings per verb and per shape from
INSIDE the verb; if it does not show ~48 `softmax` backward firings at `[1, 16, N, N]` per taped
pass on this arm, this paragraph is wrong and the prediction below is scored as refuted, not
repaired.

## Candidate 1 (primary) — the `softmax` verb's backward

`_v_softmax` -> `autograd.softmax_bw_inner`: `dx = y * (g - sum_j g_j y_j / sum_j y_j)`. It is the
only verb in that chain whose backward carries a REDUCTION over the axis that grows, and it is a
cancelling one: `g - inner` subtracts a weighted mean of `g` from `g`. Going 64 -> 384 multiplies
the number of summands by 6 and adds 328 masked columns whose `y` is near zero and whose `g` is not.

Predicted: `softmax` carries the largest single share of the per-block injected error at 384, and
its share grows between the widths.

Falsified if: the per-verb injection table (each verb's own backward on device operands, differenced
against the same closure recomputed in float64 on the same operands) puts `softmax` under a quarter
of the block's injected error at 384; or if `dev_cot.py --lever softmax_fp32` at 384 leaves the
per-block factor above 2.5x; or if the census shows the verb does not fire at that shape.

## Candidate 2 — the two `matmul` verbs

`q @ k^T` and `probs @ v`, and their four backward matmuls. The accumulation depth over `k_len`
goes from 2 K-tiles to 12, which is `of3t-shapekey`'s own residual candidate list.

Predicted: second largest share, and unlike candidate 1 its share does NOT grow much with width,
because a matmul reduction is not cancelling.

Falsified if: `matmul` is the largest injector at 384 (then it, not softmax, is the answer), or if
its injection ratio 384/64 exceeds `softmax`'s.

## Candidate 3 — the elementwise verbs and the transpose

`add_`, `multiply_`, `transpose`, `_v_create_qkv_heads` / `_v_concat_heads` if the census finds them.
No reduction, so no width dependence beyond element count.

Predicted INERT: together under 10 % of the block's injected error at 384, ratio 384/64 under 1.2x.

Falsified if: any of them is above 10 %.

## Candidate 4 — the leaf reductions, predicted to be the WRONG place

`_sum_leading` in `_taped_layer_norm` and `linear` sums over 6x more rows at 384. It is a leaf-side
reduction and cannot move a cotangent, so if the cotangent ladder shows the ~2.9x, candidate 4 is
excluded by construction rather than by an arm. Stated so that a later reading of the leaves cannot
be presented as a confirmation of this row.

## The ladder's shape, pre-registered

`of3t-bwdaccum`'s discriminator: cotangent flat at the bf16 floor at every rung means a wrong leaf
backward; degrading with depth means an injection per block. From shapekey's four blocks (47:
1.0789, 44: 2.9456, 4: 2.9272, 0: 1.8463) the factor is already near its full size four blocks into
the backward and does not keep climbing.

Predicted: at 384 the ds over-floor ratio rises sharply over the first few rungs (47 -> ~44) and
then runs FLAT, i.e. branch C, a regime-dependent injection that saturates, not branch A. The
per-block factor is then a factor on a quantity injected early, not a factor applied 48 times.

Falsified if: the over-floor ratio rises monotonically across the whole ladder with end-to-end
growth >= 4.0 (branch A), or if it never leaves the floor (branch B, which would put the object back
at the leaves and refute this row's premise).

## Floors, before any arm is scored

1. A/A determinism: two independent A64 device runs, output compared byte for byte (sha256). If it
   is not exactly 0, every arm delta is read against whatever it is and no smaller delta is claimed.
2. Wall clock: the spread between those same two byte-identical runs. `of3t-shapekey` measured 9.1 %
   and declined to read a 7.7 % difference as a difference. Same rule here.
3. Every float64 reference and every bf16 floor is built on the SAME host as the arm it scores
   (D189: two c64 floors off different boxes read 0.4007 and 0.3739, 7.2 % apart).

## What this row will NOT claim

It will not claim a fix, and nothing it measures moves a shipped default. A verb identified as the
carrier is an attribution, not a repair; any change to `tt_bio/` here is instrumentation that must
be shown inert on the shipped path.
