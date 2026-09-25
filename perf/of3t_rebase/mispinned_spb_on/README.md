# Not the result. The first ladder run, pinned to the wrong bias convention.

These six arms were taken with `--scale-pair-bias on`. The six published crop-64 arms they
replace were taken with `scale_pair_bias = false` — audited, not assumed, from their own
`shipped_config` blocks; it is only the three 48-block *stack* arms that were at `true`. So
these six moved the bias convention and the upstream revision together, which is the
two-moving-parts defect this row exists to undo.

Moved here rather than left beside the result with a note, because a label next to a wrong
number does not stop the number being quoted. The result is
`perf/of3t_rebase/instrument_a_bundle_043spb_block*_crop64_tb*.json`, scored in
`score_arms_043.json`.

Kept because they are evidence about the flag rather than about the revision. At block 47 the
shipped arm reads 9.439e-01 here and 2.013e-01 with the convention matched, so the bias scale
is worth a factor of 4.7 at depth 47 and almost nothing at depth 0 (1.207e-02 against
1.214e-02). The mis-pin also drops `attn_pair_bias.linear_z.weight` out of the bijection — 52
compared instead of 53 — which is how it was caught.
