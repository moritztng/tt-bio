# Not the 0.4.3 result. The pair-bias convention and the revision moved together.

These six are byte-identical to `perf/of3t_rebase/mispinned_spb_on/`, which `of3t-rebase`
quarantined for exactly that reason, and to
`perf/of3t_orchestrator/block47/instrument_a_bundle_043_block*_crop64_tb*.json`, which carries
the same bytes under a name that does not say so. They ran `--scale-pair-bias on`; all six
published crop-64 arms they would replace ran `scale_pair_bias = false`.

Kept because they are evidence about the flag, not about the revision: block 47 shipped reads
9.439e-01 here against 2.013e-01 with the convention matched, and block 0 reads 1.207e-02
against 1.214e-02 — the bias scale is worth 4.7x at depth 47 and nothing at depth 0. The mis-pin
also drops `attn_pair_bias.linear_z.weight` out of the bijection, 51 compared instead of 52.

The 0.4.3 result is `../instrument_a_bundle_043spb_*`.
