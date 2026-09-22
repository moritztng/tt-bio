# Scored against openfold3 0.5.0, not the 0.4.3 this row reports

Every file here was scored against a reference side that resolved `openfold3` from
`/home/ttuser/of3t_refprec/pylibs` (0.5.0), not from `/home/ttuser/of3t_refprec/of3pkg043`
(0.4.3). The device side in them is fine; the reference side is a different model.

Cause: all five scripts in this directory installed the reference trees with
`sys.path.insert(1, p)` in a loop over `(OF3PKG, deps, pylibs)`. Three inserts at the same index
reverse the order, so `pylibs` went in last and landed at index 1, ahead of `of3pkg043` at index
3. Proof at process level, from `theirs_aa2.log` of the run archived here:

    /home/ttuser/of3t_refprec/pylibs/openfold3/core/model/primitives/attention.py:54: UserWarning

This matters because 0.4.3 and 0.5.0 are a different function at this boundary rather than a
different rounding: D120 measured 1.94959719e-05 (0.5.0) against 7.66979728e-01 (0.4.3), f32
against f32. `grads_f64_043.pt` and `of3t-modeltraj`'s `SCOPE_LADDER.json` are both on 0.4.3.

The `1 missing / 24 unexpected` load and the all-ones shared `layer_norm_z` reported in these
artifacts are a symptom of the same thing. 0.4.3 builds the shared norm only under
`use_cross_attention` (`diffusion_transformer.py:253`), so the main diffusion transformer has
none and each block's `AttentionPairBias` owns its own (`attention_pair_bias.py:107`, applied
156), which is exactly how this checkpoint stores it. Against 0.4.3 the load is total: 0
missing, 0 unexpected, 761 parameters, no rewiring.

Fixed in `refpath.py`: the dep trees are appended and only the tree under test is prepended, and
`assert_resolved()` reads `openfold3.__file__` and refuses if it did not come from `OF3PKG`.
Every artifact now records the resolved path as `ref_tree`, and `--score` refuses a reference
arm whose `ref_tree` is not the tree under test. Recorded as D149.
