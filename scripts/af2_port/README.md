# AF2-IG port — reference notes

AF2-IG is PXDesign's confidence filter: AlphaFold2 monomer `model_1_ptm`, run single-sequence
through ColabDesign with the design's own coordinates as the initial guess and as a template. It
is the pipeline's only success counter, so the port's acceptance metric is the filter's accept/
reject decision, not a structure metric.

That decision is `af2_easy`, four conditions and all four scored: pLDDT > 0.8, i_pTM > 0.5,
i_pAE < 0.35 from the complex pass, and bound-unbound RMSD < 3.5 from a second binder-only pass
(`filter_tolerance.py --stage monomer` plus `bound_unbound_rmsd.py`). On 50 real designs across two
targets the device arm flips none of them, pooled 0 of 50 with a Wilson CI95 of [0, 0.071].

Score a population in one process and the arm is the same arm for every design in it. That is a
property of the template memo's key, not an accident: without one, the second design in a process
gets the first design's template embedding, which for two backbones against the same target is a
binder 34 A away. `tests/test_af2_template_cache.py` pins the key.

`af2ig_spec.json` is the configuration, read from ColabDesign 1.1.3, PXDesignBench and
`params_model_1_ptm.npz` rather than inferred: every dimension, block count, runtime flag, filter
threshold and metric formula the port has to reproduce, each with the source file and line.

Two things in there are easy to get wrong and expensive to discover late:

- The trunk runs in **bfloat16**. `global_config.bfloat16` is a ColabDesign default, so the
  Evoformer and template stack compute in bf16 while the structure module and heads stay fp32. A
  reference built in fp32 throughout is not the thing PXDesign runs.
- The extra-MSA stack runs on an **all-zero mask**. Its pair track still transforms the pair
  representation, so it cannot be skipped, but the only MSA-to-pair path is the outer product
  mean, whose masked output is a per-channel constant.

The plan, the work packages and the acceptance checks live in
`~/.coworker/state/pxdesign-af2ig-port.md`. No JAX enters tt-bio: the reference arm is torch, and
the one-time cross-check against ColabDesign runs in an external environment
(`~/pxd_af2_cpu` on qb2) from a script that `tt_bio` never imports.
