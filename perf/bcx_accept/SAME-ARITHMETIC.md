# `--exact off` is the shipped arithmetic, so the two populations differ only in their grader

Settled 2026-09-26 04:4xZ, card-free, at runtime rather than from source. Evidence:
`perf/bcx_accept/ARMED-643cabcf9.json`, produced by `bcx-exact`'s own `perf/bcx_exact/armed.py`
on commit `643cabcf9` — the tree the three qb1 acceptance arms run.

## The question this closes

`state/bcx/RATE.md` reports two populations. Population A ran `c014a2a7d`, which **predates** the
host float64 `exact_training` instrument (`502ed112e` is not an ancestor). Population B runs trees
that carry it, with `traj_arm.py --exact off`. If "off" were not exactly "absent", the two would
differ in arithmetic as well as in grader, and the Fisher comparison between them would be
confounded.

The orchestrator asked for this to be checked rather than assumed. It could not be read off the
arms: `traj_arm.py` writes `exact_counters` and `arm_held` only when the arm exits, and none of the
population B arms has exited.

## What was measured

    inside tape(), module default          softmax_installed true,  layer_norm_installed true
    inside tape(), exact_training(False)   softmax_installed FALSE, layer_norm_installed FALSE
                                           all six bindings (3 raw, 3 taped verbs) SHIPPED
    innermost_wins                         true -- an inner exact_training(True) does re-arm,
                                           so the outermost scope has to be the one that owns it
    counter_on_the_path                    true -- calling the INSTALLED binding moves
                                           EXACT_SOFTMAX_STATS["raw"] 0 -> 1
    bc2_harness_calls_exact_training       false
    opened_a_device                        false

`counter_on_the_path` is the part that matters most and is easy to skip: a counter that reads zero
proves nothing unless it is wired to the callable that actually gets installed. It is.

The only `exact_training` call site in the engine outside `autograd.py` itself is
`tt_bio/train/cli.py:251`, which is not on BindCraft 2's path. `tt_bio/bindcraft2.py` has zero
`install(` calls, so the documented escape hatch at `autograd.py:2761` — `install(exact_softmax=True)`
arms the exact softmax even inside `exact_training(False)` — is never taken.

## The conclusion

Under `--exact off` the exact host-float64 path is **not installed**, so a population B arm runs the
same ops as a tree from before the instrument existed. **Populations A and B are the same
arithmetic and differ only in which model grades acceptance** — BC2's device-pool folds against
BC2's own JAX folds.

That is what makes their agreement worth quoting: 2/7 against 2/11, Fisher exact two-sided
**p = 1.000**, costs per accepted design 20,742 / 21,028 / 22,472 s. With the arithmetic held
fixed, that comparison is about the grader alone, and the two graders do not disagree.

## One operational note

`armed.py` writes its output into its own worktree, `wt/bcx-exact`, and `bcx-exact` has concluded,
so fleet hygiene will eventually remove it. The copy here is the durable one.
