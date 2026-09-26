# OF3T training-outcome grade: the bar, fixed before any graded arm ran

`of3t-p10trainout`, opened 2026-09-26. OF3T ships device-resident and is graded on training
OUTCOME, not on the pre-registered tensor clause (`state/ask-11666-decision.md`). This file is
the bar, committed here rather than only in `~/.coworker/state/`, because that tree is
gitignored, shared with every other row and committed by whoever runs `git add -A` next.

**Arm A** is the device-native trunk: the card's own bf16 softmax and layer norm,
`exact_training(False)`, which is what ships. **Arm B** is the same run with the float64 host
softmax and layer norm installed. Everything else is identical, including the seed.

## The harness is the shipped loop, not the step harness

`perf/of3t_stepfloor/fullstep.py` cannot grade this. Its ground truth is the model's own
prediction plus noise, so its loss value and gradient direction are meaningless by construction
and its own docstring says so. It times a step; it does not grade one. The graded arms run
`tt_bio.train.recipes.train_loop` through `perf/of3t_p10trainout/trainarm.py`, on deposited
structures.

## Fixed

| | |
|---|---|
| start checkpoint | `of3-p2-155k.pt` |
| train set | `training_cache_with_templates_subset_8.json`: 1kvu 1wyc 1xbs 210l 2wig 3wnm 4ky2 5ron, which expand to 292 datapoints |
| held-out set | `validation_cache_with_templates_subset_4.json`: 7fb8 7kud 7ohe 7vus, one datapoint each. Never in a training batch, in either arm |
| crop / stage | 384 tokens, `initial_training`, `--loss-shape model` |
| corpus | drawn once by `featurise.py` from upstream's `datapoint_probabilities`, with replacement, `--draw-seed 20260926`. The manifest carries the step order and a sha256 per file; both arms replay the files |
| data order | `sharding.batches(len(dataset), global_batch=1, steps=N, seed=S)`, reproducible from those three numbers alone |
| optimizer | `train="weights"`, AdamW at OpenFold3's own settings: lr 3e-4, betas (0.9, 0.95), weight decay 0, `plateau_until` 50000, per-sample clipping at the shipped 10.0 |
| warmup | `warmup_steps=0`. The 1000-step default leaves the rate at 9e-6 by step 30, and two arms that never moved agree perfectly |
| forward | `rollout` 20, `num_cycles` 1 |
| metric | `af3_loss` at `initial_training` weights on the 4 held-out targets, diffusion draws fixed by a recorded seed |
| evaluator | one evaluator for both arms, independent of either. Neither arm grades itself: the device's own softmax and layer norm are the variable under test, so they cannot also be the instrument |
| tolerance | the gap between the arms must not exceed the larger of the replicate floor and the seed floor, both measured on arm A |

## The two floors, and why both

**Replicate floor**: arm A twice, same seed, same order. Measured on the first two-step arms and
it is not zero. Identical config and identical step-0 loss (1.352114 both) gave step 1 at
1.583343 and 1.611312, so the divergence enters with the first optimizer step.

**Seed floor**: arm A twice, seeds differing. A seed floor below the replicate floor would mean
the seed is not the variable being measured, and the pass reports that instead of a tolerance.

## The precondition that makes the grade mean anything

Two arms that both failed to move agree perfectly, so the run is gradable only if
`|val(A) - val(start checkpoint)|` exceeds the floor. If it does not, N or the rate was too
small, and the pass reports that instead of a verdict.

Direction is recorded, not scored. Arm A landing BETTER than arm B by more than the floor is
not a pass either; it is a result that needs its own explanation.

## Preconditions asserted by the harness, per arm, into the artifact

1. `ce0f78d60` is an ANCESTOR of the tree (`trainarm.py` refuses the arm otherwise). It is the
   slot-ordering fix, without which `rebind()` writes AdamW's new weight into a dict the forward
   never reads and 270 diffusion weights train on nothing. `params_with_grad == 2944` stopped
   being sufficient at `42a664fa6`: the AdamW write-skip drops writes that round away, so fewer
   handles move and a tree that still has the bug reads 2,944. A cherry-pick does not satisfy
   this and cannot -- three shas carry the same two edits -- so the branch is merged.
2. `exact_training_ops()` is read INSIDE the run and recorded. `--exact on` must show both ops,
   `--exact off` must show none. The switch has no environment variable, so an arm that believed
   its own argument and never looked would be the whole experiment.

## What the grade will report whatever the verdict

The cumulative displacement ratio. The first two-step arm read **0.7468** against the optimizer's
(0.9, 1.1) band: the fp32 master moved 2.9485 where the bf16 device weight the forward reads
moved 2.2021. A quarter of the update dies in the cast at lr 3e-4. Both arms store weights in
bf16 on the card so it cannot bias A against B, but it is part of what device-resident training
means.

## Verdict

GO if arm A is inside the tolerance and the movement control cleared. NO-GO names the layer,
softmax or layer norm, that the gap traces to, and hands it to `of3t-p10exact`. Shared `tt_bio/`
code is unified, not per-model: tell `bcx-perf10`.
