# Amendment to the OF3T training-outcome bar: which floor the tolerance uses

Written 2026-09-26 18:5x UTC, BEFORE arm B produced a number. Arm B was launched at 18:55:34Z and
was still in its discovery forward when this was written; its `eval_before` had reproduced arm A's
held-out losses to the last digit and nothing else about it existed.

## What changed

The original bar said the tolerance is the LARGER of the replicate floor and the seed floor. Both
are now measured, at N=2, on the held-out mean loss:

| | |
|---|---|
| movement, arm A seed 0 | 6.802440 -> 9.629875, **+2.827435** |
| replicate floor (same seed, two runs) | 9.629875 vs 9.630438, **5.63e-4** |
| seed floor (seed 0 vs seed 1) | 9.629875 vs 6.803615, **2.826260** |

The seed floor is the same size as the movement, and for a good reason rather than a bad one: the
seed sets the batch ORDER, and at two steps *which two samples you drew* is the whole of the
outcome. Seed 1 drew a pair that moved the held-out metric by 0.001175 where seed 0's pair moved
it by 2.827435.

## So the tolerance is the REPLICATE floor, 5.63e-4

Taking the larger would make the test unable to fail: a tolerance of 2.83 on a movement of 2.83
passes any two arms whatsoever, including two that trained different models. A bar that cannot
fail is not a bar.

The seed floor is the variance of a nuisance factor **the A/B design holds fixed**. Arm A and arm B
run the same seed, the same corpus and the same order by construction, so the data draw cannot
differ between them and its variance does not belong in their tolerance. The replicate floor is
what is left when everything the design controls is controlled, and it is therefore the yardstick
for a difference the design did not control -- which is the arithmetic, and is the variable under
test.

This TIGHTENS the bar, by a factor of 5000, and it is written down before the arm it judges.

## What gets reported either way

The pass/fail against 5.63e-4, and the gap as a FRACTION OF THE MOVEMENT. A difference can exceed
the replicate floor and still be small against what the training did, and those are different
statements: the first says device precision changes the trajectory more than re-running the same
arm does, the second says whether it matters. Both go in the verdict.

The seed floor stays in the record as the reason N=2 cannot be read as a statement about training
in general. At two steps the draw dominates; a claim about what OF3T training converges to needs
the arm to run long enough for the draw to average out, and that is blocked on the ~0.98 GB/step
DRAM growth that caps the arm at 6 steps.
