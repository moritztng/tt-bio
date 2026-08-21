# Pass 8 predictions, written before the runs

Committed before `p8_b5.sh` produced a single number, so the batch-5 cells are scored against a
written-down prediction the way passes 4-7 were.

## Batch 5 with lever 8, qb2 card 0 (p300c, 12/14 tensix columns)

Lever 8 is a TRUNK lever and the trunk runs once per fold whatever the diffusion batch is (pass 6
measured 47.47 s in both the batch-1 and batch-5 cells at 768 aa, to 0.01 %). So its absolute
saving at batch 5 should be the same seconds it is at batch 1, and its RATIO should be smaller
because the wall is longer.

Base: the qb2 batch-1 folds at the shipped config, and the diffusion phase inside them
(512 aa 25.302 s fold / 3.622 s diffusion, 768 aa 58.6665 / 5.301, 1024 aa ~113.0 / 7.140).
Pass 6 measured the batch-5 rollout at 5.41x the batch-1 rollout at 512 aa with lever 7 in
(19.6129 against 3.6258), so scale the diffusion phase by 5.41 and leave everything else.

| aa | predicted fold, batch 5 | target | predicted ratio |
|---|---|---|---|
| 512 | 41.3 s | 34.204 | 1.21x |
| 768 | 82.1 s | 71.2036 | 1.15x |
| 1024 | 144.5 s | 127.5092 | 1.13x |

128 and 256 aa are predicted loosely (~7 s and ~15 s, 0.5x and 0.9x of target) because their
batch-1 cells are pass-3 numbers, taken before levers 6, 7 and 8.

Lever 8 at 768 aa batch 5: 2.63 s, the same absolute saving as at batch 1, i.e. 1.032x rather
than the 1.045x it is worth there.

## What would falsify these

A batch-5 fold that is not batch-1 plus 4.41 extra rollouts means the rollout is not the only
thing D multiplies: either residency, or a per-member trunk cost that pass 6's 0.01 % agreement
says does not exist. Either would be worth more than the ladder cell.
