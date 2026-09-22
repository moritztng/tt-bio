# Five paths the frozen batch could not reach, fired

The SS6 coverage census read 4 of 11 conditional paths covered. Five of the seven that did not
fire were a batch selection problem: four of its five reasons are properties of 5nw3, the only
structure in the frozen bundle, and the fifth attributes a per-target gate to a per-dataset
default. Picking a different target inside the same `(stage, dataset)` pair fires all five.
`COVERAGE_UNION.json` reads **9 of 11**.

The two that stay uncovered are `diffusion_rollout` and `model_forward`. They need the OF3
training forward wired in `tt_bio/train/`, and no choice of batch reaches them.

## What each path needed

| path | target | why 5nw3 could not do it |
|---|---|---|
| `templates` | 1kvu, 6 template ids on chain 1 | 5nw3's four template slots are mask-zero |
| `nucleotide` | 4hj5, protein + DNA at 2.04 Ang | 5nw3 is one protein chain plus Fe and Na |
| `disabled_parameters` | 5oid at 4.6 Ang | nothing to do with the dataset: `set_loss_weights` zeroes the confidence weights when the resolution leaves `[0.1, 4.0]`, and 5nw3 is inside it |
| `bond` | 4g5j chain 1, settled by `of3t-bondcov` | no inter-token bond on 5nw3 |
| `multichain_permutation` | both batches | it was never broken; a harness removed the key |

Every target comes from upstream's own `training_cache_with_templates.json`, so the selection is
theirs and not ours. The corpus is four structures, 525 files, 82.6 MB, digest
`da6442618bafea83a324abfc38aa73a4226671e225d97f63516192f6fe7c3b54`.

## A path is covered when a gradient moves

One forward, two losses, two backwards, differenced per parameter tensor. openfold3 0.4.3,
torch 2.11.0+cpu, float32, seed 20260921, crop 384, CPU on qb1.

| path | control | share of the squared gradient norm | tensors moved |
|---|---|---|---|
| `templates` | the two template masks zeroed | 4.708550e-02 | 4,168 of 4,170 |
| `nucleotide` | DNA tokens relabelled as protein for the loss | 2.784922e-01 | 4,164 of 4,170 |
| `disabled_parameters` | the four confidence weights restored to 1e-4 | 5.738758e-07 | 243 of 4,170 |

`template_embedder` moves 6.589196 of its own squared gradient norm, more than the whole of it,
because the arm with the masks zeroed puts a much larger gradient there than the arm that reads
real templates. `disabled_parameters` moves exactly the 243 confidence-head parameters and
nothing in the trunk, which is the shape a correct result has.

Both feature edits are bit-exact no-ops on a batch their path cannot act on: the nucleotide edit
moves 0 of 350 tensors on protein-only 1kvu, the template edit 0 of 424 on a crop that drew no
template. `covpath_gradient.py` also refuses to run an arm whose control cannot move rather than
reporting a zero that was zero by construction.

## The confidence head holds a zero gradient, not a missing one

`PREDICTION.md` expected `grad is None` on the 243 disabled parameters, reading that off
LEDGER R6. Measured: 0 of 243 are None on either arm and all 243 are exactly zero on the arm
where the gate fires. Autograd still builds the confidence terms and multiplies them by zero.
R6's `None` is the runner's own list of disabled parameter NAMES, used for counting gradients
across ranks and for dropping them from the clip norm, and it is not what autograd produces.

That has a consequence. Because those gradients are exactly zero on the batch that trips the
gate, dropping them from `grad_manager._clip_grads`' `params_enabled` changes the clip norm by
nothing. The recorded "their norm 15.099, ours 92.493" comes from a synthetic case with one large
disabled tensor.

## The permutation alignment was never broken

Ten arms, seven completing and three configured to break. The alignment completes inside the real
checkpointed forward on both batches the campaign holds, moving 8 ground-truth atoms on 5nw3 and
74 on 4g5j, and it resolves atom symmetry that the naive fallback leaves alone: with the
prediction flipped in 58 ref spaces it moves 146 atoms, 2.386 Ang from naive.

`KeyError: 'ref_space_uid_to_perm'` needs the key removed first. `model.py:670` pops it off the
caller's batch dict and restores it onto a new one, so a second forward over the same dict finds
it gone, `pop(..., None)` stores `None`, and `reshape_per_sample_inputs` re-adds it only when it
is not None. `permutation_alignment.py:1412` then raises. Arm J reproduces that by calling the
forward twice; arm F reproduces the collation defect `of3t-bondcov` hit. The key itself is
written by `conformer.py:181` on every arm here, with 56 ref spaces on 5nw3 and 197 on 4g5j.

## Files

| file | what |
|---|---|
| `PREDICTION.md` | the five expectations and their source evidence, committed before the first run |
| `covpath_probe.py`, `presence_n4.json` | featurisation only: masks, molecule-type counts, loss weights, the disable gate |
| `covpath_gradient.py`, `covpath_gradrun.sh` | the gradient arms |
| `gradient_*.json` | one per arm |
| `null_control.py`, `null_control_*.json` | each edit applied where its path is absent |
| `permalign_rerun/` | the ten permutation arms re-run here |
| `make_coverage_union.py`, `COVERAGE_UNION.json` | the union, derived from the artifacts; no figure is typed in |
| `scan_cache.py`, `scan_candidates.py` | how the targets were found in the 180,975-structure cache |

`COVERAGE_UNION.json` is not a re-emission of `coverage_census.json`. That artifact belongs to
`of3t-gradients` and that row concluded; this one cites it by path and sha256 and says per path
whether it confirms, extends or contradicts it.
