# The off switch BindCraft 2 already has, and the one it does not

Two different claims hide behind "BindCraft 2 cannot turn `exact_training` off". Only one is true,
and the measurement this row owes is blocked by neither.

## The measurement needs no edit to `tt_bio/bindcraft2.py`

`exact_training` is a dynamic-extent stack, not a construction-time argument. It pushes onto
`autograd._EXACT_TRAINING` and pops on exit (`autograd.py:1606-1618`), and `tape()` reads the top
of that stack **when it opens**, not when the model was built: `taped_ttnn.py:1237` is
`with ag._training_exact("tape"):`, and `_training_exact` is `exact(ops, owner) if ops else
nullcontext()` over `exact_training_ops()`, which returns `()` inside `exact_training(False)`
(`autograd.py:1621-1631`).

So an arm wraps the campaign, not the predictor:

    with ag.exact_training(False):
        campaign.run_campaign(settings, project, ...)

Every `tape()` BindCraft 2 opens inside that extent, in every round and every block, leaves
softmax and layer norm on the device ops. Proved at runtime, card-free, in `ARMED.json`
(`inside_tape_exact_training_False`: all six bindings shipped) and `REACH.json`
(`off_switch_closes_it`, through the module global AF2's triangle attention actually calls).

Because it is a stack, an interleaved A/B in one process is the natural form and the arms cannot
leak into each other as long as each arm is its own `with`. `round_ab.py` asserts that directly:
the OFF arm's counters must stay flat and the ON arm's must move.

## What BindCraft 2 genuinely lacks is a PRODUCT surface

`tt_bio/bindcraft2.py` exposes no parameter for it, so a user of the shipped API cannot make the
choice. That is a real gap and it is worth closing once the acceptance evidence justifies a
recommendation, as one parameter defaulting to today's behaviour:

    def predictor(*, trunk="device", card=None, checkpoints=None, resident=None,
                  blocks=EVOFORMER_BLOCKS, recompute=True, exact: bool = True):
        ...
        with autograd.exact_training(exact):
            with evoformer_on_device(evo):
                yield _factory(trunk="device", pool=pool, evoformer=evo)

and the same on `campaign_predictor`. Defaulting to `True` changes nothing for any caller.

**Not landed here, deliberately.** `wk/land-standing` has 281 lines pending in this exact file
(260 insertions, 21 deletions against the merge base) and is one parity-gate run from landing.
The change belongs behind it, on its tree, after the numbers exist.

An earlier version of this file said that branch was "12 commits behind main and deletes 79 files",
and used it as a second reason not to write against it. That reading was wrong and the reason it
was wrong is worth keeping. It came from a two-dot `git diff main branch`, which shows everything
main has that the branch lacks, including every file main gained after the fork. Measured today
against `origin/main` `b10cd9504`:

| reading | files | insertions | deletions | files called deleted |
|---|---|---|---|---|
| two-dot `git diff main branch` | 173 | 8,789 | 61,130 | 104 |
| three-dot, `git diff $(git merge-base main branch) branch` | 65 | 8,775 | 251 | **1** |

The branch is 28 ahead and 31 behind, merge base `ce44136bc`, and the one file it really deletes is
`perf/bcx_predictor/ttbio_predictor.py`. Among the 104 the two-dot reading called deletions are
`perf/bcx_exact/ARMED.json`, `LINEAGE.json`, `PCIE.json`, `PC_FLOOR.json` and this file -- this
row's own work, which landed on main after the fork. **When the question is "what would this merge
do", ask it against the merge base.** A two-dot diff against a moving main shows a branch deleting
work it has never seen.
