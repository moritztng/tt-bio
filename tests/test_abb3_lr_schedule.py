"""The learning-rate schedule the reproduction runs, and the property that makes a resume exact.

`params.yaml`'s `optimiser:` block is RAdam at 5e-4 *plus* `T_0: 50`, `T_mult: 1` and
`eta_min: 0`, which `lightning_module.py:144` turns into `CosineAnnealingWarmRestarts` stepped
once per epoch. The reproduction's recipe once carried lr and weight decay alone, so nothing
wrapped the optimizer and the run held a constant 5e-4: 1.96x upstream's mean over a full
schedule, and none of its troughs. Four things are checked, and each one is a way that failure
stayed invisible:

1. the recipe carries the whole `optimiser:` block, compared against `params.yaml` itself
   rather than against a second copy of its numbers;
2. the lr is a pure function of the global step, so a resume lands where the step index says
   and no scheduler state has to survive in the checkpoint;
3. it agrees with upstream's per-epoch staircase at the epoch boundaries, which is the one
   place a per-step schedule and a per-epoch one are comparable;
4. the cycle is 50 epochs: the trough falls at the end of it and the restart is a full reset to
   the base lr.

No device: all of it is host arithmetic.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tt_bio.train.abb3_run import CosineRestartsByStep
from tt_bio.train.abodybuilder3_step import RECIPE
from tt_bio.train.sharding import steps_per_epoch

#: Upstream's own config, as they ship it. Machine-local, so the comparison skips where the
#: checkout is absent rather than asserting numbers copied out of it.
PARAMS = Path("/home/ttuser/abb3_src/ABodyBuilder3/params.yaml")

#: 8,395 training structures at global batch 64 with the short final batch kept. The released
#: checkpoint's own `global_step` 193,512 is 132 x 1466, which is where the 132 is confirmed.
STEPS_PER_EPOCH = 132


def sched(steps_per_epoch_=STEPS_PER_EPOCH) -> CosineRestartsByStep:
    p = torch.zeros(1, requires_grad=True)
    return CosineRestartsByStep(torch.optim.RAdam([p], lr=RECIPE["lr"]),
                                steps_per_epoch=steps_per_epoch_, T_0=RECIPE["T_0"],
                                T_mult=RECIPE["T_mult"], eta_min=RECIPE["eta_min"])


@pytest.mark.skipif(not PARAMS.is_file(), reason=f"{PARAMS} is not on this host")
def test_the_recipe_carries_the_whole_optimiser_block():
    yaml = pytest.importorskip("yaml")
    block = yaml.safe_load(PARAMS.read_text())["optimiser"]
    assert block["optimiser"] == "RAdam", (
        f"params.yaml selects {block['optimiser']!r}, and the schedule this module installs is "
        f"the one lightning_module.py builds on the RAdam branch")
    for key in ("lr", "weight_decay", "T_0", "eta_min", "T_mult"):
        assert float(RECIPE[key]) == float(block[key]), (
            f"RECIPE[{key!r}] is {RECIPE[key]} and params.yaml says {block[key]}. The three "
            f"scheduler keys went missing once and the run trained at a constant lr")


def test_the_number_of_steps_in_an_epoch_is_the_one_the_batcher_counts():
    assert steps_per_epoch(8395, 64, drop_last=False) == STEPS_PER_EPOCH
    assert STEPS_PER_EPOCH * 1466 == 193_512, (
        "upstream's released checkpoint sits at global_step 193,512, which is 1,466 epochs of "
        "132 steps only if the short final batch is kept")


def test_the_learning_rate_is_a_pure_function_of_the_global_step():
    """So a resume needs no scheduler state, which is why the fractional form was taken.

    A call-counted scheduler restarts at the top of the cosine on every resume, and this run
    expects 9 to 47 of them.
    """
    walked = sched()
    for step in range(1, 4001):
        value = walked.set_step(step)
    jumped = sched().set_step(4000)
    assert jumped == value, (
        "the lr after walking to step 4,000 differs from the lr set by jumping there, so a "
        "resumed run would not be on the same schedule as an uninterrupted one")
    # And out of order, which is what a resume onto an earlier checkpoint does.
    assert sched().set_step(1) == pytest.approx(RECIPE["lr"], rel=1e-15)
    back = walked.set_step(1)
    assert back == pytest.approx(RECIPE["lr"], rel=1e-15)


def test_it_agrees_with_upstreams_per_epoch_staircase_at_the_epoch_boundaries():
    p = torch.zeros(1, requires_grad=True)
    opt = torch.optim.RAdam([p], lr=RECIPE["lr"])
    upstream = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=RECIPE["T_0"], T_mult=RECIPE["T_mult"], eta_min=RECIPE["eta_min"])
    ours = sched()
    for epoch in range(120):
        theirs = opt.param_groups[0]["lr"]
        mine = ours.set_step(epoch * STEPS_PER_EPOCH + 1)
        assert mine == pytest.approx(theirs, rel=1e-12), (
            f"at the first step of epoch {epoch} the schedules disagree: {mine} against "
            f"upstream's {theirs}")
        upstream.step()


def test_the_cycle_is_fifty_epochs_and_the_restart_resets_to_the_base_lr():
    cycle = RECIPE["T_0"] * STEPS_PER_EPOCH
    assert cycle == 6600
    s = sched()
    assert s.set_step(1) == pytest.approx(RECIPE["lr"], rel=1e-15)
    assert s.set_step(cycle // 2 + 1) == pytest.approx(RECIPE["lr"] / 2, rel=1e-3)
    assert s.set_step(cycle) < 1e-8, "the trough sits at the end of the cycle"
    assert s.set_step(cycle + 1) == pytest.approx(RECIPE["lr"], rel=1e-15), (
        "a warm restart returns to the base lr, which is what makes the troughs a feature of "
        "the recipe rather than the end of it")


def test_the_schedule_round_trips_through_the_file_a_reader_checks_it_with():
    spec = sched().spec
    assert CosineRestartsByStep.load(spec).set_step(3_000) == sched().set_step(3_000)
    assert spec["steps_per_epoch"] == STEPS_PER_EPOCH and spec["lr"] == RECIPE["lr"]


# ------------------------------------------------------------------ the wiring, not the class

class _FakeStep:
    """The host half of a ``TrainStep``: a mirror, an optimizer, a dropout RNG and the recipe.

    The defect was never in the schedule, it was in nobody calling one, so this checks the loop
    and not the arithmetic: what lr does the optimizer actually see on step ``k``, and does the
    history say so.
    """

    def __init__(self, n=2, size=4, accumulate=16):
        from tt_bio.abodybuilder3 import Dropout
        self.mirror = [torch.zeros(size, size, requires_grad=True) for _ in range(n)]
        self.params = []
        self.optimizer = torch.optim.RAdam(self.mirror, lr=RECIPE["lr"],
                                           weight_decay=RECIPE["weight_decay"])
        self.dropout = Dropout.__new__(Dropout)
        self.dropout.rate, self.dropout.calls = RECIPE["dropout_rate"], 0
        self.dropout.generator = torch.Generator().manual_seed(0)
        self.recipe = dict(RECIPE)
        self.accumulate = accumulate
        self.loss_terms = {}
        #: The lr in the parameter group at the moment the update is taken, per step.
        self.seen = []

    def step(self, micros):
        from tt_bio.train.abodybuilder3_step import StepTiming
        self.seen.append(float(self.optimizer.param_groups[0]["lr"]))
        for i, m in enumerate(self.mirror):
            m.grad = torch.full(m.shape, 0.01 * (i + 1))
        self.optimizer.step()
        return {"loss": 1.0}, StepTiming(micro_batches=len(micros))


class _FakeData:
    def __init__(self, n=8395):
        self.n = n

    def __len__(self):
        return self.n

    def batch(self, indices):
        return {"indices": list(indices)}


def test_the_run_loop_drives_the_schedule_and_writes_it_into_every_row(tmp_path):
    """The check the missing scheduler would have failed: read the lr out of the run.

    Both halves matter. ``seen`` is what the optimizer's parameter group held when it stepped,
    so a scheduler that is built and never called fails here even though every test of the
    class above passes. The history is the other half: the lr went unlogged for the run's first
    444 steps, which is why the defect survived two readers.
    """
    import json

    from tt_bio.train.abb3_run import RunConfig, run

    step = _FakeStep()
    cfg = RunConfig(out_dir=tmp_path, steps=6, global_batch=64, micro_batch=4,
                    rendezvous=tmp_path / "rv", checkpoint_minutes=1e6)
    out = run(step, _FakeData(), cfg, resume=False)

    want = [sched().set_step(k) for k in range(1, 7)]
    assert step.seen == want, (
        "the lr the optimizer stepped at does not follow the recipe's cosine, so the schedule "
        "is built but not driven")
    rows = out["history"]
    assert [r["lr"] for r in rows] == want
    assert all(r["grad_norm"] > 0 for r in rows), "the gradient norm is not reaching the history"
    assert step.seen[0] == pytest.approx(RECIPE["lr"], rel=1e-15)
    assert step.seen[-1] < step.seen[0], "the cosine is flat, which is the defect itself"

    spec = json.loads((tmp_path / "schedule.json").read_text())
    assert spec["steps_per_epoch"] == STEPS_PER_EPOCH
    assert CosineRestartsByStep.load(spec).set_step(6) == want[-1]
