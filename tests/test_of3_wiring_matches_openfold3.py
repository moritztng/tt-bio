"""The four wiring divergences `of3t-traj20` found, as checks that can fail.

A wiring divergence is a component applied in the wrong order, called at the wrong time, or
not called at all, and no per-parameter agreement test can see one: every piece here was
already correct in isolation and three of the four had passing unit tests beside them while
the assembled recipe diverged. So these assert CALL SITES and DEFAULTS, which is where the
four actually lived.

1. `recipes.py` took `AdamW`s 0.01 weight decay where upstreams `configure_optimizers`
   builds `torch.optim.Adam` with none.
2. `AdamW.clip_and_accumulate` implemented upstreams per-sample clipping and had zero
   callers in `tt_bio/`, against three in the test that proved it correct.
3. `AdamW` wrote `self.participation` and read it zero times; upstreams
   `_sync_and_average_grads` divides each parameter by its own count.
4. `recipes.py` selected Protenixs schedule family where OpenFold3 ships AF2s.
"""

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

pytest.importorskip("ttnn")

from tt_bio.train import optim as tt_optim                           # noqa: E402
from tt_bio.train.optim import AdamW                                 # noqa: E402
from tt_bio.train.recipes import train_loop                          # noqa: E402


def _defaults():
    return {k: v.default for k, v in inspect.signature(train_loop).parameters.items()}


def test_the_recipe_carries_no_weight_decay():
    """Upstream `runner.py:852-860` builds a plain Adam. Removing it alone moved the 20-step
    endpoint from 2.938e-01 to 1.105e-01 and the growth exponent from +0.406 to -0.152."""
    assert _defaults()["weight_decay"] == 0.0
    src = inspect.getsource(train_loop)
    assert "weight_decay=weight_decay" in src, "the argument exists but is not passed to AdamW"
    # The class default stays AdamW`s own. A class called AdamW whose decay defaults to zero
    # is a worse lie than a recipe that names the value its reference recipe uses.
    assert inspect.signature(AdamW).parameters["weight_decay"].default == 0.01


def test_the_recipe_pins_openfold3s_second_moment_horizon():
    """The fifth divergence, and the one that survived closing the other four.

    `AdamW`s class default is Adams usual `(0.9, 0.999)`; OpenFold3 runs `(0.9, 0.95)`
    (`model_config.py:143-146`, read by `runner.py:855-860`). A twenty-times-longer
    second-moment horizon moves no gradient, no loss and no norm -- it changes the SIZE of
    every update, which is why a per-parameter gradient check cannot see it. Over a 20-step
    trajectory with bit-identical gradients on both sides it was a uniform 0.9 % excess, and
    passing the right betas took d_2 from 9.095550e-03 to 4.884661e-06.
    """
    assert _defaults()["betas"] == (0.9, 0.95)
    assert "betas=betas" in inspect.getsource(train_loop)
    assert inspect.signature(AdamW).parameters["betas"].default == (0.9, 0.999)


def test_the_recipe_selects_openfold3s_schedule_family():
    """`plateau_until` is the whole difference between AF2`s schedule and Protenix`s. Over
    0..200,004 the two families agree on 200,004 of 200,005 steps and disagree at 50,000, so
    a 20-step window cannot see this and only the whole domain can."""
    assert _defaults()["plateau_until"] == 50000
    assert "plateau_until=plateau_until" in inspect.getsource(train_loop)


def test_per_sample_clipping_has_a_caller():
    """The defect was not a wrong implementation, it was no call. So this is a call-site
    assertion and it is the only kind that could have caught it."""
    src = inspect.getsource(train_loop)
    assert "opt.clip_and_accumulate()" in src
    # One sample at a time, or there is nothing per-sample about it.
    assert "dataset.batch([index])" in src
    # The tape is cleared after the last sample, so `replicas()` is empty at the step and the
    # axis reduces the accumulator rather than whichever sample happened to run last.
    assert src.index("opt.zero_grad()") < src.index("opt.step(")


class _HostValue:
    def __init__(self, arr):
        self.arr = np.ascontiguousarray(arr, dtype=np.float32)
        self.dtype = "float32"

    def device(self):
        return None


class _Param:
    def __init__(self, arr):
        self.value = _HostValue(arr)
        self.grad = None


@pytest.fixture
def host_stubs():
    orig = (tt_optim.to_host, tt_optim.to_device)
    tt_optim.to_host = lambda t, dtype=None: (
        t.arr if isinstance(t, _HostValue) else orig[0](t, dtype=dtype))
    tt_optim.to_device = lambda arr, device, dtype=None, layout=None: _HostValue(arr)
    yield
    tt_optim.to_host, tt_optim.to_device = orig


def test_step_divides_each_parameter_by_its_own_participation_count(host_stubs):
    """Upstream `grad_manager.py:225-232`. The counts genuinely spread -- their runner
    disables the confidence head on any sample whose confidence weight is zero, which
    `initial_training.yml` does on 4 of its 5 datasets -- so this is not a uniform scaling
    and Adam does not cancel it.

    Read off `last_grad_norm`, which `step()` computes from the accumulator AFTER the
    division. Two tensors with identical per-sample gradients and different counts: averaged,
    both land at ||g||; summed, `a` lands at 2||g|| and the norm is sqrt(5) instead of
    sqrt(2) times ||g||.
    """
    g = np.full(16, 0.25, np.float32)                       # ||g|| = 1
    params = {"a": _Param(np.zeros(16, np.float32)), "conf": _Param(np.zeros(16, np.float32))}
    opt = AdamW(params, lr=1.8e-3, weight_decay=0.0, clip_norm=0.0)

    params["a"].grad = g.copy()
    params["conf"].grad = g.copy()
    opt.clip_and_accumulate()                               # sample 0: both active
    opt.zero_grad()
    params["a"].grad = g.copy()
    opt.clip_and_accumulate()                               # sample 1: the head is disabled
    opt.zero_grad()

    assert opt.participation == {"a": 2, "conf": 1}
    opt.step()
    assert opt.last_per_sample is True
    assert opt.last_grad_norm == pytest.approx(np.sqrt(2.0), rel=1e-6), (
        "step() used the SUMMED accumulator; upstream divides each parameter by its own "
        "participation count before the optimizer sees it")


def test_a_uniform_count_is_still_divided(host_stubs):
    """The degenerate case has to be right too: every parameter in every sample means the
    divisor is the batch size, which is what an accumulation step is supposed to do."""
    g = np.full(16, 0.25, np.float32)
    params = {"a": _Param(np.zeros(16, np.float32))}
    opt = AdamW(params, lr=1.8e-3, weight_decay=0.0, clip_norm=0.0)
    for _ in range(4):
        params["a"].grad = g.copy()
        opt.clip_and_accumulate()
        opt.zero_grad()
    assert opt.participation == {"a": 4}
    opt.step()
    assert opt.last_grad_norm == pytest.approx(1.0, rel=1e-6)
