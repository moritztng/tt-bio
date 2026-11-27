"""The default recipe refuses a run that trains nothing, and it reads the right thing to decide.

The guard itself is graded in `test_training_parity_bars.py` and driven on upstream RFD4's own
blocks in `test_rfd4_gradient_coverage.py`. What is asserted here is the WIRING, because the
loop rather than the function is where a dead parameter has to be noticed: ABB3 ran 1,388 steps
with 356 of 436 parameters dead behind a healthy loss, a plausible grad_norm and a digest that
changed every step.

`train_loop` decides from the optimizer's accumulator, so the contract it depends on is tested
against a REAL `AdamW`: a parameter whose gradient never arrived is ABSENT from `accum`, and one
whose gradient is identically zero is PRESENT and zero. Both read as dead; anything else would
make the guard pass a dead run or refuse a live one.

Host-only -- `to_host` / `to_device` are numpy, no card.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
np = pytest.importorskip("numpy")

from tt_bio.train import optim  # noqa: E402
from tt_bio.train.parity import GradientCoverage, zero_or_missing  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SHAPES = {"live": (4, 8), "zeroed": (16,), "never": (3, 2)}


class _DType:
    name = "bfloat16"


class _Value:
    dtype = _DType()

    def __init__(self, arr):
        self.arr = np.asarray(arr, np.float32)

    def device(self):
        return None


class _Grad:
    def __init__(self, arr):
        self.arr = np.asarray(arr, np.float32)


class _Param:
    def __init__(self, arr):
        self.value, self.grad = _Value(arr), None


@pytest.fixture
def host_optimizer(monkeypatch):
    monkeypatch.setattr(optim, "to_host",
                        lambda t, dtype=None: (t.arr if isinstance(t, (_Grad, _Value))
                                               else np.asarray(t)).copy())
    monkeypatch.setattr(optim, "to_device", lambda arr, dev, dtype=None: _Value(arr))
    rng = np.random.default_rng(0)
    params = {n: _Param(rng.standard_normal(s)) for n, s in SHAPES.items()}
    return params, optim.AdamW(params, lr=1e-2)


def _accumulate_one_step(params, opt, rng):
    """One step's worth of gradients: one live, one identically zero, one that never arrives."""
    params["live"].grad = _Grad(rng.standard_normal(SHAPES["live"]))
    params["zeroed"].grad = _Grad(np.zeros(SHAPES["zeroed"]))
    params["never"].grad = None
    opt.clip_and_accumulate()


def test_the_accumulator_tells_a_missing_gradient_from_a_zero_one(host_optimizer):
    """The contract `train_loop` reads. If this changes, the guard goes blind in silence."""
    params, opt = host_optimizer
    _accumulate_one_step(params, opt, np.random.default_rng(1))
    assert "never" not in opt.accum, "a parameter with no gradient reached the accumulator"
    assert "zeroed" in opt.accum and not np.any(opt.accum["zeroed"])
    assert np.any(opt.accum["live"])
    assert set(opt.accum) <= set(opt.master), "the accumulator named something the master lacks"


def test_the_dead_set_the_loop_builds_names_both_kinds(host_optimizer):
    params, opt = host_optimizer
    _accumulate_one_step(params, opt, np.random.default_rng(2))
    dead = zero_or_missing({name: opt.accum.get(name) for name in opt.master})
    assert dead == {"never", "zeroed"}


def test_five_such_steps_refuse_the_run(host_optimizer):
    """What the loop does, with the loop's own two lines, over the guard's five steps."""
    params, opt = host_optimizer
    rng = np.random.default_rng(3)
    coverage = GradientCoverage(sorted(opt.master), steps=5)
    with pytest.raises(RuntimeError, match=r"2 of 3 parameters received no gradient"):
        for _ in range(5):
            _accumulate_one_step(params, opt, rng)
            coverage.observe(zero_or_missing({n: opt.accum.get(n) for n in opt.master}))
            opt.step()
            opt.zero_grad()
            opt.accum.clear()
            opt.participation.clear()
            opt.accum_count = 0


def test_a_healthy_run_passes_the_same_five_steps(host_optimizer):
    params, opt = host_optimizer
    rng = np.random.default_rng(4)
    coverage = GradientCoverage(sorted(opt.master), steps=5)
    for _ in range(5):
        for name, shape in SHAPES.items():
            params[name].grad = _Grad(rng.standard_normal(shape))
        opt.clip_and_accumulate()
        coverage.observe(zero_or_missing({n: opt.accum.get(n) for n in opt.master}))
        opt.step()
        opt.zero_grad()
        opt.accum.clear()
        opt.participation.clear()
        opt.accum_count = 0
    assert coverage.done and not coverage.never


def test_the_recipe_still_runs_the_guard():
    """The positive assertion: the Tier-1 body has not quietly dropped it.

    A guard nothing calls is the defect it exists to catch, one level up.
    """
    src = (REPO_ROOT / "tt_bio" / "train" / "recipes.py").read_text()
    body = src.split("def train_loop", 1)[1]
    assert "GradientCoverage(" in body and "zero_or_missing(" in body, (
        "tt_bio.train.recipes.train_loop no longer runs the every-parameter-trains guard, so "
        "a run whose tape reaches nothing produces a plausible loss curve and trains nothing")
    assert body.index("coverage.observe(") < body.index("opt.step("), (
        "the guard reads the accumulator, and step() consumes it: observing after the step "
        "reads an emptied accumulator and calls every parameter dead")
