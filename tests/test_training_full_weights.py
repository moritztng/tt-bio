"""Full-weight training: one argument away from LoRA, and the hazard that separates them.

`train="weights"` hands the optimizer the model's own weights instead of a factor pair beside
them. Everything else -- the loop, the objective, the optimizer, the checkpointer, the
data-parallel axis -- is the same code, and the escape-hatch tests in
`tests/test_train_interface.py` already prove there is one body rather than two.

What is NOT shared is how a parameter gets its name. A LoRA site takes one adapter however
many blocks read it; a weight cannot, and the difference is a correctness bug rather than a
preference. These run host-only: the substitution hook composes over whatever hook is
installed under it, so a recording stand-in in that slot is enough to exercise every path
without a card.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

ttnn = pytest.importorskip("ttnn")

from tt_bio import ops                                              # noqa: E402
from tt_bio.train import lora                                       # noqa: E402
from tt_bio.train.lora import LoraConfig, attach, trainable, weights_for   # noqa: E402

import _stacked_model as M                                          # noqa: E402


class _Weight:
    """The least a weight has to be for the substitution path: a shape and an identity."""

    def __init__(self, shape=(8, 8)):
        self.shape = shape


class _Recorder:
    """The hook the substitution composes over. Records what reached ops.linear."""

    def __init__(self):
        self.weights = []

    def __call__(self, name, shipped, args, kwargs):
        if name == "linear":
            self.weights.append(args[1])
        return args[0]              # a value, so `shipped` is never called and no card opens


def _run(fn, *args, **kwargs):
    """Call `fn` with a recorder installed, and restore whatever was there."""
    rec = _Recorder()
    prev = ops.set_grad_hook(rec)
    try:
        return rec, fn(*args, **kwargs)
    finally:
        ops.set_grad_hook(prev)


def test_one_site_many_weights_gets_one_parameter_each():
    """The 48-block hazard. One call site, N weights, N parameters -- never one.

    `_site_name` is file:line:qualname, so every block of a stacked trunk reports the same
    site. Keying the parameter on the site alone would train block 0's weight and apply it to
    all of them: a model that trains, loses nothing measurable, and predicts something else.
    """
    ws = [_Weight() for _ in range(5)]
    rec, params = _run(lambda: weights_for(M.forward, None, object(), ws))
    assert len(params) == 5, (
        f"5 distinct weights at one call site collapsed to {len(params)} parameter(s). "
        f"Each block's own weight has to be its own parameter.")
    # One bare name and the rest numbered by first appearance, so a data-parallel rank's
    # `#3` is every other rank's `#3`.
    sites = sorted(params)
    assert sum("#" not in n for n in sites) == 1, sites
    assert {n.rsplit("#", 1)[1] for n in sites if "#" in n} == {"1", "2", "3", "4"}, sites
    assert all(params[n].requires_grad for n in params)


def test_one_site_one_weight_read_many_times_is_one_parameter():
    """The other half of the same rule: a genuinely shared weight is ONE parameter.

    Minting per call instead would hand the optimizer three parameters that are the same
    tensor, and three Adam states racing to write one weight.
    """
    w = _Weight()
    rec, params = _run(lambda: weights_for(M.shared, None, object(), w, blocks=3))
    assert len(params) == 1, f"one weight read three times became {len(params)} parameters"
    assert len(rec.weights) == 3, "the stand-in did not actually run three blocks"


def test_the_forward_reads_the_parameter_and_not_the_checkpoint_weight():
    """After `attach`, the object reaching ops.linear is the parameter, not the model's own.

    This is what makes an optimizer step visible to the next forward: `AdamW.step` replaces
    `t.value`, while the model's cache still holds the tensor it uploaded once. A forward that
    kept reading the passed weight would train happily and never move.
    """
    ws = [_Weight() for _ in range(3)]
    _, params = _run(lambda: weights_for(M.forward, None, object(), ws))
    rec = _Recorder()
    prev = ops.set_grad_hook(rec)
    try:
        with attach(params, None):
            M.forward(object(), ws)
    finally:
        ops.set_grad_hook(prev)
    assert len(rec.weights) == 3
    assert all(w in params.values() for w in rec.weights), (
        "the forward was handed the checkpoint's weight, so an optimizer step would be "
        "invisible to it")
    assert not any(w in ws for w in rec.weights)


def test_a_reuploading_forward_is_refused_rather_than_trained_silently():
    """A forward that mints a fresh weight per call cannot be trained, and says so.

    The failure it replaces is the quiet one: every gradient is real, every step is taken,
    and the model does not move because the optimizer owns tensors nothing reads twice.
    """
    _, params = _run(lambda: weights_for(M.reuploading, None, object(), _Weight))
    rec = _Recorder()
    prev = ops.set_grad_hook(rec)
    try:
        with pytest.raises(ValueError, match="does not hold its device weights"):
            with attach(params, None):
                M.reuploading(object(), _Weight)
    finally:
        ops.set_grad_hook(prev)


def test_a_forward_with_no_adaptable_site_refuses_by_name():
    with pytest.raises(ValueError, match="no adaptable linear site"):
        _run(lambda: weights_for(lambda: None, None))


def test_targets_select_the_same_way_they_do_for_lora():
    ws = [_Weight() for _ in range(3)]
    cfg = LoraConfig(targets=("_stacked_model.py",))
    _, params = _run(lambda: weights_for(M.forward, cfg, object(), ws))
    assert len(params) == 3
    miss = LoraConfig(targets=("no_such_file",))
    with pytest.raises(ValueError, match="no adaptable linear site"):
        _run(lambda: weights_for(M.forward, miss, object(), ws))


def test_trainable_returns_the_same_shape_for_both_modes():
    """One call, both modes, so the recipe body does not branch on which one it is."""
    ws = [_Weight() for _ in range(2)]
    _, (installed, params) = _run(lambda: trainable(M.forward, None, object(), object(), ws))
    assert installed is params and len(params) == 2


def test_tier1_refuses_an_unknown_train_mode_before_a_device_opens():
    from tt_bio.train.loop import TRAIN_MODES, finetune

    assert TRAIN_MODES == ("adapters", "weights")
    with pytest.raises(ValueError, match="no train mode"):
        finetune(None, None, out_dir="x", global_batch=1, steps=1, train="full")


def test_tier0_and_tier1_agree_on_the_modes():
    """Duplicated so `--train` can be validated without importing the tape. Pinned here."""
    from tt_bio.train.cli import TRAIN_MODES as CLI
    from tt_bio.train.loop import TRAIN_MODES as LOOP

    assert CLI == LOOP


def test_plan_prices_the_two_modes_apart():
    """`--train weights` is not a free rename: it changes what the planner can answer.

    A LoRA adapter on a frozen trunk has a measured replica behind it. A trained trunk has
    only a feasibility memo, so the honest answer is UNMEASURED and the dry run says which it
    got rather than printing a projection shaped like a measurement.
    """
    from tt_bio.train.dryrun import UNMEASURED, plan

    adapters = plan(tokens=256, chips=1, global_batch=8, frozen_trunk=True)
    weights = plan(tokens=256, chips=1, global_batch=8, frozen_trunk=False)
    assert adapters.verdict != UNMEASURED and adapters.measured
    assert weights.verdict == UNMEASURED and not weights.measured
    assert "projection" in weights.why


def test_the_substitution_composes_over_the_tape_rather_than_replacing_it():
    """Every non-linear op, and every site this run does not train, reaches the hook below.

    Declining instead would run the rest of the model untaped: the gradient would reach
    whatever sits after the last substituted site and nothing before it, with no error.
    """
    seen = []

    def base(name, shipped, args, kwargs):
        seen.append(name)
        return args[0]

    params = {}
    hook = lora._Substitute(params, base)
    assert hook("layer_norm", None, (object(),), {}) is not None
    assert seen == ["layer_norm"]


def test_a_measured_oom_is_reported_even_when_the_trunk_is_trained():
    """"We measured this failing" and "we have no measurement" are different answers.

    The forward OOM at 384 aa is a fact about the FORWARD, which both modes run. Deciding the
    trained-trunk case first would answer UNMEASURED there and hide a measurement behind the
    absence of one -- on the exact crop Protenix's own recipe uses.
    """
    from tt_bio.train.dryrun import FORWARD_OOM, UNMEASURED, plan

    assert 384 in FORWARD_OOM
    for frozen in (True, False):
        p = plan(tokens=384, chips=1, global_batch=8, frozen_trunk=frozen)
        assert p.verdict == "refused" and not p.fits, (frozen, p.verdict)
        assert "OOMs at 384 aa" in p.why
    # and a crop the forward does fit at still reports the honest UNMEASURED for the tape
    assert plan(tokens=256, chips=1, global_batch=8,
                frozen_trunk=False).verdict == UNMEASURED
