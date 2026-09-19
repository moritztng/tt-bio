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

    A LoRA adapter on a frozen trunk has a measured replica behind it. A trained trunk has a
    measured retention bound and no measured peak, because the backward that would set the peak
    is not built yet. So the answer stays UNMEASURED and carries the gigabytes it does have,
    rather than the 27.58 GB feasibility projection it used to answer with.
    """
    from tt_bio.train.dryrun import UNMEASURED, plan

    adapters = plan(tokens=256, chips=1, global_batch=8, frozen_trunk=True)
    weights = plan(tokens=256, chips=1, global_batch=8, frozen_trunk=False)
    assert adapters.verdict != UNMEASURED and adapters.measured
    assert weights.verdict == UNMEASURED and not weights.measured
    assert "4.074 GB" in weights.why and "retention" in weights.why
    assert "27.58" not in weights.why, "the retired projection is answering again"


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


def test_the_recipe_crop_is_not_refused_and_a_real_measured_oom_still_would_be():
    """384 aa is Protenix's own crop, and `plan()` used to turn a user away at it.

    That refusal was measured on a differentiable twin of the pair track, a module since
    deleted. The shipped 48-block forward peaks at 0.877 GB of 34.23 GB there, measured on two
    chips in two campaigns, so the entry is gone. The two halves are tested apart: the crop no
    longer refuses, and the mechanism that refuses a genuinely measured OOM is untouched --
    "we measured this failing" is still a different answer from "we have no measurement", and
    it is still decided before the trained-trunk branch because the forward is what both modes
    run.
    """
    from tt_bio.train import dryrun
    from tt_bio.train.dryrun import FORWARD_FITS, FORWARD_OOM, UNMEASURED, plan

    for tokens in (384, 512):
        assert tokens not in FORWARD_OOM and tokens in FORWARD_FITS
        for frozen in (True, False):
            p = plan(tokens=tokens, chips=1, global_batch=8, frozen_trunk=frozen)
            assert p.verdict == UNMEASURED, (tokens, frozen, p.verdict)
            assert p.fits is not False and "OOM" not in p.why

    # The mechanism, with an entry present for the length of these assertions.
    entry = (1.00, 123_456_789, "state/concluded/ptx-crop -- a hypothetical, for this test")
    FORWARD_OOM[900] = entry
    try:
        for frozen in (True, False):
            p = plan(tokens=900, chips=1, global_batch=8, frozen_trunk=frozen)
            assert p.verdict == "refused" and not p.fits, (frozen, p.verdict)
            assert "OOMs at 900 aa" in p.why and "123,456,789 B refused" in p.why
            assert p.sources == [entry[2]], "a refusal that does not say where it came from"
    finally:
        del FORWARD_OOM[900]
    assert dryrun.FORWARD_OOM == {}

    # and a crop the forward does fit at still reports the honest UNMEASURED for the tape
    assert plan(tokens=256, chips=1, global_batch=8,
                frozen_trunk=False).verdict == UNMEASURED


def _provenance_gaps(table):
    """Every entry in a measurement table that does not say where its number came from."""
    from tt_bio.train.dryrun import provenance_gap

    gaps = []
    for key, entry in table.items():
        if not isinstance(entry, tuple) or not isinstance(entry[-1], str):
            gaps.append(f"{key}: no `where` at all")
            continue
        gap = provenance_gap(entry[-1])
        if gap:
            gaps.append(f"{key}: {gap}")
    return gaps


def test_every_measured_entry_names_where_its_number_came_from():
    """A number that outlives the module it was measured on is this planner's failure mode.

    `FORWARD_OOM`'s 384 and 512 were two bare tuples under a shared comment. The comment was
    honest and the entries carried nothing of their own, so when the twin they were measured on
    was deleted there was nothing to check them against, and `tt-bio finetune` turned users away
    at the recipe's own crop for two more campaigns. This asks every entry in every table for a
    source a reader can open.
    """
    from tt_bio.train.dryrun import FORWARD_FITS, FORWARD_OOM, TRAINED_TRUNK_BOUND

    for name, table in (("FORWARD_FITS", FORWARD_FITS), ("FORWARD_OOM", FORWARD_OOM),
                        ("TRAINED_TRUNK_BOUND", TRAINED_TRUNK_BOUND)):
        assert _provenance_gaps(table) == [], name

    # The negative control, and not optional: FORWARD_OOM is empty today, so the assertion
    # above passes over it vacuously and would keep passing if the old entries came back. These
    # are those two entries verbatim, plus the two ways a citation gets faked -- prose that
    # reads like provenance, and a bare figure.
    assert _provenance_gaps({384: (4.14, 75_497_472), 512: (7.15, 536_870_912)}) == [
        "384: no `where` at all", "512: no `where` at all"]
    assert _provenance_gaps({384: (4.14, 75_497_472, "re-measure on the shipped forward")})
    assert _provenance_gaps({384: (4.14, 75_497_472, "")})
    assert _provenance_gaps({256: (34.23, 4_278_190_016, "4278190016 B per bank, 8 banks")})
    assert _provenance_gaps({384: (0.877, 0, "state/concluded/ptx-crop")}) == []
    assert _provenance_gaps({384: (0.877, 0, "wk/ptx-crop 9c567b590")}) == []
