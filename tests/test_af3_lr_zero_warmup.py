"""`warmup_steps=0` means no warmup, and it has to be sayable.

`af3_lr` computed `step / warmup_steps` before testing it, so 0 raised `ZeroDivisionError` in
`AdamW.step` -- after the forward, the loss and the backward had all run. A short arm is the
case that needs it: at 30 steps the 1000-step default leaves the rate at 9e-6 of 3e-4, so the
weights barely move and two arms compared over those steps agree because neither trained. That
is an agreement about nothing, which is the one result a grading run must not be able to
produce by accident.
"""
import pytest

from tt_bio.train.optim import af3_lr


def test_no_warmup_is_full_rate_from_step_zero():
    for step in (0, 1, 5):
        assert af3_lr(step, 3e-4, warmup_steps=0, plateau_until=50000) == pytest.approx(3e-4)


def test_warmup_still_ramps_when_it_is_asked_for():
    assert af3_lr(0, 1e-3, warmup_steps=1000) == pytest.approx(0.0)
    assert af3_lr(500, 1e-3, warmup_steps=1000) == pytest.approx(5e-4)
    assert af3_lr(1000, 1e-3, warmup_steps=1000) == pytest.approx(1e-3)


def test_the_default_would_have_flattened_a_thirty_step_arm():
    """The number that motivates the flag, pinned so nobody re-picks the default by accident."""
    assert af3_lr(30, 3e-4, warmup_steps=1000) == pytest.approx(9e-6)
