"""The training-parity bars are pre-registered, and every check can fail.

The numbers below were fixed in ``state/r4t/PLAN.md`` (G2, G3) before any RFD4 training loss or
curve existed, and committed here with the checks that apply them. This file's first commit is
the pre-registration proof: changing a pinned value is changing a bar, and that needs a reason
recorded before the data it grades.

Each check is shown failing on the shape of defect it exists for before it is trusted passing.
"""

from __future__ import annotations

import numpy as np
import pytest

from tt_bio.train import parity as P


# ------------------------------------------------------------------------ pinned values

def test_bars_are_the_preregistered_values():
    assert P.LOSS_PARITY_BARS == {"bfloat16": 1.0e-2, "float32": 1.0e-4}
    assert P.LOSS_PARITY_MIN_BATCHES == 8
    assert P.CURVE_MIN_STEPS == 2000
    assert P.CURVE_TAIL == 200
    assert P.CURVE_REL == 0.02
    assert P.CURVE_SEED_FACTOR == 1.5
    assert P.CURVE_WINDOW == 100
    assert P.CURVE_DRIFT == 0.10
    assert P.TRAINS_MIN_STEPS == 5


# ------------------------------------------------------------------------ loss parity

TERMS = ("total", "bb_ca", "local_latents", "seq", "signed_volume")


def _batches(n=8, seed=0):
    rng = np.random.default_rng(seed)
    return [{k: float(rng.uniform(0.1, 3.0)) for k in TERMS} for _ in range(n)]


def _scaled(batches, rel, batch=None, term=None):
    out = [dict(b) for b in batches]
    for i, b in enumerate(out):
        for k in b:
            if (batch is None or i == batch) and (term is None or k == term):
                b[k] *= 1.0 + rel
    return out


def test_loss_parity_passes_inside_the_bar():
    ref = _batches()
    r = P.loss_parity(_scaled(ref, 0.009), ref, "bfloat16")
    assert r.passed, str(r)
    assert r.n == 8 and 0.0089 < r.worst_rel < 0.0091


def test_loss_parity_fails_one_term_on_one_batch_over_the_bar():
    ref = _batches()
    r = P.loss_parity(_scaled(ref, 0.011, batch=5, term="seq"), ref, "bfloat16")
    assert not r.passed
    assert (r.worst_batch, r.worst_term) == (5, "seq")


def test_fp32_arm_is_a_hundred_times_tighter():
    ref = _batches()
    assert P.loss_parity(_scaled(ref, 9e-5), ref, "float32").passed
    assert not P.loss_parity(_scaled(ref, 2e-4), ref, "float32").passed
    assert P.loss_parity(_scaled(ref, 2e-4), ref, "bfloat16").passed


def test_loss_parity_fails_below_eight_batches():
    ref = _batches(7)
    assert not P.loss_parity(ref, ref, "bfloat16").passed


def test_loss_parity_fails_a_dropped_term():
    """A loss that silently drops a term matches on everything it still computes."""
    ref = _batches()
    ours = [dict(b) for b in ref]
    del ours[3]["signed_volume"]
    assert not P.loss_parity(ours, ref, "bfloat16").passed


def test_loss_parity_fails_nan_and_a_nonzero_against_zero():
    ref = _batches()
    ours = [dict(b) for b in ref]
    ours[0]["bb_ca"] = float("nan")
    assert not P.loss_parity(ours, ref, "bfloat16").passed
    ref2 = [dict(b) for b in ref]
    ref2[1]["seq"] = 0.0
    assert not P.loss_parity(ref, ref2, "bfloat16").passed
    assert P.loss_parity(ref2, ref2, "bfloat16").passed


# ------------------------------------------------------------------------ the curve

def _curve(n=2000, seed=0, floor=1.0, noise=0.02):
    rng = np.random.default_rng(seed)
    s = np.arange(n)
    return floor + 2.0 * np.exp(-s / 300.0) + noise * rng.standard_normal(n)


def test_curve_passes_a_reseeded_copy_of_upstream():
    a, b, ours = _curve(seed=1), _curve(seed=2), _curve(seed=3)
    r = P.curve_match(ours, a, b, warmup=100)
    assert r.passed, str(r)


def test_curve_fails_a_run_that_plateaus_higher():
    """Converged 5 % above upstream: over the 2 % bar and over 1.5x the seed gap."""
    a, b = _curve(seed=1), _curve(seed=2)
    ours = _curve(seed=3, floor=1.05)
    r = P.curve_match(ours, a, b, warmup=100)
    assert not r.passed and "gap" in r.failures[0]


def test_curve_seed_control_widens_the_bar_when_upstream_itself_is_noisy():
    """A 5 % gap is not a finding if re-seeding upstream moves it 4 %: bar = 1.5 x 4 % = 6 %."""
    a, ours = _curve(seed=1), _curve(seed=3, floor=1.05)
    b = _curve(seed=2, floor=1.04)
    assert P.curve_match(ours, a, b, warmup=100).passed


def test_curve_fails_a_run_that_ends_right_but_wanders():
    """Same end point, but 30 % above upstream for a stretch mid-run: the shape check catches it."""
    a, b, ours = _curve(seed=1), _curve(seed=2), _curve(seed=3)
    ours[800:1000] *= 1.3
    r = P.curve_match(ours, a, b, warmup=100)
    assert not r.passed
    assert any("running mean" in f for f in r.failures)
    assert 800 <= r.worst_drift_step < 1100


def test_curve_drift_is_not_scored_inside_warmup():
    a, b, ours = _curve(seed=1), _curve(seed=2), _curve(seed=3)
    ours[:300] *= 1.5
    assert P.curve_match(ours, a, b, warmup=400).passed
    assert not P.curve_match(ours, a, b, warmup=100).passed


def test_curve_fails_short_mismatched_or_nonfinite():
    a, b = _curve(seed=1), _curve(seed=2)
    assert not P.curve_match(a[:1999], a[:1999], b[:1999], warmup=100).passed
    assert not P.curve_match(a[:1900], a, b, warmup=100).passed
    bad = a.copy()
    bad[1500] = np.inf
    assert not P.curve_match(bad, a, b, warmup=100).passed
    assert not P.curve_match(a, a, b, warmup=1900).passed


# ------------------------------------------------------------------------ every parameter trains

def test_guard_refuses_fewer_than_five_steps():
    with pytest.raises(ValueError):
        P.GradientCoverage(range(10), steps=1)


def test_guard_ignores_the_zero_init_transient():
    """Dead at step 1 behind zero-initialised output projections, live from step 2."""
    g = P.GradientCoverage(range(500))
    for dead in ({0, 1, 4, 5, 6, 7}, {0, 1}, set(), set(), set()):
        g.observe(dead)
    assert g.done


def test_guard_fails_on_one_detached_parameter():
    g = P.GradientCoverage([f"p{i}" for i in range(500)], where="[rank 0] ")
    for _ in range(4):
        g.observe({"p499"})
    with pytest.raises(RuntimeError, match=r"1 of 500 parameters.*p499"):
        g.observe({"p499"})


def test_guard_honours_a_named_exemption_and_rejects_an_unknown_one():
    names = ["a", "b"]
    g = P.GradientCoverage(names, exempt={"b": "unconditional-only input, never in a batch"})
    for _ in range(5):
        g.observe({"b"})
    with pytest.raises(ValueError):
        P.GradientCoverage(names, exempt={"c": "typo"})


def test_zero_or_missing_counts_none_and_all_zero():
    import torch
    grads = {"none": None, "zero": torch.zeros(3), "np0": np.zeros(2),
             "live": torch.tensor([0.0, 1e-30]), "nplive": np.array([0.0, -1.0])}
    assert P.zero_or_missing(grads) == {"none", "zero", "np0"}
