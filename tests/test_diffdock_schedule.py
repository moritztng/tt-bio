"""The expbeta t schedule and the sigma curves it drives."""

import numpy as np
import pytest
from scipy.stats import beta as beta_dist

from tt_bio.diffdock import sampling


def test_expbeta_schedule_bit_matches():
    """At the recipe's alpha=beta=1 the beta CDF is the identity, so the schedule is linear.

    Checked bit for bit against that closed form, and against the beta quantile recompute at
    a shape where the two differ, so the test fails if the alpha/beta arguments are dropped.
    """
    recipe = sampling.recipe()
    cfg = sampling.SamplingConfig.from_recipe(recipe)
    assert (cfg.inf_sched_alpha, cfg.inf_sched_beta) == (1.0, 1.0)

    sched = sampling.t_schedule(cfg.inference_steps, inf_sched_alpha=cfg.inf_sched_alpha,
                                inf_sched_beta=cfg.inf_sched_beta)
    assert np.array_equal(sched, np.linspace(1.0, 0.0, cfg.inference_steps + 1)[:-1])

    shaped = sampling.t_schedule(20, inf_sched_alpha=2.0, inf_sched_beta=3.0)
    expect = beta_dist.ppf(np.linspace(beta_dist.cdf(1.0, a=2.0, b=3.0), 0, 21)[:-1],
                           a=2.0, b=3.0)
    assert np.array_equal(shaped, expect)
    assert not np.allclose(shaped, sched)


def test_schedule_starts_at_t_max_and_descends():
    sched = sampling.t_schedule(20)
    assert sched[0] == pytest.approx(1.0)
    assert np.all(np.diff(sched) < 0)
    assert sched[-1] > 0.0          # the last step jumps to 0, the schedule never reaches it


def test_one_schedule_per_head_unless_different_schedules():
    cfg = sampling.SamplingConfig.from_recipe()
    bounds = sampling.SigmaBounds(0.1, 19.0, 0.03, 1.55, 0.0314, 3.14)
    tr, rot, tor = cfg.schedules(bounds)
    assert tr is rot is tor

    pocket = sampling.SamplingConfig.from_recipe(
        different_schedules=True, pocket_knowledge=True, pocket_tr_max=3.0)
    tr, rot, tor = pocket.schedules(bounds)
    assert rot is tor and tr is not rot
    # The translation schedule starts where tr_sigma equals pocket_tr_max, not at the
    # receptor-wide sigma: a pocket-conditioned run does not re-scatter the pose.
    assert bounds.sigmas(float(tr[0]), 0, 0)[0] == pytest.approx(3.0, rel=1e-6)


def test_sigma_interpolation_and_g_coefficient():
    bounds = sampling.SigmaBounds(0.1, 19.0, 0.03, 1.55, 0.0314, 3.14)
    assert bounds.sigmas(1.0, 1.0, 1.0) == pytest.approx((19.0, 1.55, 3.14))
    assert bounds.sigmas(0.0, 0.0, 0.0) == pytest.approx((0.1, 0.03, 0.0314))
    mid = bounds.sigmas(0.5, 0.5, 0.5)
    assert mid[0] == pytest.approx(np.sqrt(0.1 * 19.0))
    assert bounds.g("tr", 19.0) == pytest.approx(19.0 * np.sqrt(2 * np.log(190.0)))
