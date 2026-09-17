#!/usr/bin/env python3
"""Controls for size_scaling.py.

Two jobs. Known-answer cases prove the arithmetic recovers an exponent and a size-independent term
it was not told; staleness guards FAIL if the finding this directory reports ever stops being true,
so the prose cannot quietly go stale behind the numbers.
"""
import math
import random

import pytest

import size_scaling as S


# --- known-answer: recover an A and a p the code was never told -------------------------------
@pytest.mark.parametrize("A,B,p", [(3000.0, 0.04, 2.0), (4469.0, 12.0, 1.0),
                                   (0.0, 1e-4, 3.0), (9000.0, 5e-5, 3.0)])
def test_recovers_exact_A_from_a_synthetic_pure_power_pair(A, B, p):
    n_big, n_small = 512, 298
    w_big, w_small = A + B * n_big ** p, A + B * n_small ** p
    got = S.fixed_part(w_big, w_small, n_big / n_small, p)
    assert got == pytest.approx(A, rel=1e-9, abs=1e-6)


def test_exponent_inverts_the_ratio_it_was_given():
    for r in (512 / 298, 1.6, 2.0):
        for p in (0.5, 1.0, 2.37, 3.0):
            q = r ** p
            got, _ = S.exponent(q, 0.0, r)
            assert got == pytest.approx(p, rel=1e-12)


def test_pure_power_exponent_is_the_p_at_which_A_is_exactly_zero():
    res = S.analyse()
    for name, rec in res["scaling"].items():
        A = S.fixed_part(S.MEAS[512]["W_Mcyc"], S.MEAS[298]["W_Mcyc"], rec["r"],
                         rec["pure_power_exponent"])
        assert abs(A) < 1e-6, f"{name}: A at the pure-power exponent is {A}, not zero"


# --- the lower-bound property, which is the whole reason p=1 is quoted -------------------------
def test_smallest_exponent_gives_a_lower_bound_under_a_mixture():
    """W = A + B1*N + B2*N**2 with B2 > 0. Evaluating at p=1 must UNDERSTATE A, by exactly
    B2 * N_small**2 * r -- so the quoted p=1 figure is a floor, never an overstatement."""
    A, B1, B2 = 5000.0, 10.0, 0.03
    n_big, n_small = 512, 298
    r = n_big / n_small
    w_big = A + B1 * n_big + B2 * n_big ** 2
    w_small = A + B1 * n_small + B2 * n_small ** 2
    got = S.fixed_part(w_big, w_small, r, 1.0)
    assert got < A
    assert got == pytest.approx(A - B2 * n_small ** 2 * r, rel=1e-9)


def test_A_rises_monotonically_with_the_assumed_exponent():
    r = 512 / 298
    vals = [S.fixed_part(S.MEAS[512]["W_Mcyc"], S.MEAS[298]["W_Mcyc"], r, p)
            for p in (1.0, 1.5, 2.0, 2.5, 3.0)]
    assert vals == sorted(vals)
    assert all(v < S.MEAS[512]["W_Mcyc"] for v in vals), "A cannot exceed the total work term"


def test_degenerate_exponent_raises_instead_of_dividing_by_zero():
    with pytest.raises(ValueError):
        S.fixed_part(100.0, 90.0, 1.7, 0.0)


# --- uncertainty is propagated, not asserted --------------------------------------------------
def test_propagated_ratio_error_matches_a_monte_carlo():
    random.seed(7)
    a, sa, b, sb = 14665.0, 121.1, 10403.4, 44.9
    _, se = S.ratio(a, sa, b, sb)
    draws = [random.gauss(a, sa) / random.gauss(b, sb) for _ in range(40000)]
    m = sum(draws) / len(draws)
    mc = math.sqrt(sum((d - m) ** 2 for d in draws) / (len(draws) - 1))
    assert se == pytest.approx(mc, rel=0.05), f"analytic {se} vs monte carlo {mc}"


def test_two_identical_measurements_give_a_zero_exponent_not_a_crash():
    q, se = S.ratio(100.0, 1.0, 100.0, 1.0)
    p, _ = S.exponent(q, se, 512 / 298)
    assert p == pytest.approx(0.0, abs=1e-12)


# --- inputs are quoted, so guard them against drift -------------------------------------------
def test_inputs_still_match_the_numbers_c10_fixed_cost_published():
    assert S.MEAS[512]["W_Mcyc"] == 14665.0 and S.MEAS[512]["W_se"] == 121.1
    assert S.MEAS[298]["W_Mcyc"] == 10403.4 and S.MEAS[298]["W_se"] == 44.9
    assert S.MEAS[512]["F_s"] == 3.9830 and S.MEAS[298]["F_s"] == 1.9500
    # F + W/f must reproduce that row's own 14.846 s reading at 1350 MHz.
    T = S.MEAS[512]["F_s"] + S.MEAS[512]["W_Mcyc"] / S.PIN_MHZ
    assert T == pytest.approx(14.846, abs=0.002)


def test_the_fold_has_one_size_axis_so_atoms_and_tokens_cannot_be_separated():
    res = S.analyse()
    rt, ra = res["size_ratios"]["tokens"], res["size_ratios"]["atoms"]
    assert abs(rt - ra) / rt < 1e-3, "atom and token axes have diverged; re-derive before quoting"


# --- staleness guards: these FAIL if the finding stops being true -----------------------------
def test_work_still_grows_strictly_SLOWER_than_the_target():
    """The entire finding. If the work ratio ever reaches the size ratio there is no evidence of a
    size-independent term and every conclusion in the README is void."""
    res = S.analyse()
    q = res["work_ratio"]["value"] + 2 * res["work_ratio"]["se"]
    assert q < res["size_ratios"]["padded32"], (
        "work ratio has reached the size ratio: the size-independent term is gone, rewrite README")
    assert res["scaling"]["tokens"]["p_work"]["value"] < 1.0


def test_fixed_term_is_still_size_DEPENDENT_so_it_is_not_pure_host_dispatch():
    """Host dispatch is size-independent: same call count at both sizes. A p_fixed indistinguishable
    from zero would revive that reading and void the discriminator this directory claims."""
    res = S.analyse()
    pf = res["scaling"]["tokens"]["p_fixed"]
    assert pf["sigma_from_zero"] > 5.0, f"p_fixed only {pf['sigma_from_zero']:.1f} sigma from zero"
    assert pf["value"] > 1.0


def test_lower_bound_is_still_a_material_share_of_the_campaigns_deletion_target():
    res = S.analyse()
    assert res["headline"]["lower_bound_covers_pct_of_cut"] > 25.0


# --- negative controls: break the input, the conclusion must break too ------------------------
def test_shrinking_W298_until_the_ratio_exceeds_the_size_ratio_kills_the_finding():
    r = 512 / 298
    w_small = S.MEAS[512]["W_Mcyc"] / (r ** 1.2)      # now scaling FASTER than linear
    A = S.fixed_part(S.MEAS[512]["W_Mcyc"], w_small, r, 1.0)
    assert A < 0, "a superlinear pair must yield a negative A, i.e. no size-independent evidence"


def test_equal_fixed_terms_at_both_sizes_would_restore_the_host_dispatch_reading():
    q, se = S.ratio(3.9830, 0.1181, 3.9830, 0.1181)
    p, sp = S.exponent(q, se, 512 / 298)
    assert abs(p / sp) < 1.0, "a flat F must read as indistinguishable from size-independent"


def test_demand_arithmetic_is_the_one_c10_fixed_cost_published():
    res = S.analyse()["demand"]
    assert res["fold_s_at_pin"] == pytest.approx(14.846, abs=0.002)
    assert res["cut_pct"] == pytest.approx(44.6, abs=0.2)
    assert res["allowed_W_Mcyc_at_target"] == pytest.approx((10.0 - 3.9830) * 1350.0, rel=1e-12)


def test_trace_lever_result_corroborates_the_p_fixed_reading():
    """p_fixed >> 0 predicted that removing host dispatch would NOT move the fold. c10-trace-lever
    then measured exactly that. If either half of this pair ever changes, the README's central
    claim that F is not collapsible host overhead has to be rewritten."""
    res = S.analyse()
    assert res["scaling"]["tokens"]["p_fixed"]["sigma_from_zero"] > 5.0
    c = res["corroboration"]
    assert c["inside_floor"], "the trace lever no longer measures a null; re-read the row"
    assert abs(c["measured_s"]) < 0.1 * c["predicted_s"], "measured is no longer ~zero vs predicted"
