"""The planner decides what a half-funded session leaves behind, so its ordering is the thing to test."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bc4_session_plan as P  # noqa: E402

REQUEST = "/home/moritz/.coworker/state/bc4/QUALITY-REF-REQUEST.json"


def make_request():
    """The shape of the real request: pdl1 banked on A1, three A2 campaigns, three on a second target."""
    runs = ([{"id": f"pdl1/A1_s{s}", "target": "pdl1", "arm": "A1", "campaign_seed": s, "trajectories": 10}
             for s in (104, 105, 106)]
            + [{"id": f"pdl1/A2_s{s}", "target": "pdl1", "arm": "A2", "campaign_seed": s, "trajectories": 10}
               for s in (201, 202, 203)]
            + [{"id": f"il2r/R1_s{s}", "target": "il2r", "arm": "R1", "campaign_seed": s, "trajectories": 10}
               for s in (101, 102, 103)])
    banked = [{"id": f"pdl1/A1_s{s}", "target": "pdl1", "arm": "A1", "campaign_seed": s, "trajectories": n}
              for s, n in ((101, 4), (102, 5), (103, 5))]
    return {"runs": runs, "banked_runs": banked}


def test_the_aa_controls_missing_arm_goes_first():
    """A1 has 14 banked trajectories and A2 has none. Without A2 there is no A/A and the gate fails outright."""
    p = P.plan(make_request(), budget_s=10 ** 9)
    assert all(i.startswith("pdl1/A2_") for i in p["order"][:3])


def test_topups_to_a_banked_arm_go_last():
    """They only rebalance an arm that is already past 30 when the target is pooled: power, not pass/fail."""
    p = P.plan(make_request(), budget_s=10 ** 9)
    assert all(i.startswith("pdl1/A1_") for i in p["order"][-3:])
    assert not any(i.startswith("pdl1/A1_") for i in p["gate_needs"])


def test_the_second_target_is_gate_critical():
    """The gate needs two targets and the second one has nothing banked, so every one of its campaigns is required."""
    p = P.plan(make_request(), budget_s=10 ** 9)
    assert {i for i in p["gate_needs"] if i.startswith("il2r/")} == {f"il2r/R1_s{s}" for s in (101, 102, 103)}


def test_a_budget_that_buys_everything_drops_nothing():
    p = P.plan(make_request(), budget_s=10 ** 9)
    assert p["dropped"] == [] and p["gate_satisfiable"]


def test_a_budget_one_minute_under_the_gate_minimum_does_not_claim_the_gate():
    """The advertised minimum has to be the real boundary, or a buyer funds a session that cannot pass."""
    req = make_request()
    floor = P.plan(req, budget_s=10 ** 9)["min_budget_min_for_gate"]
    assert P.plan(req, budget_s=floor * 60)["gate_satisfiable"]
    assert not P.plan(req, budget_s=(floor - 1) * 60)["gate_satisfiable"]


def test_a_tiny_budget_still_spends_it_on_the_highest_priority_wave():
    """Running out of money must leave the MOST useful campaigns on disk, not whichever came first in the file."""
    p = P.plan(make_request(), budget_s=P.SETUP_S + P.campaign_span_s(10, 3) + 1)
    assert [i for w in p["waves"] for i in w["runs"]] == [f"pdl1/A2_s{s}" for s in (201, 202, 203)]


def test_a_budget_that_buys_nothing_schedules_nothing():
    p = P.plan(make_request(), budget_s=P.SETUP_S)
    assert p["waves"] == [] and len(p["dropped"]) == 9 and not p["gate_satisfiable"]


def test_the_rate_is_the_measured_one():
    """If someone edits the constant, the cost model stops matching the box it was measured on."""
    assert P.DELIVERED_S_PER_TRAJECTORY == pytest.approx(85.2)
    assert P.campaign_span_s(10, 3) == pytest.approx(2556.0)


@pytest.mark.skipif(not os.path.exists(REQUEST), reason="the real request is not on this host")
def test_the_real_request_plans_and_its_gate_minimum_is_under_the_full_cost():
    p = P.plan(json.load(open(REQUEST)), budget_s=150 * 60)
    assert p["gate_satisfiable"] and p["dropped"] == []
    assert p["min_cost_usd_for_gate"] < p["projected_cost_usd"], "the gate must be cheaper than the whole request"
