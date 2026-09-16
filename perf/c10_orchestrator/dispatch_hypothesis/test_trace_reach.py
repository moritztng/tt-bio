"""Controls for the trace-reach arithmetic."""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def report():
    return json.loads((HERE / "trace_reach.json").read_text())


def test_the_two_shares_use_the_two_different_denominators():
    import trace_reach as m
    d = report()["diffusion_loop"]
    assert abs(d["share_of_launching_calls_pct"] - 100 * d["programs"] / m.LAUNCHING_CALLS) < 1e-9
    assert abs(d["share_of_top_level_calls_pct"] - 100 * d["programs"] / m.TOP_LEVEL_CALLS) < 1e-9
    assert d["share_of_launching_calls_pct"] > d["share_of_top_level_calls_pct"]


def test_the_attributed_share_never_exceeds_the_whole_term():
    c = report()["clock_immune_cost"]
    assert 0 < c["attributable_to_the_diffusion_loop_s"] < c["total_s"]
    assert abs(c["attributable_to_the_diffusion_loop_s"] + c["remainder_s"] - c["total_s"]) < 1e-9


def test_per_call_costs_are_consistent_with_their_denominators():
    import trace_reach as m
    c = report()["clock_immune_cost"]
    assert abs(c["per_top_level_call_us"] - 1e6 * m.FIXED_S / m.TOP_LEVEL_CALLS) < 1e-9
    assert abs(c["per_launching_call_us"] - 1e6 * m.FIXED_S / m.LAUNCHING_CALLS) < 1e-9
    assert c["per_launching_call_us"] > c["per_top_level_call_us"]


def test_the_limits_name_the_uniform_cost_assumption():
    joined = " ".join(report()["limits"]).lower()
    assert "uniform" in joined and "instrument" in joined


def test_it_does_not_claim_device_programs_disappear():
    t = report()["what_this_predicts"].lower()
    assert "does not remove the device" in t


def test_report_reproduces_byte_for_byte():
    got = subprocess.run([sys.executable, str(HERE / "trace_reach.py")],
                         capture_output=True, text=True, check=True).stdout
    assert json.loads(got) == report()
