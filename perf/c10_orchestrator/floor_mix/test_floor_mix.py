"""Controls for the floor ceiling analysis."""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def report():
    return json.loads((HERE / "floor_mix.json").read_text())


def test_floor_is_at_least_the_larger_sum_because_max_is_per_call():
    t = report()["totals_s"]
    assert t["floor"] >= max(t["arithmetic"], t["traffic"]) - 1e-9
    assert abs(t["floor"] - t["reported_total_floor"]) < 1e-6


def test_floor_is_at_most_the_sum_of_both_terms():
    t = report()["totals_s"]
    assert t["floor"] <= t["arithmetic"] + t["traffic"] + 1e-9


def test_each_ceiling_is_consistent_with_its_own_arithmetic():
    d = report()
    t, c = d["totals_s"], d["ceilings"]
    for key, base in (("delete_every_byte", t["arithmetic"]), ("make_arithmetic_free", t["traffic"])):
        e = c[key]
        assert abs(e["floor_becomes_s"] - base) < 1e-9
        assert abs(e["wins_s"] - (t["floor"] - base)) < 1e-9
        assert abs(e["ratio"] - t["floor"] / base) < 1e-9


def test_byte_headroom_never_exceeds_a_class_floor_and_is_never_negative():
    for x in report()["where_the_byte_axis_lives"]:
        assert 0.0 <= x["byte_headroom_s"] <= x["floor_s"] + 1e-9, x


def test_byte_headroom_ranking_is_not_just_the_floor_ranking():
    # The point of the analysis: the biggest classes are not where the byte axis lives.
    rows = report()["where_the_byte_axis_lives"]
    assert rows[0]["op"] != max(rows, key=lambda x: x["floor_s"])["op"]


def test_the_roof_clock_is_reported_as_absent_not_guessed():
    p = report()["roof_provenance"]
    assert p["recorded_clock"] is None and p["cube4096_TFLOPs"] > 0


def test_report_reproduces_byte_for_byte():
    got = subprocess.run([sys.executable, str(HERE / "floor_mix.py")],
                         capture_output=True, text=True, check=True).stdout
    assert json.loads(got) == report()
