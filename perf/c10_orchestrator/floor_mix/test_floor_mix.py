"""Controls for the floor split. It must not invent a binding term or drop a class."""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def report():
    return json.loads((HERE / "floor_mix.json").read_text())


def test_the_two_binding_terms_sum_to_the_total():
    d = report()
    b = d["by_binding_term"]
    assert abs(b["arithmetic_s"] + b["traffic_s"] - d["total_floor_s"]) < 1e-9


def test_percentages_sum_to_a_hundred():
    b = report()["by_binding_term"]
    assert abs(b["arithmetic_pct"] + b["traffic_pct"] - 100.0) < 1e-9


def test_every_top_class_binds_on_its_larger_term():
    for x in report()["top_classes"]:
        larger = "arithmetic" if x["arith_s"] >= x["traffic_s"] else "traffic"
        assert x["binds"] == larger, x
        assert abs(x["floor_s"] - max(x["arith_s"], x["traffic_s"])) < 1e-9, x


def test_the_roof_clock_is_reported_as_absent_not_guessed():
    p = report()["roof_provenance"]
    assert p["recorded_clock"] is None
    assert p["cube4096_TFLOPs"] > 0


def test_concentration_is_a_subset_not_a_restatement():
    d = report()
    assert d["concentration"]["pct_of_calls"] < 50.0
    assert d["concentration"]["pct_of_floor"] > 50.0


def test_report_reproduces_byte_for_byte():
    got = subprocess.run([sys.executable, str(HERE / "floor_mix.py")],
                         capture_output=True, text=True, check=True).stdout
    assert json.loads(got) == report()
