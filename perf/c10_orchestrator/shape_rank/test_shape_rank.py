"""Controls for the shape ranking."""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def report():
    return json.loads((HERE / "shape_rank.json").read_text())


def test_rows_are_sorted_and_the_largest_is_the_first():
    t = report()["top"]
    assert t == sorted(t, key=lambda r: -r["floor_s"])
    assert report()["largest_single_item"] == t[0]


def test_per_call_cost_is_consistent_with_calls_and_floor():
    for r in report()["top"]:
        assert abs(r["us_per_call"] - 1e6 * r["floor_s"] / r["calls"]) < 1e-6


def test_no_shape_dominates_so_the_no_giant_claim_holds():
    d = report()
    assert d["top"][0]["pct_of_floor"] < 10.0
    assert any("no giant hiding" in o for o in d["observations"])


def test_the_top_slice_does_not_exceed_the_whole_floor():
    d = report()
    assert 0 < d["top_share_pct"] < 100.0
    assert sum(r["floor_s"] for r in d["top"]) <= d["total_floor_s"] + 1e-9


def test_pure_traffic_subset_is_a_subset():
    d = report()
    assert d["pure_traffic_elementwise"]["floor_s"] < d["total_floor_s"]
    assert d["pure_traffic_elementwise"]["calls"] < d["total_calls"]


def test_limits_name_the_missing_clock():
    joined = " ".join(report()["limits"]).lower()
    assert "no recorded clock" in joined and "not measured time" in joined


def test_report_reproduces_byte_for_byte():
    got = subprocess.run([sys.executable, str(HERE / "shape_rank.py")],
                         capture_output=True, text=True, check=True).stdout
    assert json.loads(got) == report()
