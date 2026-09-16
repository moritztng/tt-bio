"""Controls: the check must read the tree, and must fail loudly if the gate moves."""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def report():
    return json.loads((HERE / "route_reach.json").read_text())


def test_it_found_the_gate_in_the_actual_source():
    d = report()
    assert d["gate_found_in_source"] is True
    assert d["threshold"]["value"] > 0
    assert d["flag"]["env"] == "TT_BIO_SDPA_FUSED_LARGE_S"


def test_reachability_agrees_with_the_threshold():
    d = report()
    t = d["threshold"]["value"]
    for size, reach in d["reachable_at"].items():
        assert reach == (int(size) > t), (size, reach, t)


def test_the_campaign_sizes_cannot_reach_it():
    r = report()["reachable_at"]
    assert r["512"] is False and r["298"] is False


def test_a_larger_size_can_reach_it_so_the_check_is_not_vacuous():
    assert report()["reachable_at"]["1536"] is True


def test_report_reproduces_byte_for_byte():
    got = subprocess.run([sys.executable, str(HERE / "check_route_reach.py")],
                         capture_output=True, text=True, check=True).stdout
    assert json.loads(got) == report()
