"""Controls: the provenance read must not silently accept a cross-architecture transfer."""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def report():
    return json.loads((HERE / "grid_evidence.json").read_text())


def test_the_ranked_claim_is_wormhole_and_says_so():
    p = report()["its_provenance"]
    assert "WORMHOLE" in p["arch"].upper()
    assert p["cores"] == 72 and p["grid"] == [8, 9]


def test_blackhole_is_a_different_grid():
    d = report()
    assert "BLACKHOLE" in d["what_blackhole_actually_has"]["arch"].upper()
    assert d["what_blackhole_actually_has"]["grid"] != d["its_provenance"]["grid"]


def test_no_blackhole_core_sweep_is_claimed():
    b = report()["what_blackhole_actually_has"]
    assert b["core_count_sweep_exists"] is False
    assert b["every_arm_ran_the_full_grid"] is True


def test_the_two_classes_move_in_opposite_directions_on_the_same_sweep():
    d = report()
    assert d["its_provenance"]["product_win_32_over_72"] > 1.0
    assert d["the_same_sweep_moves_the_other_way_for_sdpa"]["sdpa_ratio_32_over_72"] > 1.0


def test_neither_source_recorded_a_clock():
    d = report()
    assert d["its_provenance"]["recorded_clock"] is None
    assert d["what_blackhole_actually_has"]["recorded_clock"] is None


def test_rate_ratios_are_consistent_with_their_inputs():
    b = report()["what_blackhole_actually_has"]
    cube = b["dense_cube_same_session"]["TFLOPs_at_min"]
    for cls, key in (("trimul", "trimul_vs_cube_rate_ratio"), ("triatt", "triatt_vs_cube_rate_ratio")):
        assert abs(b[key] - cube / b[cls]["TFLOPs_at_min"]) < 1e-9


def test_report_reproduces_byte_for_byte():
    got = subprocess.run([sys.executable, str(HERE / "grid_evidence.py")],
                         capture_output=True, text=True, check=True).stdout
    assert json.loads(got) == report()
