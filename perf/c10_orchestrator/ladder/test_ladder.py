"""Controls for the ladder consolidation."""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import ladder as L


def report():
    return json.loads((HERE / "ladder.json").read_text())


def test_mutually_exclusive_rows_are_never_both_counted():
    p = report()["priced_total"]
    assert "TT_BIO_DIT_FUSED_QKV" in p["dropped_as_mutually_exclusive"]
    assert "TT_BIO_DIT_FUSED_QKV" not in p["items"]
    assert p["fold_s"] < p["naive_sum_including_excluded_s"]


def test_the_exclusion_drops_the_cheaper_side():
    by = {r["item"]: r["fold_s"] for r in L.LADDER if r["fold_s"] is not None}
    assert by["TT_BIO_DIT_FUSED_QKV"] < by["TT_BIO_HEAD_PAD_TAIL"]


def test_exclusions_are_declared_symmetrically():
    by = {r["item"]: r for r in L.LADDER}
    for r in L.LADDER:
        for other in r.get("excludes", ()):
            assert r["item"] in by[other].get("excludes", ()), (r["item"], other)


def test_the_ceiling_row_is_not_counted_as_a_candidate():
    p = report()["priced_total"]
    assert not any("byte axis" in i for i in p["items"])


def test_every_row_carries_an_evidence_class_and_a_basis():
    classes = {L.MEASURED_BH_CLOCKED, L.MEASURED_BH_UNCLOCKED, L.MEASURED_WH,
               L.TRANSFERRED, L.DERIVED, L.UNMEASURED}
    for r in report()["ladder"]:
        assert r["evidence"] in classes, r
        assert r["basis"].strip() and r["accuracy_spend"].strip(), r


def test_mcycles_track_seconds_at_the_stated_clock():
    d = report()
    for r in d["ladder"]:
        if r["fold_s"] is None:
            assert r["Mcycles"] is None
        else:
            assert abs(r["Mcycles"] - r["fold_s"] * d["baseline"]["clock_MHz"]) < 1e-6


def test_the_total_does_not_reach_the_target():
    # If this ever fails the campaign has changed and the honest_reading text must be rewritten.
    d = report()
    assert d["priced_total"]["would_read_s"] > d["target_s"]


def test_the_reading_refuses_to_call_the_total_a_forecast():
    joined = " ".join(report()["honest_reading"]).lower()
    assert "not a prediction" in joined or "not a forecast" in joined
    assert "sub-additive" in joined


def test_report_reproduces_byte_for_byte():
    got = subprocess.run([sys.executable, str(HERE / "ladder.py")],
                         capture_output=True, text=True, check=True).stdout
    assert json.loads(got) == report()
