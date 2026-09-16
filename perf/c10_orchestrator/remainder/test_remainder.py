"""Controls: this note must not present a cross-machine cost model as a measurement."""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def report():
    return json.loads((HERE / "remainder.json").read_text())


def test_the_remainder_is_the_stated_subtraction():
    d = report()
    assert abs(d["unowned_remainder_s"]
               - (d["clock_immune_term_s"] - d["diffusion_trace_reach_s"])) < 1e-9


def test_the_device_resident_stage_is_excluded_from_the_total():
    d = report()
    assert "diffusion_conditioning" in d["moved_to_device_since"]
    expected = sum(v for k, v in d["host_stages_measured_on_pc_s"].items()
                   if k not in d["moved_to_device_since"])
    assert abs(d["still_host_total_s"] - expected) < 1e-9
    assert d["still_host_total_s"] < sum(d["host_stages_measured_on_pc_s"].values())


def test_it_does_not_claim_the_seconds_transfer():
    joined = " ".join(report()["limits"]).lower()
    assert "do not transfer" in joined
    assert "not on qb2" in joined or "not qb2" in joined


def test_it_admits_both_inputs_are_partly_derived():
    joined = " ".join(report()["limits"]).lower()
    assert "derived" in joined and "coincidence" in joined


def test_the_source_is_a_cpu_only_synthetic_census():
    s = report()["source"]
    assert s["host"] == "pc"
    assert "CPU only" in s["note"] or "no device" in s["note"]


def test_the_accounted_fraction_is_under_one():
    assert 0.0 < report()["fraction_of_remainder_accounted"] < 1.0


def test_report_reproduces_byte_for_byte():
    got = subprocess.run([sys.executable, str(HERE / "remainder.py")],
                         capture_output=True, text=True, check=True).stdout
    assert json.loads(got) == report()
