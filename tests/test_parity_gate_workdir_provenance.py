"""A parity-gate workdir is bound to the code it scored.

`full_parity_gate.py` caches one JSON report per leg in the workdir and resumes from it by
default, keyed on the leg id alone. The default workdir is a fixed path, so pointing a second
release gate at it replays the first release's verdicts: on 2026-08-22 the v0.6.6 cut resumed
all 32 legs from a workdir that predated the accurate-softmax default flip and printed a full
green tally, and the only thing that gave it away was `total wall 0.0 min`.

A gate that cannot tell whose code it scored is not a gate, so the workdir carries a
fingerprint of `tt_bio/` + `scripts/` and a resume across a change to either is refused.
"""
import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts" / "full_parity_gate.py"


@pytest.fixture(scope="module")
def gate():
    """Import the gate module without running it (it has no import-time side effects)."""
    spec = importlib.util.spec_from_file_location("_fpg_under_test", GATE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # dataclasses resolves annotations via sys.modules
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(spec.name, None)
    return mod


def test_fingerprint_is_stable_and_covers_model_code(gate):
    assert gate.code_fingerprint() == gate.code_fingerprint()
    assert set(gate._FINGERPRINT_ROOTS) >= {"tt_bio", "scripts"}, (
        "the fingerprint must cover every tree that can move a number")


def test_first_run_stamps_the_workdir(gate, tmp_path):
    gate.check_workdir_provenance(tmp_path, resume=True)
    stamp = json.loads((tmp_path / "GATE_CODE.json").read_text())
    assert stamp["code_fingerprint"] == gate.code_fingerprint()


def test_resume_on_the_same_code_is_allowed(gate, tmp_path):
    gate.check_workdir_provenance(tmp_path, resume=True)
    gate.check_workdir_provenance(tmp_path, resume=True)   # must not raise


def test_resume_across_a_code_change_is_refused(gate, tmp_path):
    (tmp_path / "GATE_CODE.json").write_text(json.dumps({"code_fingerprint": "deadbeef"}))
    with pytest.raises(SystemExit) as e:
        gate.check_workdir_provenance(tmp_path, resume=True)
    msg = str(e.value)
    assert "REFUSING TO RESUME" in msg
    assert "deadbeef" in msg and "--fresh" in msg, "the refusal must name both keys and the way out"


def test_fresh_rewrites_the_stamp_instead_of_refusing(gate, tmp_path):
    (tmp_path / "GATE_CODE.json").write_text(json.dumps({"code_fingerprint": "deadbeef"}))
    gate.check_workdir_provenance(tmp_path, resume=False)  # --fresh: no cached verdict is read
    stamp = json.loads((tmp_path / "GATE_CODE.json").read_text())
    assert stamp["code_fingerprint"] == gate.code_fingerprint()
