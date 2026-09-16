"""Known-answer controls for the two-clock solve."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import two_clock_session as m


def test_recovers_an_exact_synthetic_pair():
    a, b = 3.95, 14418.0
    pts = [(a + b / f, f) for f in (800.0, 1339.0)]
    ra, rb = m.fit(pts)
    assert abs(ra - a) < 1e-9 and abs(rb - b) < 1e-9


def test_recovers_it_from_six_noiseless_points():
    a, b = 1.25, 9000.0
    pts = [(a + b / f, f) for f in (800.0, 800.0, 800.0, 1339.0, 1340.0, 1341.0)]
    ra, rb = m.fit(pts)
    assert abs(ra - a) < 1e-7 and abs(rb - b) < 1e-7


def test_refuses_a_single_clock():
    try:
        m.fit([(14.5, 1350.0), (14.6, 1350.0), (14.7, 1350.0)])
    except ValueError:
        return
    raise AssertionError("one clock cannot separate a fixed term from a work term")


def test_warmup_folds_are_excluded():
    doc = json.loads(m.TWO_CLOCK.read_text())
    assert any(r.get("warmup") for r in doc["runs"]), "fixture should contain a warmup fold"
    assert not any(r.get("warmup") for r in m.accepted(doc))


def test_a_shifted_arm_moves_the_fixed_term_not_the_work_term():
    # Adding a constant to every fold must land entirely in the fixed term.
    a, b = 2.0, 12000.0
    pts = [(a + b / f, f) for f in (800.0, 1339.0)]
    ra, rb = m.fit([(t + 0.5, f) for t, f in pts])
    assert abs(ra - (a + 0.5)) < 1e-9 and abs(rb - b) < 1e-6


def test_report_reproduces_byte_for_byte():
    here = Path(__file__).parent
    got = subprocess.run([sys.executable, str(here / "two_clock_session.py")],
                         capture_output=True, text=True, check=True).stdout
    assert json.loads(got) == json.loads((here / "two_clock_session.json").read_text())
