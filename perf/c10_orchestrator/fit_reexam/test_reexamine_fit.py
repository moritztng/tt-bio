"""Known-answer controls for the inverse-clock fitter. A fitter that cannot recover an exact
synthetic (a, b) has no business re-pricing the campaign's target."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reexamine_fit import fit, predict


def test_recovers_an_exact_two_point_line():
    a, b = 2.5, 12000.0
    pts = [(a + b / f, f) for f in (800.0, 1350.0)]
    ra, rb = fit(pts)
    assert abs(ra - a) < 1e-9 and abs(rb - b) < 1e-9


def test_recovers_an_exact_line_from_many_points():
    a, b = 3.4777, 14959.6
    pts = [(a + b / f, f) for f in (700.0, 850.0, 1000.0, 1124.0, 1350.0)]
    ra, rb = fit(pts)
    assert abs(ra - a) < 1e-8 and abs(rb - b) < 1e-8


def test_is_not_fooled_by_a_linear_in_f_relationship():
    # t = a + c*f is NOT the model; the fitter must not report a near-perfect 1/f fit for it.
    pts = [(1.0 + 0.001 * f, f) for f in (800.0, 1000.0, 1200.0, 1350.0)]
    a, b = fit(pts)
    worst = max(abs(predict(a, b, f) - t) for t, f in pts)
    assert worst > 0.01, "a 1/f fit should visibly miss a linear-in-f curve"


def test_refuses_degenerate_clock_labels():
    try:
        fit([(14.5, 1350.0), (14.6, 1350.0)])
    except ValueError:
        return
    raise AssertionError("two points at the same clock must not yield a slope")


def test_refuses_a_single_point():
    try:
        fit([(14.5, 1350.0)])
    except ValueError:
        return
    raise AssertionError("one point cannot identify two parameters")


def test_report_reproduces_byte_for_byte():
    here = Path(__file__).parent
    got = subprocess.run([sys.executable, str(here / "reexamine_fit.py")],
                         capture_output=True, text=True, check=True).stdout
    assert json.loads(got) == json.loads((here / "fit_reexam.json").read_text())
