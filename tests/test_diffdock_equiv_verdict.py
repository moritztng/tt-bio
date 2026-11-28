"""The SE(3) verdict the device grader gives a set of per-transform gaps (card-free)."""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")

_SPEC = importlib.util.spec_from_file_location(
    "device_pose_grade",
    Path(__file__).resolve().parents[1] / "scripts" / "diffdock_port" / "device_pose_grade.py")


@pytest.fixture(scope="module")
def grade():
    mod = importlib.util.module_from_spec(_SPEC)
    try:
        _SPEC.loader.exec_module(mod)
    except ImportError as e:      # the grader imports the device stack at module level
        pytest.skip(str(e))
    return mod


def _steps(gaps, host=1e-14):
    cell = lambda g: {"gap_b": g, "gap_a": gaps[0], "host_residual": host}
    return {"18": {str(i): {n: cell(g) for n in ("tr", "rot", "tor")} for i, g in enumerate(gaps)}}


# The two populations that started this: 17 torsion gaps at step 18, identity first.
STEPPED = [0.0303, 0.0206, 0.0221, 0.0226, 0.0261, 0.0282, 0.0301, 0.0307, 0.0307, 0.0308,
           0.0309, 0.0315, 0.0372, 0.0383, 0.0455, 0.0518, 0.0530]
RESIDENT = [0.0248, 0.0208, 0.0249, 0.0253, 0.0264, 0.0267, 0.0290, 0.0301, 0.0316, 0.0325,
            0.0337, 0.0347, 0.0350, 0.0368, 0.0423, 0.0470, 0.0594]


def test_same_population_passes_wherever_the_identity_draw_lands(grade):
    ok, table = grade.equiv_verdict(_steps(STEPPED))
    assert ok and table["18"]["tor"]["worst_over_median"] < 2
    ok, table = grade.equiv_verdict(_steps(RESIDENT), _steps(STEPPED), needs_ref=True)
    assert ok
    assert RESIDENT[-1] > 2 * RESIDENT[0]          # the single-draw rule failed this one
    assert table["18"]["tor"]["median_over_stepped"] < 1.25


def test_a_broken_transform_fails(grade):
    ok, table = grade.equiv_verdict(_steps(STEPPED[:-1] + [0.9]))
    assert not ok and not table["18"]["tor"]["pass"]


def test_a_uniformly_worse_form_fails_the_median_guard(grade):
    worse = [1.4 * g for g in STEPPED]
    assert grade.equiv_verdict(_steps(worse))[0]   # the median alone cannot see it
    ok, table = grade.equiv_verdict(_steps(worse), _steps(STEPPED), needs_ref=True)
    assert not ok and table["18"]["tor"]["median_over_stepped"] > 1.25


def test_a_resident_run_without_the_stepped_reference_fails_closed(grade):
    assert not grade.equiv_verdict(_steps(RESIDENT), None, needs_ref=True)[0]


def test_a_host_reference_that_misses_its_bar_fails(grade):
    assert not grade.equiv_verdict(_steps(STEPPED, host=1e-4))[0]
    assert not grade.equiv_verdict(_steps(STEPPED, host=1e-4), device=False)[0]
