"""Every model behind a CLI --model choice is perf-gated or exempted, with a reason.

The assertion this covers was written against the three model tuples that existed then, so
a new CLI verb with its own tuple reopened exactly the hole it was built to close: nesso1
could have shipped with zero perf coverage and the check would have stayed green. It now
discovers the tuples, and this test is what keeps it discovering them.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _perf_regression():
    sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location(
        "tt_bio_perf_regression", REPO / "scripts" / "perf_regression.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_shipped_model_is_covered_or_exempt():
    _perf_regression()._assert_full_model_coverage()


def test_a_new_model_tuple_is_not_a_free_pass(monkeypatch):
    """A future verb bringing its own *_MODELS tuple must fail loudly, not silently."""
    from tt_bio import main as tt_main

    monkeypatch.setattr(tt_main, "IMAGINARY_MODELS", ("not-a-real-model",), raising=False)
    with pytest.raises(SystemExit, match="not-a-real-model"):
        _perf_regression()._assert_full_model_coverage()
