"""Every model behind a --model choice is named in the README and on the benchmark page.

The page's capability tables and the README's opening sentence are static markup: they
render even when the data fetch fails, and they go stale silently. Protenix-v1 shipped
and neither listed it until someone noticed by reading.

The model tuples are DISCOVERED from ``tt_bio.main``, not named here, for the reason
``perf_regression._assert_full_model_coverage`` discovers them: a new CLI verb brings its
own ``*_MODELS`` tuple, and a hand-copied list here would be blind to exactly the model
most likely to be undocumented. That check and this one stay separate copies of two lines
because it runs inside the gate script before any device work and this runs in pytest.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SURFACES = ("README.md", "site/benchmarks/index.html")


def _shipped() -> set:
    from tt_bio import main

    return set().union(*(getattr(main, n) for n in dir(main) if n.endswith("_MODELS")))


@pytest.mark.parametrize("surface", SURFACES)
def test_every_shipped_model_id_appears(surface):
    text = (REPO / surface).read_text()
    missing = sorted(m for m in _shipped() if m not in text)
    assert not missing, (
        f"{surface} does not name {missing}. A model reachable from a --model choice "
        f"has to be findable by the id a reader would type.")


@pytest.mark.parametrize("surface", SURFACES)
def test_a_model_the_surface_does_not_name_fails(monkeypatch, surface):
    """Negative control: the check reads the surface, it does not just count tuples."""
    from tt_bio import main

    monkeypatch.setattr(main, "IMAGINARY_MODELS", ("not-a-real-model",), raising=False)
    with pytest.raises(AssertionError, match="not-a-real-model"):
        test_every_shipped_model_id_appears(surface)
