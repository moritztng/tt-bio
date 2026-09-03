"""The README's limit table has to be the table the engine enforces.

`76919974` fixed this once by hand: the README promised "up to at least 1095 residues ... on
every structure model including OpenDDE" while OpenDDE throws at 576. The numbers were corrected
and nothing was left watching them, so the next edit to either side could drift them apart again
-- and a reader treats the README as the contract, which is what makes this class of gap worse
than a missing doc.

So this reads README.md and compares it against `size_limits.CEILINGS` directly. Both halves
matter and they fail differently:

  - A MEASURED row must appear with its number. A README that understates a ceiling turns away
    work the engine does; one that overstates it promises a fold that throws.
  - An UNMEASURED model must be named as never refused. This is the half that was actually still
    wrong after 76919974: the README named four of the eleven, so a user reading it would think
    `protenix-v1` and `esmfold2-fast` had a ceiling when nothing refuses them at any size.

Models are DISCOVERED from CEILINGS, never listed here, for the reason
`test_shipped_models_are_documented` discovers its own: a hand-copied list is blind to exactly
the model most likely to have been forgotten -- the new one. Card-free.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tt_bio import size_limits as sl

README = Path(__file__).resolve().parents[1] / "README.md"


_HEADER = "| model | Wormhole limit | first measured failure |"


def _rows(text: str) -> list[str]:
    """The body rows of README's limit table, and only that table.

    Scoped by its header rather than by "any markdown row", because the capability matrix a few
    lines above also has a `opendde` row -- matching across both tables found two hits per model
    and made every lookup ambiguous."""
    lines = text.splitlines()
    assert _HEADER in lines, (
        f"README no longer has the limit table with header {_HEADER!r}; this whole file is "
        f"pinned to it.")
    body = lines[lines.index(_HEADER) + 1:]
    out = []
    for ln in body:
        if not ln.startswith("|"):
            break
        if "---" not in ln:
            out.append(ln)
    return out


def _measured():
    return sorted(m for m, a in sl.CEILINGS.items()
                  if a["wormhole_b0"].binds != sl.UNMEASURED)


def _unmeasured():
    return sorted(m for m, a in sl.CEILINGS.items()
                  if a["wormhole_b0"].binds == sl.UNMEASURED)


def _row_for(text: str, model: str) -> str:
    """The one table row naming `model`. Backticks, so `esmfold2` never matches
    `esmfold2-fast` -- a substring test here would silently accept a missing model."""
    hits = [ln for ln in _rows(text) if f"`{model}`" in ln]
    assert len(hits) == 1, (
        f"README's limit table should have exactly one row naming `{model}`, found "
        f"{len(hits)}. Every model with a measured ceiling needs a row a reader can look up.")
    return hits[0]


@pytest.mark.parametrize("model", _measured())
def test_a_measured_ceiling_is_in_the_readme_with_its_number(model):
    c = sl.CEILINGS[model]["wormhole_b0"]
    row = _row_for(README.read_text(), model)
    assert str(c.residues) in row, (
        f"README says {row.strip()!r} for `{model}`, but size_limits refuses above "
        f"{c.residues}. The README is the number a user plans against, so it has to be the "
        f"number the engine enforces.")


@pytest.mark.parametrize("model", _measured())
def test_the_first_measured_failure_is_in_the_readme_too(model):
    """The limit alone invites the wrong inference. OpenDDE folds 544, throws at 576 and folds
    608 again, so a reader who sees only "544" assumes everything below it is safe and
    everything above it is not; the failure column is what says the wall is not monotonic."""
    c = sl.CEILINGS[model]["wormhole_b0"]
    if not isinstance(c.fail_at, int):
        pytest.skip(f"`{model}` has no recorded first failure ({c.fail_at!r})")
    row = _row_for(README.read_text(), model)
    assert str(c.fail_at) in row, (
        f"README's row for `{model}` is {row.strip()!r} and does not carry its first measured "
        f"failure ({c.fail_at}).")


@pytest.mark.parametrize("model", _unmeasured())
def test_a_model_that_is_never_refused_says_so_in_the_readme(model):
    """Absence of a limit is not a limit, and the README has to say which models have none.
    Naming a subset is the drift this file exists for."""
    text = README.read_text()
    assert f"`{model}`" in text, f"README never names `{model}`."
    para = [p for p in text.split("\n\n") if "never refused" in p]
    assert para, "README no longer says which models are 'never refused'."
    assert any(f"`{model}`" in p for p in para), (
        f"`{model}` has no measured ceiling and is never refused at any size, but README's "
        f"never-refused list does not name it, so a reader would assume it has a limit.")


@pytest.mark.parametrize("model", _measured())
def test_a_measured_model_is_not_listed_as_never_refused(model):
    """The mirror of the check above: `esmc-6b` caps at 1968 and must not sit in the
    never-refused list just because its `esmc` siblings do."""
    para = [p for p in README.read_text().split("\n\n") if "never refused" in p]
    assert not any(f"`{model}`" in p for p in para), (
        f"README lists `{model}` as never refused, but size_limits caps it at "
        f"{sl.CEILINGS[model]['wormhole_b0'].residues}.")


def test_the_readme_claims_the_refusal_happens_before_a_device_opens():
    """The one behavioural promise in that section, and the reason the table is worth having:
    the guard runs on the host, so a refused input costs a second and leaves no chip dirty."""
    text = README.read_text()
    assert "before it opens a device" in text or "before any device" in text, \
        "README no longer says the refusal precedes the device open."
    assert "TT_BIO_SIZE_LIMIT" in text, \
        "README no longer names the escape hatch, so a false refusal has no documented way past."


def test_a_wrong_readme_number_is_caught(tmp_path, monkeypatch):
    """Negative control. It has to break what the checks above actually read -- a stub that
    merely returns nothing would pass them vacuously."""
    fake = tmp_path / "README.md"
    real = README.read_text()
    model = _measured()[0]
    row = _row_for(real, model)
    fake.write_text(real.replace(row, row.replace(
        str(sl.CEILINGS[model]["wormhole_b0"].residues), "99999")))
    monkeypatch.setattr(sys.modules[__name__], "README", fake)
    with pytest.raises(AssertionError, match="engine enforces"):
        test_a_measured_ceiling_is_in_the_readme_with_its_number(model)


def test_a_dropped_never_refused_model_is_caught(tmp_path, monkeypatch):
    """The other direction: shortening the never-refused list must fail, because that is
    exactly the state the README was in before this file existed."""
    fake = tmp_path / "README.md"
    model = _unmeasured()[0]
    fake.write_text(README.read_text().replace(f"`{model}`, ", "", 1))
    monkeypatch.setattr(sys.modules[__name__], "README", fake)
    with pytest.raises(AssertionError, match="never-refused list does not name it"):
        test_a_model_that_is_never_refused_says_so_in_the_readme(model)

