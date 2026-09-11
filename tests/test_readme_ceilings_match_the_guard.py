"""The README's published ceiling table has to say what the guard actually enforces.

The table is the number a user reads before deciding whether to submit a job;
``size_limits.CEILINGS`` is the number that refuses it. Nothing kept them equal, and this
merge is exactly the shape of change that separates them: three branches raised three
ceilings, and one of them re-added a stale table row for a model another branch had already
moved, which would have re-published 627 for rf3 after it went to 1095.

The invariant is one-directional on purpose. Every row in the table must match the guard.
The reverse is NOT required: rf3 has a measured ceiling and is deliberately absent from the
table, because the prose under it says it folds every rung to the top of its ladder, and
`boltz2` and `nesso1` are absent because they have no measured limit at all. Requiring a row
for every ceiling would force those editorial calls into the table.

The published number carries its own UNIT, read off the row rather than assumed: boltzgen's
14786 is atoms in the target, not residues, and a failure message that called it residues would
send the reader looking for the wrong stale number.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from tt_bio import size_limits

README = Path(__file__).resolve().parent.parent / "README.md"
ARCH = "wormhole_b0"
HEADER = "| model | Wormhole limit | first measured failure |"


def _table_rows(text: str) -> list[tuple[list[str], str, str]]:
    """(models, limit cell, failure cell) for every data row of the ceiling table."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip() == HEADER)
    rows = []
    for ln in lines[start + 2:]:
        if not ln.startswith("|"):
            break
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) != 3:
            break
        models = re.findall(r"`([^`]+)`", cells[0])
        rows.append((models, cells[1], cells[2]))
    return rows


def _rows():
    return _table_rows(README.read_text())


def test_the_table_was_found_and_is_not_empty():
    """A parser that silently matches nothing would make every test below vacuous."""
    rows = _rows()
    assert len(rows) >= 5
    assert all(models for models, _, _ in rows)


@pytest.mark.parametrize("row", _rows(), ids=lambda r: ",".join(r[0]))
def test_the_published_limit_is_the_enforced_limit(row):
    models, limit_cell, _ = row
    published = int(re.match(r"(\d+)", limit_cell).group(1))
    for model in models:
        c = size_limits.ceiling(model, ARCH)
        assert c is not None and c.residues is not None, f"{model}: no measured ceiling to publish"
        unit = size_limits._COUNT_NAMES[c.counts]
        assert c.residues == published, (
            f"README publishes {model} at {published} {unit}, size_limits enforces "
            f"{c.residues}. One of them is stale.")


@pytest.mark.parametrize("row", _rows(), ids=lambda r: ",".join(r[0]))
def test_the_published_failure_is_the_measured_failure(row):
    models, _, fail_cell = row
    for model in models:
        c = size_limits.ceiling(model, ARCH)
        stated = re.match(r"(\d+)", fail_cell)
        if c.fail_at is None:
            # A ladder with nothing above its top has no failure to name, and saying a number
            # here would invent one.
            assert stated is None, (
                f"README names a first failure for {model} but size_limits records fail_at=None "
                f"({c.binds}); the table is claiming a measurement that does not exist.")
        elif isinstance(c.fail_at, int):
            assert stated is not None and int(stated.group(1)) == c.fail_at, (
                f"README says {model} first fails at {fail_cell!r}, size_limits records "
                f"{c.fail_at}.")
        # a non-integer fail_at (UNRECORDED) is a prose cell by construction; nothing to compare


@pytest.mark.parametrize("row", _rows(), ids=lambda r: ",".join(r[0]))
def test_a_token_wall_says_so_in_the_table(row):
    """Where a ligand counts against the number, the cell publishing that number has to say so.

    The limit reads as a residue count, and on these rows it is a residue count PLUS the ligand's
    heavy atoms. A user comparing their sequence length against a bare "1024" would submit a
    cocrystal the guard then refuses, which is the table telling them the wrong thing.
    """
    models, limit_cell, _ = row
    for model in models:
        if size_limits.ceiling(model, ARCH).token_bound:
            assert "ligand" in limit_cell.lower(), (
                f"{model}'s ceiling counts ligand atoms as tokens, and the table cell "
                f"{limit_cell!r} does not say so.")


def test_a_drifted_number_fails(monkeypatch):
    """The negative control. Without it a parser that found no rows would pass everything."""
    models, limit_cell, fail_cell = _rows()[0]
    model = models[0]
    c = size_limits.ceiling(model, ARCH)
    import dataclasses
    bumped = dataclasses.replace(c, residues=c.residues + 64)
    monkeypatch.setitem(size_limits.CEILINGS[model], ARCH, bumped)
    with pytest.raises(AssertionError):
        test_the_published_limit_is_the_enforced_limit((models, limit_cell, fail_cell))
