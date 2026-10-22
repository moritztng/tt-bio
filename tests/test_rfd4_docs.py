"""docs/rfd4-design.md states what the CLI and the sampler actually do.

The model page is what a user reads before running RFD4. Its recipe table prices each
schedule in trunk passes and its examples name CLI flags; both are hand-written, so this
pins them to the code: every documented recipe exists and costs what the page says, every
shipped recipe is on the page, and every flag the page names is one `design` accepts.
"""
import re
from pathlib import Path

from click.testing import CliRunner

from tt_bio.main import cli
from tt_bio.rfd4.sampler import RECYCLE_SCHEDULES

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "rfd4-design.md"


def _recipe_table() -> dict[str, int]:
    """`--recipe` table rows: every backticked name in the first cell -> the pass count."""
    text = DOC.read_text()
    table = text.split("| `--recipe` | trunk passes |", 1)[1].split("\n\n", 1)[0]
    rows = {}
    for line in table.splitlines():
        m = re.match(r"\|(.*)\|\s*(\d+)\s*\|$", line)
        if m:
            for name in re.findall(r"`([^`]+)`", m.group(1)):
                rows[name] = int(m.group(2))
    return rows


def test_every_documented_recipe_costs_what_the_page_says():
    rows = _recipe_table()
    assert rows, "the recipe table was not found in docs/rfd4-design.md"
    for name, passes in rows.items():
        assert name in RECYCLE_SCHEDULES, f"the page documents {name!r}, the sampler has no such recipe"
        steps = sum(n for n, _ in RECYCLE_SCHEDULES[name])
        assert steps == 60, f"{name}: {steps} steps"
        assert sum(n * k for n, k in RECYCLE_SCHEDULES[name]) == passes, name


def test_every_shipped_recipe_is_documented():
    assert set(RECYCLE_SCHEDULES) == set(_recipe_table())


def test_every_flag_the_page_names_is_a_design_flag():
    help_text = CliRunner().invoke(cli, ["design", "--help"]).output
    named = set(re.findall(r"(?<![\w-])--[a-z][a-z_]+", DOC.read_text()))
    assert named
    missing = sorted(f for f in named if f not in help_text)
    assert not missing, f"docs/rfd4-design.md names flags `design` does not take: {missing}"


def test_the_page_is_linked_from_the_readme():
    assert "docs/rfd4-design.md" in (ROOT / "README.md").read_text()
