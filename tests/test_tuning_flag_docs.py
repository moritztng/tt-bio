"""The release notes and the README tuning table must not drift apart.

tt-bio carries ~195 environment flags. Most are internal A/B pins, tracing switches and
budgets, and the README table is deliberately a curated list of the device optimizations a
user chooses between, not a dump of every variable. So the rule here is narrow and is the
one that was actually broken: a flag the CHANGELOG *names* in the release being written is,
by the act of naming it, part of that release's user-facing story, and it needs a row.

This exists because 0.9.0 was assembled and eight named flags had no row and no
docs/tuning-flags.md section -- including two that change the coordinates a Boltz-2
prediction returns (BOLTZ2_TOKEN_DIT_SDPA, TT_BIO_ATOM_AXIS_BUCKET). Nothing caught it: the
parity, perf, UX, packaging and capacity arms all score behaviour, and a missing table row
is not behaviour.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Named in the notes but deliberately not a row, each with the reason it is not one.
# A flag belongs here only if a user never chooses between its values.
NOT_A_TUNING_ROW: dict[str, str] = {}


def _top_changelog_section(text: str) -> str:
    """The section being written: [Unreleased] if it exists, else the newest version."""
    heads = [m.start() for m in re.finditer(r"^## \[", text, re.M)]
    assert heads, "CHANGELOG.md has no version section"
    return text[heads[0]: heads[1] if len(heads) > 1 else len(text)]


def test_every_flag_the_release_notes_name_has_a_readme_row():
    changelog = (ROOT / "CHANGELOG.md").read_text()
    readme = (ROOT / "README.md").read_text()

    named = set(re.findall(
        r"`((?:TT_BIO|BOLTZ2|OPENDDE|OF3|PROTENIX|TT_PROTENIX)_[A-Z0-9_]+)",
        _top_changelog_section(changelog)))
    rows = set(re.findall(r"^\| `([A-Z0-9_]+)`", readme, re.M))

    missing = sorted(named - rows - set(NOT_A_TUNING_ROW))
    assert not missing, (
        "these flags are named in the release notes but have no row in the README tuning "
        f"table, so a user reading the notes cannot find out how to turn them off: {missing}. "
        "Add a row, or add the flag to NOT_A_TUNING_ROW in this file with the reason a user "
        "never chooses between its values."
    )


def test_the_readme_table_names_no_flag_the_package_does_not_read():
    """The other direction: a row for a flag nothing reads is a documented lie."""
    import ast

    readme = (ROOT / "README.md").read_text()
    rows = set(re.findall(r"^\| `([A-Z0-9_]+)`", readme, re.M))

    read_by_code: set[str] = set()
    for path in (ROOT / "tt_bio").rglob("*.py"):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            fn = node.func
            name = getattr(fn, "id", None) or getattr(fn, "attr", None)
            first = node.args[0]
            if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
                continue
            if name in ("env_flag", "env_int", "_site_flag"):
                read_by_code.add(first.value)
            elif name == "get" and isinstance(fn, ast.Attribute) and first.value.isupper():
                read_by_code.add(first.value)

    phantom = sorted(rows - read_by_code)
    assert not phantom, (
        f"the README tuning table documents flags nothing in tt_bio/ reads: {phantom}. "
        "Either the flag was renamed and the row was not, or the row was always wrong."
    )


def _readme_rows() -> set[str]:
    import re
    return set(re.findall(r"^\| `([A-Z0-9_]+)`", (ROOT / "README.md").read_text(), re.M))


def _tuning_flag_headings() -> set[str]:
    """Flags a `docs/tuning-flags.md` section is about.

    One heading names more than one flag where the measurement covers them
    together, so this reads every backticked name out of a heading rather than
    only the first: TT_BIO_DEVICE_CONFIDENCE and TT_BIO_DEVICE_CONF_HEADS share
    theirs.
    """
    import re
    doc = (ROOT / "docs" / "tuning-flags.md").read_text()
    named: set[str] = set()
    for heading in re.findall(r"^## (.*)$", doc, re.M):
        named |= set(re.findall(r"`([A-Z0-9_]+)`", heading))
    return named


def test_every_readme_tuning_row_has_a_tuning_flags_section():
    """The README row says what the flag does; the doc says what it was measured against.

    `docs/tuning-flags.md` opens by promising a page per flag. Twelve rows had no
    section and no mention anywhere in its body on 2026-09-20 -- including
    `BOLTZ2_TOKEN_DIT_SDPA` and `TT_BIO_ATOM_AXIS_BUCKET`, the two that move the
    coordinates a Boltz-2 prediction returns. The row's own cell is the wrong place
    for the evidence: the table is a list a user scans, and nothing was checking
    that the page behind it existed.
    """
    missing = sorted(_readme_rows() - _tuning_flag_headings())
    assert not missing, (
        f"these flags have a README tuning row but no docs/tuning-flags.md section: "
        f"{missing}. The row states what the flag does; the section states what it was "
        f"measured against, and the page promises one for every flag."
    )


def test_the_heading_scan_reads_a_second_flag_off_one_heading():
    """Negative control for `_tuning_flag_headings`, added 2026-09-20.

    Reading only the first backticked name off each heading is the bug this
    replaced, and it reports `TT_BIO_DEVICE_CONF_HEADS` as undocumented.
    """
    import re
    combined = "## `TT_BIO_DEVICE_CONFIDENCE`, `TT_BIO_DEVICE_CONF_HEADS` -- both on"
    assert re.findall(r"`([A-Z0-9_]+)`", combined)[:1] == ["TT_BIO_DEVICE_CONFIDENCE"]
    assert "TT_BIO_DEVICE_CONF_HEADS" in _tuning_flag_headings()
