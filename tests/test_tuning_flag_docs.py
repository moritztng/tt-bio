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

import pytest

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
            if name in ("env_flag", "env_int"):
                read_by_code.add(first.value)
            elif name == "get" and isinstance(fn, ast.Attribute) and first.value.isupper():
                read_by_code.add(first.value)

    phantom = sorted(rows - read_by_code)
    assert not phantom, (
        f"the README tuning table documents flags nothing in tt_bio/ reads: {phantom}. "
        "Either the flag was renamed and the row was not, or the row was always wrong."
    )

def _env_flags_at(rev: str) -> dict:
    """{FLAG: default-expression} for every `env_flag(...)` under tt_bio/ at a git revision."""
    import subprocess
    out: dict[str, str] = {}
    names = subprocess.run(["git", "-C", str(ROOT), "ls-tree", "-r", "--name-only", rev, "tt_bio/"],
                           capture_output=True, text=True)
    if names.returncode:
        return {}
    for f in names.stdout.split():
        if not f.endswith(".py"):
            continue
        src = subprocess.run(["git", "-C", str(ROOT), "show", f"{rev}:{f}"],
                             capture_output=True, text=True).stdout
        for m in re.finditer(r'env_flag\(\s*"([A-Z0-9_]+)"\s*,\s*([^)\n]+)\)', src):
            out[m.group(1)] = m.group(2).strip()
    return out


def _previous_release_tag() -> str:
    """The tag of the newest RELEASED section, i.e. the one this release is measured against."""
    versions = re.findall(r"^## \[(\d+\.\d+\.\d+)\]", (ROOT / "CHANGELOG.md").read_text(), re.M)
    return f"v{versions[0]}" if versions else ""


def test_a_flag_that_became_a_default_this_release_is_in_the_release_notes():
    """The other hole, and the one the rule above cannot see.

    The test at the top of this file keys on the CHANGELOG *naming* a flag. A flag that ships ON
    and is never named escapes it entirely -- and that is not hypothetical. Between v0.9.0 and
    2026-09-19 main gained TT_BIO_TRIMUL_BACK_ONE_PASS_L1 and TT_BIO_TRIMUL_GATED_MOVE_L1, both
    default ON, and flipped TT_BIO_DIT_COND_HOIST from off to on. None of the three was in the
    notes, so three behaviour changes shipped that a user could neither find nor turn off.

    Scope is deliberately small: only flags that are ON for someone who sets nothing, and only
    the delta against the last release. A new opt-in flag is not a behaviour change and is not
    asked for.
    """
    import subprocess

    tag = _previous_release_tag()
    if not tag or subprocess.run(["git", "-C", str(ROOT), "rev-parse", "-q", "--verify", tag],
                                 capture_output=True).returncode:
        pytest.skip(f"no {tag or 'release'} tag in this checkout; the rule is a git delta")

    before, after = _env_flags_at(tag), _env_flags_at("HEAD")
    if not before or not after:
        pytest.skip("tt_bio/ is not readable at one of the two revisions")

    def is_on(expr: str) -> bool:
        # A bare True, or a module constant, which in this file is always a shipped-on default.
        return expr == "True" or (expr.isupper() and expr.replace("_", "").isalnum())

    became_on = {f for f, d in after.items() if is_on(d) and (f not in before or not is_on(before[f]))}
    named = set(re.findall(
        r"`((?:TT_BIO|BOLTZ2|OPENDDE|OF3|PROTENIX|TT_PROTENIX|RFD3)_[A-Z0-9_]+)",
        _top_changelog_section((ROOT / "CHANGELOG.md").read_text())))

    missing = sorted(became_on - named - set(NOT_A_TUNING_ROW))
    assert not missing, (
        f"these flags are ON by default at HEAD and were not at {tag}, but the release notes do "
        f"not name them: {missing}. Each one changes what a user gets without being asked for, so "
        "it needs a CHANGELOG entry (and therefore a README row, by the test above)."
    )


def test_that_rule_can_fail():
    """Negative control: the detector must fire on a flag that became a default and is unnamed."""
    fake_before = {"A_FLAG": "False", "B_FLAG": "True"}
    fake_after = {"A_FLAG": "True", "B_FLAG": "True", "C_FLAG": "False"}

    def is_on(expr):
        return expr == "True" or (expr.isupper() and expr.replace("_", "").isalnum())

    became_on = {f for f, d in fake_after.items()
                 if is_on(d) and (f not in fake_before or not is_on(fake_before[f]))}
    assert became_on == {"A_FLAG"}, (
        "the detector must catch a flipped default (A_FLAG), ignore one that was already on "
        f"(B_FLAG) and ignore a new opt-in (C_FLAG); it returned {became_on}")
