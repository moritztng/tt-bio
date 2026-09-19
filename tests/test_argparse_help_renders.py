"""Every argparse help string must survive argparse's own %-expansion.

argparse runs ``help % params`` when it formats ``--help``, so a literal ``%`` in a help string
raises and the script exits 1 with no usage printed. Nothing else breaks: the script runs fine,
only its usage cannot be shown, which is why three of these sat in the tree unnoticed --
``scripts/abb3_port/tripwire.py`` (``ValueError: incomplete format``),
``perf/attn_sites/rfd3_esm_replay.py`` (``TypeError: %c requires int or char``) and
``perf/c12_mm_attrib/attrib.py``.

Static rather than executed: running ``--help`` on every script would import ttnn a few hundred
times and need a card for some of them. The AST carries everything the check needs.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
#: argparse expands against a dict, so these are the only well-formed conversions.
OK_NEXT = "%("


def _help_strings(path: Path):
    """(lineno, value) for every ``help=`` keyword whose value is a plain string literal."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg != "help":
                continue
            # An f-string or a name is skipped: argparse expands those too, but the literal
            # is not knowable from the AST.
            try:
                value = ast.literal_eval(kw.value)
            except (ValueError, TypeError, SyntaxError):
                continue
            if isinstance(value, str):
                yield node.lineno, value


def _bad_percents(text: str) -> list[int]:
    """Offsets of every ``%`` argparse cannot expand. ``%%`` and ``%(name)s`` are fine."""
    bad, i = [], 0
    while i < len(text):
        if text[i] != "%":
            i += 1
            continue
        nxt = text[i + 1:i + 2]
        if nxt == "%":
            i += 2
        elif nxt == "(":
            i += 2
        else:
            bad.append(i)
            i += 1
    return bad


SOURCES = sorted(
    p for d in ("scripts", "perf", "tt_bio", "examples") for p in (ROOT / d).rglob("*.py"))


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_help_strings_survive_percent_expansion(path: Path):
    offenders = [(lineno, value) for lineno, value in _help_strings(path)
                 if _bad_percents(value)]
    assert not offenders, "\n".join(
        f"{path.relative_to(ROOT)}:{lineno} help={value!r} -- argparse expands this and raises, "
        f"so --help exits 1 with no usage. Write %% for a literal percent."
        for lineno, value in offenders)
