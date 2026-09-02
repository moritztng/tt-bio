"""An empty input YAML is a user error, not an AttributeError from inside a parser.

`yaml.safe_load` returns None for an empty or comment-only file. Every reader then reached
for a key, so `tt-bio predict` on a commented-out YAML died with `AttributeError: 'NoneType'
object has no attribute 'get'` raised from `data/parse.py`, naming nothing the user typed.
`main.py`'s rfd3 reader already checked the type and named the file; the check just was not
anywhere else.

Stdlib plus pyyaml, no device.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tt_bio.data.yaml_input import load_mapping, require_mapping  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "tt_bio"


@pytest.mark.parametrize("body,label", [
    ("", "an empty file"),
    ("# everything commented out\n", "a comment-only file"),
    ("\n\n   \n", "whitespace only"),
    ("null\n", "an explicit null"),
])
def test_a_document_with_no_mapping_is_refused_by_name(tmp_path, body, label):
    p = tmp_path / "in.yaml"
    p.write_text(body)
    with pytest.raises(ValueError, match=r"expected a YAML mapping"):
        load_mapping(p)
    # The file the user typed has to be in the message; that is the whole complaint
    # about the AttributeError this replaces.
    try:
        load_mapping(p)
    except ValueError as exc:
        assert str(p) in str(exc), label


@pytest.mark.parametrize("body,kind", [("- a\n- b\n", "list"), ("7\n", "int"), ("hi\n", "str")])
def test_a_document_that_is_not_a_mapping_says_what_it_got(tmp_path, body, kind):
    p = tmp_path / "in.yaml"
    p.write_text(body)
    with pytest.raises(ValueError, match=f"got a {kind}, not a mapping"):
        load_mapping(p)


def test_a_real_mapping_passes_through_unchanged(tmp_path):
    p = tmp_path / "in.yaml"
    p.write_text("version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: MKT\n")
    doc = load_mapping(p)
    assert doc == {"version": 1, "sequences": [{"protein": {"id": "A", "sequence": "MKT"}}]}


def test_an_open_handle_gets_the_same_check(tmp_path):
    """The `file=` form, for readers that dispatch on the suffix inside a `with` block."""
    p = tmp_path / "in.yaml"
    p.write_text("a: 1\n")
    with p.open() as fh:
        assert load_mapping(p, file=fh) == {"a": 1}
    p.write_text("")
    with p.open() as fh:
        with pytest.raises(ValueError, match="expected a YAML mapping"):
            load_mapping(p, file=fh)


def test_require_mapping_accepts_an_already_loaded_document():
    assert require_mapping({"a": 1}, "x.yaml") == {"a": 1}
    with pytest.raises(ValueError, match="expected a YAML mapping"):
        require_mapping(None, "x.yaml")


#: Sites that call ``yaml.safe_load`` and are right not to want a mapping. A row with a
#: written reason, not a second code path.
_NOT_READING_A_DOCUMENT = {
    "tt_bio/boltzgen/_config.py": (
        "parses the value half of a `key=value` CLI override as a YAML scalar, so an int, "
        "a bool or `null` coming back is the point. It reads no file."),
}


def _safe_load_call_lines(path: Path) -> list[int]:
    """Line numbers of real ``yaml.safe_load(...)`` calls, from the parse tree.

    Read via ``ast`` rather than by grepping the text, because the text of this very
    module and of ``yaml_input.py`` discusses ``yaml.safe_load`` in prose. A substring
    scanner cannot tell a call from a sentence about calls, and flags both.
    """
    out = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr == "safe_load":
            out.append(node.lineno)
    return out


def test_every_shipped_yaml_reader_guards_its_result():
    """The scanner. A `yaml.safe_load` whose result is used unguarded is the bug above.

    Narrow on purpose (the cache-contract scanner's own note says why a broad guard full
    of exemptions is worse): it accepts `or {}`, an `isinstance` check within the same
    block, or going through `load_mapping`/`require_mapping`. Anything else is a finding.
    """
    offenders = []
    for path in sorted(ROOT.rglob("*.py")):
        rel = str(path.relative_to(ROOT.parent))
        if "_vendor" in path.parts or rel in _NOT_READING_A_DOCUMENT:
            continue
        lines = path.read_text().splitlines()
        for lineno in _safe_load_call_lines(path):
            line = lines[lineno - 1]
            window = "\n".join(lines[lineno - 1:lineno + 7])
            guarded = (
                "or {}" in line
                or "load_mapping" in line
                or "require_mapping" in window
                or "isinstance" in window
            )
            if not guarded:
                offenders.append(f"{rel}:{lineno}: {line.strip()}")
    assert not offenders, (
        "these dereference yaml.safe_load's result without checking it is a mapping; "
        "an empty or comment-only file makes it None:\n  " + "\n  ".join(offenders))


def test_the_scanner_would_catch_the_bug_it_was_written_for(tmp_path):
    """A guard nobody has seen fail is not a guard. The shape below is verbatim what
    `data/parse.py` and `pxdesign/inputs.py` shipped before this change."""
    bad = tmp_path / "reader.py"
    bad.write_text("import yaml\n\n\ndef read(p):\n    cfg = yaml.safe_load(p.read_text())\n"
                   "    return cfg.get('target')\n")
    assert _safe_load_call_lines(bad) == [5]
    line = bad.read_text().splitlines()[4]
    assert not ("or {}" in line or "load_mapping" in line), "the fixture must be unguarded"


def test_the_scanner_ignores_prose_about_safe_load(tmp_path):
    """The false positive the first cut of this scanner had: it flagged its own docstring."""
    prose = tmp_path / "doc.py"
    prose.write_text('"""yaml.safe_load returns None for an empty file."""\n\nX = 1\n')
    assert _safe_load_call_lines(prose) == []
