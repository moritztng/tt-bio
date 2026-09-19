"""Every ``perf/...`` path cited from shipped source or docs must exist.

Tuning constants in ``tt_bio/`` carry the measurement that set them. When a lever
is hand-ported onto main from a worker branch, the comment comes across and the
artifact it names does not, leaving a pointer to nothing. Three directories had
gone that way (``perf/esm3p4``, ``perf/odde4x``, ``perf/odde512``) before this
test existed.

``site/`` is read too: a published cell that names the raw reports behind it is
making the same promise a tuning comment does, and the periodic tidy deletes by
citation.

The tidy is the reason the second half exists. A gate or a test that READS a
measurement tree does not spell the path as one string -- ``scripts/release_gate.py``
builds ``REPO_ROOT / "perf" / "ceilrfd3" / "targets"`` and the two
``test_chip_determinism_*`` files load ``ROOT / "perf" / "wh-parity"``. The
one-string regex below cannot see any of that, so a tidy censusing citations the
way this file does reads those directories as uncited and deletes them: 22 MB of
the release gate's RFD3 ladder targets and the module two tests import. That
nearly happened on 2026-09-11. Joined citations are collected as well now, from
``tests/`` and ``scripts/`` too, and only as far as the directory -- the segment
after it is usually an f-string.

That second half reads the AST, not a regex. A regex matching ``"perf", "<dir>"``
for ``os.path.join`` also matches a tuple of sibling directory names, and
``("scripts", "perf", "tt_bio", "examples")`` is four trees to walk, not a path
to ``perf/tt_bio``. The first ordinary tuple the comma reader met was the one in
``tests/test_argparse_help_renders.py``, and the gate went red against a correct
line.
"""

import ast
import re
import warnings
from pathlib import Path

import pytest

from conftest import git_tracked

REPO = Path(__file__).resolve().parents[1]

# Directory names are lowercase, so this skips prose like "the perf/UX gate".
CITATION = re.compile(r"perf/[a-z0-9][A-Za-z0-9_.\-]*(?:/[A-Za-z0-9_.\-]+)*")

# ``Path(...) / "perf" / "<dir>"``, for a file that is not Python and so has no
# AST to read. The comma form -- ``os.path.join(..., "perf", "<dir>")`` -- is not
# here on purpose: a comma between two string literals is also what a tuple of
# sibling directory names looks like, and only the AST can tell them apart.
DIVIDED = re.compile(r"""['"]perf['"]\s*/\s*['"]([A-Za-z0-9_.\-]+)['"]""")

# Nothing here holds prose, and two of them are large enough to matter.
BINARY = {".cif", ".pdb", ".npz", ".a3m", ".sto", ".pt", ".parquet",
          ".png", ".ico", ".svg", ".woff2"}
# What the one-string half has always skipped on top of that: a JSON or a data
# dump mostly re-reports paths the run itself wrote.
DATA = {".json", ".txt", ".jsonl", ".csv"}


def _tracked(*prefixes):
    """Files under *prefixes*, from git where there is a git, else from disk.

    Unlike the two repo-hygiene guards, this one must not skip outside a work
    tree: it reads shipped source for dangling `perf/...` citations, and those
    files are all present in an export. Walking the tree names the same set
    minus anything untracked, which is what this wants to read anyway.
    """
    tracked = git_tracked(REPO, *prefixes)
    if tracked is not None:
        return tracked
    found = []
    for prefix in prefixes:
        for path in sorted((REPO / prefix).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                found.append(str(path.relative_to(REPO)))
    return found


def _perf_files():
    root = REPO / "perf"
    return [str(p.relative_to(REPO)) for p in root.rglob("*") if p.is_file()]


def _text(name):
    path = REPO / name
    if path.suffix in BINARY:
        return None
    return path.read_text(errors="ignore")


def _citations():
    """`perf/...` written as one string, from the surfaces that make claims."""
    for name in _tracked("tt_bio", "docs", "site"):
        if (REPO / name).suffix in DATA:
            continue
        body = _text(name)
        if body is None:
            continue
        for match in CITATION.finditer(body):
            yield name, match.group(0).rstrip(".,;:")


def _string(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _div_segments(node):
    """``a / "perf" / "x"`` -> ``[a, "perf", "x"]``. The chain nests leftwards."""
    parts = []
    while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        parts.append(node.right)
        node = node.left
    parts.append(node)
    return parts[::-1]


def _is_join(func):
    """``os.path.join``, ``posixpath.join``, a bare imported ``join``.

    ``",".join(parts)`` lands here too and yields nothing: its one argument is
    an iterable, never two adjacent literals.
    """
    if isinstance(func, ast.Attribute):
        return func.attr == "join"
    return isinstance(func, ast.Name) and func.id == "join"


def _joined_dirs(source):
    """Directories under ``perf`` that Python *source* builds from segments.

    ``ROOT / "perf" / "whceil"`` and ``os.path.join(REPO, "perf", "whceil")``
    both open ``perf/whceil``. ``("scripts", "perf", "tt_bio")`` does not open
    ``perf/tt_bio``: it is three sibling names, and whatever indexes it is a
    loop variable no static reader can resolve. A regex sees the same two
    quoted words with a comma between them in both, which is why this reads
    the AST. ``None`` if *source* is not parseable Python.
    """
    try:
        with warnings.catch_warnings():
            # Reading a file is not importing it. An invalid escape sequence in
            # some other module's string is that module's problem, and it would
            # arrive here labelled `<unknown>:2`.
            warnings.simplefilter("ignore")
            tree = ast.parse(source)
    except SyntaxError:
        return None
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            segments = _div_segments(node)
        elif isinstance(node, ast.Call) and _is_join(node.func):
            segments = node.args
        else:
            continue
        for left, right in zip(segments, segments[1:]):
            name = _string(right)
            if _string(left) == "perf" and name:
                found.append(name)
    # A chain nests, so `ast.walk` reaches `a / "perf" / "x"` once whole and
    # once as the left side of the next `/`.
    return list(dict.fromkeys(found))


def _joined_reads():
    """`"perf" / "<dir>"` and its `join` form, from everything that opens one."""
    for name in _tracked("tt_bio", "docs", "site", "tests", "scripts"):
        body = _text(name)
        if body is None:
            continue
        dirs = _joined_dirs(body) if name.endswith(".py") else None
        if dirs is None:
            dirs = [m.group(1) for m in DIVIDED.finditer(body)]
        for cited in dirs:
            yield name, "perf/" + cited


def _resolves(cited):
    # A citation may be truncated by a brace or glob the regex stops at, e.g.
    # `reblock_window_band_qb1c0{,_r2}.json` or `ops_*.json`, so a prefix counts.
    if (REPO / cited).exists():
        return True
    return any(p.startswith(cited) for p in _perf_files())


@pytest.mark.parametrize("source,cited", sorted(set(_citations())))
def test_cited_perf_artifact_exists(source, cited):
    assert _resolves(cited), (
        f"{source} cites {cited}, which is not in the repo. Either commit the "
        f"artifact or state the measurement without naming a path."
    )


@pytest.mark.parametrize("source,cited", sorted(set(_joined_reads())))
def test_perf_tree_a_gate_opens_by_path_join_exists(source, cited):
    assert _resolves(cited), (
        f"{source} builds {cited} from path segments and it is not in the repo. "
        f"This is the form the periodic tidy's citation census used to miss, so "
        f"the directory reads as uncited and gets deleted."
    )


def test_the_census_sees_a_joined_citation_a_one_string_regex_cannot():
    """Negative control for the half added 2026-09-11.

    It has to fail on what the old scan passed, or the fix is decorative. The
    two literals below are exactly the shapes `scripts/release_gate.py` and
    `tests/test_token_axis_bucketing_hw.py` use.
    """
    line = 'path = REPO_ROOT / "perf" / "ceilrfd3" / "targets"'
    assert not CITATION.search(line), "the one-string regex should not see this"
    assert _joined_dirs(line) == ["ceilrfd3"]

    call = 'p = os.path.join(REPO, "perf", "wh-correctness", "results")'
    assert not CITATION.search(call), "the one-string regex should not see this"
    assert _joined_dirs(call) == ["wh-correctness"]


def test_a_dangling_joined_citation_still_fails():
    """The gate's whole reason to exist, on a directory that is really absent.

    A reader narrow enough to stop mis-firing can also be narrow enough to see
    nothing, which would delete the gate rather than fix it. This is the case
    the tidy loses money on: literal segments, directory not in the repo.
    """
    source = 'REF = ROOT / "perf" / "no_such_tree" / "targets"'
    assert _joined_dirs(source) == ["no_such_tree"]
    assert not _resolves("perf/no_such_tree")
    with pytest.raises(AssertionError, match="no_such_tree"):
        test_perf_tree_a_gate_opens_by_path_join_exists(
            "fixture.py", "perf/no_such_tree")


def test_a_tuple_of_sibling_directories_is_not_a_joined_citation():
    """`("scripts", "perf", "tt_bio", "examples")` cites no `perf/tt_bio`.

    The comma reader read the two adjacent elements as one path and failed the
    gate against correct code. Four sibling trees, one rglob each, and the name
    that indexes them is a loop variable.
    """
    line = ('SOURCES = [p for d in ("scripts", "perf", "tt_bio", "examples")\n'
            '           for p in (ROOT / d).rglob("*.py")]')
    assert _joined_dirs(line) == []


def test_a_file_that_is_not_python_still_gets_the_divided_form():
    """A shell script or a fenced snippet has no AST; the `/` form is safe there.

    Only the `/` form: a comma outside a parsed file is the ambiguity above
    with nothing left to resolve it.
    """
    assert [m.group(1) for m in DIVIDED.finditer(
        'sys.path.insert(0, str(here / "perf" / "bgsdpa"))')] == ["bgsdpa"]
    assert _joined_dirs("this is not python ===") is None
