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
"""

import re
from pathlib import Path

import pytest

from conftest import git_tracked

REPO = Path(__file__).resolve().parents[1]

# Directory names are lowercase, so this skips prose like "the perf/UX gate".
CITATION = re.compile(r"perf/[a-z0-9][A-Za-z0-9_.\-]*(?:/[A-Za-z0-9_.\-]+)*")

# ``Path(...) / "perf" / "<dir>"`` and ``os.path.join(..., "perf", "<dir>")``.
JOINED = re.compile(r"""['"]perf['"]\s*(?:/|,)\s*['"]([A-Za-z0-9_.\-]+)['"]""")

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


def _joined_reads():
    """`"perf" / "<dir>"`, from everything that could open one."""
    for name in _tracked("tt_bio", "docs", "site", "tests", "scripts"):
        body = _text(name)
        if body is None:
            continue
        for match in JOINED.finditer(body):
            yield name, "perf/" + match.group(1)


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
    literal below is exactly the shape `scripts/release_gate.py` uses.
    """
    line = 'path = REPO_ROOT / "perf" / "ceilrfd3" / "targets"'
    assert not CITATION.search(line), "the one-string regex should not see this"
    assert [m.group(1) for m in JOINED.finditer(line)] == ["ceilrfd3"]
