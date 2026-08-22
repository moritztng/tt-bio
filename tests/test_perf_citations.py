"""Every ``perf/...`` path cited from shipped source or docs must exist.

Tuning constants in ``tt_bio/`` carry the measurement that set them. When a lever
is hand-ported onto main from a worker branch, the comment comes across and the
artifact it names does not, leaving a pointer to nothing. Three directories had
gone that way (``perf/esm3p4``, ``perf/odde4x``, ``perf/odde512``) before this
test existed.
"""

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Directory names are lowercase, so this skips prose like "the perf/UX gate".
CITATION = re.compile(r"perf/[a-z0-9][A-Za-z0-9_.\-]*(?:/[A-Za-z0-9_.\-]+)*")


def _tracked(*prefixes):
    """Files under *prefixes*, from git where there is a git, else from disk.

    This runs at COLLECTION time (``parametrize`` calls it), so anything it
    raises aborts the whole session, not just this file. ``git ls-files``
    exits 128 outside a work tree, and a release gate runs the suite against a
    ``git archive`` export with no ``.git`` in it, as does anyone running the
    tests from an unpacked sdist. Fall back to walking the tree: it names the
    same shipped files, minus anything untracked, which is exactly the set this
    test wants to read anyway.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "ls-files", *prefixes],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.split()
    except (subprocess.CalledProcessError, FileNotFoundError):
        found = []
        for prefix in prefixes:
            for path in sorted((REPO / prefix).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    found.append(str(path.relative_to(REPO)))
        return found


def _perf_files():
    root = REPO / "perf"
    return [str(p.relative_to(REPO)) for p in root.rglob("*") if p.is_file()]


def _citations():
    for name in _tracked("tt_bio", "docs"):
        path = REPO / name
        if path.suffix in {".json", ".cif", ".pdb", ".a3m", ".sto", ".npz", ".txt"}:
            continue
        for match in CITATION.finditer(path.read_text(errors="ignore")):
            yield name, match.group(0).rstrip(".,;:")


@pytest.mark.parametrize("source,cited", sorted(set(_citations())))
def test_cited_perf_artifact_exists(source, cited):
    # A citation may be truncated by a brace or glob the regex stops at, e.g.
    # `reblock_window_band_qb1c0{,_r2}.json` or `ops_*.json`, so a prefix counts.
    if (REPO / cited).exists():
        return
    assert any(p.startswith(cited) for p in _perf_files()), (
        f"{source} cites {cited}, which is not in the repo. Either commit the "
        f"artifact or state the measurement without naming a path."
    )
