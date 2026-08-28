"""The repo root is an allowlist, not a dumping ground.

Campaign one-offs kept landing in the root: `bh_ab384.sh` and `bh_ab_chain.sh`
were A/B drivers hardcoded to `/home/ttuser/.coworker/wt/wh-perf-opendde`, and a
top-level `results/` held GPU concurrency JSON. Measurement artifacts belong
under `perf/`, tooling under `scripts/`. This test fails on anything new in the
root so it is caught in review rather than by a human reading the file list.

Directories are checked too, and were not at first: the allowlist read only
top-level files, so `results/` -- the directory this test was written about --
survived it and took another artifact four months later. Anything that is not one
of the eight trees below is a new root directory and fails here.

To add a genuinely root-level file or directory, add it to ALLOWED or
ALLOWED_DIRS with a reason.
"""
from pathlib import Path

import pytest

from conftest import git_tracked

REPO = Path(__file__).resolve().parent.parent

ALLOWED = {
    "README.md",          # entry point
    "LICENSE",            # MIT
    "NOTICE",             # third-party attribution, required by the licences we bundle
    "CHANGELOG.md",       # release history
    "RELEASING.md",       # release runbook
    "pyproject.toml",     # package definition
    "MANIFEST.in",        # sdist contents; kernel packaging has regressed via this 3x
    ".gitignore",
}


ALLOWED_DIRS = {
    "tt_bio",        # the package
    "tests",
    "scripts",       # tooling, gates, per-port harnesses
    "docs",
    "examples",      # input fixtures the README and the gates fold
    "perf",          # every measurement artifact; see perf/README.md
    "site",          # tt-bio.com
    "site-redirect", # the old tt-boltz domain
    ".github",
}


def _tracked_root(kind):
    out = git_tracked(REPO, "--full-name")
    if out is None:
        pytest.skip("not a git work tree; nothing is tracked here to be stray")
    if kind == "file":
        return {p for p in out if "/" not in p and p}
    return {p.split("/", 1)[0] for p in out if "/" in p}


def test_repo_root_has_no_stray_files():
    stray = sorted(_tracked_root("file") - ALLOWED)
    assert not stray, (
        "Unexpected files tracked in the repo root: %s\n"
        "Measurement artifacts go under perf/, tooling under scripts/. "
        "If one genuinely belongs at the root, add it to ALLOWED in this test "
        "with a reason." % ", ".join(stray)
    )


def test_allowlist_entries_all_exist():
    """A stale allowlist hides the next regression."""
    missing = sorted(n for n in ALLOWED if not (REPO / n).exists())
    assert not missing, "ALLOWED lists files that no longer exist: %s" % ", ".join(missing)


def test_repo_root_has_no_stray_directories():
    stray = sorted(_tracked_root("dir") - ALLOWED_DIRS)
    assert not stray, (
        "Unexpected directories tracked in the repo root: %s\n"
        "Measurement artifacts go under perf/, tooling under scripts/. "
        "If one genuinely belongs at the root, add it to ALLOWED_DIRS in this "
        "test with a reason." % ", ".join(stray))
