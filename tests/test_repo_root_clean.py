"""The repo root is an allowlist, not a dumping ground.

Campaign one-offs kept landing in the root: `bh_ab384.sh` and `bh_ab_chain.sh`
were A/B drivers hardcoded to `/home/ttuser/.coworker/wt/wh-perf-opendde`, and a
top-level `results/` held GPU concurrency JSON. Measurement artifacts belong
under `perf/`, tooling under `scripts/`. This test fails on anything new in the
root so it is caught in review rather than by a human reading the file list.

To add a genuinely root-level file, add it to ALLOWED with a reason.
"""
import subprocess
from pathlib import Path

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


def _tracked_root_files():
    out = subprocess.run(
        ["git", "ls-files", "--full-name"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    return {p for p in out if "/" not in p and p}


def test_repo_root_has_no_stray_files():
    stray = sorted(_tracked_root_files() - ALLOWED)
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
