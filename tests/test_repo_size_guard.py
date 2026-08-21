"""Nothing large or machine-generated gets committed again.

The pack is ~2 GB against a ~7 MB working tree, and all of it is history:
`press/pmol-env/` was a committed Python virtualenv (libLLVM 186 MB, libclang-cpp
88 MB, libclang 49 MB, qmake 27 MB, ...) totalling ~976 MB, plus ~70 MB of landing-page
video. Every one of those files is already deleted from the tree, so only a history
rewrite can reclaim the space -- and that breaks 12 forks and 21 release tags, so it
is a deliberate decision rather than a cleanup.

What IS in our control is that it never grows again. Largest legitimately tracked
file today is 5.65 MB (a ColabFold MSA), so 10 MB is comfortable headroom and would
have caught every blob above.
"""
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MAX_BYTES = 10 * 1024 * 1024

# Directory names that mean "machine-generated, never commit this".
BANNED_DIR_NAMES = {
    "site-packages", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist-info", ".mypy_cache", ".pytest_cache", ".ruff_cache",
}
# Extensions that are build output or bulk media, not source.
BANNED_SUFFIXES = {".so", ".dylib", ".dll", ".a", ".o", ".pyd", ".whl", ".pyc",
                   ".mp4", ".webm", ".mov", ".avi"}


def _tracked():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO,
                         capture_output=True, check=True).stdout
    return [p for p in out.decode().split("\0") if p]


def test_no_tracked_file_over_10mb():
    big = []
    for rel in _tracked():
        f = REPO / rel
        if f.is_file() and f.stat().st_size > MAX_BYTES:
            big.append("%s (%.1f MB)" % (rel, f.stat().st_size / 1048576))
    assert not big, (
        "Files over 10 MB are tracked: %s\n"
        "The repo already carries ~2 GB of history from committed binaries. "
        "Put large artifacts outside git, or raise MAX_BYTES here deliberately."
        % ", ".join(sorted(big))
    )


def test_no_virtualenvs_or_build_output_tracked():
    bad = []
    for rel in _tracked():
        parts = Path(rel).parts
        if any(p in BANNED_DIR_NAMES for p in parts[:-1]):
            bad.append(rel)
        elif Path(rel).suffix.lower() in BANNED_SUFFIXES:
            bad.append(rel)
    assert not bad, (
        "Machine-generated files are tracked: %s\n"
        "A committed virtualenv (press/pmol-env) is what put ~976 MB into this "
        "repo's history. Add to .gitignore instead." % ", ".join(sorted(bad)[:10])
    )
