"""release_train.py: notes, version and the release commit, on strings and a throwaway tree."""
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
spec = importlib.util.spec_from_file_location("release_train", REPO / "scripts" / "release_train.py")
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)

RELEASED = """# Changelog

## [Unreleased]

## [0.14.0] - 2026-10-09

### Changed
- **Old lever.** Faster.
"""
CANDIDATE = """# Changelog

## [Unreleased]

### Changed
- **New lever.** Even faster,
  on two lines.
- **Old lever.** Faster.

### Fixed
- **Old lever.** Faster.

## [0.14.0] - 2026-10-09

### Changed
- **Old lever.** Faster.
"""


def test_notes_drop_what_an_earlier_release_carried_and_empty_headings():
    body = rt.notes(CANDIDATE, RELEASED)
    assert body == "### Changed\n- **New lever.** Even faster,\n  on two lines."


def test_version_is_minor_for_changes_patch_for_fixes_and_skips_taken_tags():
    assert rt.next_version("v0.14.0", "### Changed\n- x", set()) == "0.15.0"
    assert rt.next_version("v0.14.0", "### Fixed\n- x", set()) == "0.14.1"
    assert rt.next_version("v0.14.0", "### Added\n- x", {"v0.15.0"}) == "0.15.1"


def test_release_files_touch_only_the_notes_version_and_install_tag(tmp_path):
    (tmp_path / "CHANGELOG.md").write_text(CANDIDATE)
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "tt-bio"\nversion = "0.14.0"\n')
    (tmp_path / "README.md").write_text('pip install "x @ git+https://github.com/moritztng/tt-bio.git@v0.14.0"\n')
    rt.release_files(tmp_path, "0.15.0", "### Changed\n- **New lever.**", "2026-10-12")
    cl = (tmp_path / "CHANGELOG.md").read_text()
    assert cl.startswith("# Changelog\n\n## [Unreleased]\n\n## [0.15.0] - 2026-10-12\n\n### Changed\n- **New lever.**\n\n## [0.14.0]")
    assert 'version = "0.15.0"' in (tmp_path / "pyproject.toml").read_text()
    assert "tt-bio.git@v0.15.0" in (tmp_path / "README.md").read_text()


def test_release_commit_changes_only_the_release_files_and_keeps_the_content_key(tmp_path, monkeypatch):
    import subprocess

    def git(*a):
        return subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True,
                              text=True).stdout.strip()
    git("init", "-q"), git("config", "user.email", "t@t"), git("config", "user.name", "t")
    (tmp_path / "CHANGELOG.md").write_text(CANDIDATE)
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.14.0"\n')
    (tmp_path / "README.md").write_text("tt-bio.git@v0.14.0\n")
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "perf" / "x").mkdir(parents=True)
    for name in ("gate", "other"):
        (tmp_path / "perf" / "x" / f"{name}.txt").write_text(
            f"RECORDED-AT: {'0' * 40} scripts/{name}.py\nRERUN: PY scripts/{name}.py\nEXPECT: ok\n")
    git("add", "."), git("commit", "-qm", "c")
    sha = git("rev-parse", "HEAD")
    monkeypatch.setattr(rt, "REPO", tmp_path)
    monkeypatch.setattr(rt, "git", lambda *a, **k: git(*a))
    gate = tmp_path / "gate"
    gate.mkdir()
    (gate / "verdict.json").write_text(json.dumps({"results": [
        {"leg": "recorded:gate", "verdict": "PASS", "worker": "qb1:3", "ended": "2026-10-12T12:00:00Z"},
        {"leg": "recorded:other", "verdict": "FAIL", "worker": "qb1:2", "ended": "2026-10-12T12:00:00Z"}]}))
    records = rt.reproduced(str(gate), sha)
    assert records == {"perf/x/gate.txt": "recorded:gate qb1:3 2026-10-12T12:00:00Z"}
    rel = rt.release_commit(sha, "0.15.0", "### Changed\n- **New lever.**", "2026-10-12", records)
    assert git("diff", "--name-only", sha, rel).split() == ["CHANGELOG.md", "README.md", "perf/x/gate.txt",
                                                            "pyproject.toml"]
    assert f"RECORDED-AT: {sha} scripts/gate.py" in git("show", f"{rel}:perf/x/gate.txt")
    assert git("rev-parse", f"{rel}^") == sha and git("rev-parse", "HEAD") == sha   # nothing checked out
    assert rt.content_hash(rel, tmp_path) == rt.content_hash(sha, tmp_path)
    assert 'version = "0.15.0"' in git("show", f"{rel}:pyproject.toml")
