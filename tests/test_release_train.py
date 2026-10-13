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
    (tmp_path / "docs" / "size_ladder_baseline.d").mkdir(parents=True)
    (tmp_path / "docs" / "size_ladder_baseline.d" / "m.json").write_text(
        json.dumps({"cards": {"p150a": {"old": 1}, "p300c": {"old": 3}}}))
    (tmp_path / "perf" / "x" / "gate.txt").write_text(f"RECORDED-AT: {'0' * 40} scripts/gate.py\n")
    git("add", "."), git("commit", "-qm", "c")
    sha = git("rev-parse", "HEAD")
    monkeypatch.setattr(rt, "REPO", tmp_path)
    monkeypatch.setattr(rt, "git", lambda *a, **k: git(*a))
    gate = tmp_path / "gate"
    (gate / "abb3" / "perf" / "x").mkdir(parents=True)
    (gate / "abb3" / "perf" / "x" / "gate.txt").write_text(f"RECORDED-AT: {sha} scripts/gate.py\n")
    for verdict, want in (("FAIL", {}), ("PASS", {"perf/x/gate.txt": f"RECORDED-AT: {sha} scripts/gate.py\n"})):
        (gate / "verdict.json").write_text(json.dumps({"results": [{"leg": "record:abb3", "verdict": verdict}]}))
        assert rt.rerecorded_files(str(gate), sha) == want
    rec = tmp_path / "rec"                                  # a record run re-recorded one card type
    (rec / "recorded" / "p150a").mkdir(parents=True)
    (rec / "recorded" / "p150a" / "m.json").write_text(json.dumps({"cards": {"p150a": {"new": 1}, "p300c": {"x": 9}}}))
    frag = "docs/size_ladder_baseline.d/m.json"
    want = {**want, frag: json.dumps({"cards": {"p150a": {"new": 1}, "p300c": {"old": 3}}}, indent=2) + "\n"}
    assert rt.rerecorded_files(str(gate), sha, str(rec)) == want
    rel = rt.release_commit(sha, "0.15.0", "### Changed\n- **New lever.**", "2026-10-12", want)
    assert git("diff", "--name-only", sha, rel).split() == ["CHANGELOG.md", "README.md",
                                                            "docs/size_ladder_baseline.d/m.json",
                                                            "perf/x/gate.txt", "pyproject.toml"]
    assert git("show", f"{rel}:perf/x/gate.txt") == f"RECORDED-AT: {sha} scripts/gate.py"
    assert git("rev-parse", f"{rel}^") == sha and git("rev-parse", "HEAD") == sha   # nothing checked out
    assert rt.content_hash(rel, tmp_path) == rt.content_hash(sha, tmp_path)
    assert 'version = "0.15.0"' in git("show", f"{rel}:pyproject.toml")


def test_a_new_head_waits_for_the_running_gate_instead_of_starting_a_second(tmp_path, monkeypatch, capsys):
    import subprocess
    started = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0))
    monkeypatch.setattr(rt, "start_gate", lambda *a: started.append(a))
    cfg = tmp_path / "gate.json"
    cfg.write_text("{}")
    argv = ["release_train.py", "--gates", str(tmp_path), "--gate-config", str(cfg)]
    monkeypatch.setattr(sys, "argv", argv)
    for running, want in ((["a" * 40], []), ([], [("b" * 40, tmp_path, cfg, None)])):
        monkeypatch.setattr(rt, "decide", lambda **k: {"do": "GATE", "sha": "b" * 40, "why": "", "running": running})
        assert rt.main() == 0 and started == want
    assert "WAIT bbbbbbbbbbbb: gate main's head after the running gate on aaaaaaaaa ends" in capsys.readouterr().out


def test_a_gate_runs_from_its_own_worktree_removed_once_the_gate_ended(tmp_path, monkeypatch):
    import fcntl
    import subprocess
    repo = tmp_path / "clone"
    repo.mkdir()

    def git(*a):
        return subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True,
                              text=True).stdout.strip()
    git("init", "-q"), git("config", "user.email", "t@t"), git("config", "user.name", "t")
    (repo / "a.py").write_text("x = 1\n")
    git("add", "."), git("commit", "-qm", "c")
    monkeypatch.setattr(rt, "REPO", repo)
    gates = tmp_path / "gates"
    live, done = gates / "rel-a", gates / "rel-b"
    for d in (live, done):
        d.mkdir(parents=True)
        assert rt.own_runner(d) == d / "runner" and (d / "runner" / "a.py").exists()
    (live / "VERDICT.md").write_text(""), (done / "VERDICT.md").write_text("")
    with open(live / "runner.lock", "w") as held:      # a runner still holds rel-a
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rt.own_runner(gates / "rel-c")
    assert (live / "runner").exists() and not (done / "runner").exists()
    assert git("rev-parse", "HEAD") == git("-C", str(gates / "rel-c" / "runner"), "rev-parse", "HEAD")
