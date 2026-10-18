"""release_train.py: notes, version and the release commit, on strings and a throwaway tree."""
import importlib.util
import json
import subprocess
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


def test_merge_back_takes_the_release_files_main_left_alone_and_keeps_mains_notes(tmp_path, monkeypatch):
    """rel/v0.17.0 and rel/v0.18.0 were never merged back, so main graded every later gate against
    the pre-0.18.0 ladder baselines and failed census lines 0.18.0 had accepted. And 0.19.0's merge
    kept main's whole README and pyproject, which main had touched, so main still pinned v0.18.0."""
    origin, repo = tmp_path / "origin.git", tmp_path / "repo"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    subprocess.run(["git", "clone", "-q", str(origin), str(repo)], check=True, capture_output=True)

    def git(*a):
        return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a],
                              check=True, capture_output=True, text=True).stdout.strip()

    def commit(files, msg):
        for f, text in files.items():
            (repo / f).parent.mkdir(parents=True, exist_ok=True)
            (repo / f).write_text(text)
        git("add", *files)
        git("commit", "-q", "-m", msg)
        return git("rev-parse", "HEAD")

    sha = commit({"CHANGELOG.md": "## [Unreleased]\n- a\n", "base.json": "old\n", "touched.json": "old\n",
                  "pyproject.toml": 'version = "0.1.0"\n', "README.md": "pip tt-bio.git@v0.1.0\n"}, "gated")
    git("push", "-q", "origin", "HEAD:main")
    rel = commit({"CHANGELOG.md": "## [Unreleased]\n\n## [0.2.0]\n- a\n", "base.json": "recorded\n",
                  "touched.json": "recorded\n", "pyproject.toml": 'version = "0.2.0"\n',
                  "README.md": "pip tt-bio.git@v0.2.0\n"}, "Release 0.2.0")
    git("checkout", "-q", "--detach", sha)
    commit({"CHANGELOG.md": "## [Unreleased]\n- b\n- a\n", "touched.json": "main's\n",
            "README.md": "new line\npip tt-bio.git@v0.1.0\n"}, "later on main")
    git("push", "-q", "origin", "HEAD:main")
    monkeypatch.setattr(rt, "REPO", repo)

    m = rt.merge_back(rel, sha, "origin")
    assert m and git("rev-parse", "origin/main") == m
    assert git("merge-base", "--is-ancestor", rel, m) == ""
    assert git("show", f"{m}:base.json") == "recorded"
    assert git("show", f"{m}:pyproject.toml") == 'version = "0.2.0"'
    assert git("show", f"{m}:touched.json") == "main's"
    assert git("show", f"{m}:README.md") == "new line\npip tt-bio.git@v0.2.0"
    assert git("show", f"{m}:CHANGELOG.md") == "## [Unreleased]\n\n- b\n\n## [0.2.0]\n- a"
    assert rt.merge_back(rel, sha, "origin") == m          # already merged: nothing new


def test_a_gate_waits_while_no_galaxy_is_lent(tmp_path, monkeypatch, capsys):
    """A "galaxies" config with nothing lent would start a gate with no WH worker: wait for a lend instead."""
    import subprocess
    started = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0))
    monkeypatch.setattr(rt, "start_gate", lambda *a: started.append(a))
    monkeypatch.setattr(rt, "decide", lambda **k: {"do": "GATE", "sha": "b" * 40, "why": "", "running": []})
    (tmp_path / "hosts.json").write_text(json.dumps({"g103": {}}))
    alloc = tmp_path / "ALLOC.json"
    cfg = tmp_path / "gate.json"
    cfg.write_text(json.dumps({"hosts": str(tmp_path / "hosts.json"),
                               "galaxies": {"alloc": str(alloc), "owners": ["spd"], "slots": 16}}))
    monkeypatch.setattr(sys, "argv", ["release_train.py", "--gates", str(tmp_path), "--gate-config", str(cfg)])
    alloc.write_text(json.dumps({"galaxies": {"103": {"owner": "japanfold", "phase": "serving"}}}))
    assert rt.main() == 0 and started == []
    assert "WAIT bbbbbbbbbbbb: no Galaxy lent to spd; WH legs need one" in capsys.readouterr().out
    alloc.write_text(json.dumps({"galaxies": {"103": {"owner": "spd", "phase": "lent"}}}))
    assert rt.main() == 0 and len(started) == 1


def test_galaxy_pools_follow_the_lent_galaxy(tmp_path):
    """Pool slots go to the Galaxy lent to the gate's owners now, not to a Galaxy named by hand."""
    hosts = tmp_path / "hosts.json"
    hosts.write_text(json.dumps({"qb2": {}, "g107": {}, "g108": {}}))
    alloc = tmp_path / "ALLOC.json"
    alloc.write_text(json.dumps({"galaxies": {
        "107": {"owner": "japanfold", "phase": "serving"},
        "108": {"owner": "spd", "phase": "lent"},
        "110": {"owner": "spd", "phase": "lent"},       # lent, but no hosts entry to reach it by
        "111": {"owner": "cfd", "phase": "lent"}}}))
    cfg = {"hosts": str(hosts), "workers": "qb2:1", "timed": "qb2:1",
           "galaxies": {"alloc": str(alloc), "owners": ["spd"], "slots": 2}}
    got = rt.galaxy_pools(cfg)
    assert got["workers"] == "qb2:1,g108:pool,g108:pool"
    assert got["timed"] == "qb2:1,g108:pool"
    alloc.write_text(json.dumps({"galaxies": {"108": {"owner": "japanfold", "phase": "serving"}}}))
    assert rt.galaxy_pools(cfg)["workers"] == "qb2:1"
    assert rt.galaxy_pools({"workers": "qb2:1"}) == {"workers": "qb2:1"}
