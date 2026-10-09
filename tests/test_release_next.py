"""release_next.py: the release train's decision, on a throwaway repo."""
import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
spec = importlib.util.spec_from_file_location("release_next", REPO / "scripts" / "release_next.py")
rn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rn)


def _repo(tmp_path):
    def git(*a):
        return subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True,
                              text=True).stdout.strip()
    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@t"), git("config", "user.name", "t")
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "pyproject.toml").write_text('version = "0.1.0"\n')
    git("add", "."), git("commit", "-qm", "base")
    git("tag", "v0.1.0")
    return git


def test_nothing_when_main_only_moved_docs_and_version(tmp_path):
    git = _repo(tmp_path)
    (tmp_path / "README.md").write_text("docs\n")
    (tmp_path / "pyproject.toml").write_text('version = "0.2.0"\n')
    git("add", "."), git("commit", "-qm", "docs")
    d = rn.decide(tmp_path, None, "main")
    assert d["do"] == "NOTHING" and d["last_release"] == "v0.1.0"


def _plan(gates, name, sha, repo, **kw):
    (gates / name).mkdir(parents=True)
    (gates / name / "plan.json").write_text(json.dumps(
        {"sha": sha, "content": rn.content_hash(sha, repo), "pid": os.getpid(),
         "host": socket.gethostname(), **kw}))
    return gates / name


def test_gate_the_newest_staging_and_never_restart_a_running_gate(tmp_path):
    git = _repo(tmp_path)
    for n in (8, 10):
        git("checkout", "-qb", f"s{n}")
        (tmp_path / "a.py").write_text(f"x = {n}\n")
        git("commit", "-qam", f"lever {n}")
        git("checkout", "-q", "main")
        git("merge", "-q", "--no-ff", f"s{n}", "-m", f"Merge wk/spd-int-next{n} (staging{n}): levers")
    s8 = git("rev-parse", "main^")
    head = git("rev-parse", "main")
    gates = tmp_path / "gates"
    _plan(gates, "g8", s8, tmp_path)
    d = rn.decide(tmp_path, gates, "main")
    assert d["stagings_since"] == ["8", "10"]
    assert d["do"] == "GATE" and d["sha"] == head
    assert "finish the running gate on " + s8[:9] in d["why"]

    _plan(gates, "cpu10", head, tmp_path, partial=True)
    (gates / "cpu10" / "verdict.json").write_text(json.dumps({"pass": True}))
    assert rn.decide(tmp_path, gates, "main")["do"] == "GATE"     # a --legs run is no verdict
    _plan(gates, "g10", head, tmp_path)
    assert rn.decide(tmp_path, gates, "main")["do"] == "WAIT"
    (gates / "g10" / "verdict.json").write_text(json.dumps({"pass": False}))
    assert rn.decide(tmp_path, gates, "main")["do"] == "FIX"
    (gates / "g10" / "verdict.json").write_text(json.dumps({"pass": True}))
    assert rn.decide(tmp_path, gates, "main")["do"] == "CUT"


def test_a_docs_commit_rides_its_parents_gate_and_a_dead_run_is_none(tmp_path):
    git = _repo(tmp_path)
    (tmp_path / "a.py").write_text("x = 2\n")
    git("commit", "-qam", "lever")
    code = git("rev-parse", "main")
    (tmp_path / "docs.md").write_text("docs\n")
    git("add", "."), git("commit", "-qm", "docs")
    gates = tmp_path / "gates"
    g = _plan(gates, "g", code, tmp_path)
    d = rn.decide(tmp_path, gates, "main")
    assert d["do"] == "WAIT" and "finish" not in d["why"]
    plan = json.loads((g / "plan.json").read_text())
    plan["pid"] = 2 ** 22 + 1                                       # above pid_max: never alive
    (g / "plan.json").write_text(json.dumps(plan))
    assert rn.decide(tmp_path, gates, "main")["do"] == "GATE"
    _plan(gates, "wh", code, tmp_path, partial=True)
    assert "partial runs on " + code[:9] in rn.decide(tmp_path, gates, "main")["why"]
