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
    for n, msg in ((8, "Merge wk/spd-int-next8 (staging8): levers"),
                   (10, "Merge wk/spd-int-next10 (staging10): levers"),
                   ("11d", "Merge staging11d (wk/spd-orch-s11d 92f99b28f): levers")):
        git("checkout", "-qb", f"s{n}")
        (tmp_path / "a.py").write_text(f"x = '{n}'\n")
        git("commit", "-qam", f"lever {n}")
        git("checkout", "-q", "main")
        git("merge", "-q", "--no-ff", f"s{n}", "-m", msg)
    s8 = git("rev-parse", "main^^")
    head = git("rev-parse", "main")
    gates = tmp_path / "gates"
    _plan(gates, "g8", s8, tmp_path)
    d = rn.decide(tmp_path, gates, "main")
    assert d["stagings_since"] == ["8", "10", "11d"]
    assert d["do"] == "GATE" and d["sha"] == head
    assert "finish the running gate on " + s8[:9] in d["why"]

    _plan(gates, "cpu10", head, tmp_path, partial=True)
    (gates / "cpu10" / "verdict.json").write_text(json.dumps({"pass": True}))
    assert rn.decide(tmp_path, gates, "main")["do"] == "GATE"     # a --legs run is no verdict
    _plan(gates, "g10", head, tmp_path)
    assert rn.decide(tmp_path, gates, "main")["do"] == "WAIT"
    (gates / "g10" / "verdict.json").write_text(json.dumps({"pass": False, "results": [
        {"leg": "ladder:m", "verdict": "OWED", "owed": ["SDPA_FUSED_PADDED"]}]}))
    d = rn.decide(tmp_path, gates, "main")
    assert d["do"] == "FIX" and "record SDPA_FUSED_PADDED first" in d["why"]
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


def test_a_passed_gate_ships_even_after_main_moved_on(tmp_path):
    git = _repo(tmp_path)
    (tmp_path / "a.py").write_text("x = 2\n")
    git("commit", "-qam", "staging A")
    a = git("rev-parse", "main")
    git("checkout", "-qb", "rel")                                 # the release branch's own fix
    (tmp_path / "b.py").write_text("y = 1\n")
    git("add", "."), git("commit", "-qm", "gate fix")
    fix = git("rev-parse", "rel")
    git("checkout", "-q", "main")
    (tmp_path / "a.py").write_text("x = 3\n")
    git("commit", "-qam", "staging B")
    head = git("rev-parse", "main")
    gates = tmp_path / "gates"
    (_plan(gates, "ga", a, tmp_path) / "verdict.json").write_text(json.dumps({"pass": True}))
    d = rn.decide(tmp_path, gates, "main")
    assert d["do"] == "CUT" and d["sha"] == a and "gate it now" in d["why"]
    (_plan(gates, "gfix", fix, tmp_path) / "verdict.json").write_text(json.dumps({"pass": True}))
    assert rn.decide(tmp_path, gates, "main")["sha"] == fix      # newest passed candidate
    git("tag", "v0.2.0", fix)                                     # cut: neither is a candidate now
    d = rn.decide(tmp_path, gates, "main")
    assert d["do"] == "GATE" and d["sha"] == head
    (_plan(gates, "gb", head, tmp_path) / "verdict.json").write_text(json.dumps({"pass": True}))
    (tmp_path / "a.py").write_text("x = 4\n")                       # main never merged the release branch
    git("commit", "-qam", "staging C")
    d = rn.decide(tmp_path, gates, "main")
    assert d["do"] == "CUT" and d["sha"] == head                   # B is after v0.2.0 all the same


def test_a_gate_failing_only_on_ladder_drift_is_re_recorded_then_cut(tmp_path):
    git = _repo(tmp_path)
    (tmp_path / "a.py").write_text("x = 2\n")
    git("commit", "-qam", "staging with a new lever")
    head = git("rev-parse", "main")
    gates = tmp_path / "gates"
    g = _plan(gates, f"rel-{head[:9]}", head, tmp_path)
    logs = {"drift": "    FAIL m/512 SDPA: frac 0.000 -> 1.000 (started firing)\n"
                     "    FAIL m/768 MM: decline clause [] -> ['k_tiles=2']\n"
                     "    FAIL m/1024 SW: frac 0.861 -> 0.981 exceeds the 0.05 band (partial darkness)\n",
            "dark": "    FAIL m/512 SDPA: frac 1.000 -> 0.000 (went dark on pad)\n",
            "partial": "    FAIL m/1024 SW: frac 0.981 -> 0.861 exceeds the 0.05 band (partial darkness)\n"}
    for name, text in logs.items():
        (g / f"{name}.log").write_text(text + "GATE FAIL — size-ladder drift vs the recorded baseline\n")

    def verdict(log):
        (g / "verdict.json").write_text(json.dumps({"pass": False, "results": [
            {"leg": "parity:m", "verdict": "PASS"},
            {"leg": "ladder:m@512-768", "verdict": "FAIL", "log": str(g / log)}]}))
    verdict("dark.log")
    assert rn.decide(tmp_path, gates, "main")["do"] == "FIX"              # darkness is not drift
    verdict("partial.log")
    assert rn.decide(tmp_path, gates, "main")["do"] == "FIX"
    verdict("drift.log")
    d = rn.decide(tmp_path, gates, "main")
    assert (d["do"], d["sha"], d["models"]) == ("RECORD", head, ["m"])
    r = _plan(gates, f"rec-{head[:9]}", head, tmp_path, partial=True)
    assert rn.decide(tmp_path, gates, "main")["do"] == "WAIT"
    (r / "verdict.json").write_text(json.dumps({"pass": True}))
    d = rn.decide(tmp_path, gates, "main")
    assert (d["do"], d["sha"], d["records"]) == ("CUT", head, str(r))
    (tmp_path / "a.py").write_text("x = 3\n")                          # main moved on: still the candidate
    git("commit", "-qam", "next staging")
    d = rn.decide(tmp_path, gates, "main")
    assert (d["do"], d["sha"], d["records"]) == ("CUT", head, str(r))


def test_a_card_failed_ladder_leg_of_a_drifted_model_records_alongside_its_rerun(tmp_path, monkeypatch):
    git = _repo(tmp_path)
    (tmp_path / "a.py").write_text("x = 2\n")
    git("commit", "-qam", "staging")
    head = git("rev-parse", "main")
    gates = tmp_path / "gates"
    g = _plan(gates, f"rel-{head[:9]}", head, tmp_path)
    (g / "drift.log").write_text("    FAIL m/512 SDPA: frac 0.000 -> 1.000 (started firing)\n"
                                 "GATE FAIL — size-ladder drift vs the recorded baseline\n")
    (g / "hang.log").write_text("fold timed out\n")

    def verdict(*extra):
        (g / "verdict.json").write_text(json.dumps({"pass": False, "results": [
            {"leg": "ladder:m", "arch": "wh", "verdict": "FAIL", "log": str(g / "drift.log")},
            {"leg": "ladder:m", "arch": "bh", "verdict": "FAIL", "log": str(g / "hang.log"), "key": "k"},
            *extra]}))
    verdict({"leg": "parity:m", "arch": "bh", "verdict": "FAIL"})
    monkeypatch.setattr(rn, "LEDGER", tmp_path / "none")
    assert rn.decide(tmp_path, gates, "main")["do"] == "FIX"          # a non-ladder failure: no record
    verdict()
    d = rn.decide(tmp_path, gates, "main")
    assert (d["do"], d["models"]) == ("RECORD", ["m"]) and "ladder:m bh must pass" in d["why"]
    r = _plan(gates, f"rec-{head[:9]}", head, tmp_path, partial=True)
    (r / "verdict.json").write_text(json.dumps({"pass": True}))
    assert rn.decide(tmp_path, gates, "main")["do"] == "FIX"          # recorded, but the bh leg is owed
    monkeypatch.setattr(rn, "LEDGER", tmp_path / "ledger")
    rn.LEDGER.mkdir()
    (rn.LEDGER / "k.json").write_text("{}")                           # the rerun passed
    d = rn.decide(tmp_path, gates, "main")
    assert (d["do"], d["sha"], d["records"]) == ("CUT", head, str(r))


def test_a_failed_gate_is_fixed_forward_from_its_fixes_not_regated_at_head(tmp_path):
    git = _repo(tmp_path)
    (tmp_path / "a.py").write_text("x = 2\n")
    (tmp_path / "harness.py").write_text("bug = 1\n")
    git("add", "."), git("commit", "-qm", "staging1")
    x = git("rev-parse", "HEAD")
    (tmp_path / "harness.py").write_text("bug = 0\n")
    git("commit", "-qam", "fix the harness")
    fix = git("rev-parse", "HEAD")
    (tmp_path / "a.py").write_text("x = 3\n")
    git("commit", "-qam", "staging2")
    gates = tmp_path / "gates"
    g = _plan(gates, "gx", x, tmp_path)
    (g / "verdict.json").write_text(json.dumps({"pass": False, "results": []}))
    d = rn.decide(tmp_path, gates, "main")
    assert d["do"] == "FIX" and d["sha"] == x and "FIXES" in d["why"]   # not a full gate on staging2
    old = (g / "verdict.json").stat().st_mtime - rn.FIX_HOLD_S - 60
    os.utime(g / "verdict.json", (old, old))
    assert rn.decide(tmp_path, gates, "main")["sha"] == git("rev-parse", "HEAD")   # nobody fixed it
    (g / "FIXES").write_text(f"{fix[:9]}  # harness\n")
    d = rn.decide(tmp_path, gates, "main")
    c = d["sha"]
    assert d["do"] == "GATE" and d["fixes"]["base"] == x and c != git("rev-parse", "HEAD")
    assert rn.decide(tmp_path, gates, "main")["sha"] == c                 # the same list, the same commit
    assert git("show", f"{c}:harness.py") == "bug = 0" and git("show", f"{c}:a.py") == "x = 2"
    assert git("rev-parse", f"{c}^") == x
    gc = _plan(gates, "gc", c, tmp_path)
    assert rn.decide(tmp_path, gates, "main")["do"] == "WAIT"
    (gc / "verdict.json").write_text(json.dumps({"pass": True}))
    d = rn.decide(tmp_path, gates, "main")
    assert d["do"] == "CUT" and d["sha"] == c


def test_a_test_only_fix_forward_is_gated_not_judged_by_the_verdict_it_fixes(tmp_path):
    git = _repo(tmp_path)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text("assert 0\n")
    (tmp_path / "a.py").write_text("x = 2\n")
    git("add", "."), git("commit", "-qm", "staging1")
    x = git("rev-parse", "HEAD")
    (tmp_path / "tests" / "test_a.py").write_text("assert 1\n")
    git("commit", "-qam", "fix the test")
    fix = git("rev-parse", "HEAD")
    gates = tmp_path / "gates"
    g = _plan(gates, "gx", x, tmp_path)
    (g / "verdict.json").write_text(json.dumps({"pass": False, "results": [{"leg": "pytest_cpu", "verdict": "FAIL"}]}))
    (g / "FIXES").write_text(fix + "\n")
    d = rn.decide(tmp_path, gates, "main")
    c = d["sha"]
    assert rn.content_hash(c, tmp_path) == rn.content_hash(x, tmp_path)   # the case: same content key
    assert d["do"] == "GATE" and d["fixes"]["base"] == x and c != x
    gc = _plan(gates, "gc", c, tmp_path)
    assert rn.decide(tmp_path, gates, "main")["do"] == "WAIT"
    (gc / "verdict.json").write_text(json.dumps({"pass": True}))
    d = rn.decide(tmp_path, gates, "main")
    assert d["do"] == "CUT" and d["sha"] == c


def test_a_fix_that_does_not_apply_on_the_frozen_commit_is_a_fix(tmp_path):
    git = _repo(tmp_path)
    (tmp_path / "a.py").write_text("x = 2\n")
    git("commit", "-qam", "staging1")
    x = git("rev-parse", "HEAD")
    (tmp_path / "a.py").write_text("x = 3\n")
    git("commit", "-qam", "staging2")
    (tmp_path / "a.py").write_text("x = 4\n")
    git("commit", "-qam", "fix on top of staging2")
    gates = tmp_path / "gates"
    g = _plan(gates, "gx", x, tmp_path)
    (g / "verdict.json").write_text(json.dumps({"pass": False, "results": []}))
    (g / "FIXES").write_text(git("rev-parse", "HEAD") + "\n")
    d = rn.decide(tmp_path, gates, "main")
    assert d["do"] == "FIX" and d["sha"] == x and "does not apply" in d["why"]


def test_a_waiver_in_the_base_gate_covers_every_candidate_fixed_forward_from_it(tmp_path):
    git = _repo(tmp_path)
    (tmp_path / "a.py").write_text("x = 2\n")
    (tmp_path / "h1.py").write_text("bug = 1\n")
    (tmp_path / "h2.py").write_text("bug = 1\n")
    git("add", "."), git("commit", "-qm", "staging1")
    x = git("rev-parse", "HEAD")
    fixes = []
    for h in ("h1", "h2"):
        (tmp_path / f"{h}.py").write_text("bug = 0\n")
        git("commit", "-qam", f"fix {h}")
        fixes.append(git("rev-parse", "HEAD"))
    gates = tmp_path / "gates"
    log = tmp_path / "dark.log"
    log.write_text("size-ladder drift vs the recorded baseline\n"
                   "    FAIL m/256 SOME_LEVER: frac 0.942 -> 0.000 (went dark on memory)\n")
    fail = {"pass": False, "results": [
        {"leg": "ladder:m", "arch": "wh", "verdict": "FAIL", "log": str(log)},
        {"leg": "perf", "arch": "wh", "verdict": "FAIL"},
        {"leg": "ux:e", "arch": "wh", "verdict": "FAIL"}]}
    g = _plan(gates, "gx", x, tmp_path)
    (g / "verdict.json").write_text(json.dumps(fail))
    d = rn.decide(tmp_path, gates, "main")
    assert d["do"] == "FIX" and "failed: ladder:m wh, perf wh, ux:e wh" in d["why"]
    assert f"{g}/WAIVE" in d["why"] and f"{g}/FIXES" in d["why"]

    (g / "FIXES").write_text(fixes[0] + "\n")
    c1 = rn.decide(tmp_path, gates, "main")["sha"]
    g1 = _plan(gates, "gc1", c1, tmp_path)
    (g1 / "verdict.json").write_text(json.dumps(fail))
    (g / "FIXES").write_text(fixes[0] + "\n" + fixes[1] + "\n")
    c2 = rn.decide(tmp_path, gates, "main")["sha"]
    assert git("rev-parse", f"{c2}^") == c1                     # candidates chain onto each other
    g2 = _plan(gates, "gc2", c2, tmp_path)
    (g2 / "verdict.json").write_text(json.dumps(fail))
    d = rn.decide(tmp_path, gates, "main")
    assert d["do"] == "FIX" and f"{g}/WAIVE" in d["why"]       # waivers go to the base, not gc2

    (g / "WAIVE").write_text("perf wh  # no Wormhole baseline yet\nux:e wh\n")   # no reason: no waiver
    (g1 / "WAIVE").write_text("ladder:m wh  # SOME_LEVER superseded by OTHER_LEVER, e2e faster\n")
    assert rn.waivers(g2, gates, tmp_path, "main") == {
        "perf wh": "no Wormhole baseline yet", "ladder:m wh": "SOME_LEVER superseded by OTHER_LEVER, e2e faster"}
    d = rn.decide(tmp_path, gates, "main")
    assert d["do"] == "FIX" and "failed: ux:e wh" in d["why"] and "perf wh" not in d["why"].split("failed:")[1]
    (g / "WAIVE").write_text("perf wh  # no Wormhole baseline yet\nux:e wh  # flaky host, passed on rerun\n")
    d = rn.decide(tmp_path, gates, "main")
    assert (d["do"], d["sha"], d["models"]) == ("RECORD", c2, ["m"])   # a waived ladder is re-recorded
    r = _plan(gates, f"rec-{c2[:9]}", c2, tmp_path, partial=True)
    (r / "verdict.json").write_text(json.dumps({"pass": True}))
    d = rn.decide(tmp_path, gates, "main")
    assert (d["do"], d["sha"], d["records"]) == ("CUT", c2, str(r))


def test_a_gate_whose_every_failure_is_waived_is_cut(tmp_path):
    git = _repo(tmp_path)
    (tmp_path / "a.py").write_text("x = 2\n")
    git("commit", "-qam", "staging1")
    x = git("rev-parse", "HEAD")
    gates = tmp_path / "gates"
    g = _plan(gates, "gx", x, tmp_path)
    (g / "verdict.json").write_text(json.dumps({"pass": False, "results": [
        {"leg": "perf", "arch": "wh", "verdict": "FAIL"}]}))
    assert rn.decide(tmp_path, gates, "main")["do"] == "FIX"
    (g / "WAIVE").write_text("perf wh  # no Wormhole baseline yet\n")
    d = rn.decide(tmp_path, gates, "main")
    assert (d["do"], d["sha"], d["waived"]) == ("CUT", x, {"perf wh": "no Wormhole baseline yet"})


def test_a_failed_leg_whose_key_passed_since_stops_blocking(tmp_path, monkeypatch):
    """Gate 8b2f8acaa (10-13): ladder:boltz2 bh died on a card a previous leg left dirty; a partial
    run of the same key on a clean card filed the pass the verdict lacked."""
    git = _repo(tmp_path)
    (tmp_path / "a.py").write_text("x = 2\n")
    git("commit", "-qam", "staging1")
    x = git("rev-parse", "HEAD")
    gates, ledger = tmp_path / "gates", tmp_path / "ledger"
    ledger.mkdir()
    monkeypatch.setattr(rn, "LEDGER", ledger)
    g = _plan(gates, "gx", x, tmp_path)
    (g / "verdict.json").write_text(json.dumps({"pass": False, "results": [
        {"leg": "ux:a", "arch": "bh", "verdict": "PASS", "key": "k0"},
        {"leg": "rg:b", "arch": "bh", "verdict": "FAIL", "key": "k1"}]}))
    assert rn.decide(tmp_path, gates, "main")["do"] == "FIX"
    (ledger / "k1.json").write_text("{}")
    d = rn.decide(tmp_path, gates, "main")
    assert (d["do"], d["sha"]) == ("CUT", x)


def test_a_failed_leg_is_judged_by_its_latest_run_on_the_same_commit(tmp_path, monkeypatch):
    """Gate 8b2f8acaa (10-13): the BH ladder reruns ran from newer runner scripts, so their keys
    differed from the gate's; boltz2's pass and opendde's drift-only rerun still read as card faults."""
    git = _repo(tmp_path)
    (tmp_path / "a.py").write_text("x = 2\n")
    git("commit", "-qam", "staging1")
    x = git("rev-parse", "HEAD")
    gates = tmp_path / "gates"
    monkeypatch.setattr(rn, "LEDGER", tmp_path / "none")
    g = _plan(gates, "gx", x, tmp_path)
    (g / "verdict.json").write_text(json.dumps({"pass": False, "results": [
        {"leg": "rg:b", "arch": "bh", "verdict": "FAIL", "key": "k1"}]}))
    assert rn.failures(g) == ([], ["rg:b bh"])
    rerun = gates / "gx-rerun"
    rerun.mkdir()
    rows = [{"leg": "rg:b", "arch": "bh", "verdict": "FAIL", "sha": x, "ended": "2026-10-13T20:00:00Z"},
            {"leg": "rg:b", "arch": "bh", "verdict": "PASS", "sha": x, "ended": "2026-10-13T21:00:00Z"},
            {"leg": "rg:c", "arch": "bh", "verdict": "PASS", "sha": "0" * 40, "ended": "2026-10-13T22:00:00Z"}]
    (rerun / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[:1]))
    assert rn.failures(g) == ([], ["rg:b bh"])
    (rerun / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[::-1]))
    assert rn.failures(g) == ([], [])
