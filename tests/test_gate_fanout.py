"""gate_fanout.py: leg split, result keys, reuse, scheduling and verdict, all card-free."""
import importlib.util
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gate_fanout", REPO / "scripts" / "gate_fanout.py")
gf = importlib.util.module_from_spec(spec)
sys.modules["gate_fanout"] = gf
spec.loader.exec_module(gf)

ROSTER = {"parity": ["esmc-300m", "boltzgen", "capacity"],
          "rg": ["boltz2", "boltzgen", "opendde-abag", "capacity", "nesso1", "rf3-1024aa",
                 "l1-budget", "size-ladder"],
          "ladder": ["boltz2", "rf3"], "capacity": ["boltz2"], "ux": ["boltz2"]}


@pytest.fixture(autouse=True)
def _no_real_bad_marks(tmp_path, monkeypatch):
    """Keep the bad-chip check off this machine's real ~/spd/locks and ~/.coworker/state."""
    monkeypatch.setattr(gf, "LOCAL_BLOCKS", tmp_path / "pc-state")
    monkeypatch.setattr(gf, "BAD_MARKS", (f"{tmp_path}/locks/chip{{card}}.BAD",
                                          f"{tmp_path}/host-state/cardblock-{{host}}-{{card}}"))


def _legs():
    return gf.build_legs(ROSTER, ["tests/test_a.py", "tests/test_b.py", "tests/test_c.py"], 2)


def test_every_arm_becomes_a_leg_and_duplicates_of_parity_are_dropped():
    names = [lg.name for lg in _legs()]
    assert "rg:boltz2" in names and "rg:l1-budget" in names
    # covered by the parity gate's in-process call of the same release_gate runner
    for arm in gf.PARITY_COVERS_RG:
        assert f"rg:{arm}" not in names
    assert "rg:size-ladder" not in names
    assert {"ladder:boltz2", "ladder:rf3", "capacity:boltz2", "ux:boltz2", "perf"} <= set(names)
    assert {"check", "packaging_smoke", "pytest_cpu"} <= set(names)


def test_pytest_shards_cover_every_file_exactly_once():
    shards = [lg for lg in _legs() if lg.family == "pytest_device"]
    files = [a for lg in shards for a in lg.argv if a.startswith("tests/") and a != gf.STALE_TEST]
    assert sorted(files) == ["tests/test_a.py", "tests/test_b.py", "tests/test_c.py"]


def test_pytest_leaves_the_abb3_staleness_test_to_record_abb3_only_when_the_gate_runs_it():
    for abb3 in (False, True):
        legs = gf.build_legs(ROSTER, ["tests/test_a.py"], 1, record_abb3=abb3)
        assert any(lg.name == gf.ABB3_LEG for lg in legs) is abb3
        assert all((gf.STALE_TEST in lg.argv) is abb3 for lg in legs if lg.family in gf.PYTEST)


def test_only_perf_is_timed_and_card_free_legs_take_no_card():
    legs = _legs()
    assert [lg.name for lg in legs if lg.timed] == ["perf"]
    assert {lg.name for lg in legs if not lg.card} == set(gf.CARD_FREE)


def test_select_globs():
    assert [lg.name for lg in gf.select(_legs(), ["rg:*"])] == ["rg:boltz2", "rg:l1-budget"]
    names = [lg.name for lg in gf.select(_legs(), ["!ladder:*", "!rg:l1-*"])]
    assert "rg:boltz2" in names and "perf" in names
    assert not [n for n in names if n.startswith("ladder:") or n == "rg:l1-budget"]


def test_record_lever_legs_splice_into_a_copy_and_are_only_built_on_request():
    assert not [lg for lg in _legs() if lg.family == "record"]
    legs = [lg for lg in gf.build_legs(ROSTER, [], 1, "SDPA_FUSED_PADDED") if lg.family == "record"]
    assert [lg.name for lg in legs] == ["record:boltz2", "record:rf3"]
    h = gf.Host("h", {"arch": "wh", "card_type": "w", "lock": "/l{card}", "root": "/r"}, "a" * 40)
    cmd = h.command(legs[0], 3, "/o")
    assert "cp -r docs/size_ladder_baseline.json docs/size_ladder_baseline.d /o/ && " in cmd
    assert "--size-ladder-baseline /o/size_ladder_baseline.json" in cmd
    assert "--size-ladder-record-lever SDPA_FUSED_PADDED" in cmd


def test_record_full_re_records_a_refused_model_and_splices_the_rest():
    legs = {lg.name: lg for lg in gf.build_legs(ROSTER, [], 1, "SDPA_FUSED_PADDED", "rf3")
            if lg.family == "record"}
    assert "--size-ladder-record" in legs["record:rf3"].argv
    assert "--size-ladder-record-lever" not in legs["record:rf3"].argv
    assert "--size-ladder-record-lever" in legs["record:boltz2"].argv
    only = [lg.name for lg in gf.build_legs(ROSTER, [], 1, "", "rf3") if lg.family == "record"]
    assert only == ["record:rf3"]
    with pytest.raises(SystemExit):
        gf.build_legs(ROSTER, [], 1, "", "nope")


def test_key_moves_with_code_env_card_and_argv_not_with_markdown(tmp_path):
    def git(*a):
        return subprocess.run(["git", "-C", str(tmp_path), *a], check=True,
                              capture_output=True, text=True).stdout.strip()
    git("init", "-q")
    git("config", "user.email", "t@t"), git("config", "user.name", "t")
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "README.md").write_text("one\n")
    git("add", "."), git("commit", "-qm", "1")
    c1 = gf.content_hash("HEAD", tmp_path)
    (tmp_path / "README.md").write_text("two\n")
    git("commit", "-qam", "2")
    assert gf.content_hash("HEAD", tmp_path) == c1          # docs-only commit: same key
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "t"\nversion = "0.1.0"\n')
    git("add", "."), git("commit", "-qm", "3")
    c3 = gf.content_hash("HEAD", tmp_path)
    assert c3 != c1
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "t"\nversion = "0.2.0"\n')
    git("commit", "-qam", "4")
    assert gf.content_hash("HEAD", tmp_path) == c3          # release version bump: same key
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "t"\nversion = "0.2.0"\ndeps = 1\n')
    git("commit", "-qam", "5")
    assert gf.content_hash("HEAD", tmp_path) != c3          # any other pyproject change: new key
    (tmp_path / "a.py").write_text("x = 2\n")
    git("commit", "-qam", "6")
    assert gf.content_hash("HEAD", tmp_path) != c1          # code change: new key
    c6 = gf.content_hash("HEAD", tmp_path)
    perf = tmp_path / "perf"
    perf.mkdir()
    (perf / "gate.txt").write_text("RECORDED-AT: abcdef1 scripts/x.py\n1.0 s\n")
    git("add", "."), git("commit", "-qm", "7")
    assert gf.content_hash("HEAD", tmp_path) == c6          # a re-recorded measurement: same key
    (perf / "data.txt").write_text("fixture\n")
    git("add", "."), git("commit", "-qm", "8")
    assert gf.content_hash("HEAD", tmp_path) != c6          # any other perf/ file: new key
    c8, l8 = gf.content_hash("HEAD", tmp_path), gf.baseline_hash("HEAD", "ladder", tmp_path)
    b8, o8 = (gf.baseline_hash("HEAD", "ladder", tmp_path, model=m) for m in ("boltz2", "other"))
    p8 = gf.baseline_hash("HEAD", "perf", tmp_path)
    (tmp_path / "docs" / "size_ladder_baseline.d").mkdir(parents=True)
    (tmp_path / "docs" / "size_ladder_baseline.d" / "boltz2.json").write_text("{}\n")
    git("add", "."), git("commit", "-qm", "9")
    assert gf.content_hash("HEAD", tmp_path) == c8          # a re-recorded baseline: same code key,
    assert gf.baseline_hash("HEAD", "ladder", tmp_path) != l8  # a new key for the ladder legs only
    assert gf.baseline_hash("HEAD", "perf", tmp_path) == p8
    assert gf.baseline_hash("HEAD", "ladder", tmp_path, model="boltz2") != b8   # that model's leg reruns,
    assert gf.baseline_hash("HEAD", "ladder", tmp_path, model="other") == o8    # another model's does not
    assert gf.content_hash("HEAD", tmp_path, baselines=True) != c8
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "gate_fanout.py").write_text("# runner fix\n")
    git("add", "."), git("commit", "-qm", "10")
    assert gf.content_hash("HEAD", tmp_path) == c8          # a fix to this runner: same key
    (tmp_path / "scripts" / "flock_first.sh").write_text("# how a leg waits for its card\n")
    git("add", "."), git("commit", "-qm", "10b")
    assert gf.content_hash("HEAD", tmp_path) == c8          # so is a fix to the card wait,
    assert gf.content_hash("HEAD", tmp_path, runner=gf.RUNNER_FILES_0) != c8  # once part of the key

    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_only_pytest.py").write_text("def test_a(): pass\n")
    (tmp_path / "tests" / "test_loaded.py").write_text("def test_b(): pass\n")
    (tmp_path / "scripts" / "gate.py").write_text("load('tests/test_loaded.py')\n"
                                                  "# covered by tests/test_only_pytest.py\n")
    (tmp_path / "scripts" / "run.sh").write_text("pytest tests/test_only_pytest.py\n")
    git("add", "."), git("commit", "-qm", "11")
    c11 = gf.content_hash("HEAD", tmp_path)
    t11 = gf.tests_hash("HEAD", "pytest_device", tmp_path)
    assert gf.test_only("HEAD", tmp_path) == {"tests/test_only_pytest.py"}
    assert gf.tests_hash("HEAD", "parity", tmp_path) == ""
    (tmp_path / "tests" / "test_only_pytest.py").write_text("def test_a(): assert 1\n")
    git("commit", "-qam", "12")
    assert gf.content_hash("HEAD", tmp_path) == c11         # a test fix: same code key,
    assert gf.tests_hash("HEAD", "pytest_device", tmp_path) != t11  # the pytest legs rerun
    assert gf.content_hash("HEAD", tmp_path, tests=True) != gf.content_hash("HEAD~", tmp_path, tests=True)
    (tmp_path / "tests" / "test_loaded.py").write_text("def test_b(): assert 1\n")
    git("commit", "-qam", "13")
    assert gf.content_hash("HEAD", tmp_path) != c11         # a test a gate script loads is code

    leg = _legs()[3]
    k = gf.leg_key(c1, "env", "p150a", leg)
    assert k == gf.leg_key(c1, "env", "p150a", leg, "", "")   # an empty input keeps the old key
    assert k == gf.leg_key(c1, "env", "p150a", leg)
    assert k != gf.leg_key(c1, "env2", "p150a", leg)
    assert k != gf.leg_key(c1, "env", "tt-galaxy-wh-l", leg)
    other = gf.Leg(leg.name, leg.argv + ["--seeds", "1"], leg.family)
    assert k != gf.leg_key(c1, "env", "p150a", other)


def test_a_ladder_key_sees_only_its_own_card_types_rows(tmp_path):
    def git(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)
    frag = tmp_path / "docs" / "size_ladder_baseline.d" / "nesso1.json"
    frag.parent.mkdir(parents=True)

    def write(cards):
        frag.write_text(json.dumps({"cards": cards}))
        git("add", "."), git("commit", "-qm", "x")
    git("init", "-q"), git("config", "user.email", "t@t"), git("config", "user.name", "t")
    bh = {"p150a": {"host_threads": 2, "models": {"nesso1": {"256": 1}}}}
    write({**bh, "wh": {"host_threads": 2, "models": {"nesso1": {"256": 2}}}})
    key = lambda cards, model="nesso1": gf.baseline_hash("HEAD", "ladder", tmp_path, model=model, cards=cards)
    b1, w1, blob1 = key(("p150a", "p300c")), key(("wh",)), gf.baseline_hash("HEAD", "ladder", tmp_path, model="nesso1")
    write({**bh, "wh": {"host_threads": 2, "models": {"nesso1": {"256": 3}}}})       # a WH re-record
    assert key(("p150a", "p300c")) == b1                                             # BH legs reuse
    assert key(("wh",)) != w1                                                        # WH legs rerun
    assert gf.baseline_hash("HEAD", "ladder", tmp_path, model="nesso1") != blob1     # the blob moved
    write({"p150a": {"host_threads": 4, "models": bh["p150a"]["models"]}, "wh": {}})
    assert key(("p150a", "p300c")) != b1                                             # the cap is read too


def test_a_commit_with_the_same_dependencies_installs_the_last_gated_resolution(tmp_path):
    led = gf.Ledger(tmp_path)
    assert led.pins("a" * 40, "spec1") is None                  # nothing recorded: resolve from PyPI
    led.save_pins("a" * 40, "spec1", {"venv": "beartype==0.23.0\n"})
    led.save_pins("a" * 40, "spec1", {"venv": "beartype==0.23.1\n"})   # the first resolution stays
    assert led.pins("b" * 40, "spec1") == {"venv": "beartype==0.23.0\n"}
    assert led.pins("b" * 40, "spec2") is None                  # a dependency changed: resolve again
    assert led.pins("a" * 40, "spec2") == {"venv": "beartype==0.23.0\n"}   # its own, whatever the spec
    assert not led.shas()                                       # the pins are not ledger rows


def test_env_hash_ignores_order_only():
    a = {"python": "3.12.1", "dists": ["ttnn==0.68.0", "torch==2.13.0"]}
    b = {"python": "3.12.1", "dists": ["ttnn==0.68.0", "torch==2.13.0"]}
    c = {"python": "3.12.1", "dists": ["ttnn==0.67.4", "torch==2.13.0"]}
    assert gf.env_hash(a) == gf.env_hash(b) != gf.env_hash(c)


def test_parity_blocked_is_not_a_failure_but_a_scored_gap_is():
    leg = next(lg for lg in _legs() if lg.family == "parity")
    blocked = {"legs": [{"verdict": "BLOCKED-REF-REGEN-NEEDED"}]}
    assert gf.classify(leg, 1, blocked) == "BLOCKED"
    assert gf.classify(leg, 1, {"legs": [{"verdict": "GAP"}]}) == "FAIL"
    assert gf.classify(leg, 0, {"legs": [{"verdict": "PASS"}]}) == "PASS"
    assert gf.classify(leg, 1, None) == "FAIL"


def test_capacity_size_guard_refusal_passes_only_when_the_baseline_records_it():
    leg = next(lg for lg in _legs() if lg.family == "capacity")
    refused = {"model": "opendde", "verdict": "FAIL", "mechanism": "size_guard"}
    report = {"counts": {"fail_like": 1}, "coverage_gaps": [], "results": [refused]}
    recorded = {"opendde": {"verdict": "FAIL", "mechanism": "size_guard"}}
    assert gf.classify(leg, 1, report, recorded) == "PASS"
    assert gf.classify(leg, 1, report, {}) == "FAIL"
    assert gf.classify(leg, 1, report, None) == "FAIL"
    oom = {**refused, "mechanism": "dram_oom"}
    assert gf.classify(leg, 1, {**report, "results": [oom]}, recorded) == "FAIL"
    assert gf.classify(leg, 1, {**report, "coverage_gaps": ["x"]}, recorded) == "FAIL"


def test_a_dropped_ssh_connection_is_not_a_leg_result():
    assert gf.SSH_LOST.search("# leg\nssh: connect to host 192.168.178.70 port 22: No route to host\n")
    assert gf.SSH_LOST.search("Timeout, server 192.168.178.70 not responding.\n")
    assert gf.SSH_LOST.search("ssh_dispatch_run_fatal: Connection to 192.168.178.70 port 22: Broken pipe\n")
    assert not gf.SSH_LOST.search("FAILED tests/test_x.py::test_ssh_connect_to_host\n")


def test_a_full_disk_is_not_a_leg_result():
    assert gf.DISK_FULL.search("mkdir: cannot create directory '/x': No space left on device\n")
    assert gf.DISK_FULL.search("OSError: [Errno 28] No space left on device: '/tmp/x'\n")
    assert not gf.DISK_FULL.search("RuntimeError: Out of Memory: Not enough space to allocate\n")


def test_a_refused_card_reset_is_matched_from_release_gates_own_words():
    assert gf.CARD_STUCK.search((REPO / "scripts/release_gate.py").read_text())


def test_a_record_leg_returns_the_entry_it_recorded_not_the_staged_copy():
    staged = {"cards": {"p150a": {"recorded": "old", "models": {"m": 1}},
                        "p300c": {"recorded": "old", "models": {"m": 2}}}}
    written = {"cards": {"p150a": {"recorded": "x", "models": {}},
                         "p300c": {"recorded": "new", "host_threads": 5, "models": {"m": 3}}}}
    frag = json.loads(gf.record_fragment(json.dumps(staged), json.dumps(written), "p300c", "m"))
    assert frag["cards"]["p300c"] == written["cards"]["p300c"]
    assert frag["cards"]["p150a"] == staged["cards"]["p150a"]
    assert gf.record_fragment(json.dumps(staged), json.dumps(written), "p150a", "m") is None


def test_a_contended_card_is_not_a_leg_result():
    assert gf.CONTENDED.search("[pxdesign] measurement error: design [pxdesign] rep 2/3 exited 75 after 123s: "
                               "device contention, nothing ran: physical card 3 on tt-quietbox is in use\n")
    assert gf.CONTENDED.search("waited 120s. Refusing to open it concurrently (would wedge)\n")
    assert not gf.CONTENDED.search("GATE FAIL — regressed beyond ±15%: boltzgen\n")


def _host(name, arch, tmp_path):
    return gf.Host(name, {"arch": arch, "card_type": arch, "root": str(tmp_path)}, "f" * 40)


def test_scheduler_runs_every_leg_once_per_arch_and_timed_legs_alone(tmp_path):
    bh, wh = _host("qb1", "bh", tmp_path), _host("g", "wh", tmp_path)
    legs = [lg for lg in _legs() if lg.card]
    todo = {"bh": list(legs), "wh": list(legs)}
    live = {"bh": set(), "wh": set()}
    mu = threading.Lock()
    violations = []

    def execute(host, card, leg):
        with mu:
            if leg.timed and live[host.arch]:
                violations.append(("timed leg started beside", set(live[host.arch])))
            if any(n == "perf" for n in live[host.arch]):
                violations.append(("leg started beside timed", leg.name))
            if gf.out_group(leg.name) in {gf.out_group(n) for n in live[host.arch]}:
                violations.append(("leg started beside its output dir's other writer", leg.name))
            live[host.arch].add(leg.name)
        time.sleep(0.01)
        with mu:
            live[host.arch].discard(leg.name)
        return {"leg": leg.name, "arch": host.arch, "verdict": "PASS", "worker": f"{host.name}:{card}"}

    workers = [(bh, 0), (bh, 1), (bh, 3), (wh, 5), (wh, 6)]
    res = gf.Gate(todo, workers, {("qb1", 3), ("g", 6)}, execute, tmp_path).run()
    assert not violations
    got = sorted((r["leg"], r["arch"]) for r in res)
    assert got == sorted((lg.name, a) for lg in legs for a in ("bh", "wh"))
    perf = [r for r in res if r["leg"] == "perf"]
    assert {r["worker"] for r in perf} == {"qb1:3", "g:6"}
    assert (tmp_path / "results.jsonl").read_text().count("\n") == len(res)


def test_a_models_fast_arm_shares_its_output_dir():
    assert gf.out_group("rg:esmfold2-fast") == gf.out_group("rg:esmfold2")
    assert gf.out_group("rg:esmfold2") != gf.out_group("ux:esmfold2")
    assert gf.out_group("rg:protenix-v1") == gf.out_group("rg:protenix-v2")
    assert gf.out_group("rg:opendde") != gf.out_group("rg:opendde-abag")


def test_legs_sharing_an_output_dir_never_overlap_on_one_host(tmp_path):
    host = _host("g", "wh", tmp_path)
    legs = [gf.Leg(n, [], "rg") for n in ("rg:esmfold2", "rg:esmfold2-fast", "rg:boltz2")]
    spans = {}

    def execute(h, card, leg):
        t0 = time.monotonic()
        time.sleep(0.05)
        spans[leg.name] = (t0, time.monotonic())
        return {"leg": leg.name, "arch": h.arch, "verdict": "PASS", "worker": f"g:{card}"}

    res = gf.Gate({"wh": legs}, [(host, 0), (host, 1), (host, 2)], set(), execute, tmp_path).run()
    assert len(res) == 3
    (a0, a1), (b0, b1) = spans["rg:esmfold2"], spans["rg:esmfold2-fast"]
    assert a1 <= b0 or b1 <= a0


def test_verdict_counts_reused_and_blocked_as_pass_and_fails_on_any_fail(tmp_path):
    rows = [{"leg": "rg:boltz2", "arch": "bh", "verdict": "PASS", "wall_s": 600, "worker": "qb1:0"},
            {"leg": "rg:boltz2", "arch": "wh", "verdict": "REUSED", "evidence": "g:5 x"},
            {"leg": "parity:x", "arch": "bh", "verdict": "BLOCKED", "wall_s": 1, "worker": "qb1:1"},
            {"leg": "parity:x", "arch": "wh", "verdict": "PASS", "wall_s": 1, "worker": "g:6"},
            {"leg": "check", "arch": "any", "verdict": "PASS", "wall_s": 9, "worker": "qb1:None"}]
    assert gf.write_verdict(tmp_path, "a" * 40, ["bh", "wh"], rows)
    md = (tmp_path / "VERDICT.md").read_text()
    assert md.startswith("# Gate aaaaaaaaaaaa: PASS") and "REUSED (g:5 x)" in md
    rows.append({"leg": "ux:boltz2", "arch": "wh", "verdict": "FAIL", "wall_s": 1, "worker": "g:5"})
    assert not gf.write_verdict(tmp_path, "a" * 40, ["bh", "wh"], rows)
    assert json.loads((tmp_path / "verdict.json").read_text())["pass"] is False


def test_ledger_roundtrip_and_history_takes_the_latest_wall(tmp_path):
    led = gf.Ledger(tmp_path / "l")
    assert led.get("k") is None
    led.put("k", {"verdict": "PASS", "leg": "ux:boltz2", "wall_s": 100, "ended": "2026-10-09T01:00:00Z"})
    led.put("j", {"verdict": "PASS", "leg": "ux:boltz2", "wall_s": 300, "ended": "2026-10-09T02:00:00Z"})
    assert led.get("k")["wall_s"] == 100
    assert led.history() == {"ux:boltz2": 300}


def test_a_first_host_takes_its_card_through_flock_first():
    h = gf.Host("qb1", {"arch": "bh", "card_type": "p150a", "root": "/r", "first": True,
                        "lock": "/l/card{card}.lock"}, "a" * 40)
    leg = next(lg for lg in _legs() if lg.family == "parity")
    assert "bash scripts/flock_first.sh /l/card3.lock env " in h.command(leg, 3, "/t/out/x")


def test_flock_first_runs_ahead_of_a_waiter_and_continues_it_after_the_grace(tmp_path):
    """A holder, then a foreign waiter, then the gate leg: the gate leg runs second, the waiter
    is stopped meanwhile and continued once the grace passes with the card free."""
    lock, order = tmp_path / "card.lock", tmp_path / "order"
    lock.touch()
    holder = subprocess.Popen(["flock", str(lock), "sleep", "3"])
    time.sleep(0.3)
    foreign = subprocess.Popen(["flock", str(lock), "sh", "-c", f"echo foreign >> {order}"])
    time.sleep(0.3)
    gate = subprocess.Popen(["bash", str(REPO / "scripts" / "flock_first.sh"), str(lock),
                             "sh", "-c", f"echo gate >> {order}"], env={"FLOCK_FIRST_GRACE": "1", "PATH": "/usr/bin:/bin"})
    time.sleep(1.5)
    assert Path(f"{lock}.gate-stopped").read_text().split() == [str(foreign.pid)]
    assert gate.wait(10) == 0 and holder.wait(10) == 0 and foreign.wait(15) == 0
    assert order.read_text().split() == ["gate", "foreign"]
    assert Path(f"{lock}.gate-stopped").read_text() == ""


def test_flock_first_never_stops_another_gates_wait(tmp_path):
    """Two gates on one card stopping each other's waits would both stay frozen."""
    lock = tmp_path / "card.lock"
    lock.touch()
    holder = subprocess.Popen(["flock", str(lock), "sleep", "3"])
    time.sleep(0.3)
    other = subprocess.Popen(["bash", "-c", f'exec -a gate_fanout-wait flock -w 20 {lock} true'])
    time.sleep(0.3)
    gate = subprocess.Popen(["bash", str(REPO / "scripts" / "flock_first.sh"), str(lock), "true"],
                            env={"FLOCK_FIRST_GRACE": "1", "PATH": "/usr/bin:/bin"})
    time.sleep(1.5)
    assert not Path(f"{lock}.gate-stopped").exists() or Path(f"{lock}.gate-stopped").read_text() == ""
    assert holder.wait(10) == 0 and gate.wait(10) == 0 and other.wait(20) == 0

def test_flock_first_continues_its_waiters_when_the_leg_is_sigkilled(tmp_path):
    """A leg killed by SIGKILL or a timeout runs no exit trap; its waiters must not stay stopped."""
    lock = tmp_path / "card.lock"
    lock.touch()
    holder = subprocess.Popen(["flock", str(lock), "sleep", "2"])
    time.sleep(0.3)
    foreign = subprocess.Popen(["flock", str(lock), "true"])
    time.sleep(0.3)
    gate = subprocess.Popen(["bash", str(REPO / "scripts" / "flock_first.sh"), str(lock), "sleep", "60"],
                            env={"FLOCK_FIRST_GRACE": "1", "PATH": "/usr/bin:/bin"}, start_new_session=True)
    time.sleep(3)
    os.killpg(gate.pid, signal.SIGKILL)
    gate.wait(5)                                  # a zombie still answers kill -0
    assert holder.wait(10) == 0 and foreign.wait(15) == 0


def test_flock_first_continues_its_waiters_while_an_ordinary_waiter_holds_the_card(tmp_path):
    """qb1 10-10: a process that queued after the stop pass took the card during the grace, the
    reaper read that as a gate leg and left 64 waiters stopped for up to 9 h."""
    lock = tmp_path / "card.lock"
    lock.touch()
    holder = subprocess.Popen(["flock", str(lock), "sleep", "2"])
    time.sleep(0.3)
    foreign = subprocess.Popen(["flock", str(lock), "true"])
    time.sleep(0.3)
    assert subprocess.run(["bash", str(REPO / "scripts" / "flock_first.sh"), str(lock), "true"],
                          env={"FLOCK_FIRST_GRACE": "1", "PATH": "/usr/bin:/bin"}).returncode == 0
    late = subprocess.Popen(["flock", str(lock), "sleep", "3"])
    assert holder.wait(10) == 0 and late.wait(10) == 0 and foreign.wait(15) == 0


def test_flock_first_frees_the_card_when_its_leg_ends(tmp_path):
    """The grace keeps paused waiters paused; it must not keep the card locked."""
    lock = tmp_path / "card.lock"
    lock.touch()
    assert subprocess.run(["bash", str(REPO / "scripts" / "flock_first.sh"), str(lock), "true"],
                          env={"FLOCK_FIRST_GRACE": "5", "PATH": "/usr/bin:/bin"}).returncode == 0
    assert subprocess.run(["flock", "-n", str(lock), "true"]).returncode == 0

def test_an_ab_pass_rerecords_only_the_cells_the_reference_measured():
    old = {"note": "seed 0.6.2", "date": "2026-08-01", "value": 1.0}
    tree = {"cards": {"p150a": {"machines": {"qb1": {"note": "seed 0.6.2", "date": "2026-08-01",
                                                     "models": {"boltz2": old, "nesso1": old}}}},
                      "tt-galaxy-wh-l": {"machines": {"g": {"note": "wh seed", "models": {"x": old}}}}}}
    ab = {"note": "A/B reference for /r/trees/x, same card and session", "date": "2026-10-10", "value": 2.0}
    ref = {"cards": {"p150a": {"machines": {"qb1": {**ab, "models": {"boltz2": ab}},
                                            "pc": {"note": "pc", "models": {"boltz2": old}}}}}}
    out = json.loads(gf.rerecorded(json.dumps(tree), json.dumps(ref), "re-recorded"))
    qb1 = out["cards"]["p150a"]["machines"]["qb1"]
    assert qb1["models"]["boltz2"]["value"] == 2.0 and qb1["models"]["boltz2"]["note"] == "re-recorded"
    assert qb1["models"]["nesso1"] == old and qb1["note"] == "re-recorded" and qb1["date"] == "2026-10-10"
    assert "pc" not in out["cards"]["p150a"]["machines"]
    assert out["cards"]["tt-galaxy-wh-l"] == tree["cards"]["tt-galaxy-wh-l"]


def test_perf_against_runs_the_timed_leg_against_the_last_release_tree():
    h = gf.Host("qb1", {"arch": "bh", "card_type": "p150a", "root": "/r", "perf_against": "v0.13.1",
                        "lock": "/l/card{card}.lock"}, "a" * 40)
    perf = next(lg for lg in _legs() if lg.family == "perf")
    assert h.leg(perf).argv[-2:] == ["--against", "/r/trees/v0.13.1"]
    assert "--against" not in h.leg(next(lg for lg in _legs() if lg.family == "parity")).argv


def test_a_dropped_ssh_handshake_is_retried_not_recorded(monkeypatch):
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        err = "kex_exchange_identification: Connection closed by remote host" if len(calls) < 3 else ""
        return subprocess.CompletedProcess(argv, 255 if err else 0, "ok", err)
    monkeypatch.setattr(gf.subprocess, "run", run)
    monkeypatch.setattr(gf.time, "sleep", lambda s: None)
    h = gf.Host("g108", {"ssh": "10.0.0.8", "arch": "wh", "card_type": "x", "root": "/r"}, "a" * 40)
    assert h.ssh("true", capture_output=True).returncode == 0 and len(calls) == 3


def test_an_ssh_timeout_is_retried_and_never_raises(monkeypatch):
    """qb1 off the LAN timed out a report fetch; the exception killed the worker thread and four
    BH parity legs were never recorded (rel-59c374cf7-b, 10-10)."""
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        if len(calls) < 3:
            raise subprocess.TimeoutExpired(argv, kw.get("timeout"))
        return subprocess.CompletedProcess(argv, 0, "{}", "")
    monkeypatch.setattr(gf.subprocess, "run", run)
    monkeypatch.setattr(gf.time, "sleep", lambda s: None)
    h = gf.Host("qb1", {"ssh": "qb1", "arch": "bh", "card_type": "p", "root": "/r"}, "a" * 40)
    assert h.ssh("cat r", capture_output=True, timeout=1).stdout == "{}" and len(calls) == 3
    calls.clear()

    def dark(argv, **kw):
        calls.append(argv)
        raise subprocess.TimeoutExpired(argv, 1)
    monkeypatch.setattr(gf.subprocess, "run", dark)
    assert h.ssh("cat r", capture_output=True, timeout=1).returncode == 255 and len(calls) == 6


def test_every_family_has_a_budget_and_an_expectation():
    for lg in _legs():
        assert lg.budget > 0 and lg.family in gf.EXPECT


def test_a_leg_known_to_need_longer_than_its_budget_gets_more_and_a_resume_after_a_timeout_does_too():
    hist = {"record:rf3@tt-galaxy-wh-l": 3000, "ladder:opendde": 15087}
    prior = [{"leg": "record:rf3@tt-galaxy-wh-l", "rc": gf.TIMED_OUT, "wall_s": 14442},
             {"leg": "record:boltz2@p150a", "rc": 1, "wall_s": 20000}]   # a failure, not a timeout
    known = gf.known_walls(hist, prior)
    assert known == {"record:rf3@tt-galaxy-wh-l": 14442, "ladder:opendde": 15087}
    leg = gf.Leg("record:rf3@tt-galaxy-wh-l", [], "record", known_s=known["record:rf3@tt-galaxy-wh-l"])
    assert leg.budget == round(gf.BUDGET_X * 14442) > gf.BUDGET["record"]
    assert gf.Leg("record:nesso1@p300c", [], "record", known_s=249).budget == gf.BUDGET["record"]


def test_command_pins_card_under_its_flock_and_substitutes_placeholders(tmp_path):
    h = gf.Host("qb1", {"arch": "bh", "card_type": "p150a", "root": "/r",
                        "lock": "/l/card{card}.lock", "env": {"ESM_ROOT": "/esm"},
                        "pythonpath_extra": ["/bc2"]}, "a" * 40)
    leg = next(lg for lg in _legs() if lg.family == "parity")
    cmd = h.command(leg, 2, "/t/out/x")
    assert "flock /l/card2.lock" in cmd and "TT_VISIBLE_DEVICES=2" in cmd
    assert "PYTHONPATH=/r/trees/aaaaaaaaaaaa:/bc2" in cmd and "ESM_ROOT=/esm" in cmd
    assert "localhost:2" in cmd and "/t/out/x/report.json" in cmd
    assert "RELEASE_GATE_SIZE_WORKDIR=/t/out/x/sizegate-work" in cmd
    assert "/r/venv-aaaaaaaaaaaa/bin/python scripts/full_parity_gate.py" in cmd
    free = shlex.split(h.command(next(lg for lg in _legs() if lg.name == "pytest_cpu"), None, "/t/out/y"))[-1]
    assert "flock" not in free and "TT_VISIBLE_DEVICES='' " in free
    assert "/r/venv312-aaaaaaaaaaaa/bin/python -m pytest" in free
    bc2 = h.command(next(lg for lg in _legs() if lg.family == "bc2"), 1, "/t/out/z")
    assert "PYTHONPATH=/bc2 " in bc2 and "/r/trees" not in bc2.split("&&")[-1].split("bash -c")[0]


def test_a_rerun_ends_the_attempt_it_replaces(tmp_path):
    """A dropped connection leaves the remote attempt waiting on the card; the rerun must end
    it, leg and all, or the leg runs twice into one directory."""
    tree = tmp_path / "trees" / ("a" * 12)
    (tree / "scripts").mkdir(parents=True)
    (tree / "scripts" / "flock_first.sh").write_text((REPO / "scripts" / "flock_first.sh").read_text())
    h = gf.Host("h", {"arch": "bh", "card_type": "p", "root": str(tmp_path), "first": True,
                      "lock": str(tmp_path / "card{card}.lock")}, "a" * 40)
    out = tmp_path / "out"
    cmd = h.command(gf.Leg("x", ["sleep", "57"], "parity"), 0, str(out))
    first = subprocess.Popen(["bash", "-c", cmd])
    pidf = out / "attempt.pid"
    deadline = time.time() + 20
    while not (pidf.exists() and subprocess.run(["pgrep", "-s", pidf.read_text().strip(), "-x", "sleep"],
                                                 capture_output=True).returncode == 0):
        assert time.time() < deadline
        time.sleep(0.1)
    sid1 = pidf.read_text().strip()
    second = subprocess.Popen(["bash", "-c", cmd])
    try:
        assert first.wait(timeout=20) != 0
        assert subprocess.run(["pgrep", "-s", sid1], capture_output=True).returncode == 1
        while pidf.read_text().strip() == sid1:
            assert time.time() < deadline + 20
            time.sleep(0.1)
        assert second.poll() is None
    finally:
        subprocess.run(["pkill", "-TERM", "-s", pidf.read_text().strip()])
        second.wait(timeout=20)


def test_a_seeding_host_adds_its_flags_to_the_key_and_reports_seeded_not_pass(tmp_path):
    seed = ["--update-baseline", "--note", "seed tt-galaxy-wh-l"]
    wh = gf.Host("g", {"arch": "wh", "card_type": "w", "root": "/r", "lock": "/l{card}",
                       "args": {"perf": seed}}, "a" * 40)
    bh = gf.Host("q", {"arch": "bh", "card_type": "p", "root": "/r", "lock": "/l{card}"}, "a" * 40)
    perf = next(lg for lg in _legs() if lg.name == "perf")
    assert wh.seeding(perf) and not bh.seeding(perf)
    assert "--update-baseline" in wh.command(perf, 5, "/o") and "--update-baseline" not in bh.command(perf, 3, "/o")
    assert gf.leg_key("c", "e", "w", wh.leg(perf)) != gf.leg_key("c", "e", "w", perf)
    ux = next(lg for lg in _legs() if lg.family == "ux")
    assert wh.leg(ux) is ux
    rows = [{"leg": "perf", "arch": "wh", "verdict": "SEEDED", "wall_s": 60, "worker": "g:5",
             "card_type": "w", "seeded": "/o/seeded/wh/perf_baselines.json"},
            {"leg": "perf", "arch": "bh", "verdict": "PASS", "wall_s": 60, "worker": "q:3"}]
    assert gf.write_verdict(tmp_path, "a" * 40, ["bh", "wh"], rows)
    md = (tmp_path / "VERDICT.md").read_text()
    assert "SEEDED 1 min g:5" in md and "/o/seeded/wh/perf_baselines.json" in md


def test_a_pool_leg_is_one_job_file_that_runs_on_the_chip_the_pool_picks(tmp_path, monkeypatch):
    """The pool, not the runner, chooses the chip: the job reads $CHIP for the card, its flock and
    the leg's own arguments, and the runner takes the exit code and log from what the job left."""
    monkeypatch.setattr(gf, "POOL_POLL_S", 0.05)
    queue = tmp_path / "queue"
    queue.mkdir()
    sha = "b" * 40
    (tmp_path / "trees" / sha[:12]).mkdir(parents=True)
    h = gf.Host("g", {"arch": "wh", "card_type": "w", "root": str(tmp_path), "ssh": "localhost",
                      "lock": str(tmp_path / "chip{card}.lock"), "python": sys.executable,
                      "pool": {"queue": str(queue), "prio": 5, "row": "spd-shipflow"}}, sha)
    leg = gf.Leg("ux:x", ["PY", "-c", "import os, sys; print('card', os.environ['TT_VISIBLE_DEVICES'], "
                          "sys.argv[1]); sys.exit(3)", "localhost:{CARD}"], "ux")

    def pool():  # what the pool runner does: start the job on chip 7
        while not (jobs := list(queue.glob("*.sh"))):
            time.sleep(0.02)
        assert jobs[0].name == "5-spd-shipflow-run1-ux_x.sh"
        subprocess.run(["bash", str(jobs[0])], env={"CHIP": "7", "PATH": "/usr/bin:/bin"}, check=True)
    t = threading.Thread(target=pool)
    t.start()
    log = tmp_path / "leg.log"
    with open(log, "w") as f:
        rc = gf.run_in_pool(h, gf.POOL, leg, str(tmp_path / "o" / "run1" / "wh" / "ux_x"), f)
    t.join()
    assert rc == 3
    text = log.read_text()
    assert "##CHIP 7" in text and "card 7 localhost:7" in text and "##LEG-START" in text
    assert (tmp_path / "chip7.lock").exists()
    assert gf.parse_workers("g:pool,g:pool", {"g": h}) == [(h, "pool"), (h, "pool")]


def test_a_pool_leg_whose_host_rebooted_under_it_is_reported_not_waited_for(tmp_path, monkeypatch):
    """g114 rebooted under five legs on 2026-10-11 and their runners waited for an exit code that
    could not come. The job notes the boot id it started under; a new one and no rc ends the wait."""
    monkeypatch.setattr(gf, "POOL_POLL_S", 0.05)
    boot = tmp_path / "boot_id"
    boot.write_text("first\n")
    monkeypatch.setattr(gf, "BOOT_ID", str(boot))
    queue = tmp_path / "queue"
    queue.mkdir()
    h = gf.Host("g", {"arch": "wh", "card_type": "w", "root": str(tmp_path), "ssh": "localhost",
                      "lock": str(tmp_path / "chip{card}.lock"), "python": sys.executable,
                      "pool": {"queue": str(queue), "prio": 0}}, "b" * 40)
    leg = gf.Leg("ux:x", ["PY", "-c", "pass"], "ux")
    rdir = tmp_path / "o" / "run1" / "wh" / "ux_x"

    def pool():  # starts the job's first line only (it dies with the host), then the host reboots
        while not (jobs := list(queue.glob("*.sh"))):
            time.sleep(0.02)
        first = jobs[0].read_text().splitlines()[2]
        subprocess.run(["bash", "-c", first], check=True)
        assert (rdir / "boot").read_text() == "first\n"
        boot.write_text("second\n")
    t = threading.Thread(target=pool)
    t.start()
    with open(tmp_path / "leg.log", "w") as f:
        rc = gf.run_in_pool(h, gf.POOL, leg, str(rdir), f)
    t.join()
    assert rc == 1 and gf.REBOOTED in (tmp_path / "leg.log").read_text()


def test_a_pool_leg_requeued_after_a_reboot_waits_for_its_new_run(tmp_path, monkeypatch):
    """The retry after a reboot queues into the same rdir. The last run's boot file must not read
    as this run having started under another boot: rel-5e250928a recorded nine WH legs FAIL that way
    within seconds (10-16 11:01Z) and the pool later ran their jobs for nobody."""
    monkeypatch.setattr(gf, "POOL_POLL_S", 0.05)
    boot = tmp_path / "boot_id"
    boot.write_text("second\n")
    monkeypatch.setattr(gf, "BOOT_ID", str(boot))
    queue = tmp_path / "queue"
    queue.mkdir()
    sha = "b" * 40
    (tmp_path / "trees" / sha[:12]).mkdir(parents=True)
    h = gf.Host("g", {"arch": "wh", "card_type": "w", "root": str(tmp_path), "ssh": "localhost",
                      "lock": str(tmp_path / "chip{card}.lock"), "python": sys.executable,
                      "pool": {"queue": str(queue), "prio": 0}}, sha)
    leg = gf.Leg("ux:x", ["PY", "-c", "import sys; sys.exit(4)"], "ux")
    rdir = tmp_path / "o" / "run1" / "wh" / "ux_x"
    rdir.mkdir(parents=True)
    (rdir / "boot").write_text("first\n")       # the run the reboot killed

    def pool():  # the pool is busy for a while, then starts the job
        while not (jobs := list(queue.glob("*.sh"))):
            time.sleep(0.02)
        time.sleep(0.3)
        subprocess.run(["bash", str(jobs[0])], env={"CHIP": "7", "PATH": "/usr/bin:/bin"}, check=True)
    t = threading.Thread(target=pool)
    t.start()
    with open(tmp_path / "leg.log", "w") as f:
        rc = gf.run_in_pool(h, gf.POOL, leg, str(rdir), f)
    t.join()
    assert rc == 4 and gf.REBOOTED not in (tmp_path / "leg.log").read_text()


def test_a_pool_leg_notes_a_give_back_that_stopped_it(tmp_path, monkeypatch):
    """glx_elastic SIGINTs a lent box's pool jobs and touches the pool's RETURNED (.107, 10-16 07:34Z,
    14 WH legs recorded FAIL). A job whose leg ended after that touch says so in its log."""
    monkeypatch.setattr(gf, "POOL_POLL_S", 0.05)
    pool_dir = tmp_path / "pool"
    queue = pool_dir / "queue"
    queue.mkdir(parents=True)
    sha = "b" * 40
    (tmp_path / "trees" / sha[:12]).mkdir(parents=True)
    h = gf.Host("g", {"arch": "wh", "card_type": "w", "root": str(tmp_path), "ssh": "localhost",
                      "lock": str(tmp_path / "chip{card}.lock"), "python": sys.executable,
                      "pool": {"queue": str(queue), "prio": 0}}, sha)
    # the leg itself stands in for the give-back: RETURNED is touched while it runs, then it dies on SIGINT
    leg = gf.Leg("ux:x", ["PY", "-c", f"import pathlib, sys, time; time.sleep(1.1); "
                          f"pathlib.Path({str(pool_dir / 'RETURNED')!r}).touch(); sys.exit(130)"], "ux")

    def pool():
        while not (jobs := list(queue.glob("*.sh"))):
            time.sleep(0.02)
        subprocess.run(["bash", str(jobs[0])], env={"CHIP": "7", "PATH": "/usr/bin:/bin"}, check=True)
    t = threading.Thread(target=pool)
    t.start()
    with open(tmp_path / "leg.log", "w") as f:
        rc = gf.run_in_pool(h, gf.POOL, leg, str(tmp_path / "o" / "run1" / "wh" / "ux_x"), f)
    t.join()
    assert rc == 130 and gf.GIVEN_BACK in (tmp_path / "leg.log").read_text()


def test_a_pool_leg_stopped_by_a_give_back_runs_again(tmp_path, monkeypatch):
    calls, backs = [], gf.GIVEN_BACK_TRIES

    def fake_run(host, card, leg, rdir, f):
        calls.append(card)
        if len(calls) <= backs:
            f.write(f"##LEG-START 5\ncensus fold exited -2\n##{gf.GIVEN_BACK}\n")
            return 1
        f.write("##LEG-START 5\nok\n")
        return 0
    monkeypatch.setattr(gf, "run_in_pool", fake_run)
    h = gf.Host("g", {"arch": "wh", "card_type": "w", "root": "/r"}, "c" * 40)
    ex = gf.make_executor("c" * 40, tmp_path, gf.Ledger(tmp_path / "l"), {("ux:x", "wh"): "k"}, "o")
    assert ex(h, gf.POOL, gf.Leg("ux:x", ["PY"], "ux"))["verdict"] == "PASS"
    assert len(calls) == gf.GIVEN_BACK_TRIES + 1
    calls.clear()
    monkeypatch.setattr(gf, "GIVEN_BACK_TRIES", 1)   # the next give-back after the last try is the leg's result
    assert ex(h, gf.POOL, gf.Leg("ux:x", ["PY"], "ux"))["verdict"] == "FAIL"


def test_a_leg_refused_for_host_load_starts_again_instead_of_failing(tmp_path, monkeypatch):
    monkeypatch.setattr(gf, "LOAD_RETRY_S", 0)
    calls = []

    def fake_run(host, card, leg, rdir, f):
        calls.append(card)
        if len(calls) < 3:
            f.write("PREFLIGHT - " + gf.LOAD_REFUSAL + " 1-min loadavg 131.89\n")
            return 1
        f.write("##LEG-START 5\nok\n")
        return 0
    monkeypatch.setattr(gf, "run_over_ssh", fake_run)
    h = gf.Host("g", {"arch": "wh", "card_type": "w", "root": "/r"}, "c" * 40)
    leg = gf.Leg("ux:x", ["PY"], "ux")
    ex = gf.make_executor("c" * 40, tmp_path, gf.Ledger(tmp_path / "l"), {("ux:x", "wh"): "k"}, "o")
    res = ex(h, 3, leg)
    assert len(calls) == 3 and res["verdict"] == "PASS"


def test_a_leg_that_met_another_process_on_its_card_runs_again(tmp_path, monkeypatch):
    monkeypatch.setattr(gf, "CONTENDED_RETRY_S", 0)
    calls = []

    def fake_run(host, card, leg, rdir, f):
        calls.append(card)
        if len(calls) == 1:
            f.write("##LEG-START 5\n[pxdesign] measurement error: device contention, nothing ran\n")
            return 1
        f.write("##LEG-START 5\nok\n")
        return 0
    monkeypatch.setattr(gf, "run_over_ssh", fake_run)
    h = gf.Host("q", {"arch": "bh", "card_type": "p", "root": "/r"}, "c" * 40)
    ex = gf.make_executor("c" * 40, tmp_path, gf.Ledger(tmp_path / "l"), {("perf", "bh"): "k"}, "o")
    res = ex(h, 3, gf.Leg("perf", ["PY"], "perf", timed=True))
    assert len(calls) == 2 and res["verdict"] == "PASS"


def test_a_leg_that_reports_its_card_unresettable_retires_the_card_without_a_probe(tmp_path, monkeypatch):
    """qb2:1, 10-13: ladder:opendde stopped on a refused reset, the probe passed, and the next leg
    on that card died at its first open."""
    def fake_run(host, card, leg, rdir, f):
        f.write("##LEG-START 5\nrelease gate stopping: a fold timed out and its card could not be "
                "reset (refused).\n")
        return 1
    monkeypatch.setattr(gf, "run_over_ssh", fake_run)
    monkeypatch.setattr(gf.Host, "card_dead", lambda self, card: pytest.fail("probed"))
    h = gf.Host("q", {"arch": "bh", "card_type": "p", "root": "/r"}, "c" * 40)
    ex = gf.make_executor("c" * 40, tmp_path, gf.Ledger(tmp_path / "l"), {("ladder:m", "bh"): "k"}, "o")
    with pytest.raises(gf.CardDead) as e:
        ex(h, 1, gf.Leg("ladder:m", ["PY"], "ladder"))
    assert "could not be reset" in e.value.why


def test_a_second_run_of_a_leg_keeps_the_first_runs_log(tmp_path, monkeypatch):
    def fake_run(host, card, leg, rdir, f):
        f.write("##LEG-START 5\nrun\n")
        return 1
    monkeypatch.setattr(gf, "run_over_ssh", fake_run)
    h = gf.Host("q", {"arch": "bh", "card_type": "p", "root": "/r"}, "c" * 40)
    ex = gf.make_executor("c" * 40, tmp_path, gf.Ledger(tmp_path / "l"), {("perf", "bh"): "k"}, "o")
    first = ex(h, 1, gf.Leg("perf", ["PY"], "perf", timed=True))["log"]
    second = ex(h, 1, gf.Leg("perf", ["PY"], "perf", timed=True))["log"]
    assert first != second and Path(first).exists() and Path(second).name == "perf.1.log"


def test_a_lever_an_imported_module_registers_is_owed_by_a_baseline_without_its_row(tmp_path):
    """SDPA_FUSED_PADDED reached main with no size-ladder row; the WH gate learned it after 40 min
    per ladder model. The baseline already says which modules each model's fold imports."""
    levers = [("A", "tt_bio.tenstorrent", "", "", ""), ("B", "tt_bio.tenstorrent", "", "", ""),
              ("C", "tt_bio.esmc", "", "", "")]
    base = {"cards": {"w": {"models": {
        "m": {"levers": {"256": {"A": {"resolved": "True"}, "C": {"resolved": "not-imported"}}}},
        "n": {"levers": {"256": {"A": {"resolved": "not-imported"}, "C": {"resolved": "not-imported"}}}},
        "x": {"levers": {"256": {"A": {"resolved": "True"}}}}}}}}
    (tmp_path / "size_ladder_baseline.d").mkdir()
    (tmp_path / "size_ladder_baseline.json").write_text(json.dumps(base))
    assert gf.owed_levers(tmp_path, levers, ["m", "n"]) == {"w": {"m": ["B"]}}
    rows = base["cards"]["w"]["models"]["m"]["levers"]["256"]
    rows["B"] = {"resolved": "True"}
    (tmp_path / "size_ladder_baseline.d" / "m.json").write_text(json.dumps(
        {"cards": {"w": {"models": {"m": {"levers": {"256": rows}}}}}}))
    assert gf.owed_levers(tmp_path, levers, ["m", "n"]) == {}        # the fragment overrides


def test_an_owed_ladder_leg_fails_the_verdict_and_names_the_record_command(tmp_path):
    results = [{"leg": "ladder:m", "arch": "wh", "verdict": "OWED", "owed": ["B"], "card_type": "w"},
               {"leg": "perf", "arch": "wh", "verdict": "PASS", "wall_s": 60, "worker": "h:1"}]
    assert not gf.write_verdict(tmp_path, "a" * 40, ["wh"], results)
    text = (tmp_path / "VERDICT.md").read_text()
    assert "| ladder:m | OWED B |" in text and "--record-lever B" in text


def test_a_ladder_leg_runs_only_on_a_host_timing_it_at_its_baselines_cap(tmp_path):
    """rel-19d30971f: four p150a ladders were folded for 30-80 min each on qb1 at --host-threads 8
    against baselines timed uncapped, and each failed "re-record at this cap" at the end. The cap
    is in the baseline, so the plan knows it before folding, and a host at another cap (qb2 at 5,
    keyed as p150a but timed against its own p300c rows) must not take the leg."""
    base = {"cards": {"p": {"host_threads": None, "models": {"m": {}, "n": {}}},
                      "c": {"host_threads": 5, "models": {"m": {}}}}}
    (tmp_path / "size_ladder_baseline.d").mkdir()
    (tmp_path / "size_ladder_baseline.json").write_text(json.dumps(base))
    (tmp_path / "size_ladder_baseline.d" / "n.json").write_text(json.dumps(
        {"cards": {"p": {"host_threads": 8, "models": {"n": {}}}}}))
    roster = {"caps": gf.ladder_caps(tmp_path, ["m", "n"])}
    assert roster["caps"] == {"p": {"m": None, "n": 8}, "c": {"m": 5}}   # the fragment wins

    def host(card_type, cap):
        cfg = {"arch": "bh", "card_type": card_type, "root": "/r"}
        if cap:
            cfg["args"] = {"ladder": ["--host-threads", str(cap)]}
        return gf.Host(card_type + str(cap), cfg, "f" * 40)
    m, n, rec = (gf.Leg("ladder:m", [], "ladder"), gf.Leg("ladder:n", [], "ladder"),
                 gf.Leg("record:m", [], "record"))
    assert host("p", 8).ladder_cap == 8 and host("p", None).ladder_cap is None
    assert gf.cap_fits(roster, host("p", 8), n) and not gf.cap_fits(roster, host("p", 8), m)
    assert gf.cap_fits(roster, host("p", None), m) and gf.cap_fits(roster, host("c", 5), m)
    assert gf.cap_fits(roster, host("p", 8), rec)                     # a record sets the cap
    assert gf.cap_fits(roster, host("c", 5), n)                       # no row: the leg says so
    results = [{"leg": "ladder:m", "arch": "bh", "verdict": "OWED", "owed": [gf.CAP_OWED]}]
    assert not gf.write_verdict(tmp_path, "a" * 40, ["bh"], results)
    text = (tmp_path / "VERDICT.md").read_text()
    assert "--record-full m" in text and "--record-lever" not in text


def test_a_gate_never_hands_a_ladder_leg_to_a_host_at_another_cap(tmp_path):
    roster = {"caps": {"bh": {"m": 8}}}
    off, on = _host("qb2", "bh", tmp_path), _host("qb1", "bh", tmp_path)
    on.cfg["args"] = {"ladder": ["--host-threads", "8"]}
    ran = []

    def execute(host, card, leg):
        ran.append(host.name)
        return {"leg": leg.name, "arch": host.arch, "verdict": "PASS"}
    gf.Gate({"bh": [gf.Leg("ladder:m", [], "ladder")]}, [(off, 0), (on, 0)], set(), execute,
            tmp_path, roster=roster).run()
    assert ran == ["qb1"]


def test_only_a_ladder_leg_owes_levers_and_a_bare_leg_name_does_not_crash_the_plan():
    roster = {"owed": {"w": {"m": ["B"]}}}
    assert gf.owed_by(roster, "w", gf.Leg("ladder:m", ["PY"], "ladder")) == ["B"]
    assert gf.owed_by(roster, "w", gf.Leg("ladder:n", ["PY"], "ladder")) == []
    assert gf.owed_by(roster, "w", gf.Leg("perf", ["PY"], "perf")) == []
    assert gf.owed_by(roster, "w", gf.Leg("check", ["PY"], "check")) == []
    assert gf.owed_by({}, "w", gf.Leg("ladder:m", ["PY"], "ladder")) == []


def test_splice_takes_only_the_fragments_card_type(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "splice", Path(__file__).resolve().parent.parent / "scripts" / "splice_ladder_fragments.py")
    sp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sp)
    base = tmp_path / "base"
    base.mkdir()
    (base / "m.json").write_text(json.dumps({"rungs": [1, 2, 3], "cards": {"wh": {"v": 1}, "bh": {"v": 2}}}))
    frag = tmp_path / "rec" / "wh"
    frag.mkdir(parents=True)
    (frag / "m.json").write_text(json.dumps({"rungs": [1], "cards": {"wh": {"v": 9}, "bh": {"v": 0}}}))
    assert sp.splice([tmp_path / "rec"], base) == [("m", "wh")]
    got = json.loads((base / "m.json").read_text())
    assert got == {"rungs": [1, 2, 3], "cards": {"wh": {"v": 9}, "bh": {"v": 2}}}
    assert sp.splice([tmp_path / "rec"], base) == []          # idempotent


def _key(name):
    return gf.sha256(name)


def _runner(host, legs, ledger, me, tmp_path, ran, mu, verdict="PASS"):
    """A Gate over `legs` on one bh card whose ledger claims as runner `me`."""
    ledger.ME = me
    keys = {(lg.name, "bh"): _key(lg.name) for lg in legs}

    def execute(h, card, leg):
        with mu:
            ran.append((me["pid"], leg.name))
        time.sleep(0.02)
        res = {"leg": leg.name, "arch": "bh", "verdict": verdict, "worker": f"{h.name}:{card}",
               "ended": "t", "sha": "f" * 40, "log": "l", "key": keys[(leg.name, "bh")]}
        if verdict in gf.OK:
            ledger.put(res["key"], res)
        return res

    out = tmp_path / str(me["pid"])
    out.mkdir()
    return gf.Gate({"bh": list(legs)}, [(host, 0)], set(), execute, out, ledger, keys)


def test_a_second_runner_joins_a_gate_and_no_leg_runs_twice(tmp_path, monkeypatch):
    monkeypatch.setattr(gf, "POOL_POLL_S", 0.01)
    legs = [lg for lg in _legs() if lg.card and not lg.timed and lg.family != "record"][:12]
    host = _host("qb1", "bh", tmp_path)
    ran, mu = [], threading.Lock()
    a = _runner(host, legs, gf.Ledger(tmp_path / "ledger"), {"host": "pc", "pid": 1}, tmp_path, ran, mu)
    b = _runner(host, legs[::-1], gf.Ledger(tmp_path / "ledger"), {"host": "pc", "pid": 2}, tmp_path, ran, mu)
    ts = [threading.Thread(target=g.run) for g in (a, b)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=30)
    assert sorted(n for _, n in ran) == sorted(lg.name for lg in legs)
    assert {p for p, _ in ran} == {1, 2}
    for g in (a, b):
        assert sorted(r["leg"] for r in g.results) == sorted(lg.name for lg in legs)
        assert all(r["verdict"] in ("PASS", "REUSED") for r in g.results)
    assert not list((tmp_path / "ledger").glob("*.claim"))


def test_a_leg_its_holder_left_without_a_pass_runs_again_and_a_dead_holder_is_dropped(tmp_path, monkeypatch):
    monkeypatch.setattr(gf, "POOL_POLL_S", 0.01)
    legs = [lg for lg in _legs() if lg.card and not lg.timed and lg.family != "record"][:2]
    host = _host("qb1", "bh", tmp_path)
    ledger = gf.Ledger(tmp_path / "ledger")
    me = {"host": gf.socket.gethostname(), "pid": os.getpid()}
    held, dead = (_key(lg.name) for lg in legs)
    (ledger.path / f"{held}.claim").write_text(json.dumps({"host": "elsewhere", "pid": 7}))
    (ledger.path / f"{dead}.claim").write_text(json.dumps({"host": me["host"], "pid": 2 ** 22 + 12345}))
    ran, mu = [], threading.Lock()
    g = _runner(host, legs, ledger, me, tmp_path, ran, mu)
    t = threading.Thread(target=g.run)
    t.start()
    time.sleep(0.3)
    assert [n for _, n in ran] == [legs[1].name]      # the dead holder's leg ran, the live one waits
    (ledger.path / f"{held}.claim").unlink()           # the holder stopped without a row
    t.join(timeout=10)
    assert sorted(n for _, n in ran) == sorted(lg.name for lg in legs)
    assert all(r["verdict"] == "PASS" for r in g.results)


def test_a_host_keyed_as_another_card_type_keeps_its_own_for_the_record():
    h = gf.Host("qb2", {"arch": "bh", "card_type": "p300c", "key_card_type": "p150a", "root": "/r"}, "f" * 40)
    assert (h.cfg["card_type"], h.key_card_type) == ("p300c", "p150a")
    assert gf.Host("qb1", {"arch": "bh", "card_type": "p150a", "root": "/r"}, "f" * 40).key_card_type == "p150a"


def _pool_host(name, tmp_path, lend_at=None, paused=False):
    h = gf.Host(name, {"arch": "wh", "card_type": "wh", "root": str(tmp_path),
                       "pool": {"queue": str(tmp_path / name / "queue")}}, "f" * 40)
    h.lend = (time.time(), lend_at, paused)      # fresh, so no ssh
    return h


def test_a_pool_leg_skips_the_load_ceiling_and_a_flock_leg_keeps_it(tmp_path):
    """A busy pool holds its Galaxy above the QuietBox load ceiling by itself; its legs must not
    refuse on it. The flag stays out of the leg's key, so pool evidence keeps matching."""
    pool = _pool_host("g114", tmp_path)
    pool.cfg["lock"] = "/l/chip{card}.lock"
    flock = gf.Host("qb1", {"arch": "bh", "card_type": "p", "root": "/r", "lock": "/l{card}"}, "a" * 40)
    for fam in ("parity", "rg", "ladder"):
        leg = next(lg for lg in _legs() if lg.family == fam)
        assert "--load-ceiling 0" in pool.command(leg, gf.POOL_CARD, "/o")
        assert "--load-ceiling" not in flock.command(leg, 3, "/o")
        assert pool.leg(leg) is leg
    perf = next(lg for lg in _legs() if lg.name == "perf")
    assert "--load-ceiling" not in pool.command(perf, gf.POOL_CARD, "/o")


def test_a_lent_pool_takes_no_leg_the_lend_would_hold_and_another_host_runs_it(tmp_path):
    lent = _pool_host("g114", tmp_path, lend_at=int(time.time()) + 3600)
    free = _pool_host("g107", tmp_path)
    short, long_ = gf.Leg("check", [], "check", card=True), gf.Leg("capacity:opendde", [], "capacity")
    assert short.budget < 3600 < long_.budget

    def execute(h, card, leg):
        return {"leg": leg.name, "arch": h.arch, "verdict": "PASS", "worker": h.name}

    res = gf.Gate({"wh": [long_, short]}, [(lent, gf.POOL), (free, gf.POOL)], set(), execute, tmp_path).run()
    assert {r["leg"]: r["worker"] for r in res}["capacity:opendde"] == "g107"
    assert len(res) == 2
    assert _pool_host("p", tmp_path, paused=True).lent(1) and not free.lent(10 ** 6)


def test_card_free_legs_run_on_a_free_host_when_the_first_one_is_lent(tmp_path):
    lent = _pool_host("g107", tmp_path, paused=True)
    free = _pool_host("g114", tmp_path)
    workers = [(lent, gf.POOL), (free, gf.POOL)]
    assert gf.cardfree(workers + [(free, gf.POOL)], "wh") == [(lent, None), (free, None)]

    def execute(h, card, leg):
        return {"leg": leg.name, "arch": "any", "verdict": "PASS", "worker": h.name}

    legs = [gf.Leg(n, [], n) for n in ("check", "packaging_smoke", "pytest_cpu")]
    res = gf.Gate({"any": legs}, workers + gf.cardfree(workers, "wh"), set(), execute, tmp_path).run()
    assert sorted(r["leg"] for r in res) == ["check", "packaging_smoke", "pytest_cpu"]
    assert {r["worker"] for r in res} == {"g114"}


def test_a_leg_queued_before_the_lend_was_booked_is_taken_back(tmp_path, monkeypatch):
    monkeypatch.setattr(gf, "POOL_POLL_S", 0.01)
    monkeypatch.setattr(gf, "LEND_READ_S", 0)
    lend = tmp_path / "LEND-AT"
    monkeypatch.setattr(gf, "LEND_AT", str(lend))
    queue = tmp_path / "queue"
    queue.mkdir()
    h = gf.Host("g", {"arch": "wh", "card_type": "w", "root": str(tmp_path), "ssh": "localhost",
                      "lock": str(tmp_path / "chip{card}.lock"), "python": sys.executable,
                      "pool": {"queue": str(queue), "prio": 0}}, "b" * 40)
    leg = gf.Leg("capacity:opendde", ["PY", "-c", "pass"], "capacity")
    lend.write_text(str(int(time.time()) + 600))
    with open(tmp_path / "leg.log", "w") as f:
        rc = gf.run_in_pool(h, gf.POOL, leg, str(tmp_path / "o" / "run1" / "wh" / "capacity_opendde"), f)
    assert rc == gf.WITHDRAWN
    assert [p.name for p in queue.iterdir()] == ["0-gate-run1-capacity_opendde.sh.lent"]

    lend.unlink()                # the lend ended
    tries = []

    def execute(host, card, lg):
        tries.append(host.name)
        if len(tries) == 1:
            return None          # what execute returns for a withdrawn leg
        return {"leg": lg.name, "arch": host.arch, "verdict": "PASS", "worker": host.name}

    res = gf.Gate({"wh": [leg]}, [(_pool_host("a", tmp_path), gf.POOL)], set(), execute, tmp_path).run()
    assert len(tries) == 2 and len(res) == 1


def test_end_attempt_stops_the_leg_it_started_and_nothing_else(tmp_path):
    """A stopped runner ends each remote leg it started (10-11: a stopped gate left 86 processes
    on 23 cards). It ends only its own attempt: a pid file whose process is not that leg's is left."""
    out = str(tmp_path / "leg")
    os.makedirs(out)
    mark = f"GATE_ATTEMPT={out}"
    attempt = f"echo $$ > {shlex.quote(out + '/attempt.pid')}; exec sleep 60"
    leg = subprocess.Popen(["bash", "-c", f"env {shlex.quote(mark)} setsid -w bash -c {shlex.quote(attempt)}"])
    for _ in range(50):
        if Path(out, "attempt.pid").exists():
            break
        time.sleep(0.1)
    stranger = subprocess.Popen(["sleep", "60"])
    other = str(tmp_path / "other")
    os.makedirs(other)
    Path(other, "attempt.pid").write_text(f"{stranger.pid}\n")
    subprocess.run(["bash", "-c", gf.Host.end_attempt(out) + "; " + gf.Host.end_attempt(other)], check=True)
    assert leg.wait(10) != 0
    assert stranger.poll() is None
    stranger.kill()


def test_one_runner_per_out(tmp_path):
    held = gf.hold_out(tmp_path / "g")
    with pytest.raises(SystemExit, match="another gate_fanout.py"):
        gf.hold_out(tmp_path / "g")
    held.close()
    gf.hold_out(tmp_path / "g").close()


def test_a_pass_is_reused_on_a_commit_that_changed_nothing_the_leg_read(tmp_path):
    """gate_reads.py records what a leg's Python read; a later commit reuses the PASS when none of
    it changed, and any non-Python file change keys every leg anew."""
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*a):
        return subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True,
                              text=True).stdout.strip()

    def commit(files, msg):
        for name, body in files.items():
            (repo / name).parent.mkdir(parents=True, exist_ok=True)
            (repo / name).write_text(body)
        git("add", "."), git("commit", "-qm", msg)
        return git("rev-parse", "HEAD")
    git("init", "-q"), git("config", "user.email", "t@t"), git("config", "user.name", "t")
    a = commit({"m/a.py": "1", "m/b.py": "1", "k/x.cpp": "1", "pyproject.toml": 'version = "1"\n'}, "a")
    b = commit({"m/b.py": "2", "pyproject.toml": 'version = "2"\n'}, "b: a file the leg never read")
    c = commit({"m/new.py": "1"}, "c: a module added where the leg listed")
    d = commit({"m/a.py": "2"}, "d: a file the leg read")
    assert gf.reads_match(a, b, ["m/a.py", "gone.json"], repo)
    assert not gf.reads_match(a, c, ["m/a.py", "dir:m"], repo)
    assert gf.reads_match(a, c, ["m/a.py"], repo)                  # import-system listings are not kept
    assert not gf.reads_match(a, d, ["m/a.py"], repo)
    np = [gf.content_hash(s, repo, python=False) for s in (a, b, d)]
    assert np[0] == np[1] == np[2]                                 # .py and the version line only
    e = commit({"k/x.cpp": "2"}, "e")
    assert gf.content_hash(e, repo, python=False) != np[0]
    f = commit({"k/fix.json": "1"}, "f: a fixture only Python reads")
    assert gf.content_hash(f, repo, python=False) != gf.content_hash(e, repo, python=False)
    assert gf.content_hash(f, repo, python=False, data=False) == gf.content_hash(e, repo, python=False, data=False)

    led = gf.Ledger(tmp_path / "led")
    row = {"leg": "parity:m", "verdict": "PASS", "sha": a, "ended": "1", "worker": "h:0", "log": "l",
           "scope": "s", "reads": ["m/a.py"]}
    led.put("k1", row)
    led.put("k2", {**row, "verdict": "FAIL", "reads": []})
    assert led.read_hit("s", b, repo)["sha"] == a
    assert gf.Ledger(tmp_path / "led").read_hit("s", d, repo) is None
    assert led.read_hit("other", b, repo) is None


def test_gate_reads_records_reads_of_every_python_process_but_not_writes(tmp_path):
    tree, out = tmp_path / "tree", tmp_path / "reads"
    (tree / "pkg").mkdir(parents=True)
    (tree / "pkg" / "__init__.py").write_text("")
    (tree / "pkg" / "m.py").write_text("X = 1\n")
    (tree / "data.json").write_text("{}")
    (tree / "main.py").write_text(
        "import gate_reads, os, subprocess, sys\n"
        "import pkg.m\n"
        "open('data.json').read(); open('out.txt', 'w').write('x'); os.listdir('pkg')\n"
        "subprocess.run([sys.executable, '-c', 'import gate_reads; open(\"pkg/m.py\").read()'], check=True)\n"
        "if os.fork() == 0:\n"
        "    open('pkg/__init__.py').read(); os._exit(0)\n"
        "os.wait()\n")
    env = {**os.environ, "GATE_READS": str(out), "GATE_TREE": str(tree),
           "PYTHONPATH": f"{REPO / 'scripts'}:{tree}", "PYTHONDONTWRITEBYTECODE": "1"}
    subprocess.run([sys.executable, "main.py"], cwd=tree, env=env, check=True, timeout=60)
    files = list(out.glob("*.txt"))
    seen = {ln for f in files for ln in f.read_text().split("\n") if ln}
    assert len(files) == 3                                          # parent, subprocess, fork
    # main.py itself is seen only where a venv's .pth installed the hook before the interpreter read
    # the script (a gate venv); here the script's own import installs it after
    assert seen - {"main.py"} == {"pkg/__init__.py", "pkg/m.py", "data.json", "dir:pkg"}


def test_perf_against_last_is_the_newest_release_tag(tmp_path):
    def git(*a):
        return subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True, text=True)
    git("init", "-q"), git("config", "user.email", "t@t"), git("config", "user.name", "t")
    git("commit", "-q", "--allow-empty", "-m", "a")
    for tag in ("v0.9.0", "v0.15.1", "v0.15.0", "nightly"):
        git("tag", tag)
    assert gf.last_tag(tmp_path) == "v0.15.1"


def test_a_dead_card_is_retired_and_its_leg_runs_on_another_card(tmp_path):
    host = _host("qb2", "bh", tmp_path)
    legs = [gf.Leg(f"ux:m{i}", [], "ux") for i in range(6)]
    ran = []

    def execute(h, card, leg):
        time.sleep(0.01)
        res = {"leg": leg.name, "arch": h.arch, "verdict": "PASS", "worker": f"qb2:{card}"}
        if card == 1:
            raise gf.CardDead({**res, "verdict": "FAIL"}, "TT_THROW: failed to initialize FW")
        ran.append((card, leg.name))
        return res

    res = gf.Gate({"bh": list(legs)}, [(host, 0), (host, 1)], set(), execute, tmp_path).run()
    assert sorted(r["leg"] for r in res) == sorted(lg.name for lg in legs)
    assert {r["verdict"] for r in res} == {"PASS"}
    assert {c for c, _ in ran} == {0}


def test_a_leg_that_kills_two_cards_is_recorded_as_it_ran(tmp_path):
    host = _host("qb2", "bh", tmp_path)
    killed = []

    def execute(h, card, leg):
        killed.append(card)
        raise gf.CardDead({"leg": leg.name, "arch": h.arch, "verdict": "FAIL",
                           "worker": f"qb2:{card}"}, "ARC core (8, 0) failed to start.")

    res = gf.Gate({"bh": [gf.Leg("ux:killer", [], "ux")]}, [(host, 0), (host, 1), (host, 2)], set(),
                  execute, tmp_path).run()
    assert [r["verdict"] for r in res] == ["FAIL"] and len(killed) == 2


def test_a_leg_no_card_is_left_to_run_is_unrun_and_fails_the_gate(tmp_path):
    host = _host("qb2", "bh", tmp_path)

    def execute(h, card, leg):
        raise gf.CardDead({"leg": leg.name, "arch": h.arch, "verdict": "FAIL",
                           "worker": f"qb2:{card}"}, "TT_THROW: failed to initialize FW")

    legs = [gf.Leg("ux:a", [], "ux"), gf.Leg("ux:b", [], "ux")]
    res = gf.Gate({"bh": legs}, [(host, 0)], set(), execute, tmp_path).run()
    assert {r["leg"]: r["verdict"] for r in res} == {"ux:a": "UNRUN", "ux:b": "UNRUN"}
    assert not gf.write_verdict(tmp_path, "f" * 40, ["bh"], res)


def test_the_card_probe_reads_dead_only_when_it_ran_and_did_not_dispatch(tmp_path, monkeypatch):
    monkeypatch.setattr(gf, "CARD_PROBE_GAP_S", 0)
    host = gf.Host("qb2", {"arch": "bh", "card_type": "p300c", "root": str(tmp_path),
                           "lock": "/tmp/chip{card}.lock"}, "f" * 40)
    seen = []

    def fake(stdout):
        def ssh(cmd, **kw):
            seen.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout, "")
        return ssh

    host.ssh = fake("CARD_PROBE_START\nCARD_OPEN\nCARD_HEALTHY\n")
    assert host.card_dead(1) is None
    assert "TT_VISIBLE_DEVICES=1" in seen[-1] and "flock -w" in seen[-1] and "/tmp/chip1.lock" in seen[-1]
    host.ssh = fake("")                                       # never got the flock
    assert host.card_dead(1) is None
    host.ssh = fake("CARD_PROBE_START\nTT_THROW: Device 0 init: failed to initialize FW!\n")
    assert "failed to initialize FW" in host.card_dead(1)
    assert len(seen) == 1 + 1 + gf.CARD_PROBE_TRIES
    outs = iter(["CARD_PROBE_START\nRuntimeError: lease held\n", "CARD_PROBE_START\nCARD_OPEN\nCARD_HEALTHY\n"])
    host.ssh = lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, next(outs), "")
    assert host.card_dead(1) is None              # a failed leg's workers let go: not dead


SHARED = [256, 512, 640, 768, 896, 1024]
RUNGS = {"p150a": {"boltz2": SHARED + [1152, 1280, 1408, 1536], "rf3": SHARED + [1088, 1152, 1280, 1408, 1536]},
         "tt-galaxy-wh-l": {"boltz2": SHARED + [1280, 1536], "rf3": SHARED + [1088, 1280, 1536]},
         "*": {"boltz2": SHARED, "rf3": SHARED + [1088]}}


def test_a_ladder_runs_as_rung_groups_that_keep_the_exponent_rungs_together():
    legs = {lg.name: lg for lg in gf.build_legs({**ROSTER, "rungs": RUNGS}, [], 1) if lg.family == "ladder"}
    assert sorted(legs) == ["ladder:boltz2@1152-1280", "ladder:boltz2@1408-1536", "ladder:boltz2@256-768",
                            "ladder:boltz2@896-1024", "ladder:rf3@1152-1280", "ladder:rf3@1408-1536",
                            "ladder:rf3@256-768", "ladder:rf3@896-1088"]
    a = legs["ladder:boltz2@256-768"]
    assert a.rungs == (256, 512, 640, 768)
    assert a.argv[-3:] == ["--size-ladder-part", "--size-ladder-rungs", "256,512,640,768"]
    assert gf.ladder_model(a.name) == "boltz2" and gf.out_group(a.name) == a.name
    # a tree whose roster carries no rungs keeps one leg per model
    assert [lg.name for lg in gf.build_legs(ROSTER, [], 1) if lg.family == "ladder"] == ["ladder:boltz2", "ladder:rf3"]


def test_a_rung_group_goes_only_to_a_card_whose_ladder_walks_it(tmp_path):
    roster = {"rungs": RUNGS}
    legs = {lg.name: lg for lg in gf.build_legs({**ROSTER, **roster}, [], 1) if lg.family == "ladder"}

    def host(ct):
        return gf.Host(ct, {"arch": "bh", "card_type": ct, "root": "/r"}, "f" * 40)

    p300c, p150a, wh = host("p300c"), host("p150a"), host("tt-galaxy-wh-l")
    assert gf.rungs_fit(roster, p300c, legs["ladder:boltz2@896-1024"])
    assert not gf.rungs_fit(roster, p300c, legs["ladder:boltz2@1408-1536"])
    assert gf.rungs_fit(roster, p150a, legs["ladder:boltz2@1408-1536"])
    assert gf.rungs_fit(roster, wh, legs["ladder:boltz2@1408-1536"])       # 1536
    assert gf.rungs_fit(roster, wh, legs["ladder:boltz2@1152-1280"])        # 1280
    assert not gf.rungs_fit(roster, p300c, legs["ladder:rf3@1152-1280"])
    assert gf.rungs_fit(roster, p300c, gf.Leg("rg:x", [], "rg"))



def test_record_abb3_leg_only_with_its_flag_and_only_on_p300c():
    assert gf.ABB3_LEG not in [lg.name for lg in gf.build_legs(ROSTER, [], 1)]
    leg = {lg.name: lg for lg in gf.build_legs(ROSTER, [], 1, record_abb3=True)}[gf.ABB3_LEG]
    assert leg.family == "record" and leg.workdir and leg.card_types == ("p300c",)

    class H:
        def __init__(self, ct):
            self.cfg = {"card_type": ct}
    assert gf.type_fits(H("p300c"), leg) and not gf.type_fits(H("p150a"), leg)
    assert gf.type_fits(H("p150a"), gf.Leg("x", [], "rg"))


MODEL_REC = """# The model gate.
RECORDED-AT: {sha} scripts/abb3_port/model_gate.py
#
=== raw ===
{rows}"""
STEP_REC = """# The step gate.
RECORDED-AT: {sha} scripts/abb3_port/step_gate.py
#
== --micro 8 ==
TT_FATAL: Out of Memory: Not enough space to allocate 100663296 B DRAM buffer
== default: --micro 4 ==
{step}"""
ARM = "  ok   frames  1.0e-02  0.0e+00\n  {v} angles  {a}  0.0e+00\n  grad on   atom14 positions: max 0.2 A\n"
STEP = ("  taped device parameters: 500\n  step   1  19.7 s  loss 4.9937\n  step  10  19.0 s  loss {l10}\n"
        "STEP: median 19.379 s over 10 COMPLETE steps\n"
        "CLOCK: AICLK during the run: card 0 min 1343 median 1350 max 1350 MHz over 80\n"
        "DRAM: peak 30981 MB of the card's 32640 MB\n")


def _abb3_tree(tmp_path, arms, step, step_rc=0, oom=True):
    """A tree whose abb3 gate scripts print canned output, and its committed records."""
    (tmp_path / "scripts/abb3_port").mkdir(parents=True)
    (tmp_path / "perf/abb3_port").mkdir(parents=True)
    (tmp_path / "out").mkdir()
    (tmp_path / "arms.json").write_text(json.dumps(arms))
    (tmp_path / "scripts/abb3_port/model_gate.py").write_text(
        "import json, sys\nfrom pathlib import Path\n"
        "a = json.loads((Path(__file__).parents[2] / 'arms.json').read_text())[sys.argv[4] + '/' + sys.argv[2]]\n"
        "print(a); sys.exit('FAIL' in a)\n")
    (tmp_path / "scripts/abb3_port/step_gate.py").write_text(
        "import sys\n"
        f"if '--micro' in sys.argv:\n    print({'TT_FATAL: Out of Memory: Not enough space to allocate 67108864 B' if oom else 'ok'!r}); sys.exit({int(oom)})\n"
        f"print({step!r}); sys.exit({step_rc})\n")
    return tmp_path


def _abb3_run(tree, sha="b" * 40):
    return subprocess.run([sys.executable, "-c", gf.ABB3_RECORD, str(tree / "out"), sha, "qb2:3", "p300c",
                           str(gf.ABB3_LOSS_TOL), *gf.ABB3_FILES], cwd=tree, capture_output=True, text=True)


def _abb3_commit(tree, arms, step):
    order = [arms[f"{s}/{t}"] for s in ("0.05", "0.3") for t in ("64", "256")]
    (tree / gf.ABB3_FILES[0]).write_text(MODEL_REC.format(sha="a" * 40, rows="".join(order)))
    (tree / gf.ABB3_FILES[1]).write_text(STEP_REC.format(sha="a" * 40, step=step))


ARMS = {f"{s}/{t}": ARM.format(v="ok  ", a="1.5e-02") for s in ("0.05", "0.3") for t in ("64", "256")}
ARMS["0.3/256"] = ARM.format(v="FAIL", a="5.11e-02")


def test_record_abb3_rerecords_when_every_row_matches(tmp_path):
    tree = _abb3_tree(tmp_path, ARMS, STEP.format(l10="4.9911"))
    _abb3_commit(tree, ARMS, STEP.format(l10="4.9910"))   # a last-digit loss move is within tolerance
    r = _abb3_run(tree)
    assert r.returncode == 0, r.stderr
    model = (tree / "out/model_gate_qb1c3.txt").read_text()
    step = (tree / "out/step_gate_default_qb1c3.txt").read_text()
    assert f"RECORDED-AT: {'b' * 40} scripts/abb3_port/model_gate.py" in model and "a" * 40 not in model
    assert "identical to the aaaaaaaaa record, which still FAILs angles at 5.11e-02 (sigma 0.3, 256 tokens)" \
        in " ".join(ln.lstrip("# ") for ln in model.splitlines())
    assert model.index("RE-RECORDED") < model.index("=== raw ===")
    flat = " ".join(ln.lstrip("# ") for ln in step.splitlines())
    assert "Losses 4.9937 / 4.9911 at steps 1, 10, 500 taped parameters and the 30981 MB DRAM peak" in flat
    assert "67,108,864 B request" in flat and "AICLK median 1350 (min 1343)" in flat
    assert (tree / "out/model_gate.txt").exists() and (tree / "out/step_gate_micro8.txt").exists()


@pytest.mark.parametrize("change", ["row", "loss", "peak", "fits"])
def test_record_abb3_fails_on_any_difference(tmp_path, change):
    arms, step, oom = dict(ARMS), STEP.format(l10="4.9911"), True
    committed = dict(ARMS)
    if change == "row":
        arms["0.05/64"] = ARM.format(v="ok  ", a="1.6e-02")
    elif change == "loss":
        step = STEP.format(l10="4.9921")
    elif change == "peak":
        step = step.replace("30981", "31002")
    else:
        oom = False
    tree = _abb3_tree(tmp_path, arms, step, oom=oom)
    _abb3_commit(tree, committed, STEP.format(l10="4.9911"))
    r = _abb3_run(tree)
    assert r.returncode != 0 and "abb3 records do not match" in r.stderr
    assert not (tree / "out/model_gate_qb1c3.txt").exists()


def test_a_record_runs_once_per_card_type_the_gate_holds():
    legs = {lg.name: lg for lg in gf.build_legs(ROSTER, [], 1, "", "rf3", record_cards=("p150a", "p300c"))
            if lg.family == "record"}
    assert sorted(legs) == ["record:rf3@p150a", "record:rf3@p300c"]
    assert legs["record:rf3@p300c"].card_types == ("p300c",)
    assert gf.ladder_model("record:rf3@p300c") == "rf3"
    qb1 = gf.Host("qb1", {"arch": "bh", "card_type": "p150a", "lock": "/l{card}", "root": "/r"}, "a" * 40)
    assert gf.type_fits(qb1, legs["record:rf3@p150a"]) and not gf.type_fits(qb1, legs["record:rf3@p300c"])


def test_a_chip_marked_bad_by_any_of_the_three_marks_is_reported(tmp_path):
    h = _host("qb2", "bh", tmp_path)
    assert h.marked_bad(0) is None
    for mark in (tmp_path / "locks" / "chip0.BAD", tmp_path / "host-state" / "cardblock-qb2-0",
                 tmp_path / "pc-state" / "cardblock-qb2-0"):
        mark.parent.mkdir(parents=True, exist_ok=True)
        mark.touch()
        assert h.marked_bad(0) == str(mark) and h.marked_bad(1) is None
        mark.unlink()
    (tmp_path / "host-state" / "cardblock-qb2-0.cleared-2026-10-14").touch()
    assert h.marked_bad(0) is None


def test_a_worker_on_a_chip_marked_bad_runs_no_leg_and_its_legs_run_elsewhere(tmp_path):
    """rec-c2b4b898e, 10-14: qb2:0 carried chip0.BAD and a cardblock, gate.json still listed it,
    and protenix-v2 ran (and deadlocked) there."""
    (tmp_path / "locks").mkdir()
    (tmp_path / "locks" / "chip0.BAD").touch()
    h = _host("qb2", "bh", tmp_path)
    legs = [lg for lg in _legs() if lg.card]
    ran = []

    def execute(host, card, leg):
        ran.append(card)
        return {"leg": leg.name, "arch": "bh", "verdict": "PASS", "worker": f"qb2:{card}"}

    res = gf.Gate({"bh": list(legs)}, [(h, 0), (h, 1)], {("qb2", 1)}, execute, tmp_path).run()
    assert set(ran) == {1} and len(res) == len(legs)
