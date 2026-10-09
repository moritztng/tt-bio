"""gate_fanout.py: leg split, result keys, reuse, scheduling and verdict, all card-free."""
import importlib.util
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gate_fanout", REPO / "scripts" / "gate_fanout.py")
gf = importlib.util.module_from_spec(spec)
sys.modules["gate_fanout"] = gf
spec.loader.exec_module(gf)

ROSTER = {"parity": ["esmc-300m", "boltzgen", "capacity"],
          "rg": ["boltz2", "boltzgen", "opendde-abag", "capacity", "nesso1", "rf3-1024aa",
                 "l1-budget", "size-ladder"],
          "ladder": ["boltz2", "rf3"], "capacity": ["boltz2"], "ux": ["boltz2"]}


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
    files = [a for lg in shards for a in lg.argv if a.startswith("tests/")]
    assert sorted(files) == ["tests/test_a.py", "tests/test_b.py", "tests/test_c.py"]


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
    (tmp_path / "perf").mkdir()
    (tmp_path / "perf" / "gate.txt").write_text("RECORDED-AT: abcdef1 scripts/x.py\n1.0 s\n")
    git("add", "."), git("commit", "-qm", "7")
    assert gf.content_hash("HEAD", tmp_path) == c6          # a re-recorded measurement: same key
    (tmp_path / "perf" / "data.txt").write_text("fixture\n")
    git("add", "."), git("commit", "-qm", "8")
    assert gf.content_hash("HEAD", tmp_path) != c6          # any other perf/ file: new key
    c8, l8 = gf.content_hash("HEAD", tmp_path), gf.baseline_hash("HEAD", "ladder", tmp_path)
    p8 = gf.baseline_hash("HEAD", "perf", tmp_path)
    (tmp_path / "docs" / "size_ladder_baseline.d").mkdir(parents=True)
    (tmp_path / "docs" / "size_ladder_baseline.d" / "boltz2.json").write_text("{}\n")
    git("add", "."), git("commit", "-qm", "9")
    assert gf.content_hash("HEAD", tmp_path) == c8          # a re-recorded baseline: same code key,
    assert gf.baseline_hash("HEAD", "ladder", tmp_path) != l8  # a new key for the ladder legs only
    assert gf.baseline_hash("HEAD", "perf", tmp_path) == p8
    assert gf.content_hash("HEAD", tmp_path, baselines=True) != c8

    leg = _legs()[3]
    k = gf.leg_key(c1, "env", "p150a", leg)
    assert k == gf.leg_key(c1, "env", "p150a", leg)
    assert k != gf.leg_key(c1, "env2", "p150a", leg)
    assert k != gf.leg_key(c1, "env", "tt-galaxy-wh-l", leg)
    other = gf.Leg(leg.name, leg.argv + ["--seeds", "1"], leg.family)
    assert k != gf.leg_key(c1, "env", "p150a", other)


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


def test_every_family_has_a_budget_and_an_expectation():
    for lg in _legs():
        assert lg.budget > 0 and lg.family in gf.EXPECT


def test_command_pins_card_under_its_flock_and_substitutes_placeholders(tmp_path):
    h = gf.Host("qb1", {"arch": "bh", "card_type": "p150a", "root": "/r",
                        "lock": "/l/card{card}.lock", "env": {"ESM_ROOT": "/esm"},
                        "pythonpath_extra": ["/bc2"]}, "a" * 40)
    leg = next(lg for lg in _legs() if lg.family == "parity")
    cmd = h.command(leg, 2, "/t/out/x")
    assert "flock /l/card2.lock" in cmd and "TT_VISIBLE_DEVICES=2" in cmd
    assert "PYTHONPATH=/r/trees/aaaaaaaaaaaa:/bc2" in cmd and "ESM_ROOT=/esm" in cmd
    assert "localhost:2" in cmd and "/t/out/x/report.json" in cmd
    assert "/r/venv-aaaaaaaaaaaa/bin/python scripts/full_parity_gate.py" in cmd
    free = h.command(next(lg for lg in _legs() if lg.name == "pytest_cpu"), None, "/t/out/y")
    assert "flock" not in free and "TT_VISIBLE_DEVICES='' " in free
    assert "/r/venv312-aaaaaaaaaaaa/bin/python -m pytest" in free
    bc2 = h.command(next(lg for lg in _legs() if lg.family == "bc2"), 1, "/t/out/z")
    assert "PYTHONPATH=/bc2 " in bc2 and "/r/trees" not in bc2.split("&&")[-1].split("bash -c")[0]


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
