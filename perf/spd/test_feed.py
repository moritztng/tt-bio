"""perf/spd/feed.py without a box: fake pools, check fill order, done-skip, twin-skip and the fast-fail quarantine."""
import json
import shutil
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import feed  # noqa: E402

GRID = """
row: spd-x
tier: {tier}
experiments:
  - name: g
    arch: wh
    prio: 4
    axes: {{seed: [1, 2], cpx: [A, B]}}
    vars: {{out: "~/o/{{seed}}/{{cpx}}"}}
    done: ["{{out}}/b/bench.jsonl"]
    also_named: ["4-spd-x-g-{{cpx}}-s{{seed}}.sh"]
    command: |
      run_chip.sh $CHIP {{out}} "b:fast" {{cpx}} 60 --seed {{seed}} ${{HOME}}
"""


class Box:
    def __init__(self, healthy=1):
        self.healthy, self.queue, self.ended, self.exists, self.live = healthy, [], {}, set(), []

    def probe(self, paths):
        jobs = {n: {"state": "queued"} for n in self.queue}
        jobs.update(self.ended)
        return {"healthy": self.healthy, "queued": len(self.queue), "paused": False, "jobs": jobs, "swept": 0,
                "live": self.queue + self.live, "exists": {p: p in self.exists for p in paths}}


def make_pool(tmp_path, monkeypatch, tier="red", healthy=1):
    (tmp_path / "state/spd/grids").mkdir(parents=True)
    (tmp_path / "state/glx").mkdir(parents=True)
    (tmp_path / "state/spd/grids/spd-x.yaml").write_text(GRID.format(tier=tier))
    (tmp_path / "state/glx/ALLOC.json").write_text(json.dumps({"galaxies": {"9": {"owner": "spd", "phase": "lent"}}}))
    box = Box(healthy)
    monkeypatch.setattr(feed, "BH_POOLS", {})

    def remote(ssh, script, payload, timeout=120):
        if script is feed.PROBE:
            return box.probe(payload["paths"])
        box.queue += [j["name"] for j in payload["jobs"]]
        box.placed = payload["jobs"]
        return len(payload["jobs"])
    monkeypatch.setattr(feed, "remote", remote)
    return box


def run(tmp_path):
    assert feed.main(["--root", str(tmp_path)]) == 0


def test_render_keeps_shell_vars():
    assert feed.render("a {x} ${HOME} $CHIP", {"x": 1}) == "a 1 ${HOME} $CHIP"


def test_fills_to_depth_and_skips_done_and_twins(tmp_path, monkeypatch):
    box = make_pool(tmp_path, monkeypatch, healthy=1)
    box.exists.add("~/o/1/A/b/bench.jsonl")          # cell (1, A) measured already
    box.live.append("4-spd-x-g-B-s1.sh")             # cell (1, B) hand-queued by the row
    run(tmp_path)
    assert [n.split("-")[5] for n in box.queue] == ["2_A", "2_B"]   # depth 2 x 1 healthy chip
    body = box.placed[0]["body"]
    assert "# mem: 24" in body and "--seed 2" in body and "${HOME}" in body and "TT_BIO_LEASE_HOLDER=spd-x" in body
    run(tmp_path)                                      # nothing new: both pending
    assert len(box.queue) == 2


def test_two_fast_fails_quarantine_and_wake(tmp_path, monkeypatch):
    box = make_pool(tmp_path, monkeypatch, healthy=1)
    box.exists |= {"~/o/1/A/b/bench.jsonl", "~/o/1/B/b/bench.jsonl", "~/o/2/B/b/bench.jsonl"}
    nb = tmp_path / "state/notbefore"; nb.mkdir(parents=True); (nb / "spd-x").write_text("x")
    run(tmp_path)
    name = box.queue.pop()
    for at in (100, 200):
        box.ended[name] = {"state": "ended", "dur": 12, "rc": "0", "at": at, "tail": "ImportError: no module"}
        run(tmp_path)
        if at == 100:
            assert box.queue == [name]                 # one fast fail: placed again
            box.queue.clear()
    assert box.queue == []                             # quarantined: never re-queued
    q = (tmp_path / "state/spd/QUARANTINE.md").read_text()
    assert "spd-x/g/2_A" in q and "ImportError" in q
    assert not (nb / "spd-x").exists()                 # row woken
    run(tmp_path)
    assert box.queue == []


def test_a_dirty_cards_fast_fails_do_not_quarantine_the_cell(tmp_path, monkeypatch):
    box = make_pool(tmp_path, monkeypatch, healthy=1)
    box.exists |= {"~/o/1/A/b/bench.jsonl", "~/o/1/B/b/bench.jsonl", "~/o/2/B/b/bench.jsonl"}
    run(tmp_path)
    name = box.queue.pop()
    for at in (100, 200, 300):
        box.ended[name] = {"state": "ended", "dur": 15, "rc": "1 fastfail2 reset", "at": at, "tail": "failed to initialize FW"}
        run(tmp_path)
        assert box.queue == [name]                     # placed again every time
        box.queue.clear()
    assert not (tmp_path / "state/spd/QUARANTINE.md").exists() or "2_A" not in (tmp_path / "state/spd/QUARANTINE.md").read_text()


def test_changed_grid_entry_releases_quarantine(tmp_path, monkeypatch):
    box = make_pool(tmp_path, monkeypatch, healthy=1)
    box.exists |= {"~/o/1/A/b/bench.jsonl", "~/o/1/B/b/bench.jsonl", "~/o/2/B/b/bench.jsonl"}
    run(tmp_path)
    name = box.queue.pop()
    for at in (1, 2):
        box.ended[name] = {"state": "ended", "dur": 5, "rc": "1", "at": at, "tail": ""}
        run(tmp_path); box.queue.clear()
    g = tmp_path / "state/spd/grids/spd-x.yaml"
    g.write_text(g.read_text().replace("60 --seed", "90 --seed"))
    run(tmp_path)
    assert len(box.queue) == 1 and box.queue[0] != name


def test_inputs_are_what_the_command_reads_first():
    cmd = ("[ -f ~/japanfold/env.sh ] && . ~/japanfold/env.sh\n. $HOME/spd/r/common.sh\n"
           "O=/home/u/out/{x}; mkdir -p $O; cd /home/u/trees/abc; TREE=/home/u/t2 bash ~/spd/run_cell.sh $CHIP $O")
    assert feed.inputs(cmd) == {"~/spd/r/common.sh", "/home/u/trees/abc", "/home/u/t2", "~/spd/run_cell.sh"}
    assert feed.inputs("[ -s /home/u/ref/a.npy ] || exit 1\n[ -s /home/u/opt ] && x") == {"/home/u/ref/a.npy"}


def test_tree_root_names_the_commit_dir():
    repo = str(Path(feed.__file__).resolve().parents[2])
    sha = subprocess.run(["git", "-C", repo, "rev-parse", "--short=9", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert feed.tree_root(f"/home/u/spd/r/t{sha}/perf/spd/run_chip.sh", repo) == (f"/home/u/spd/r/t{sha}", sha)
    assert feed.tree_root(f"/home/u/spd/r/tree-{sha}", repo) == (f"/home/u/spd/r/tree-{sha}", sha)
    assert feed.tree_root("/home/u/spd/r/t1b8/x", repo) is None


def test_probe_counts_an_empty_marker_and_tails_the_jobs_own_log(tmp_path):
    """`touch $O/done` and `touch $O/<arm>.done` are a finished cell (an empty result file is not), and a job that logs into its out dir gets
    that log tailed into the quarantine entry."""
    o = tmp_path / "o"; o.mkdir()
    (o / "done").touch(); (o / "a=m:fast.done").touch(); (o / "r.json").touch(); (o / "run.log").write_text("Traceback: boom\n" * 20)
    jobs = tmp_path / "spd/feed/jobs"; jobs.mkdir(parents=True)
    (jobs / "4-x.sh").write_text(""); (jobs / "4-x.sh.end").write_text("0\n"); (jobs / "4-x.sh.log").write_text("")
    a = {"kind": "bh", "cards": [], "sweep": False, "paths": [str(o / "done"), str(o / "a=m:fast.done"), str(o / "r.json")],
         "outdirs": {"4-x.sh": [str(o)]}}
    r = subprocess.run([sys.executable, "-c", feed.PROBE], input=json.dumps(a) + "\n", capture_output=True, text=True,
                       env={"HOME": str(tmp_path)}, check=True)
    out = json.loads(r.stdout)
    assert out["exists"] == {str(o / "done"): True, str(o / "a=m:fast.done"): True, str(o / "r.json"): False}
    assert "Traceback: boom" in out["jobs"]["4-x.sh"]["tail"]


def _probe_bh(tmp_path, queued):
    import fcntl
    jobs = tmp_path / "spd/feed/jobs"; jobs.mkdir(parents=True)
    me = str(subprocess.os.getpid())
    for n, card, started in (("4-a.sh", -1, False), ("4-b.sh", 1, True))[0 if queued else 1:]:
        (jobs / n).write_text(""); (jobs / f"{n}.card").write_text(str(card)); (jobs / f"{n}.pid").write_text(me)
        if started:
            (jobs / f"{n}.start").touch()
    held = open(tmp_path / "lock2", "a"); fcntl.flock(held, fcntl.LOCK_EX)     # another row's run on card 2
    a = {"kind": "bh", "cards": [0, 1, 2, 3], "lock": str(tmp_path / "lock{c}"), "sweep": False, "paths": [], "outdirs": {}}
    # no release-gate leg runs here: a live one on the test host would otherwise hold a card in the probe's pgrep
    (tmp_path / "bin").mkdir(); (tmp_path / "bin/pgrep").write_text("#!/bin/sh\nexit 1\n"); (tmp_path / "bin/pgrep").chmod(0o755)
    r = subprocess.run([sys.executable, "-c", feed.PROBE], input=json.dumps(a) + "\n", capture_output=True, text=True,
                       env={"HOME": str(tmp_path), "PATH": f"{tmp_path}/bin:/usr/bin:/bin"}, check=True)
    return json.loads(r.stdout)


def test_probe_offers_free_cards_only_when_no_waiter_of_its_own_box_will_take_them(tmp_path):
    """A BH card is free when no feed job runs on it and its CHIPS flock is free; a box with a waiter offers none."""
    out = _probe_bh(tmp_path, queued=False)
    assert out["free"] == [0, 3] and out["queued"] == 0 and out["cards"] == [0, 1, 2, 3]


def test_probe_reports_waiter_age(tmp_path):
    out = _probe_bh(tmp_path, queued=True)
    assert out["free"] == [] and out["queued"] == 1
    assert out["jobs"]["4-a.sh"]["state"] == "queued" and out["jobs"]["4-a.sh"]["age"] >= 0


BH_GRID = """
row: spd-x
experiments:
  - name: g
    arch: bh
    prio: 4
    axes: {cpx: [A]}
    vars: {out: "~/o/{cpx}"}
    done: ["{out}/r.json"]
    command: |
      echo {cpx}
"""


def test_a_long_waiting_bh_job_moves_to_a_free_card_on_the_other_box(tmp_path, monkeypatch):
    (tmp_path / "state/spd/grids").mkdir(parents=True); (tmp_path / "state/glx").mkdir(parents=True)
    (tmp_path / "state/spd/grids/spd-x.yaml").write_text(BH_GRID)
    (tmp_path / "state/glx/ALLOC.json").write_text(json.dumps({"galaxies": {}}))
    monkeypatch.setattr(feed, "BH_POOLS", {"qb1": {"ssh": "qb1", "cards": [0], "lock": "/l1_{c}"},
                                           "qb2": {"ssh": "qb2", "cards": [3], "lock": "/l2_{c}"}})
    name = feed.cells(tmp_path / "state/spd/grids")[0]["name"]
    calls = []

    def remote(ssh, script, payload, timeout=120):
        calls.append((ssh, script, payload))
        if script is feed.PROBE:
            if ssh == "qb1":
                return {"healthy": 1, "cards": [0], "queued": 1, "free": [], "paused": False, "swept": 0, "live": [],
                        "jobs": {name: {"state": "queued", "age": feed.STEAL_S + 5}}, "exists": {}}
            return {"healthy": 1, "cards": [3], "queued": 0, "free": [3], "paused": False, "swept": 0, "live": [],
                    "jobs": {}, "exists": {}}
        if script is feed.WITHDRAW:
            return payload["names"]
        return len(payload["jobs"])
    monkeypatch.setattr(feed, "remote", remote)
    (tmp_path / "state/spd/feed").mkdir(parents=True)
    key = feed.cells(tmp_path / "state/spd/grids")[0]["key"]
    (tmp_path / "state/spd/feed/ledger.json").write_text(json.dumps({key: {"host": "qb1", "name": name, "state": "queued",
                                                                            "hash": name.split("-")[-1][:-3], "fails": []}}))
    run(tmp_path)
    assert [c[0] for c in calls if c[1] is feed.WITHDRAW] == ["qb1"]
    placed = [c[2]["jobs"] for c in calls if c[1] is feed.PLACE and c[2]["jobs"]]
    assert placed == [[{"name": name, "body": feed.cells(tmp_path / "state/spd/grids")[0]["body"], "cap": 7200,
                        "cards": [3], "lock": "/l2_{c}"}]]
    assert json.loads((tmp_path / "state/spd/feed/ledger.json").read_text())[key]["host"] == "qb2"


def _place_bh(tmp_path, cards, body, cap=5, poll="0.2", leg_sid=None):
    (tmp_path / "spd/feed/jobs").mkdir(parents=True, exist_ok=True)
    # no release-gate leg but the test's own (session leg_sid): the test host's legs (pc runs some) would otherwise
    # hold a card in the waiter's pgrep
    stub = f"exec {shutil.which('pgrep')} -s {leg_sid} \"$@\"" if leg_sid else "exit 1"
    (tmp_path / "bin").mkdir(exist_ok=True); (tmp_path / "bin/pgrep").write_text(f"#!/bin/sh\n{stub}\n")
    (tmp_path / "bin/pgrep").chmod(0o755)
    job = {"name": "4-x.sh", "body": body, "cards": cards, "lock": str(tmp_path / "l{c}"), "cap": cap}
    r = subprocess.run([sys.executable, "-c", feed.PLACE], input=json.dumps({"kind": "bh", "jobs": [job]}) + "\n",
                       capture_output=True, text=True, check=True,
                       env={"HOME": str(tmp_path), "PATH": f"{tmp_path}/bin:" + os.environ["PATH"], "POLL": poll,
                            "TT_SYSFS": str(tmp_path / "sys")})
    assert r.stdout.strip() == "1"
    sh = tmp_path / "spd/feed/jobs/4-x.sh"
    for _ in range(100):
        if Path(f"{sh}.end").exists():
            break
        time.sleep(0.1)
    return sh


def test_a_bh_waiter_takes_whichever_card_of_its_box_is_free(tmp_path):
    # 10-15 qb1: waiters bound to busy cards 2 and 3 queued while card 0 sat free until a 15-min steal moved one
    import fcntl
    busy = open(tmp_path / "l5", "a"); fcntl.flock(busy, fcntl.LOCK_EX)
    (tmp_path / "spd/locks").mkdir(parents=True); (tmp_path / "spd/locks/chip6.BAD").touch()
    sh = _place_bh(tmp_path, [5, 6, 7], "echo card $CHIP\n")
    assert Path(f"{sh}.end").read_text().strip() == "0"
    assert Path(f"{sh}.log").read_text() == "card 7\n" and Path(f"{sh}.card").read_text().strip() == "7"


def test_a_bh_waiter_starts_on_the_first_card_to_come_free(tmp_path):
    import fcntl
    locks = [open(tmp_path / f"l{c}", "a") for c in (0, 1)]
    for f in locks:
        fcntl.flock(f, fcntl.LOCK_EX)
    (tmp_path / "spd/feed/jobs").mkdir(parents=True)
    t = __import__("threading").Timer(0.5, lambda: fcntl.flock(locks[1], fcntl.LOCK_UN)); t.start()
    sh = _place_bh(tmp_path, [0, 1], "echo card $CHIP\n")
    t.join()
    assert Path(f"{sh}.log").read_text() == "card 1\n"


def test_withdraw_kills_an_unstarted_waiter_and_sets_it_aside(tmp_path):
    jobs = tmp_path / "spd/feed/jobs"; jobs.mkdir(parents=True)
    w = subprocess.Popen(["setsid", "bash", "-c", "sleep 300"])
    for n, started in (("4-a.sh", False), ("4-b.sh", True)):
        (jobs / n).write_text(""); (jobs / f"{n}.pid").write_text(str(w.pid) if not started else "1"); (jobs / f"{n}.card").write_text("0")
    (jobs / "4-b.sh.start").touch()
    r = subprocess.run([sys.executable, "-c", feed.WITHDRAW], input=json.dumps({"names": ["4-a.sh", "4-b.sh"]}) + "\n",
                       capture_output=True, text=True, env={"HOME": str(tmp_path)}, check=True)
    assert json.loads(r.stdout) == ["4-a.sh"]
    assert w.wait(timeout=5) != 0
    assert (jobs / "4-a.sh.moved").exists() and not (jobs / "4-a.sh").exists() and (jobs / "4-b.sh").exists()


def test_a_cell_on_a_dark_box_is_placed_again_and_its_twin_withdrawn_on_return(tmp_path, monkeypatch):
    (tmp_path / "state/spd/grids").mkdir(parents=True); (tmp_path / "state/glx").mkdir(parents=True)
    (tmp_path / "state/spd/grids/spd-x.yaml").write_text(BH_GRID)
    (tmp_path / "state/glx/ALLOC.json").write_text(json.dumps({"galaxies": {}}))
    monkeypatch.setattr(feed, "BH_POOLS", {"qb1": {"ssh": "qb1", "cards": [0], "lock": "/l1_{c}"},
                                           "qb2": {"ssh": "qb2", "cards": [3], "lock": "/l2_{c}"}})
    c = feed.cells(tmp_path / "state/spd/grids")[0]
    calls, dark = [], [True]

    def remote(ssh, script, payload, timeout=120):
        calls.append((ssh, script, payload))
        if script is feed.PROBE:
            if ssh == "qb2":
                if dark[0]:
                    raise RuntimeError("No route to host")
                return {"healthy": 1, "cards": [3], "queued": 1, "free": [], "paused": False, "swept": 0, "live": [],
                        "jobs": {c["name"]: {"state": "queued", "card": 3, "age": 60}}, "exists": {}}
            return {"healthy": 1, "cards": [0], "queued": 0, "free": [], "paused": False, "swept": 0, "live": [],
                    "jobs": {}, "exists": {}}
        if script is feed.WITHDRAW:
            return payload["names"]
        return len(payload["jobs"])
    monkeypatch.setattr(feed, "remote", remote)
    (tmp_path / "state/spd/feed").mkdir(parents=True)
    (tmp_path / "state/qb2-offline").write_text("")
    led = tmp_path / "state/spd/feed/ledger.json"
    led.write_text(json.dumps({c["key"]: {"host": "qb2", "name": c["name"], "state": "queued", "hash": c["hash"], "fails": []}}))
    run(tmp_path)
    assert [x[0] for x in calls if x[1] is feed.PLACE and x[2]["jobs"]] == ["qb1"]
    assert json.loads(led.read_text())[c["key"]]["host"] == "qb1"
    dark[0] = False; (tmp_path / "state/qb2-offline").unlink(); calls.clear()
    run(tmp_path)
    assert [(x[0], x[2]["names"]) for x in calls if x[1] is feed.WITHDRAW] == [("qb2", [c["name"]])]


def sysfs(tmp_path, cards):
    """Fake /sys/class/tenstorrent: {bdf: (node, aiclk)}; logical id = PCI order."""
    for bdf, (node, clk) in cards.items():
        d = tmp_path / "sys" / f"tenstorrent!{node}"; dev = tmp_path / "pci" / bdf
        d.mkdir(parents=True, exist_ok=True); dev.mkdir(parents=True, exist_ok=True)
        if not (d / "device").exists():
            (d / "device").symlink_to(dev)
        (d / "tt_aiclk").write_text(clk + "\n"); (d / "tt_heartbeat").write_text(("4294967295" if clk == "4294967295" else "77") + "\n")


def _run_capped(tmp_path, body, cap, reset="", name="4-x.sh"):
    sh = tmp_path / name; sh.write_text(body)
    sysfs(tmp_path, {"0000:42:00.0": (3, "1350")})
    subprocess.run(["bash", "-c", feed.NODE + feed.run_cmd(grace=2), "_", str(sh), "0", "", str(tmp_path / "nolock"),
                    str(cap), reset], cwd=tmp_path, timeout=60, check=True,
                   env=dict(os.environ, TT_SYSFS=str(tmp_path / "sys"), TT_DEV="/dev/tt"))
    return (tmp_path / f"{name}.end").read_text().strip()


def test_a_job_under_its_cap_ends_with_its_own_rc(tmp_path):
    assert _run_capped(tmp_path, "exit 3\n", 30) == "3"


def test_a_job_that_honours_sigint_ends_on_int(tmp_path):
    assert _run_capped(tmp_path, "sleep 30\n", 1) == "124 INT"


def test_a_child_that_outlives_its_wrapper_on_sigint_gets_sigterm(tmp_path):
    # 10-15 qb1: the wrapper exited on SIGINT, the python stuck in a native wait did not and held the card's flock 2 h
    end = _run_capped(tmp_path, "(trap '' INT; exec sleep 300) & echo $! > child; wait\n", 1)
    assert end == "124 INT+TERM"
    pid = int((tmp_path / "child").read_text())
    for _ in range(50):
        if not os.path.exists(f"/proc/{pid}") or open(f"/proc/{pid}/stat").read().split()[2] == "Z":
            break
        time.sleep(0.1)
    else:
        raise AssertionError("the orphan survived SIGTERM")


def test_place_starts_a_bh_waiter_that_runs_its_job(tmp_path):
    # 10-15 05:54Z: PLACE runs on the BH host and referenced feed.py's RUN, which only exists here; every qb1/qb2
    # placement died with NameError while the unit tests, which never execute PLACE, stayed green
    sh = _place_bh(tmp_path, [7], "echo card $CHIP\n")
    assert Path(f"{sh}.end").read_text().strip() == "0"
    assert Path(f"{sh}.log").read_text() == "card 7\n"


def test_a_capped_job_resets_its_card_and_a_clean_one_does_not(tmp_path):
    # 10-15 qb1: after three hung PV2 seed cells were killed, every later job on those cards failed device init
    r = tmp_path / "reset.sh"; r.write_text('echo "$1" >> "$(dirname "$0")/resets"\n'); r.chmod(0o755)
    assert _run_capped(tmp_path, "exit 3\n", 30, str(r)) == "3"
    assert not (tmp_path / "resets").exists()
    assert _run_capped(tmp_path, "(trap '' INT; exec sleep 300) & wait\n", 1, str(r)) == "124 INT+TERM reset"
    assert (tmp_path / "resets").read_text() == "/dev/tt/3\n"     # by node path, not logical id
    assert _run_capped(tmp_path, "sleep 30\n", 1, "false") == "124 INT reset-failed"


def test_two_different_jobs_failing_fast_on_one_card_reset_it(tmp_path):
    # 10-16 00:12Z qb1 card 0: six jobs of four rows failed init in 15 s each until the card was reset by hand
    r = tmp_path / "reset.sh"; r.write_text('echo "$1" >> "$(dirname "$0")/resets"\n'); r.chmod(0o755)
    assert _run_capped(tmp_path, "exit 1\n", 30, str(r), "4-a.sh") == "1"
    assert _run_capped(tmp_path, "exit 1\n", 30, str(r), "4-a.sh") == "1"        # the same job again: the job's fault
    assert _run_capped(tmp_path, "exit 1\n", 30, str(r), "4-b.sh") == "1 fastfail2 reset"
    assert (tmp_path / "resets").read_text() == "/dev/tt/3\n"
    assert _run_capped(tmp_path, "exit 1\n", 30, str(r), "4-c.sh") == "1"        # the reset cleared the streak
    assert _run_capped(tmp_path, "exit 0\n", 30, str(r), "4-d.sh") == "0"        # a clean exit clears it too
    assert _run_capped(tmp_path, "exit 1\n", 30, str(r), "4-e.sh") == "1"
    assert _run_capped(tmp_path, "exit 127\n", 30, str(r), "4-f.sh") == "127"    # a missing command is not the card
    assert _run_capped(tmp_path, "exit 1\n", 30, "", "4-g.sh") == "1"           # qb2: no reset command, no reset
    assert (tmp_path / "resets").read_text() == "/dev/tt/3\n"


def test_qb2_boards_are_never_reset_by_the_feed():
    # a p300c board carries two chips: resetting one kills the other chip's job
    assert feed.BH_POOLS["qb1"]["reset"] and not feed.BH_POOLS["qb2"].get("reset")


DEAD = "4294967295"


def test_probe_drops_a_dead_arc_card_and_names_its_node(tmp_path):
    # 10-15 qb1: /dev/tenstorrent/3 (logical 2) read 0xFFFFFFFF for 70+ min while the feed kept landing jobs on it
    sysfs(tmp_path, {"0000:c1:00.0": (0, "1350"), "0000:01:00.0": (1, "1350"), "0000:41:00.0": (2, "1350"),
                     "0000:42:00.0": (3, DEAD)})
    r = subprocess.run([sys.executable, "-c", feed.PROBE], capture_output=True, text=True, check=True,
                       input=json.dumps({"kind": "bh", "paths": [], "cards": [0, 1, 2, 3], "lock": str(tmp_path / "l{c}")}) + "\n",
                       env=dict(os.environ, HOME=str(tmp_path), TT_SYSFS=str(tmp_path / "sys")))
    out = json.loads(r.stdout)
    assert out["dead"] == {"2": 3} and out["cards"] == [0, 1, 3] and out["healthy"] == 3


def test_a_bh_waiter_leaves_a_card_to_a_waiting_runpy_gate_leg(tmp_path):
    # 10-16 qb2: rel-fix legs bc2_boundary / parity_af2ig-trunk-device waited on chips 1 and 3 with only
    # RELEASE_GATE_RUN_DIR naming the gate, the guard missed them and a feed waiter could take the chip first
    leg = subprocess.Popen(["bash", "-c", 'exec -a "flock /h/spd_qb2_chip5.lock env TT_VISIBLE_DEVICES=5 '
                            'RELEASE_GATE_RUN_DIR=/o/run python3 -c runpy" sleep 30'], start_new_session=True)
    try:
        time.sleep(0.2)
        sh = _place_bh(tmp_path, [5, 6], "echo card $CHIP\n", leg_sid=leg.pid)
        assert Path(f"{sh}.log").read_text() == "card 6\n"
    finally:
        leg.kill()


def test_a_bh_waiter_skips_a_dead_arc_card(tmp_path):
    sysfs(tmp_path, {"0000:01:00.0": (1, DEAD), "0000:02:00.0": (0, "1350")})
    sh = _place_bh(tmp_path, [0, 1], "echo card $CHIP\n")
    assert Path(f"{sh}.log").read_text() == "card 1\n"


def _arc_pool(tmp_path, monkeypatch):
    (tmp_path / "state/spd/feed").mkdir(parents=True); (tmp_path / "state/notbefore").mkdir(parents=True)
    calls = []
    monkeypatch.setattr(feed, "remote", lambda ssh, script, payload, timeout=120: calls.append((ssh, script, payload)) or 1)
    P = {n: dict(b, kind="bh", arch="bh") for n, b in feed.BH_POOLS.items()}
    return calls, P


def test_dead_arc_resets_qb1_once_then_cardblocks_and_marks_qb2_bad(tmp_path, monkeypatch):
    calls, P = _arc_pool(tmp_path, monkeypatch)
    (tmp_path / "state/notbefore/spd-orchestrator").write_text("x")
    st = {"qb1": {"dead": {"2": 3}, "arc_busy": [], "arc": []}, "qb2": {"dead": {"1": 1}}}
    feed.arc(tmp_path, st, P, print, now=1000)
    assert [(c[0], c[1] is feed.ARCFIX, c[2]["card"]) for c in calls] == [("qb1", True, 2), ("qb2", False, 1)]
    assert calls[0][2]["node"] == 3 and calls[1][1] is feed.BAD
    calls.clear()
    st["qb1"]["arc_busy"] = [2]; feed.arc(tmp_path, {"qb1": st["qb1"]}, P, print, now=1100)
    assert calls == []                                     # its reset is in flight
    st["qb1"].update(arc_busy=[], arc=[{"card": 2, "at": 1000, "result": "ok"}], dead={})
    feed.arc(tmp_path, {"qb1": st["qb1"]}, P, print, now=1500)
    feed.arc(tmp_path, {"qb1": st["qb1"]}, P, print, now=1600)
    log = (tmp_path / "fleet.log").read_text()
    assert log.count("dead-ARC reset: ok") == 1           # each result is one fleet.log line
    st["qb1"]["dead"] = {"2": 3}
    feed.arc(tmp_path, {"qb1": st["qb1"]}, P, print, now=1000 + 3600)
    assert [(c[0], c[1] is feed.BAD) for c in calls] == [("qb1", True)]
    assert (tmp_path / "state/cardblock-qb1-2").exists() and not (tmp_path / "state/notbefore/spd-orchestrator").exists()


def test_a_dead_arc_reset_waits_for_the_flock_and_verifies_sysfs(tmp_path):
    sysfs(tmp_path, {"0000:42:00.0": (3, DEAD)})
    fix = tmp_path / "fix.sh"
    fix.write_text(f'echo "$1" > {tmp_path}/reset_arg; echo 1350 > {tmp_path}/sys/tenstorrent!3/tt_aiclk; '
                   f'echo 9 > {tmp_path}/sys/tenstorrent!3/tt_heartbeat\n')
    fix.chmod(0o755); (tmp_path / "dev").mkdir(); (tmp_path / "dev/3").touch()
    env = dict(os.environ, TT_SYSFS=str(tmp_path / "sys"), TT_DEV=str(tmp_path / "dev"), ARC_SETTLE="0", ARC_WAIT="1")
    b = tmp_path / "card2.1"
    subprocess.run(["bash", "-c", feed.ARC_SH, "_", str(b), str(tmp_path / "l2"), "3", str(fix)], env=env, check=True, timeout=60)
    assert Path(f"{b}.result").read_text().strip() == "ok" and (tmp_path / "reset_arg").read_text().strip() == f"{tmp_path}/dev/3"
    fix.write_text("true\n"); (tmp_path / "sys/tenstorrent!3/tt_aiclk").write_text(DEAD + "\n")
    subprocess.run(["bash", "-c", feed.ARC_SH, "_", str(b), str(tmp_path / "l2"), "3", str(fix)], env=env, check=True, timeout=60)
    assert Path(f"{b}.result").read_text().strip() == "still-dead"
    end = [l for l in Path(f"{b}.log").read_text().splitlines() if " end " in l]
    assert [l.split(" ", 1)[1] for l in end] == ["end rc=0 aiclk=1350 hb=9 result=ok",
                                                 f"end rc=0 aiclk={DEAD} hb=9 result=still-dead"]


def test_a_dead_arc_reset_killed_mid_reset_still_logs_its_end(tmp_path):
    sysfs(tmp_path, {"0000:42:00.0": (3, DEAD)})
    fix = tmp_path / "fix.sh"; fix.write_text("sleep 2\n"); fix.chmod(0o755)
    (tmp_path / "dev").mkdir(); (tmp_path / "dev/3").touch()
    env = dict(os.environ, TT_SYSFS=str(tmp_path / "sys"), TT_DEV=str(tmp_path / "dev"), ARC_SETTLE="0", ARC_WAIT="1")
    b = tmp_path / "card2.1"
    p = subprocess.Popen(["bash", "-c", feed.ARC_SH, "_", str(b), str(tmp_path / "l2"), "3", str(fix)], env=env)
    for _ in range(100):
        if " before " in (Path(f"{b}.log").read_text() if Path(f"{b}.log").exists() else ""): break
        time.sleep(0.1)
    p.terminate(); p.wait(timeout=60)
    assert Path(f"{b}.result").read_text().strip() == "died"
    assert Path(f"{b}.log").read_text().splitlines()[-1].split(" ", 1)[1] == f"end rc=none aiclk={DEAD} hb={DEAD} result=died"
