"""perf/spd/feed.py without a box: fake pools, check fill order, done-skip, twin-skip and the fast-fail quarantine."""
import json
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


def _place_bh(tmp_path, cards, body, cap=5, poll="0.2"):
    (tmp_path / "spd/feed/jobs").mkdir(parents=True, exist_ok=True)
    # no release-gate leg: the test host's own legs (pc runs some) would otherwise hold a card in the waiter's pgrep
    (tmp_path / "bin").mkdir(exist_ok=True); (tmp_path / "bin/pgrep").write_text("#!/bin/sh\nexit 1\n")
    (tmp_path / "bin/pgrep").chmod(0o755)
    job = {"name": "4-x.sh", "body": body, "cards": cards, "lock": str(tmp_path / "l{c}"), "cap": cap}
    r = subprocess.run([sys.executable, "-c", feed.PLACE], input=json.dumps({"kind": "bh", "jobs": [job]}) + "\n",
                       capture_output=True, text=True, check=True,
                       env={"HOME": str(tmp_path), "PATH": f"{tmp_path}/bin:" + os.environ["PATH"], "POLL": poll})
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


def _run_capped(tmp_path, body, cap):
    sh = tmp_path / "4-x.sh"; sh.write_text(body)
    subprocess.run(["bash", "-c", feed.run_cmd(grace=2), "_", str(sh), "0", "", str(tmp_path / "nolock"), str(cap)],
                   cwd=tmp_path, timeout=60, check=True)
    return (tmp_path / "4-x.sh.end").read_text().strip()


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
