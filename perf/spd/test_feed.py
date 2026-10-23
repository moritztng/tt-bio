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


def test_a_waiter_that_never_took_a_card_does_not_quarantine_the_cell(tmp_path, monkeypatch):
    # 10-16 05:24Z qb1: waiters that died "flockbusy" before taking a card quarantined two cells that had not run
    box = make_pool(tmp_path, monkeypatch, healthy=1)
    box.exists |= {"~/o/1/A/b/bench.jsonl", "~/o/1/B/b/bench.jsonl", "~/o/2/B/b/bench.jsonl"}
    run(tmp_path)
    name = box.queue.pop()
    for at in (100, 200, 300):
        box.ended[name] = {"state": "ended", "dur": 20, "rc": "flockbusy", "at": at, "tail": ""}
        run(tmp_path)
        assert box.queue == [name]
        box.queue.clear()


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
    assert feed.tree_root(f"/home/u/spd/r/trees/m{sha}/perf/spd/run_chip.sh", repo) == (f"/home/u/spd/r/trees/m{sha}", sha)
    assert feed.tree_root("/home/u/spd/r/t1b8/x", repo) is None
    assert feed.tree_root("/home/u/spd/r/trees/mbr/perf/spd/run_chip.sh", repo) is None


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


def test_probe_drops_the_booking_of_a_concluded_row(tmp_path):
    """A row that concludes leaves its booking standing (10-23: two RFD4 rows held qb2 cards 1 and 3 ~4 h after
    concluding); the probe removes bookings whose owner is in `concluded` and keeps every other one."""
    (tmp_path / "spd/feed").mkdir(parents=True)
    w = tmp_path / "spd/feed/windows.tsv"
    w.write_text("1\t9999999999\trfd4-land8\t3\tx\n1\t9999999999\trfd4-live\t0\ty\n")
    a = {"kind": "bh", "cards": [], "sweep": True, "paths": [], "outdirs": {}, "book": feed.BOOK,
         "concluded": ["rfd4-land8", "not-booked"]}
    r = subprocess.run([sys.executable, "-c", feed.PROBE], input=json.dumps(a) + "\n", capture_output=True, text=True,
                       env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}, check=True)
    assert json.loads(r.stdout)["unbooked"] == ["rfd4-land8"]
    assert [l.split("\t")[2] for l in w.read_text().splitlines()] == ["rfd4-live"]


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
    assert out["qprio"] == [0, 0, 0, 0, 1, 0, 0, 0, 0]


def test_a_bh_box_full_of_backfill_still_takes_a_lower_digit_cell(tmp_path, monkeypatch):
    """A BH waiter yields its card to a lower digit, so backfill queued past the depth must not lock a prio-3 cell out
    (10-21 qb2: five prio-4/8 waiters, depth 2 x 2 cards, every prio-3 Protenix BH cell refused)."""
    (tmp_path / "state/spd/grids").mkdir(parents=True); (tmp_path / "state/glx").mkdir(parents=True)
    (tmp_path / "state/spd/grids/spd-x.yaml").write_text(BH_GRID.replace("prio: 4", "prio: 3"))
    (tmp_path / "state/glx/ALLOC.json").write_text(json.dumps({"galaxies": {}}))
    monkeypatch.setattr(feed, "BH_POOLS", {"qb2": {"ssh": "qb2", "cards": [2, 3], "lock": "/l2_{c}"}})
    placed = []

    def remote(ssh, script, payload, timeout=120):
        if script is feed.PROBE:
            q = {f"{d}-y-f-b{i}.sh": {"state": "queued", "age": 5} for i, d in enumerate((4, 8, 8, 8, 8))}
            return {"healthy": 2, "cards": [2, 3], "queued": 5, "qprio": [0, 0, 0, 0, 1, 0, 0, 0, 4], "free": [],
                    "paused": False, "swept": 0, "live": [], "jobs": q, "exists": {}}
        placed.extend(payload.get("jobs", []))
        return len(payload.get("jobs", []))
    monkeypatch.setattr(feed, "remote", remote)
    run(tmp_path)
    assert [j["name"][0] for j in placed] == ["3"]


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
                        "cards": [3], "lock": "/l2_{c}", "outs": ["~/o/A"]}]]
    assert json.loads((tmp_path / "state/spd/feed/ledger.json").read_text())[key]["host"] == "qb2"


def _place_bh(tmp_path, cards, body, cap=5, poll="0.2", leg_sid=None, reset="", name="4-x.sh", env=None):
    (tmp_path / "spd/feed/jobs").mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(feed.__file__).parent / "book.py", tmp_path / "spd/feed/book.py")
    # no release-gate leg but the test's own (session leg_sid): the test host's legs (pc runs some) would otherwise
    # hold a card in the waiter's pgrep
    stub = f"exec {shutil.which('pgrep')} -s {leg_sid} \"$@\"" if leg_sid else "exit 1"
    (tmp_path / "bin").mkdir(exist_ok=True); (tmp_path / "bin/pgrep").write_text(f"#!/bin/sh\n{stub}\n")
    (tmp_path / "bin/pgrep").chmod(0o755)
    job = {"name": name, "body": body, "cards": cards, "lock": str(tmp_path / "l{c}"), "cap": cap}
    r = subprocess.run([sys.executable, "-c", feed.PLACE], input=json.dumps({"kind": "bh", "jobs": [job], "reset": reset}) + "\n",
                       capture_output=True, text=True, check=True,
                       env={"HOME": str(tmp_path), "PATH": f"{tmp_path}/bin:" + os.environ["PATH"], "POLL": poll,
                            "TT_SYSFS": str(tmp_path / "sys"), "TT_DEV": "/dev/tt", **(env or {})})
    assert r.stdout.strip() == "1"
    sh = tmp_path / "spd/feed/jobs" / name
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


def test_a_freed_bh_card_goes_to_the_lowest_prio_waiter_not_the_first_placed(tmp_path):
    # 10-20 qb2: prio-8 backfill waiters placed first took both cards for hours while four prio-3 row jobs waited
    import fcntl, threading
    busy = open(tmp_path / "l5", "a"); fcntl.flock(busy, fcntl.LOCK_EX)
    (tmp_path / "spd/feed/jobs").mkdir(parents=True)
    out = {}
    lo = threading.Thread(target=lambda: out.setdefault("lo", _place_bh(tmp_path, [5], "sleep 1\n", name="8-lo.sh")))
    hi = threading.Thread(target=lambda: out.setdefault("hi", _place_bh(tmp_path, [5], "sleep 1\n", name="3-hi.sh")))
    lo.start(); time.sleep(0.6); hi.start(); time.sleep(0.6)
    fcntl.flock(busy, fcntl.LOCK_UN)
    lo.join(); hi.join()
    j = tmp_path / "spd/feed/jobs"
    assert (j / "3-hi.sh.end").read_text().strip() == "0" and (j / "8-lo.sh.end").read_text().strip() == "0"
    assert (j / "3-hi.sh.start").stat().st_mtime < (j / "8-lo.sh.start").stat().st_mtime


def test_a_waiter_held_off_by_a_booking_does_not_block_a_lower_prio_one(tmp_path):
    now = int(time.time())
    _book(tmp_path, (now + 2000, now + 5000, "cfd3-k4", "5"))
    long = _place_bh(tmp_path, [5], "echo card $CHIP\n", cap=5000, name="3-l.sh")
    try:
        sh = _place_bh(tmp_path, [5], "echo card $CHIP\n", name="8-s.sh")
        assert Path(f"{sh}.log").read_text() == "card 5\n"
    finally:
        subprocess.run(["pkill", "-s", Path(f"{long}.pid").read_text()])


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


DARK_GRID = """row: spd-x
tier: red
experiments:
  - name: {name}
    arch: bh
    prio: 4
    hosts: [qb1]
    axes: {{model: [boltz2]}}
    vars: {{out: "~/o/{name}/{{model}}"}}
    done: ["{{out}}/r.json"]
    command: |
      bench --model {{model}}
"""


def test_a_dark_box_pins_clear_except_for_experiments_begun_there(tmp_path, monkeypatch):
    (tmp_path / "state/spd/grids").mkdir(parents=True); (tmp_path / "state/glx").mkdir(parents=True)
    (tmp_path / "state/spd/grids/spd-x.yaml").write_text(DARK_GRID.format(name="new") + DARK_GRID.format(name="begun")
                                                         .split("experiments:\n", 1)[1])
    (tmp_path / "state/glx/ALLOC.json").write_text(json.dumps({"galaxies": {}}))
    monkeypatch.setattr(feed, "BH_POOLS", {"qb1": {"ssh": "qb1", "cards": [0], "lock": "/l1_{c}"},
                                           "qb2": {"ssh": "qb2", "cards": [3], "lock": "/l2_{c}"}})
    cs = {c["key"]: c for c in feed.cells(tmp_path / "state/spd/grids")}
    calls, dark = [], [True]

    def remote(ssh, script, payload, timeout=120):
        calls.append((ssh, script, payload))
        assert ssh == "qb2" or not dark[0], "a dark box is never probed"
        if script is feed.PROBE:
            jobs = {"4-gone.sh": {"state": "queued", "card": 0, "age": 60}} if ssh == "qb1" else {}
            return {"healthy": 4, "cards": [3], "queued": len(jobs), "free": [], "paused": False, "swept": 0, "live": [],
                    "jobs": jobs, "exists": {}}
        if script is feed.WITHDRAW:
            return payload["names"]
        return len(payload["jobs"])
    monkeypatch.setattr(feed, "remote", remote)
    (tmp_path / "state/spd/feed").mkdir(parents=True)
    (tmp_path / "state/qb1-offline").write_text("")
    led = tmp_path / "state/spd/feed/ledger.json"
    b = cs["spd-x/begun/boltz2"]
    led.write_text(json.dumps({b["key"]: {"host": "qb1", "name": b["name"], "state": "done", "hash": b["hash"], "fails": []},
                               "spd-x/gone/x": {"host": "qb1", "name": "4-gone.sh", "state": "queued", "fails": []}}))
    run(tmp_path)
    placed = [j["name"] for x in calls if x[1] is feed.PLACE for j in x[2]["jobs"]]
    assert placed == [cs["spd-x/new/boltz2"]["name"]]
    assert json.loads(led.read_text())[b["key"]]["state"] == "done"
    assert json.loads(led.read_text())["spd-x/gone/x"]["state"] == "lost"
    dark[0] = False; (tmp_path / "state/qb1-offline").unlink(); calls.clear()
    run(tmp_path)   # back from dark: the job of the cell its row dropped meanwhile is withdrawn before it runs
    assert [(x[0], x[2]["names"]) for x in calls if x[1] is feed.WITHDRAW] == [("qb1", ["4-gone.sh"])]


def test_a_pinned_experiment_waits_for_its_dark_box(tmp_path, monkeypatch):
    (tmp_path / "state/spd/grids").mkdir(parents=True); (tmp_path / "state/glx").mkdir(parents=True)
    (tmp_path / "state/spd/grids/spd-x.yaml").write_text(DARK_GRID.format(name="lad").replace("prio: 4", "prio: 4\n    pinned: true"))
    (tmp_path / "state/glx/ALLOC.json").write_text(json.dumps({"galaxies": {}}))
    monkeypatch.setattr(feed, "BH_POOLS", {"qb1": {"ssh": "qb1", "cards": [0], "lock": "/l1_{c}"},
                                           "qb2": {"ssh": "qb2", "cards": [3], "lock": "/l2_{c}"}})
    placed = []

    def remote(ssh, script, payload, timeout=120):
        if script is feed.PROBE:
            return {"healthy": 4, "cards": [3], "queued": 0, "free": [], "paused": False, "swept": 0, "live": [],
                    "jobs": {}, "exists": {}}
        placed.extend(payload["jobs"]); return len(payload["jobs"])
    monkeypatch.setattr(feed, "remote", remote)
    (tmp_path / "state/qb1-offline").write_text("")
    run(tmp_path)
    assert placed == []


def sysfs(tmp_path, cards):
    """Fake /sys/class/tenstorrent: {bdf: (node, aiclk)}; logical id = PCI order."""
    for bdf, (node, clk) in cards.items():
        d = tmp_path / "sys" / f"tenstorrent!{node}"; dev = tmp_path / "pci" / bdf
        d.mkdir(parents=True, exist_ok=True); dev.mkdir(parents=True, exist_ok=True)
        if not (d / "device").exists():
            (d / "device").symlink_to(dev)
        (d / "tt_aiclk").write_text(clk + "\n"); (d / "tt_heartbeat").write_text(("4294967295" if clk == "4294967295" else "77") + "\n")


def _run_capped(tmp_path, body, cap, reset="", name="4-x.sh", init=True):
    sh = tmp_path / name; sh.write_text(("echo 'TT_THROW: failed to initialize FW! Try resetting the board'\n" if init else "") + body)
    sysfs(tmp_path, {"0000:42:00.0": (3, "1350")})
    subprocess.run(["bash", "-c", feed.NODE + feed.run_cmd(grace=2), "_", str(sh), "0", "", str(tmp_path / "nolock"),
                    str(cap), reset], cwd=tmp_path, timeout=60, check=True,
                   env=dict(os.environ, TT_SYSFS=str(tmp_path / "sys"), TT_DEV="/dev/tt", SPD_LOCKS=str(tmp_path / "locks")))
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
    assert _run_capped(tmp_path, "exit 1\n", 30, str(r), "4-p.sh", init=False) == "1"
    assert _run_capped(tmp_path, "exit 1\n", 30, str(r), "4-q.sh", init=False) == "1"   # two broken cells: not the card
    assert _run_capped(tmp_path, "exit 0\n", 30, str(r), "4-r.sh") == "0"
    assert _run_capped(tmp_path, "exit 1\n", 30, str(r), "4-a.sh") == "1"
    assert _run_capped(tmp_path, "exit 1\n", 30, str(r), "4-a.sh") == "1"        # the same job again: the job's fault
    assert _run_capped(tmp_path, "exit 1\n", 30, str(r), "4-b.sh") == "1 fastfail2 reset"
    assert (tmp_path / "resets").read_text() == "/dev/tt/3\n"
    assert _run_capped(tmp_path, "exit 1\n", 30, str(r), "4-c.sh") == "1"        # the reset cleared the streak
    assert _run_capped(tmp_path, "exit 0\n", 30, str(r), "4-d.sh") == "0"        # a clean exit clears it too
    assert _run_capped(tmp_path, "exit 1\n", 30, str(r), "4-e.sh") == "1"
    assert _run_capped(tmp_path, "exit 127\n", 30, str(r), "4-f.sh") == "127"    # a missing command is not the card
    assert _run_capped(tmp_path, "exit 1\n", 30, "", "4-g.sh") == "1 fastfail2 bad"  # qb2: no reset, chip off the feed
    assert "device init" in (tmp_path / "locks/chip0.BAD").read_text()           # qb2: off the feed instead
    assert (tmp_path / "resets").read_text() == "/dev/tt/3\n"


def test_an_init_failure_written_to_the_cells_own_log_counts(tmp_path):
    # 10-16 qb2 board (0,1): every cell sent the model's output to $O/log, so 12 init failures in 7 min left the chips on
    # the feed and quarantined four cells for the board's fault
    o = tmp_path / "out"; o.mkdir()
    (o / "log").write_text("TT_THROW: Device 0: Timed out while waiting for active ethernet core (x=29,y=25)\n")
    (tmp_path / "stale").mkdir(); (tmp_path / "stale/log").write_text("failed to initialize FW\n")
    def run(name, d, body):
        (tmp_path / f"{name}.outs").write_text(f"{d}\n"); time.sleep(0.02); (tmp_path / f"{name}.start").touch()
        return _run_capped(tmp_path, body, 30, "", name, init=False)
    fail = f"echo 'TT_THROW: Device 0: Timed out while waiting for active ethernet core' > {o}/log; exit 1\n"
    assert run("4-s.sh", tmp_path / "stale", "exit 1\n") == "1"     # a log from before the job started is not its own
    assert run("4-t.sh", tmp_path / "stale", "exit 1\n") == "1"
    assert not (tmp_path / "locks/chip0.BAD").exists()
    assert run("4-a.sh", o, fail) == "1"
    assert run("4-b.sh", o, fail) == "1 fastfail2 bad"


def test_a_placed_qb1_job_carries_the_reset_command(tmp_path):
    # 10-16: the waiter's `set --` dropped the reset argument, so no capped or fastfail2 reset ever ran on qb1
    # (six "124 INT+TERM" ends, none followed by a reset); the run_cmd tests passed it directly and stayed green
    sysfs(tmp_path, {"0000:42:00.0": (3, "1350")})
    r = tmp_path / "reset.sh"; r.write_text('echo "$1" >> "$(dirname "$0")/resets"\n'); r.chmod(0o755)
    fail = "echo 'failed to initialize FW! Try resetting the board'; exit 1\n"
    assert Path(f"{_place_bh(tmp_path, [0], fail, reset=str(r), name='4-a.sh')}.end").read_text().strip() == "1"
    assert Path(f"{_place_bh(tmp_path, [0], fail, reset=str(r), name='4-b.sh')}.end").read_text().strip() == "1 fastfail2 reset"
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


def _sentinels(names):
    # one session of processes named `names`; the names travel in env so the wrapper's own argv matches nothing
    env = dict(os.environ, **{f"N{i}": n for i, n in enumerate(names)})
    sh = " ".join(f'(exec -a "$N{i}" sleep 30) &' for i in range(len(names))) + " wait"
    return subprocess.Popen(["bash", "-c", sh], env=env, start_new_session=True)


def _book(tmp_path, *rows):
    (tmp_path / "spd/feed").mkdir(parents=True, exist_ok=True)
    (tmp_path / "spd/feed/windows.tsv").write_text("".join(f"{s}\t{e}\t{o}\t{c}\t\n" for s, e, o, c in rows))


def test_a_sleep_sentinel_blocks_its_card_only_for_a_while(tmp_path):
    # 10-16 qb1: cfd3 rt4 re-raised four release_gate sentinels every 5 min and cards 0-2 idled for hours behind them
    leg = _sentinels(["cfd3-rt4 spd_qb2_chip5.lock release_gate rt4 until=1 86400"])
    try:
        time.sleep(0.3)
        sh = _place_bh(tmp_path, [5, 6], "echo card $CHIP\n", leg_sid=leg.pid, name="4-a.sh")
        assert Path(f"{sh}.log").read_text() == "card 6\n"
        time.sleep(1.2)
        sh = _place_bh(tmp_path, [5], "echo card $CHIP\n", leg_sid=leg.pid, name="4-b.sh", env={"SENT_S": "1"})
        assert Path(f"{sh}.log").read_text() == "card 5\n"
    finally:
        os.killpg(leg.pid, 9)


def test_a_bh_waiter_takes_a_booked_card_only_if_it_ends_before_the_booking(tmp_path):
    now = int(time.time())
    _book(tmp_path, (now - 10, now + 3600, "cfd3-k4", "5"), (now + 2000, now + 5000, "cfd3-k4", "6"))
    sh = _place_bh(tmp_path, [5, 6, 7], "echo card $CHIP\n", name="4-a.sh")      # ends by now + 905
    assert Path(f"{sh}.log").read_text() == "card 6\n"
    sh = _place_bh(tmp_path, [5, 6, 7], "echo card $CHIP\n", cap=3000, name="4-b.sh")
    assert Path(f"{sh}.log").read_text() == "card 7\n"


def test_a_waiter_too_long_for_a_booking_gap_lets_a_shorter_one_through(tmp_path):
    # the waiter holding the dispatch flock is too long for chip 5's gap: it hands the flock on each poll
    now = int(time.time())
    _book(tmp_path, (now + 2000, now + 5000, "cfd3-k4", "5"))
    long = _place_bh(tmp_path, [5], "echo card $CHIP\n", cap=5000, name="4-l.sh")
    try:
        assert not Path(f"{long}.start").exists()
        sh = _place_bh(tmp_path, [5], "echo card $CHIP\n", name="4-s.sh")
        assert Path(f"{sh}.log").read_text() == "card 5\n"
        # and it keeps waiting: 10-16 05:24Z qb1 eight such waiters closed the flock's fd and died "flockbusy" one poll later
        time.sleep(1)
        assert not Path(f"{long}.end").exists() and not Path(f"{long}.start").exists()
    finally:
        subprocess.run(["pkill", "-s", Path(f"{long}.pid").read_text()])


def _running(tmp_path, name, card, until):
    j = tmp_path / "spd/feed/jobs"; j.mkdir(parents=True, exist_ok=True)
    (j / name).write_text(""); (j / f"{name}.card").write_text(f"{card}\n"); (j / f"{name}.start").touch()
    (j / f"{name}.until").write_text(f"{until}\n")


def _bk(tmp_path, *a):
    r = subprocess.run([sys.executable, str(Path(feed.__file__).parent / "book.py"), *map(str, a)], capture_output=True,
                       text=True, env=dict(os.environ, HOME=str(tmp_path)))
    return r.returncode, r.stdout.strip()


def test_book_starts_a_window_after_the_feed_jobs_and_other_bookings_on_its_cards(tmp_path):
    now = int(time.time())
    _running(tmp_path, "4-g.sh", 1, now + 1000); _running(tmp_path, "4-h.sh", 9, now + 99999)
    _book(tmp_path, (now + 1500, now + 2000, "rel", "2"), (now + 2100, now + 2200, "x", "3"), (now - 5, now + 9000, "y", "8"))
    rc, out = _bk(tmp_path, "add", "cfd3-k4", "0,1,2,3", 600)
    s, e = map(int, out.split())
    assert rc == 0 and s == now + 2200 and e == s + 600                 # 1000 -> past rel (2000) -> past x (2200)
    assert _bk(tmp_path, "next", 1) == (0, str(s)) and _bk(tmp_path, "next", 1, "--except", "cfd3-k4")[1] == "9999999999"
    assert _bk(tmp_path, "active", 8) == (0, "y") and _bk(tmp_path, "active", 0)[0] == 1
    # re-booking replaces the owner's line; a hold (--now) starts now whatever is booked
    assert int(_bk(tmp_path, "add", "cfd3-k4", "5", 60)[1].split()[0]) - now < 5
    assert int(_bk(tmp_path, "add", "gapboard-c8", "8", 60, "--now")[1].split()[0]) - now < 5
    assert _bk(tmp_path, "active", 8)[1] == "gapboard-c8 y" or _bk(tmp_path, "active", 8)[1] == "y gapboard-c8"
    _bk(tmp_path, "rm", "cfd3-k4")
    assert "cfd3-k4" not in (tmp_path / "spd/feed/windows.tsv").read_text()


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


SHORT = """
row: spd-x
tier: red
experiments:
  - name: s
    arch: wh
    prio: 4
    axes: {seed: [1, 2, 3, 4, 5, 6]}
    vars: {out: "~/o/{seed}"}
    done: ["{out}/r.json"]
    command: |
      run.sh {out}
"""


def finish(box, dur):
    for n in box.queue:
        box.exists.add(f"~/o/{n.split('-')[5]}/r.json")
        box.ended[n] = {"state": "ended", "dur": dur, "rc": "0", "at": 1, "tail": ""}
    box.queue = []


def test_cells_that_run_shorter_than_a_pass_queue_deeper(tmp_path, monkeypatch):
    box = make_pool(tmp_path, monkeypatch, healthy=1)
    (tmp_path / "state/spd/grids/spd-x.yaml").write_text(SHORT)
    run(tmp_path)
    assert len(box.queue) == 2                         # no run time known yet: depth 2 x 1 chip
    finish(box, 60)
    run(tmp_path)
    assert len(box.queue) == 4                         # one-minute cells weigh 0.1: the rest of the grid fits
    ledger = json.loads((tmp_path / "state/spd/feed/ledger.json").read_text())
    assert ledger["spd-x/s/1"]["dur"] == 60


def test_cells_that_run_a_full_pass_keep_the_plain_depth(tmp_path, monkeypatch):
    box = make_pool(tmp_path, monkeypatch, healthy=1)
    (tmp_path / "state/spd/grids/spd-x.yaml").write_text(SHORT)
    run(tmp_path)
    finish(box, 900)
    run(tmp_path)
    assert len(box.queue) == 2


def test_a_chip_lending_japanfold_box_is_a_pool_sized_by_its_lent_chips(tmp_path, monkeypatch):
    """.107 serves JapanFold and lends idle chips: feed places WH cells there, counts only chips lent now (not recalled),
    and its RETURNED marker does not pause it (only PAUSE does)."""
    (tmp_path / "state/glx").mkdir(parents=True)
    (tmp_path / "state/glx/ALLOC.json").write_text(json.dumps({"galaxies": {
        "7": {"owner": "japanfold", "phase": "serving"}, "8": {"owner": "japanfold", "phase": "serving"}}}))
    monkeypatch.setattr(feed, "BH_POOLS", {})
    monkeypatch.setattr(feed, "CHIP_LENDING", {"7"})
    assert feed.pools(tmp_path) == {"7": {"name": "7", "ssh": "ubuntu@172.16.102.7", "kind": "galaxy", "arch": "wh", "lend": True}}
    L = tmp_path / "japanfold/state/agent/lend"; L.mkdir(parents=True)
    (L / "0.lent").write_text(json.dumps({"recall_at": None, "node": "16"}))
    (L / "1.lent").write_text(json.dumps({"recall_at": 1.0, "node": "17"}))
    pool = tmp_path / "spd/pool"; (pool / "queue").mkdir(parents=True); (pool / "running").mkdir()
    (pool / "RETURNED").touch()
    a = {"kind": "galaxy", "lend": True, "sweep": False, "paths": [], "outdirs": {}}
    r = subprocess.run([sys.executable, "-c", feed.PROBE], input=json.dumps(a) + "\n", capture_output=True, text=True,
                       env={"HOME": str(tmp_path)}, check=True)
    out = json.loads(r.stdout)
    assert (out["healthy"], out["paused"]) == (1, False)
    (pool / "PAUSE").touch()
    r = subprocess.run([sys.executable, "-c", feed.PROBE], input=json.dumps(a) + "\n", capture_output=True, text=True,
                       env={"HOME": str(tmp_path)}, check=True)
    assert json.loads(r.stdout)["paused"] is True


LEAD = """row: spd-a
experiments:
  - name: ok
    axes: {seed: ["1", "2"]}
    done: ["/tmp/x/{seed}/done"]
    command: echo {seed}
  - name: hiprio
    prio: 1
    done: ["/tmp/y"]
    command: echo y
"""


def test_a_bad_grid_file_skips_alone_and_prio_clamps(tmp_path):
    (tmp_path / "a.yaml").write_text(LEAD)
    (tmp_path / "b.yaml").write_text("row: [unclosed\n")
    (tmp_path / "c.yaml").write_text("experiments: []\n")   # no row
    warned = []
    cs = feed.cells(tmp_path, log=warned.append)
    prio = {c["key"]: int(c["name"].split("-")[0]) for c in cs}
    assert sorted(prio) == ["spd-a/hiprio/x", "spd-a/ok/1", "spd-a/ok/2"]
    assert prio["spd-a/hiprio/x"] == 3
    assert sum(w.startswith("skip ") for w in warned) == 2, warned
    lead = {c["key"]: int(c["name"].split("-")[0]) for c in feed.cells(tmp_path, log=lambda m: None, lead={"spd-a"})}
    assert lead["spd-a/hiprio/x"] == 2


def test_lead_rows_are_live_rows_dedicated_to_a_directive(tmp_path):
    (tmp_path / "state/spd/steer").mkdir(parents=True); (tmp_path / "state/concluded").mkdir()
    (tmp_path / "state/spd/steer/config.json").write_text(json.dumps(
        {"groups": {"protenix": ["protenix-v2"], "esmc": ["esmc-300m"]}, "directives": {"protenix": {"x": 5}}}))
    (tmp_path / "state/spd/OWNERS.tsv").write_text(
        "# slug kind models\nspd-m-protenix\tcell\tprotenix-v2\nspd-bet-old\tbet\tprotenix-v2\n"
        "spd-shared\tcell\tprotenix-v2,esmc-300m\nspd-m-esmc\tcell\tesmc-300m\n")
    (tmp_path / "state/concluded/spd-bet-old").touch()
    assert feed.lead_rows(tmp_path) == {"spd-m-protenix"}
    assert feed.lead_rows(tmp_path / "missing") == set()


def test_bench_jsonl_is_done_only_with_a_rep(tmp_path):
    """An arm that crashed at device open leaves a bench.jsonl holding only its "start" record: not done."""
    src = feed.PROBE[feed.PROBE.index("def ok(p):"):feed.PROBE.index('for p in A["paths"]')]
    ns = {"os": os, "glob": __import__("glob")}
    exec(src, ns)
    ok = ns["ok"]
    b = tmp_path / "w" / "bench.jsonl"
    b.parent.mkdir()
    assert not ok(str(b))
    b.write_text(json.dumps({"ev": "start", "argv": []}) + "\n" + json.dumps({"ev": "device_open", "s": 1.0}) + "\n")
    assert not ok(str(b)) and not ok(str(tmp_path / "*" / "bench.jsonl"))
    b.write_text(b.read_text() + json.dumps({"ev": "rep", "input": "x", "fold_s": 2.0}) + "\n")
    assert ok(str(b)) and ok(str(tmp_path / "*" / "bench.jsonl"))
    (tmp_path / "ok").touch()
    assert ok(str(tmp_path / "ok")), "an empty marker is still done"
