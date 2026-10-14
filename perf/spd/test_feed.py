"""perf/spd/feed.py without a box: fake pools, check fill order, done-skip, twin-skip and the fast-fail quarantine."""
import json
import subprocess
import sys
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
    """`touch $O/done` is a finished cell (an empty result file is not), and a job that logs into its out dir gets
    that log tailed into the quarantine entry."""
    o = tmp_path / "o"; o.mkdir()
    (o / "done").touch(); (o / "r.json").touch(); (o / "run.log").write_text("Traceback: boom\n" * 20)
    jobs = tmp_path / "spd/feed/jobs"; jobs.mkdir(parents=True)
    (jobs / "4-x.sh").write_text(""); (jobs / "4-x.sh.end").write_text("0\n"); (jobs / "4-x.sh.log").write_text("")
    a = {"kind": "bh", "cards": [], "sweep": False, "paths": [str(o / "done"), str(o / "r.json")],
         "outdirs": {"4-x.sh": [str(o)]}}
    r = subprocess.run([sys.executable, "-c", feed.PROBE], input=json.dumps(a) + "\n", capture_output=True, text=True,
                       env={"HOME": str(tmp_path)}, check=True)
    out = json.loads(r.stdout)
    assert out["exists"] == {str(o / "done"): True, str(o / "r.json"): False}
    assert "Traceback: boom" in out["jobs"]["4-x.sh"]["tail"]
