"""The worker contract docs/multi-host.md publishes, held to the controller that ships it.

A platform or a scheduler of someone else's is built against that document, not against
this module, so each rule it states is checked here: the lease the controller names and
the worker paces itself to, what a host advertises, and settlement exactly once.
Host-only: the store on a temp sqlite file, time moved by monkeypatch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from click.testing import CliRunner  # noqa: E402

from tt_bio import host_controller as H  # noqa: E402


def _worker(wid: str, model: str | None = None) -> dict:
    return {"worker_id": wid, "host": "h", "accelerator": "cpu", "device_id": 0,
            "label": wid, "model": model}


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(H.time, "time", lambda: now[0])
    return now


def _run(store, *ids):
    return store.create_run({"data": "d", "out_dir": "o", "result_dir": "r",
                             "config": {"model": "esmfold2"},
                             "jobs": [{"id": i, "name": f"{i}.fasta"} for i in ids]})["run_id"]


def test_the_lease_is_named_in_every_answer_a_worker_paces_itself_by(tmp_path, clock):
    store = H.ControllerStore(tmp_path / "c.sqlite3", lease_s=30)
    assert store.lease({"worker": _worker("a")}) == {"jobs": [], "lease_s": 30}
    _run(store, "x")
    assert store.lease({"worker": _worker("a")})["lease_s"] == 30
    assert store.heartbeat({"worker": _worker("a")}) == {"ok": True, "lease_s": 30}


def test_the_default_lease_is_two_minutes_renewed_every_ten_seconds():
    assert H.LEASE_S == 120.0
    assert H.LEASE_S / H.HEARTBEAT_PER_LEASE == 10.0


def test_a_configured_lease_is_the_one_that_lapses(tmp_path, clock):
    store = H.ControllerStore(tmp_path / "c.sqlite3", lease_s=30)
    _run(store, "x")
    store.lease({"worker": _worker("a")})
    clock[0] += 29
    assert store.lease({"worker": _worker("b")})["jobs"] == []
    clock[0] += 2
    assert [j["id"] for j in store.lease({"worker": _worker("b")})["jobs"]] == ["x"]


def test_a_host_advertises_each_workers_model_and_the_jobs_it_holds(tmp_path, clock):
    store = H.ControllerStore(tmp_path / "c.sqlite3")
    run = _run(store, "x")
    store.lease({"worker": _worker("a", model="esmfold2")})
    store.heartbeat({"worker": _worker("b")})
    c = store.cluster()
    by_id = {w["worker_id"]: w for w in c["workers"]}
    assert c["online_workers"] == 2
    assert by_id["a"]["model"] == "esmfold2"
    assert by_id["a"]["running"] == [{"run_id": run, "job_id": "x"}]
    assert by_id["b"]["running"] == []
    clock[0] += 21                      # past stale_after: the host no longer counts them
    assert store.cluster()["online_workers"] == 0


def test_a_result_settles_once_and_only_from_the_lease_holder(tmp_path, clock):
    store = H.ControllerStore(tmp_path / "c.sqlite3")
    run = _run(store, "x")
    store.lease({"worker": _worker("a")})
    clock[0] += H.LEASE_S + 1           # a stopped heartbeating; b takes the job over
    assert [j["id"] for j in store.lease({"worker": _worker("b")})["jobs"]] == ["x"]
    store.complete_job({"run_id": run, "worker_id": "a",
                        "result": {"id": "x", "status": "failed", "error": "late"}})
    assert store.run_jobs(run)[0]["status"] == "running"     # the late answer changed nothing
    for _ in range(2):                  # a replayed completion settles nothing twice
        store.complete_job({"run_id": run, "worker_id": "b", "result": {"id": "x", "status": "ok"}})
    assert store.run_status(run) == {"status": "ok"}
    assert [e["event"] for e in store.events(run, 0)["events"]].count("run_done") == 1
    assert store.results(run) == [{"id": "x", "status": "ok"}]


def test_a_job_that_outlives_two_leases_fails_instead_of_hanging_a_third_chip(tmp_path, clock):
    """A Boltz-2 input that hung the device was handed to ~10 chips in turn until JapanFold's
    25 min cap (10-19). The second lapse now ends it, and the run with it."""
    store = H.ControllerStore(tmp_path / "c.sqlite3")
    run = _run(store, "x")
    for wid in ("a", "b"):
        assert [j["id"] for j in store.lease({"worker": _worker(wid)})["jobs"]] == ["x"]
        clock[0] += H.LEASE_S + 1
    assert store.lease({"worker": _worker("c")})["jobs"] == []
    assert store.run_status(run) == {"status": "failed"}
    [result] = store.results(run)
    assert result["status"] == "failed" and "2 workers" in result["error"]
    assert [e["event"] for e in store.events(run, 0)["events"]].count("run_done") == 1
    store.complete_job({"run_id": run, "worker_id": "b", "result": {"id": "x", "status": "ok"}})
    assert store.run_status(run) == {"status": "failed"}     # a late answer reopens nothing


def test_a_worker_asking_for_work_gives_back_what_it_still_held(tmp_path, clock):
    """Its completion never arrived; without this its heartbeats renewed the job forever."""
    store = H.ControllerStore(tmp_path / "c.sqlite3")
    run = _run(store, "x", "y")
    assert [j["id"] for j in store.lease({"worker": _worker("a")})["jobs"]] == ["x"]
    assert [j["id"] for j in store.lease({"worker": _worker("a")})["jobs"]] == ["x"]
    store.heartbeat({"worker": _worker("a")})
    assert [j["id"] for j in store.lease({"worker": _worker("b")})["jobs"]] == ["y"]
    assert {j["id"]: j["status"] for j in store.run_jobs(run)} == {"x": "running", "y": "running"}


def test_a_canceled_run_is_not_reopened_by_a_completion(tmp_path, clock):
    store = H.ControllerStore(tmp_path / "c.sqlite3")
    run = _run(store, "x")
    store.lease({"worker": _worker("a")})
    store.cancel_run(run)
    store.complete_job({"run_id": run, "worker_id": "a", "result": {"id": "x", "status": "ok"}})
    assert store.run_status(run) == {"status": "canceled"}


def test_a_run_created_again_under_its_id_runs_its_jobs_again(tmp_path, clock):
    """JapanFold passes its job id as the run id, so a retry that lands on the same host re-creates
    the run. The first attempt's rows and events must not answer for it (job 9c1aec9f, 10-11: the
    retry found its job already `ok`, nothing leased it, and it stalled out after 10 minutes)."""
    store = H.ControllerStore(tmp_path / "c.sqlite3")
    payload = {"run_id": "job1", "data": "d", "out_dir": "o", "result_dir": "r",
               "config": {"model": "opendde"}, "jobs": [{"id": "x", "name": "x.yaml"}]}
    store.create_run(payload)
    store.lease({"worker": _worker("a")})
    store.complete_job({"run_id": "job1", "worker_id": "a", "result": {"id": "x", "status": "ok"}})
    assert store.run_status("job1") == {"status": "ok"}
    store.create_run(payload)
    assert store.run_status("job1") == {"status": "running"}
    assert [e["event"] for e in store.events("job1", 0)["events"]] == ["run"]
    assert [j["id"] for j in store.lease({"worker": _worker("b")})["jobs"]] == ["x"]


def test_the_controller_serves_loopback_only(tmp_path):
    server = H.ControllerServer(0, tmp_path / "c.sqlite3")
    try:
        assert server.httpd.server_address[0] == "127.0.0.1"
    finally:
        server.httpd.server_close()


def test_no_command_offers_to_bind_another_interface():
    from tt_bio.main import cli

    for cmd in ("controller", "predict"):
        out = CliRunner().invoke(cli, [cmd, "--help"]).output
        assert "--listen" not in out, cmd
    assert "--lease-s" in CliRunner().invoke(cli, ["controller", "--help"]).output



def test_no_lease_heartbeat_or_poll_scans_a_table_that_grows_all_day(tmp_path):
    # A store keeps every job and event for its agent's life. A lease that scanned jobs took 50 ms after 18 h on a
    # JapanFold agent, 32 workers saturated the lock and served folds slowed by a third. Each lookup must reach its
    # rows through an index, including on a store opened on a file written before the indexes existed.
    path = tmp_path / "c.sqlite3"
    H.ControllerStore(path)
    with H.sqlite3.connect(path) as conn:
        for name in ("jobs_worker_status", "jobs_status_lease", "events_run_seq"):
            conn.execute(f"DROP INDEX {name}")
    store = H.ControllerStore(path)
    run_id = _run(store, "x")
    lookups = {
        "SELECT 1 FROM jobs j WHERE j.worker_id = ? AND j.status = 'running' AND j.lease_until >= ?": ("w", 0),
        "UPDATE jobs SET lease_until=? WHERE worker_id=? AND status='running'": (0, "w"),
        "SELECT run_id, job_id FROM jobs WHERE status='running' AND lease_until < ? AND attempts >= ?": (0, 2),
        "SELECT job_id FROM jobs WHERE status = 'pending' OR (status = 'running' AND lease_until < ?)": (0,),
        "SELECT seq, event_json FROM events WHERE run_id=? AND seq>? ORDER BY seq LIMIT 500": (run_id, 0),
    }
    with store._connect() as conn:
        for sql, args in lookups.items():
            plan = " ".join(r[3] for r in conn.execute("EXPLAIN QUERY PLAN " + sql, args))
            assert "SCAN" not in plan, (sql, plan)
    assert [e["event"] for e in store.events(run_id, 0)["events"]] == ["run"]
