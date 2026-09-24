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


def test_a_canceled_run_is_not_reopened_by_a_completion(tmp_path, clock):
    store = H.ControllerStore(tmp_path / "c.sqlite3")
    run = _run(store, "x")
    store.lease({"worker": _worker("a")})
    store.cancel_run(run)
    store.complete_job({"run_id": run, "worker_id": "a", "result": {"id": "x", "status": "ok"}})
    assert store.run_status(run) == {"status": "canceled"}


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
