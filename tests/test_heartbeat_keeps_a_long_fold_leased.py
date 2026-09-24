"""A fold longer than the lease must still finish, on the worker that started it.

Seen on the JapanFold dev Galaxies, 2026-09-24: a 1792-residue Protenix-v2 fold
leased to tt11 at 08:42 was leased again to tt1 at 09:12, exactly LEASE_SECONDS
later, while tt11 was still in its trunk. tt11 saved its structure at 09:27 and
`complete_job` dropped it because the lease had moved; tt1 would have been
replaced the same way at 09:42. The heartbeat thread was pinging every 8 s the
whole time, but a heartbeat only refreshed the worker row, never the job's lease.

Host-only: the controller store on a temp sqlite file, time moved by monkeypatch.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tt_bio import distributed as D  # noqa: E402


def _worker(wid: str) -> dict:
    return {"worker_id": wid, "host": "h", "accelerator": "cpu", "device_id": 0, "label": wid}


def _store(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(D.time, "time", lambda: clock[0])
    store = D.ControllerStore(tmp_path / "c.sqlite3")
    run = store.create_run({"data": "d", "out_dir": "o", "result_dir": "r",
                            "config": {"model": "protenix-v2"},
                            "jobs": [{"id": "big", "name": "big.yaml"}]})["run_id"]
    return store, run, clock


def test_a_heartbeating_worker_keeps_its_job_and_its_result_counts(tmp_path, monkeypatch):
    store, run, clock = _store(tmp_path, monkeypatch)
    assert [j["id"] for j in store.lease({"worker": _worker("tt11")})["jobs"]] == ["big"]
    # 45 minutes of computing, heartbeating every few minutes, another worker polling.
    for _ in range(9):
        clock[0] += 300
        store.heartbeat({"worker": _worker("tt11")})
        assert store.lease({"worker": _worker("tt1")})["jobs"] == []
    store.complete_job({"run_id": run, "worker_id": "tt11",
                        "result": {"id": "big", "status": "ok"}})
    assert store.run_jobs(run) == [{"id": "big", "status": "ok", "stage": None}]


def test_a_silent_worker_still_loses_its_job(tmp_path, monkeypatch):
    store, run, clock = _store(tmp_path, monkeypatch)
    store.lease({"worker": _worker("tt11")})
    store.heartbeat({"worker": _worker("tt1")})   # somebody else's heartbeat renews nothing
    clock[0] += D.LEASE_SECONDS + 1
    assert [j["id"] for j in store.lease({"worker": _worker("tt1")})["jobs"]] == ["big"]
