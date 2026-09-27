"""A worker is handed the work its resident weights already serve.

The controller ranks a waiting job by (the owner's fair share, whether this worker would have
to reload for it, age). "Would have to reload" is the same key the worker's own reload
decision uses, run_config_hash, which it reports as ``warm``. It used to be the model name,
which missed Boltz-2 entirely: its run config carries no ``model`` key, so its runs were
stored as model None and a worker holding Boltz-2 never matched them.
"""
import time

import pytest

from tt_bio import host_controller as H

BOLTZ2 = {"conf_kwargs": {"x": 1}, "aff_kwargs": {}, "fast": False}   # as main.predict builds it
PROTENIX = {"model": "protenix-v2", "fast": False}


def _worker(i, warm=None):
    return {"worker_id": i, "host": "h", "accelerator": "tenstorrent", "device_id": i,
            "label": i, "model": None, "warm": warm}


def _run(store, run_id, config, owner="o"):
    store.create_run({"run_id": run_id, "data": "d", "out_dir": "o", "result_dir": "r",
                      "owner": owner, "config": config,
                      "jobs": [{"id": f"{run_id}-1", "name": "t.yaml"}]})


def _next(store, worker):
    return store.lease({"worker": worker}).get("run_id")


def test_a_warm_boltz2_worker_takes_boltz2_over_an_older_job(tmp_path):
    store = H.ControllerStore(tmp_path / "c.sqlite3")
    _run(store, "protenix", PROTENIX)
    time.sleep(0.01)
    _run(store, "boltz", BOLTZ2)
    assert _next(store, _worker("a", warm=H.run_config_hash(BOLTZ2))) == "boltz"


def test_a_cold_worker_takes_the_oldest(tmp_path):
    store = H.ControllerStore(tmp_path / "c.sqlite3")
    _run(store, "protenix", PROTENIX)
    time.sleep(0.01)
    _run(store, "boltz", BOLTZ2)
    assert _next(store, _worker("a")) == "protenix"


def test_fast_is_a_different_warm_model(tmp_path):
    """--fast loads other weights under the same name, so it is not warm for the plain run."""
    store = H.ControllerStore(tmp_path / "c.sqlite3")
    _run(store, "plain", PROTENIX)
    time.sleep(0.01)
    _run(store, "fast", {**PROTENIX, "fast": True})
    assert _next(store, _worker("a", warm=H.run_config_hash({**PROTENIX, "fast": True}))) == "fast"


def test_fairness_still_outranks_a_warm_model(tmp_path):
    """An owner already on a chip waits behind one who has none, warm or not."""
    store = H.ControllerStore(tmp_path / "c.sqlite3")
    _run(store, "mine", BOLTZ2, owner="busy")
    store.create_run({"run_id": "mine2", "data": "d", "out_dir": "o", "result_dir": "r",
                      "owner": "busy", "config": BOLTZ2,
                      "jobs": [{"id": "m2", "name": "t.yaml"}]})
    assert _next(store, _worker("x")) == "mine"                   # busy now holds a chip
    _run(store, "theirs", PROTENIX, owner="idle")
    assert _next(store, _worker("a", warm=H.run_config_hash(BOLTZ2))) == "theirs"


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(H.time, "time", lambda: now[0])
    return now


WARM = H.run_config_hash(BOLTZ2)


def test_a_cold_worker_leaves_the_job_to_an_idle_warm_one(tmp_path, clock):
    """The case that reloaded production 31 times in 35: many idle chips, one of them warm."""
    store = H.ControllerStore(tmp_path / "c.sqlite3")
    store.heartbeat({"worker": _worker("warm", warm=WARM)})     # idle, polling
    _run(store, "boltz", BOLTZ2)
    clock[0] += 0.5
    assert _next(store, _worker("cold")) is None
    assert _next(store, _worker("warm", warm=WARM)) == "boltz"


def test_a_busy_warm_worker_does_not_hold_the_job_back(tmp_path, clock):
    store = H.ControllerStore(tmp_path / "c.sqlite3")
    _run(store, "first", BOLTZ2)
    assert _next(store, _worker("warm", warm=WARM)) == "first"   # now busy
    _run(store, "second", BOLTZ2)
    assert _next(store, _worker("cold")) == "second"


def test_a_silent_warm_worker_does_not_hold_the_job_back(tmp_path, clock):
    store = H.ControllerStore(tmp_path / "c.sqlite3")
    store.heartbeat({"worker": _worker("warm", warm=WARM)})
    _run(store, "boltz", BOLTZ2)
    clock[0] += H.IDLE_WARM_S + 1                               # it stopped polling
    assert _next(store, _worker("cold")) == "boltz"


def test_a_cold_worker_still_takes_work_no_idle_chip_is_warm_for(tmp_path, clock):
    store = H.ControllerStore(tmp_path / "c.sqlite3")
    store.heartbeat({"worker": _worker("warm", warm=WARM)})
    _run(store, "boltz", BOLTZ2)
    _run(store, "protenix", PROTENIX)
    assert _next(store, _worker("cold")) == "protenix"
    assert _next(store, _worker("warm", warm=WARM)) == "boltz"


def test_the_worker_reloads_on_exactly_the_key_it_reports():
    from tt_bio.worker import _WorkerState

    state = object.__new__(_WorkerState)
    state.model, state.config_hash = object(), H.run_config_hash(BOLTZ2)
    assert state.configured_for(BOLTZ2)
    assert not state.configured_for({**BOLTZ2, "fast": True})
