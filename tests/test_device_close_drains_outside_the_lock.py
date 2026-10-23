"""A hung chip must not hold the host's device-open lock while it fails to close.

`device_init_lock` serialises every device open on the host. `_close_device_locked` used to take
it before `ttnn.close_device`, whose drain waits out the op timeout on a hung chip and whose close
then spins on the same dead queue. On a Galaxy (2026-10-23) one hung OpenDDE cell held the lock for
over 15 minutes that way and 13 opens on healthy chips, JapanFold's chip checks among them, queued
behind it. The drain now runs first, outside the lock, and a chip that fails it is not closed.

Host-only. No card: the ttnn calls and the lock are replaced.
"""
import contextlib

import pytest

T = pytest.importorskip("tt_bio.tenstorrent")


@pytest.fixture
def calls(monkeypatch):
    log = []

    @contextlib.contextmanager
    def lock():
        log.append("lock")
        yield
        log.append("unlock")

    monkeypatch.setattr(T, "device_init_lock", lock)
    monkeypatch.setattr(T.ttnn, "close_device", lambda d: log.append("close"))
    monkeypatch.setattr(T, "_release_ownership_fn", lambda: lambda: log.append("release"))
    return log


def test_a_healthy_chip_drains_then_closes_under_the_lock(calls, monkeypatch):
    monkeypatch.setattr(T.ttnn, "synchronize_device", lambda d: calls.append("drain"))
    T._close_device_locked(object())
    assert calls == ["drain", "lock", "close", "release", "unlock"]


def test_a_hung_chip_never_takes_the_lock(calls, monkeypatch):
    def hung(d):
        calls.append("drain")
        raise RuntimeError("TIMEOUT: device timeout in fetch queue wait, potential hang detected")
    monkeypatch.setattr(T.ttnn, "synchronize_device", hung)
    with pytest.raises(RuntimeError, match="did not drain"):
        T._close_device_locked(object())
    assert calls == ["drain"], "a chip that cannot drain must not be closed under the host lock"


def test_cleanup_on_a_hung_chip_waits_out_the_drain_once(calls, monkeypatch):
    drains = []

    def hung(d):
        drains.append(d)
        raise RuntimeError("TIMEOUT")
    monkeypatch.setattr(T.ttnn, "synchronize_device", hung)
    monkeypatch.setattr(T, "_device", object())
    monkeypatch.setattr(T, "_device_lease", None)
    with pytest.raises(RuntimeError, match="did not drain"):
        T.cleanup()
    assert len(drains) == 1 and calls == []
    monkeypatch.setattr(T, "_device", None)   # conftest's per-test cleanup() runs before undo
