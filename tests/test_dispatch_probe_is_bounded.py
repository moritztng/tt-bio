"""The bring-up probe must fail the same way whether the chip throws or wedges.

`tenstorrent._assert_local_dispatch` fires one trivial 32x32 add at a freshly-opened chip so a
mis-initialised worker "fails HERE, at startup, and gets respawned" -- its own docstring. For as
long as it existed it guarded only the chip that THROWS. `ttnn.synchronize_device` has no
timeout, so a chip that WEDGES held the function forever, and on 2026-09-22 two arms sat in it
for 115 minutes each holding a card and computing nothing while every cheap liveness signal read
healthy (100 % CPU with CPU-time tracking elapsed, AICLK 1350 MHz against 800 on the idle cards,
9 W over idle). A fail-fast probe that can hang is worse than no probe.

Both directions are pinned here, because a bound that changes the throwing case would be a
regression and a bound that misses the wedging case would be the original defect.

Host-only. No card: the three ttnn calls are replaced, which is what lets a wedge be simulated
at all.
"""
import time

import pytest

T = pytest.importorskip("tt_bio.tenstorrent")


class _Dev:
    pass


@pytest.fixture
def probe(monkeypatch):
    """`_assert_local_dispatch` with its three ttnn calls and the device close replaced."""
    closed = []
    monkeypatch.setattr(T, "_close_device_locked", lambda d: closed.append(d))
    monkeypatch.setattr(T.ttnn, "from_torch", lambda *a, **k: "T")
    monkeypatch.setattr(T.ttnn, "add", lambda *a, **k: "T")
    return closed


def test_a_healthy_chip_passes_and_is_not_closed(probe, monkeypatch):
    monkeypatch.setattr(T.ttnn, "synchronize_device", lambda d: None)
    T._assert_local_dispatch(_Dev(), timeout_s=5)
    assert probe == [], "a healthy probe must not close the device"


def test_a_throwing_chip_still_raises_exactly_as_before(probe, monkeypatch):
    """The remote-only case this function was written for. Unchanged: same RuntimeError, same
    message, device closed on the way out."""
    def boom(d):
        raise RuntimeError("SubDeviceManagerTracker not initialized / only remote devices")
    monkeypatch.setattr(T.ttnn, "synchronize_device", boom)
    with pytest.raises(RuntimeError, match="failed the local-dispatch check"):
        T._assert_local_dispatch(_Dev(), timeout_s=5)
    assert len(probe) == 1, "a throwing chip must still be closed"


def test_a_wedging_chip_now_raises_instead_of_hanging(probe, monkeypatch):
    """The 115-minute case. Without the bound this test does not fail, it never returns."""
    monkeypatch.setattr(T.ttnn, "synchronize_device", lambda d: time.sleep(600))
    t0 = time.time()
    with pytest.raises(RuntimeError, match="failed the local-dispatch check"):
        T._assert_local_dispatch(_Dev(), timeout_s=2)
    elapsed = time.time() - t0
    assert elapsed < 30, f"the probe took {elapsed:.1f}s; the bound did not fire"
    assert len(probe) == 1, (
        "a wedging chip must be closed exactly as a throwing one is -- a wedge and a throw are "
        "one outcome for every caller, which is the whole point of the bound")


def test_the_timeout_message_names_the_wedge(probe, monkeypatch):
    """So an operator reading the traceback can tell a wedge from a remote-only init."""
    monkeypatch.setattr(T.ttnn, "synchronize_device", lambda d: time.sleep(600))
    with pytest.raises(RuntimeError) as exc:
        T._assert_local_dispatch(_Dev(), timeout_s=2)
    assert "no result from the bring-up dispatch" in str(exc.value)


def test_the_default_bound_is_far_above_a_healthy_probe():
    """A healthy open plus dispatch on a qb1 p150a is 1.6 s. The default must not need tuning,
    and must not be so tight that a slow-but-live bring-up is called a wedge."""
    assert T._DISPATCH_PROBE_TIMEOUT_S >= 60.0
    assert T._DISPATCH_PROBE_TIMEOUT_S <= 600.0


def test_the_deadline_is_inert_off_the_main_thread(monkeypatch):
    """SIGALRM only arms on the main thread. Off it the probe must still RUN -- degraded to the
    old unbounded behaviour -- rather than raising about signals."""
    import threading
    out = {}

    def run():
        monkeypatch.setattr(T.ttnn, "from_torch", lambda *a, **k: "T")
        monkeypatch.setattr(T.ttnn, "add", lambda *a, **k: "T")
        monkeypatch.setattr(T.ttnn, "synchronize_device", lambda d: None)
        monkeypatch.setattr(T, "_close_device_locked", lambda d: None)
        try:
            T._assert_local_dispatch(_Dev(), timeout_s=5)
            out["ok"] = True
        except BaseException as e:            # noqa: BLE001 - the point is that none escapes
            out["err"] = repr(e)

    th = threading.Thread(target=run)
    th.start()
    th.join(30)
    assert out.get("ok") is True, out
