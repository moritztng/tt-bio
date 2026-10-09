"""The host lane runs jobs in submission order, hands back their exceptions, and runs inline when off."""
import threading

import pytest

from tt_bio import hostlane


def test_order_and_thread(monkeypatch):
    monkeypatch.setenv("TT_BIO_HOST_LANE", "1")
    seen = []
    futs = [hostlane.submit(lambda i=i: seen.append((i, threading.current_thread().name)) or i) for i in range(20)]
    assert [f.result() for f in futs] == list(range(20))
    assert [i for i, _ in seen] == list(range(20))
    assert all(name.startswith("tt-bio-host-lane") for _, name in seen)


def test_a_later_job_can_wait_on_an_earlier_one(monkeypatch):
    monkeypatch.setenv("TT_BIO_HOST_LANE", "1")
    a = hostlane.submit(lambda: 3)
    b = hostlane.submit(lambda: a.result() * 2)
    assert b.result() == 6


@pytest.mark.parametrize("on", ["0", "1"])
def test_exception_surfaces_at_result(monkeypatch, on):
    monkeypatch.setenv("TT_BIO_HOST_LANE", on)
    f = hostlane.submit(lambda: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        f.result()


def test_off_runs_inline(monkeypatch):
    monkeypatch.setenv("TT_BIO_HOST_LANE", "0")
    seen = []
    f = hostlane.submit(lambda: seen.append(threading.current_thread().name))
    assert seen == [threading.main_thread().name]
    assert f.done()
