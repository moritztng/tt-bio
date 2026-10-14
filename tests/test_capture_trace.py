"""`capture_trace` closes a trace capture that throws, so the device is not left inside it.

A capture left open makes `close_device` spin in `synchronize_device` while it holds the
host-wide device-init lock, which stalls every other chip on the host. Host-only: ttnn's
trace calls are stubbed, no device.
"""
from __future__ import annotations

from unittest import mock

import pytest

from tt_bio import tenstorrent as T


def _stubbed():
    calls = []
    patches = [
        mock.patch.object(T.ttnn, "begin_trace_capture", lambda dev, cq_id: calls.append("begin") or 7),
        mock.patch.object(T.ttnn, "end_trace_capture", lambda dev, tid, cq_id: calls.append(("end", tid))),
        mock.patch.object(T.ttnn, "release_trace", lambda dev, tid: calls.append(("release", tid))),
    ]
    return calls, patches


def test_success_returns_tid_and_result_without_release():
    calls, patches = _stubbed()
    with patches[0], patches[1], patches[2]:
        assert T.capture_trace(object(), lambda: "out") == (7, "out")
    assert calls == ["begin", ("end", 7)]


def test_throw_closes_and_releases_then_reraises():
    calls, patches = _stubbed()

    def boom():
        raise RuntimeError("TT_FATAL: Event Synchronization is not supported during trace capture.")

    with patches[0], patches[1], patches[2], pytest.raises(RuntimeError, match="trace capture"):
        T.capture_trace(object(), boom)
    assert calls == ["begin", ("end", 7), ("release", 7)]


def test_release_runs_even_if_end_throws():
    calls = []

    def end(dev, tid, cq_id):
        raise RuntimeError("end failed")

    with mock.patch.object(T.ttnn, "begin_trace_capture", lambda dev, cq_id: 3), \
         mock.patch.object(T.ttnn, "end_trace_capture", end), \
         mock.patch.object(T.ttnn, "release_trace", lambda dev, tid: calls.append(tid)), \
         pytest.raises(RuntimeError):
        T.capture_trace(object(), lambda: 1 / 0)
    assert calls == [3]
