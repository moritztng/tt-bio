"""A campaign whose chip wedges mid-run says so, instead of a log that simply stops.

The sick-chip cases a soak has to cover, and what each one does:

  absent chip     refused before the lease, in a sentence (`runtime.visible_device_indices`)
  busy chip       refused by its lease after TT_BIO_LEASE_TIMEOUT, naming the holder: measured
                  on dev Galaxy .108 card 30, 2026-09-30, 3.0 s at a 3 s timeout, "in use by
                  worker:b2p-soak (pid 2076503)"
  wedged at open  the bring-up probe is bounded at 120 s (`tests/test_dispatch_probe_is_bounded.py`)
  wedged later    blocks inside a C call with no timeout of its own. Nothing raised, nothing
                  printed: the process holds the card and the log stops, and a researcher cannot
                  tell a hang from a slow stage. THIS file.

A campaign folder is resumable, so the right answer to the last case is a line that says what is
happening and how to get out, not a kill: a false alarm must cost nothing. Card-free.
"""
import os
import threading
import time

import pytest

from tt_bio import bindcraft2


def _run(folder, *, warn_s, every, seconds, work=None):
    said = []
    with bindcraft2._stall_reporter(str(folder), warn_s=warn_s, every=every, say=said.append):
        end = time.time() + seconds
        while time.time() < end:
            if work:
                work()
            time.sleep(every / 2)
    return said


def test_a_campaign_that_stops_moving_says_so(tmp_path):
    said = _run(tmp_path, warn_s=0.3, every=0.05, seconds=0.5)
    assert len(said) == 1, said
    line = said[0]
    assert str(tmp_path) in line
    assert "resume=true" in line and str(os.getpid()) in line
    assert "every design accepted so far is kept" in line


def test_it_speaks_once_per_silent_stretch_not_once_per_poll(tmp_path):
    """Ten polls of silence past the threshold are one stuck campaign, not ten."""
    said = _run(tmp_path, warn_s=0.4, every=0.02, seconds=0.7)
    assert len(said) == 1, said


def test_a_file_written_under_the_project_is_progress(tmp_path):
    """MPNN, validation and acceptance run no gradient rounds; they write files."""
    counter = iter(range(10**6))
    said = _run(tmp_path, warn_s=0.2, every=0.05, seconds=0.6,
                work=lambda: (tmp_path / f"f{next(counter)}.csv").write_text("x"))
    assert said == []


def test_a_gradient_round_is_progress(tmp_path):
    said = _run(tmp_path, warn_s=0.2, every=0.05, seconds=0.6, work=bindcraft2._note_progress)
    assert said == []


def test_the_watcher_ends_with_the_campaign(tmp_path):
    with bindcraft2._stall_reporter(str(tmp_path), warn_s=10, every=0.01, say=lambda _: None):
        assert any(t.name == "bindcraft2:stall" for t in threading.enumerate())
    time.sleep(0.05)
    assert not any(t.name == "bindcraft2:stall" for t in threading.enumerate())


def test_zero_turns_it_off(tmp_path):
    with bindcraft2._stall_reporter(str(tmp_path), warn_s=0, every=0.01, say=pytest.fail):
        time.sleep(0.05)
    assert not any(t.name == "bindcraft2:stall" for t in threading.enumerate())


def test_run_campaign_carries_the_reporter(monkeypatch, tmp_path):
    """The public entry point is what a researcher calls, so the reporter is on it."""
    entered = []
    real = bindcraft2._stall_reporter

    def spy(folder, *a, **kw):
        entered.append(folder)
        return real(folder, *a, **kw)

    monkeypatch.setattr(bindcraft2, "_stall_reporter", spy)
    monkeypatch.setattr(bindcraft2, "_run_campaign", lambda settings, folder, **kw: 7)
    assert bindcraft2.run_campaign({}, str(tmp_path)) == 7
    assert entered == [str(tmp_path)]
