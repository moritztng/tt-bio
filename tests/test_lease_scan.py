"""`lease_scan` has to see BOTH lease naming conventions, and the window is the whole point.

`<hostname>-cardN.json` is pid-keyed and written by the engine's device-open path; it can read
released 2.1 s after acquire, and a row that cycles arms drops it entirely between them.
`<short>-cardN.json` is the orchestrator's occupancy guard and holds 60 s past the node emptying.
The case that matters is therefore the one where ONLY the occupancy guard is on disk -- a taker
that reads just the pid-keyed name calls that card free, which is the 2026-09-26 03:40 collision.
"""
import json
import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "perf" / "bcx_bwbytes"))
from lease_scan import conflicts  # noqa: E402

HOST = "tt-quietbox"


def _write(d, name, **kw):
    (d / name).write_text(json.dumps(kw))


def test_sees_the_engines_pid_keyed_lease(tmp_path):
    _write(tmp_path, "tt-quietbox-card2.json", host=HOST, card="2", holder="worker:other", pid=1)
    assert len(conflicts(tmp_path, "2", HOST)) == 1


def test_sees_the_occupancy_guard_when_the_pid_keyed_one_is_absent(tmp_path):
    """The collision window: the card is held, and only the short-named guard says so."""
    _write(tmp_path, "qb1-card2.json", host="qb1", card="2", holder="worker:other", pid=1,
           note="guard tracks occupancy")
    assert len(conflicts(tmp_path, "2", HOST)) == 1


def test_a_released_rename_is_not_a_conflict(tmp_path):
    """Released leases are RENAMED on this fleet, not deleted, so the suffix is the live set."""
    _write(tmp_path, "qb1-card2.json.released-2026-09-26-pid-dead", host="qb1", card="2", pid=1)
    assert conflicts(tmp_path, "2", HOST) == []


def test_another_cards_lease_does_not_block_this_one(tmp_path):
    """The control: without it, a scanner that returned every file would pass every test above."""
    _write(tmp_path, "qb1-card3.json", host="qb1", card="3", holder="worker:other", pid=1)
    _write(tmp_path, "tt-quietbox-card3.json", host=HOST, card="3", holder="worker:other", pid=1)
    assert conflicts(tmp_path, "2", HOST) == []


def test_another_hosts_lease_does_not_block_this_one(tmp_path):
    """Second control: the leases directory is per-host, but the files carry a host field."""
    _write(tmp_path, "tt-quietbox2-card2.json", host="tt-quietbox2", card="2", pid=1)
    _write(tmp_path, "qb2-card2.json", host="qb2", card="2", pid=1)
    assert conflicts(tmp_path, "2", HOST) == []


def test_an_unparseable_lease_is_reported_not_skipped(tmp_path):
    """This decides whether to open a device. The safe reading of a file we cannot read is held."""
    (tmp_path / "qb1-card2.json").write_text("{not json")
    out = conflicts(tmp_path, "2", HOST)
    assert len(out) == 1 and "unparseable" in out[0]


def test_an_integer_card_field_still_matches(tmp_path):
    """The two writers disagree on the type: the engine writes "2", the guard has written 2."""
    _write(tmp_path, "qb1-card2.json", host="qb1", card=2, holder="worker:other", pid=1)
    assert len(conflicts(tmp_path, "2", HOST)) == 1


def test_a_dead_pid_lease_is_stale_not_a_conflict(tmp_path):
    """The deadlock this cost: `card_free.sh` judges a lease by pid liveness, so a scanner that
    refused on existence alone disagreed with the gate that decides whether to claim at all.
    qb1 card 2 freed at 03:35Z on 2026-09-26 and the claim bounced off a lease naming pid 187751,
    which had exited."""
    from lease_scan import scan
    dead = 999999                      # far past any live pid on these boxes
    _write(tmp_path, "tt-quietbox-card2.json", host=HOST, card="2", holder="pid:x", pid=dead)
    bad, stale = scan(tmp_path, "2", HOST)
    assert bad == [] and len(stale) == 1 and "stale" in stale[0]


def test_a_live_pid_is_still_a_conflict(tmp_path):
    """The control for the one above: without it, a scanner that called everything stale would
    pass it."""
    from lease_scan import scan
    _write(tmp_path, "tt-quietbox-card2.json", host=HOST, card="2", holder="pid:x", pid=os.getpid())
    bad, stale = scan(tmp_path, "2", HOST)
    assert len(bad) == 1 and stale == []


def test_a_lease_with_no_pid_cannot_be_shown_stale(tmp_path):
    from lease_scan import scan
    _write(tmp_path, "qb1-card2.json", host="qb1", card="2", holder="worker:other")
    bad, _ = scan(tmp_path, "2", HOST)
    assert len(bad) == 1 and "no pid" in bad[0]
