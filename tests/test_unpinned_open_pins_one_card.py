"""An unpinned get_device() on a multi-card host opens ONE card, exactly as a fanout worker does.

Tamarind eval box f12cs06 (4x p150a, cards cabled only in pairs), 2026-11-24: every in-process
open -- `tt-bio embed` (ESMC), `tt-bio saprot`, `design --model rfd3` -- died at ttnn's device
open with "Graph specified in MGD could not fit in the discovered physical topology", while
`predict` ran fine. predict pins TT_VISIBLE_DEVICES to one card per worker; the in-process
paths left it unset, so ttnn brought up and fabric-mapped all four cards for a one-chip job.
Measured there: unset -> TT_FATAL; TT_VISIBLE_DEVICES=0 -> opens.

The pin/lease half is device-free; the get_device() wiring half needs the ttnn wheel only to
import tt_bio.tenstorrent, and never opens a card (the open is stubbed).
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from tt_bio import device_lease
from tt_bio.device_lease import DeviceLease, pin_one_card

FOUR = ["0", "1", "2", "3"]


@pytest.fixture
def host(tmp_path, monkeypatch):
    """A synthetic 4-card host with its own lease dir and no ambient pin or grant."""
    monkeypatch.setenv("TT_BIO_LEASE_DIR", str(tmp_path))
    monkeypatch.setenv("TT_BIO_LEASE_HOST", "testhost")
    for k in ("TT_VISIBLE_DEVICES", "TT_BIO_LOGICAL_DEVICE_ID", "TT_BIO_LEASE_CARDS",
              "TT_BIO_LEASE_HOLDER"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(device_lease, "physical_cards", lambda: list(FOUR))
    return tmp_path


def test_an_unpinned_open_is_pinned_to_one_free_card(host):
    lease = pin_one_card()
    try:
        assert os.environ["TT_VISIBLE_DEVICES"] == "0"
        assert os.environ["TT_BIO_LOGICAL_DEVICE_ID"] == "0"
        assert lease.card == "0"
        # A second one-card job on the same host takes the next free card, not a wait on card 0.
        os.environ.pop("TT_VISIBLE_DEVICES")
        os.environ.pop("TT_BIO_LOGICAL_DEVICE_ID")
        second = pin_one_card()
        try:
            assert second.card == "1" and os.environ["TT_VISIBLE_DEVICES"] == "1"
        finally:
            second.release()
    finally:
        lease.release()


def test_the_pin_stays_inside_the_card_grant(host, monkeypatch):
    monkeypatch.setenv("TT_BIO_LEASE_CARDS", "2,3")
    lease = pin_one_card()
    try:
        assert lease.card == "2" and os.environ["TT_VISIBLE_DEVICES"] == "2"
    finally:
        lease.release()


def test_an_explicit_logical_id_names_the_card(host, monkeypatch):
    monkeypatch.setenv("TT_BIO_LOGICAL_DEVICE_ID", "3")
    lease = pin_one_card()
    try:
        assert lease.card == "3"
        assert os.environ["TT_VISIBLE_DEVICES"] == "3"
        assert os.environ["TT_BIO_LOGICAL_DEVICE_ID"] == "0"   # 0th of the one visible card
    finally:
        lease.release()
    monkeypatch.delenv("TT_VISIBLE_DEVICES")
    monkeypatch.setenv("TT_BIO_LOGICAL_DEVICE_ID", "4")
    with pytest.raises(ValueError, match="out of range"):
        pin_one_card()


def test_a_caller_pin_and_a_one_card_host_are_left_alone(host, monkeypatch):
    monkeypatch.setenv("TT_VISIBLE_DEVICES", "0,1")
    assert pin_one_card() is None and os.environ["TT_VISIBLE_DEVICES"] == "0,1"
    monkeypatch.setenv("TT_VISIBLE_DEVICES", "")          # set-and-empty is a declaration too
    assert pin_one_card() is None
    monkeypatch.delenv("TT_VISIBLE_DEVICES")
    assert pin_one_card(cards=["0"]) is None and "TT_VISIBLE_DEVICES" not in os.environ


def test_get_device_opens_one_card_when_nothing_is_pinned(host, monkeypatch):
    """The wiring: get_device() pins before it opens, and the open sees exactly one card."""
    pytest.importorskip("ttnn")
    from tt_bio import runtime
    from tt_bio import tenstorrent as tt

    monkeypatch.setattr(tt, "_device", None)
    monkeypatch.setattr(tt, "_device_lease", None)
    for name in ("refuse_dead_chips", "refuse_missing_device_nodes", "refuse_quarantined_chips"):
        monkeypatch.setattr(runtime, name, lambda *a, **k: None)
    monkeypatch.setattr(device_lease, "arm_orphan_guard", lambda: None)
    # Another process already holds card 0: the unpinned open must go to card 1, alone.
    other = DeviceLease(card="0").acquire()
    seen = {}

    def fake_open(trace_region_size):
        seen["visible"] = os.environ.get("TT_VISIBLE_DEVICES")
        seen["logical"] = os.environ.get("TT_BIO_LOGICAL_DEVICE_ID")
        return object()

    monkeypatch.setattr(tt, "_open_and_init_device", fake_open)
    try:
        tt.get_device()
        assert seen == {"visible": "1", "logical": "0"}, seen
        assert tt._device_lease.card == "1"
    finally:
        if tt._device_lease is not None:
            tt._device_lease.release()
        monkeypatch.setattr(tt, "_device", None)
        monkeypatch.setattr(tt, "_device_lease", None)
        other.release()


def test_a_failed_open_unpins_so_a_retry_picks_again(host, monkeypatch):
    pytest.importorskip("ttnn")
    from tt_bio import runtime
    from tt_bio import tenstorrent as tt

    monkeypatch.setattr(tt, "_device", None)
    monkeypatch.setattr(tt, "_device_lease", None)
    for name in ("refuse_dead_chips", "refuse_missing_device_nodes", "refuse_quarantined_chips"):
        monkeypatch.setattr(runtime, name, lambda *a, **k: None)
    monkeypatch.setattr(device_lease, "arm_orphan_guard", lambda: None)

    def boom(trace_region_size):
        raise RuntimeError("TT_FATAL")

    monkeypatch.setattr(tt, "_open_and_init_device", boom)
    with pytest.raises(RuntimeError):
        tt.get_device()
    assert tt._device_lease is None
    assert "TT_VISIBLE_DEVICES" not in os.environ
    DeviceLease(card="0", timeout=0).acquire().release()   # the card was handed back
