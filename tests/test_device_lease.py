"""Proof that the physical-card device lease serializes same-host opens and self-heals.

Device-free: exercises the tt_bio.device_lease enforcement primitive directly (the same
object get_device() acquires before ttnn.open_device), so it runs anywhere and in CI. A
companion on-hardware test (test_device_lease_hw.py) reproduces the exact flagship-vs-RFD3
two-process open on a real TT card.

Run: python3 tests/test_device_lease.py, or as part of the release suite via pytest.
"""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from tt_bio.device_lease import (CardSetLease, DeviceInUseError, DeviceLease,
                                 granted_cards, physical_card, physical_cards)

try:
    import pytest
except ImportError:                 # standalone `python3 tests/test_device_lease.py`
    pytest = None
else:
    @pytest.fixture
    def d(tmp_path, monkeypatch):
        """Per-test lease dir + host, the pytest equivalent of what __main__ sets up below.
        Every test takes `d` as its lease directory, and the spawned holders inherit the same
        dir/host via _env(), so contention happens on one file and never on the real fleet
        lease dir."""
        monkeypatch.setenv("TT_BIO_LEASE_DIR", str(tmp_path))
        monkeypatch.setenv("TT_BIO_LEASE_HOST", "testhost")
        return str(tmp_path)

# A child that acquires card 0's lease, writes a "ready" marker, holds for `hold`
# seconds, then releases (unless it is killed first).
HOLDER = r"""
import os, sys, time
sys.path.insert(0, os.environ["REPO"])
from tt_bio.device_lease import DeviceLease
lease = DeviceLease(card="0").acquire()
open(os.environ["READY"], "w").write(str(os.getpid()))
time.sleep(float(os.environ["HOLD"]))
lease.release()
"""


def _env(d, ready, hold):
    e = dict(os.environ)
    e.update(REPO=REPO, TT_BIO_LEASE_DIR=d, TT_BIO_LEASE_HOST="testhost",
             READY=ready, HOLD=str(hold))
    return e


def _spawn_holder(d, ready, hold):
    if os.path.exists(ready):
        os.remove(ready)
    p = subprocess.Popen([sys.executable, "-c", HOLDER], env=_env(d, ready, hold))
    for _ in range(200):
        if os.path.exists(ready):
            return p, int(open(ready).read())
        time.sleep(0.05)
    p.kill()
    raise AssertionError("holder never acquired the lease")


def test_serialization_blocks_then_succeeds(d):
    """A second opener WAITS while a live holder has the card, then acquires once freed."""
    p, holder_pid = _spawn_holder(d, os.path.join(d, "ready1"), hold=3.0)
    t0 = time.time()
    lease = DeviceLease(card="0", timeout=30).acquire()  # must block ~3s, not collide
    waited = time.time() - t0
    lease.release()
    p.wait()
    assert waited >= 2.5, f"second opener did not wait for the holder (waited {waited:.2f}s)"
    print(f"  serialization: 2nd open blocked {waited:.2f}s until holder (pid {holder_pid}) freed  OK")


def test_timeout_errors_cleanly(d):
    """A contended card fails LOUDLY with the holder's identity, never a silent collision."""
    p, holder_pid = _spawn_holder(d, os.path.join(d, "ready2"), hold=5.0)
    try:
        DeviceLease(card="0", timeout=1).acquire()
        raise AssertionError("expected DeviceInUseError on a contended card")
    except DeviceInUseError as e:
        assert str(holder_pid) in str(e), f"error must name the holder pid: {e}"
        print(f"  timeout: contended open raised DeviceInUseError naming pid {holder_pid}  OK")
    finally:
        p.kill(); p.wait()


def test_sigkill_reclaim(d):
    """A SIGKILLed holder (no chance to release) leaves NO phantom claim: kernel drops flock."""
    p, holder_pid = _spawn_holder(d, os.path.join(d, "ready3"), hold=60.0)
    os.kill(p.pid, signal.SIGKILL)
    p.wait()
    t0 = time.time()
    lease = DeviceLease(card="0", timeout=10).acquire()  # must succeed ~immediately
    waited = time.time() - t0
    lease.release()
    assert waited < 2.0, f"stale SIGKILLed lease was not reclaimed promptly ({waited:.2f}s)"
    print(f"  SIGKILL reclaim: card re-leased {waited:.2f}s after holder killed  OK")


def test_sigterm_release(d):
    """SIGTERM frees the card via the kernel flock-drop on process death."""
    p, holder_pid = _spawn_holder(d, os.path.join(d, "ready4"), hold=60.0)
    os.kill(p.pid, signal.SIGTERM)
    p.wait()
    t0 = time.time()
    lease = DeviceLease(card="0", timeout=10).acquire()
    waited = time.time() - t0
    lease.release()
    assert waited < 2.0, f"SIGTERMed lease not freed promptly ({waited:.2f}s)"
    print(f"  SIGTERM release: card re-leased {waited:.2f}s after holder terminated  OK")


def test_no_self_deadlock_against_dispatch_lease(d):
    """A worker opening its OWN dispatch-leased card must NOT deadlock on it.

    The fleet dispatcher writes the lease JSON WITHOUT holding an flock, so device-open
    acquires immediately and just overwrites the metadata with its live pid."""
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "testhost-card0.json")
    with open(path, "w") as f:  # simulate fleet.sh's dispatch-time write (no flock held)
        json.dump({"host": "testhost", "card": "0", "holder": "worker:flagship",
                   "pid": 999999, "acquired": time.time()}, f)
    t0 = time.time()
    lease = DeviceLease(card="0", timeout=5).acquire()
    waited = time.time() - t0
    meta = json.load(open(path))
    lease.release()
    assert waited < 1.0, f"deadlocked against own dispatch lease ({waited:.2f}s)"
    assert meta["pid"] == os.getpid(), "device-open did not claim the lease as its own pid"
    print(f"  no self-deadlock: claimed own dispatch lease in {waited:.2f}s  OK")


def test_clean_release(d):
    """Context-manager exit frees the card immediately."""
    with DeviceLease(card="0", timeout=5):
        pass
    t0 = time.time()
    with DeviceLease(card="0", timeout=5):
        pass
    assert time.time() - t0 < 1.0
    print("  clean release: re-leased immediately after context exit  OK")


def test_card_grant_refuses_card_outside_grant(d):
    """A card outside TT_BIO_LEASE_CARDS is refused, and refused FAST (no timeout wait).

    This is the hole the gate-side pin cannot close: a job that never exports
    TT_VISIBLE_DEVICES discovers cards itself and can open the whole box under a one-card
    grant (qb1, 2026-08-21). Fails against the pre-grant code, which leased card 3 happily.
    """
    os.environ["TT_BIO_LEASE_CARDS"] = "1"
    try:
        t0 = time.time()
        try:
            DeviceLease(card="3", timeout=30).acquire()
        except DeviceInUseError as e:
            waited = time.time() - t0
            assert "TT_BIO_LEASE_CARDS=1" in str(e), f"error does not name the grant: {e}"
            assert "3" in str(e), f"error does not name the refused card: {e}"
            assert waited < 1.0, f"refusal waited on the flock timeout ({waited:.2f}s)"
        else:
            raise AssertionError("card 3 was leased while the grant was card 1 only")
        # no lease file may be left behind for a card we never held
        assert not os.path.exists(os.path.join(d, "testhost-card3.json")), \
            "refused card left a lease file behind"
        print("  card grant: card 3 refused under grant=1, fast, no stray lease file  OK")
    finally:
        os.environ.pop("TT_BIO_LEASE_CARDS", None)


def test_card_grant_allows_granted_and_multi_card_grant(d):
    """A granted card still opens, and a multi-card grant admits every card in it.

    Guards the sanctioned card-fanout pattern: a worker legitimately using sibling cards
    gets a grant listing them, not a single card, and must not be refused.
    """
    os.environ["TT_BIO_LEASE_CARDS"] = "0,2 ,3"
    try:
        for card in ("0", "2", "3"):
            with DeviceLease(card=card, timeout=5):
                pass
        print("  card grant: every card in a multi-card grant opens (0,2,3)  OK")
    finally:
        os.environ.pop("TT_BIO_LEASE_CARDS", None)


def test_card_grant_absent_or_empty_is_unbounded(d):
    """Unset AND empty mean unbounded -- so nothing changes until a control plane writes it.

    This is the whole rollout safety argument: JapanFold's worker pool, the Galaxy's
    multi-card mesh opens and every manual run must be untouched by this commit.
    """
    for value in (None, "", "  ", ","):
        os.environ.pop("TT_BIO_LEASE_CARDS", None)
        if value is not None:
            os.environ["TT_BIO_LEASE_CARDS"] = value
        assert granted_cards() is None, f"TT_BIO_LEASE_CARDS={value!r} was not treated as unbounded"
        with DeviceLease(card="3", timeout=5):
            pass
    os.environ.pop("TT_BIO_LEASE_CARDS", None)
    print("  card grant: unset/empty/whitespace/bare-comma all unbounded  OK")


def test_card_set_leases_every_visible_card(d):
    """The lease set covers every VISIBLE card, because ttnn brings all of them up.

    Measured on qb2 2026-08-21: TT_VISIBLE_DEVICES=0,1 returns a 1x1 mesh
    (get_num_devices()==1) while the process holds fds on /dev/tenstorrent/0 AND
    /dev/tenstorrent/1. The old single lease named card 0 only, so fleet.sh saw one card
    held and handed card 1 to a second worker. Fails against the pre-CardSetLease code.
    """
    os.environ["TT_VISIBLE_DEVICES"] = "1,0"
    try:
        assert physical_cards() == ["0", "1"], physical_cards()   # sorted, deadlock-free order
        with CardSetLease(timeout=5) as lease:
            assert [l.card for l in lease._held] == ["0", "1"]
            for card in ("0", "1"):
                path = os.path.join(d, f"testhost-card{card}.json")
                assert os.path.exists(path), f"card {card} was held but never leased"
                assert json.load(open(path))["pid"] == os.getpid()
        print("  card set: visible 0,1 leases BOTH cards in sorted order  OK")
    finally:
        os.environ.pop("TT_VISIBLE_DEVICES", None)


def test_card_set_is_all_or_nothing(d):
    """A set containing ONE contended card leases none of them, and says which.

    Half a mesh leased is worse than none: the fleet reads the successful half as a live
    claim while the job that took it has already failed.
    """
    p, holder_pid = _spawn_holder(d, os.path.join(d, "ready_set"), hold=30.0)  # holds card 0
    try:
        try:
            CardSetLease(cards=["0", "2"], timeout=1).acquire()
            raise AssertionError("expected DeviceInUseError with card 0 contended")
        except DeviceInUseError as e:
            assert str(holder_pid) in str(e), f"error must name the holder: {e}"
        # card 2 was never left leased on the way out
        card2 = os.path.join(d, "testhost-card2.json")
        assert not os.path.exists(card2) or json.load(open(card2)).get("released"), \
            "a partial set left card 2 leased after the set failed"
        # and card 2 is really free: a fresh lease takes it immediately
        t0 = time.time()
        DeviceLease(card="2", timeout=5).acquire().release()
        assert time.time() - t0 < 1.0, "card 2 still flocked by the failed set"
        print(f"  card set: contended set (0 held by {holder_pid}) leased nothing, card 2 free  OK")
    finally:
        p.kill(); p.wait()


def test_card_set_shares_one_timeout_budget(d):
    """A contended set fails within ONE timeout, not len(cards) x timeout."""
    p, _ = _spawn_holder(d, os.path.join(d, "ready_budget"), hold=30.0)   # holds card 0
    try:
        t0 = time.time()
        try:
            CardSetLease(cards=["0", "1", "2", "3"], timeout=2).acquire()
            raise AssertionError("expected DeviceInUseError")
        except DeviceInUseError:
            waited = time.time() - t0
        assert waited < 4.0, f"set waited per-card, not once ({waited:.2f}s for timeout=2)"
        print(f"  card set: 4-card contended set failed in {waited:.2f}s on timeout=2  OK")
    finally:
        p.kill(); p.wait()


def test_card_set_refused_when_visibility_exceeds_grant(d):
    """A grant of one card refuses an open that would HOLD two, naming the extra card.

    This is the qb1 shape and the reason the grant has to be checked against the whole
    visible set: TT_BIO_LEASE_CARDS=0 with TT_VISIBLE_DEVICES=0,1 opened card 0 happily
    under the old per-card check, while holding card 1 outside the grant.
    """
    os.environ["TT_VISIBLE_DEVICES"] = "0,1"
    os.environ["TT_BIO_LEASE_CARDS"] = "0"
    try:
        t0 = time.time()
        try:
            CardSetLease(timeout=30).acquire()
            raise AssertionError("an open holding card 1 was permitted under grant=0")
        except DeviceInUseError as e:
            assert "1" in str(e) and "TT_BIO_LEASE_CARDS=0" in str(e), e
            assert time.time() - t0 < 1.0, "grant refusal waited on the flock timeout"
        # Order-independent: an earlier test may have left the file behind. What must not
        # happen is this process holding it live.
        card0 = os.path.join(d, "testhost-card0.json")
        if os.path.exists(card0):
            meta = json.load(open(card0))
            assert meta.get("pid") != os.getpid() or meta.get("released"), \
                "refused set leased the granted card anyway"
        print("  card set: visible 0,1 under grant 0 refused fast, nothing leased  OK")
    finally:
        os.environ.pop("TT_VISIBLE_DEVICES", None)
        os.environ.pop("TT_BIO_LEASE_CARDS", None)


def test_card_set_matches_grant_when_pinned(d):
    """The production shape: one visible card, one lease, grant satisfied.

    JapanFold's pool, the Galaxy's 32 workers and every gate leg pin one card each
    (worker.py:_apply_tt_environment), so this is the path that must not change.
    """
    os.environ["TT_VISIBLE_DEVICES"] = "2"
    os.environ["TT_BIO_LEASE_CARDS"] = "2"
    try:
        with CardSetLease(timeout=5) as lease:
            assert lease.cards == ["2"], lease.cards
            assert lease.card == "2"
        print("  card set: pinned open leases exactly its one card under a matching grant  OK")
    finally:
        os.environ.pop("TT_VISIBLE_DEVICES", None)
        os.environ.pop("TT_BIO_LEASE_CARDS", None)


def test_logical_device_id_out_of_range_raises(d):
    """A logical id past the visible set must not silently lease card[0] instead."""
    os.environ["TT_VISIBLE_DEVICES"] = "1,2"
    os.environ["TT_BIO_LOGICAL_DEVICE_ID"] = "3"
    try:
        try:
            physical_card()
            raise AssertionError("out-of-range logical id silently fell back to a card")
        except ValueError as e:
            assert "TT_BIO_LOGICAL_DEVICE_ID=3" in str(e) and "1, 2" in str(e).replace("[", "").replace("]", ""), e
        print("  logical id: out-of-range raises instead of leasing the wrong card  OK")
    finally:
        os.environ.pop("TT_VISIBLE_DEVICES", None)
        os.environ.pop("TT_BIO_LOGICAL_DEVICE_ID", None)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        # The main process must lease into the SAME dir/host as the spawned holders so
        # they actually contend on one file. Holders inherit these via _env().
        os.environ["TT_BIO_LEASE_DIR"] = d
        os.environ["TT_BIO_LEASE_HOST"] = "testhost"
        print(f"lease dir under test: {d}")
        test_no_self_deadlock_against_dispatch_lease(d)
        test_clean_release(d)
        test_serialization_blocks_then_succeeds(d)
        test_timeout_errors_cleanly(d)
        test_sigkill_reclaim(d)
        test_sigterm_release(d)
        test_card_grant_refuses_card_outside_grant(d)
        test_card_grant_allows_granted_and_multi_card_grant(d)
        test_card_grant_absent_or_empty_is_unbounded(d)
        test_card_set_leases_every_visible_card(d)
        test_card_set_is_all_or_nothing(d)
        test_card_set_shares_one_timeout_budget(d)
        test_card_set_refused_when_visibility_exceeds_grant(d)
        test_card_set_matches_grant_when_pinned(d)
        test_logical_device_id_out_of_range_raises(d)
    print("ALL DEVICE-LEASE UNIT TESTS PASSED")
