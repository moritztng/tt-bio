"""A test must not leave the session's device pin different from how it found it.

tt_bio.device_lease reads TT_VISIBLE_DEVICES and TT_BIO_LEASE_CARDS live on every open, so
these are session-global. Four tests in test_device_lease.py set TT_VISIBLE_DEVICES and then
*popped* it in their finally, which is not a restore: it leaves the rest of the run unpinned,
and an unpinned open brings up every card on the box. Under the one-card grant a gate leg
gets, the card lease then refuses it.

That cost a full release-gate host suite on 2026-08-23: 78 failures across test_tenstorrent.py
and test_esmc.py, every one of them "would hold physical card(s) 0,1,3 ... outside this job's
card grant", none of them about the code under test. It had never shown up before because the
gate host was pc, which has one card -- there, unpinned and granted are the same set, so the
leak is invisible. tests/conftest.py now restores the device environment around every test.
"""
import os

# Captured at import, before any test in this session has run: the pin the operator
# actually launched pytest with.
_AT_IMPORT = {k: os.environ.get(k) for k in
              ("TT_VISIBLE_DEVICES", "TT_BIO_LEASE_CARDS", "TT_BIO_LOGICAL_DEVICE_ID")}


def test_a_leaks_the_device_pin_the_way_the_old_tests_did():
    """Stand-in for the offending pattern: set the pin, then pop it instead of restoring."""
    os.environ["TT_VISIBLE_DEVICES"] = "1,0"
    os.environ["TT_BIO_LEASE_CARDS"] = "1"
    os.environ["TT_BIO_LOGICAL_DEVICE_ID"] = "1"
    os.environ.pop("TT_VISIBLE_DEVICES", None)
    os.environ.pop("TT_BIO_LEASE_CARDS", None)
    os.environ.pop("TT_BIO_LOGICAL_DEVICE_ID", None)


def test_b_the_next_test_still_sees_the_launcher_pin():
    """Runs after test_a in file order. Without the conftest fixture this fails."""
    for key, want in _AT_IMPORT.items():
        assert os.environ.get(key) == want, (
            f"{key} leaked out of the previous test: expected {want!r}, "
            f"found {os.environ.get(key)!r}")
