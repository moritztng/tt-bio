"""The three-way answer conftest.device_verdict gives a test that opens a TT card.

The helper this replaced (`_device_available()` in test_token_axis_bucketing_hw.py) asked one
question -- "is TT_VISIBLE_DEVICES set and non-empty" -- and treated the answer as if it meant
"is a card here". On a host that HAS a card and runs pytest unpinned those two disagree, and the
suite skipped every device test while the card sat right there. That is a quiet no-run standing in
for coverage, which is worse than the loud failure it replaced, so the case gets its own verdict
and its own test below.
"""
import os
import subprocess
import sys

import pytest

import conftest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def cards(tmp_path, monkeypatch):
    """Stand in for /dev/tenstorrent/ so the verdict can be asked about either kind of host."""
    monkeypatch.setattr(conftest, "_CARD_NODES", str(tmp_path / "[0-9]*"))

    def seat(n):
        for i in range(n):
            (tmp_path / str(i)).touch()
    return seat


def test_card_present_and_pinned_runs(cards):
    cards(4)
    assert conftest.device_verdict({"TT_VISIBLE_DEVICES": "3"}) == (conftest.RUN, None)


def test_card_present_and_unpinned_is_refused_not_skipped(cards):
    """The regression the old helper had. An unpinned open brings up EVERY card on the box
    (tt_bio/device_lease.py), so it must not silently happen -- and it must not silently NOT
    happen either."""
    cards(4)
    verdict, reason = conftest.device_verdict({})
    assert verdict == conftest.REFUSE
    assert "TT_VISIBLE_DEVICES=" in reason


def test_card_present_and_pin_set_empty_skips(cards):
    """`TT_VISIBLE_DEVICES= pytest` is the deliberate CPU-only run, not an accident."""
    cards(4)
    for pin in ("", "   "):
        assert conftest.device_verdict({"TT_VISIBLE_DEVICES": pin})[0] == conftest.SKIP


def test_no_card_skips_however_the_env_is_set(cards):
    """Driver loaded, nothing seated: /dev/tenstorrent exists and is empty. A pin that names a
    card which is not there is not a reason to refuse -- there is nothing to protect."""
    cards(0)
    for env in ({}, {"TT_VISIBLE_DEVICES": ""}, {"TT_VISIBLE_DEVICES": "3"}):
        assert conftest.device_verdict(env)[0] == conftest.SKIP


def test_device_marker_is_registered():
    """An unregistered mark is a warning, not an error, so a typo would go unnoticed."""
    out = subprocess.run([sys.executable, "-m", "pytest", "--markers"], cwd=REPO,
                         capture_output=True, text=True,
                         env=dict(os.environ, TT_VISIBLE_DEVICES=""))
    assert "@pytest.mark.device:" in out.stdout


def test_backstop_signature_matches_what_ttnn_actually_aborts_with(tmp_path):
    """`conftest._NO_CHIPS` is the one brittle string in the guard: the backstop that turns an
    UNMARKED device test into a skip keys on it. Pin it against the installed ttnn rather than
    against memory, so a wheel that renames the abort shows up here instead of as 60 failures
    nobody can read. Opens nothing -- with no chips visible there is no card to open."""
    child = "import ttnn; ttnn.open_device(device_id=0)"
    out = subprocess.run([sys.executable, "-c", child], cwd=REPO, capture_output=True, text=True,
                         env=dict(os.environ, TT_VISIBLE_DEVICES="",
                                  TT_BIO_LEASE_DIR=str(tmp_path)), timeout=900)
    assert out.returncode != 0, "ttnn opened a device with TT_VISIBLE_DEVICES empty"
    assert conftest._NO_CHIPS in out.stdout + out.stderr, (
        f"ttnn's no-chips abort no longer says {conftest._NO_CHIPS!r}:\n"
        f"{(out.stdout + out.stderr)[-2000:]}")
