"""The recovery a wedge-kill owes the card, and the one case where it must refuse.

qb2 hard-reset three times on 2026-09-22 because a harness killed a wedged fold's process group
and then let the next leg open the same card. These tests pin the two halves that matter: the
reset happens, and it does NOT happen when it would take a co-tenant's board-pair sibling down.
No hardware: tt-smi and the /proc scan are both substituted.
"""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "card_recovery", Path(__file__).resolve().parents[1] / "scripts" / "card_recovery.py")
cr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cr)


@pytest.mark.parametrize("card,sibling", [(0, 1), (1, 0), (2, 3), (3, 2)])
def test_a_reset_takes_the_whole_board_pair_so_every_card_knows_its_sibling(card, sibling):
    assert cr.board_sibling(card) == sibling


def test_a_card_outside_the_known_pairs_reports_no_sibling():
    # Not a pair we know, so there is nothing to protect and nothing to claim.
    assert cr.board_sibling(9) is None


@pytest.mark.parametrize("env,expected", [("0", 0), ("2,3", 2), ("", 0), ("nonsense", 0)])
def test_the_granted_card_comes_from_tt_visible_devices(monkeypatch, env, expected):
    monkeypatch.setenv("TT_VISIBLE_DEVICES", env)
    assert cr.visible_card() == expected


def _spy(monkeypatch, *, sibling_busy):
    """Substitute the /proc scan and tt-smi; return the list tt-smi calls land in."""
    calls = []
    monkeypatch.setattr(cr, "_tt_smi", lambda: "/fake/tt-smi")
    monkeypatch.setattr(cr, "fd_holders",
                        lambda card: [4242] if (sibling_busy and card == 1) else [])
    monkeypatch.setattr(cr.subprocess, "run",
                        lambda argv, **kw: calls.append(argv) or type("R", (), {"returncode": 0})())
    return calls


def test_it_refuses_when_the_board_pair_sibling_is_in_use(monkeypatch):
    calls = _spy(monkeypatch, sibling_busy=True)
    assert cr.reset_after_kill(0, say=lambda *a: None) == cr.REFUSED
    # The refusal has to be a refusal to ACT, not just a word in a return value.
    assert calls == [], "refused but reset the pair anyway"


def test_it_resets_when_the_sibling_is_free(monkeypatch):
    # The negative control for the test above: same call, sibling free, and now it must act.
    calls = _spy(monkeypatch, sibling_busy=False)
    assert cr.reset_after_kill(0, say=lambda *a: None) == cr.OK
    assert calls == [["/fake/tt-smi", "-r", "0"]]


def test_a_card_still_held_after_the_reset_is_a_failure_not_a_success(monkeypatch):
    _spy(monkeypatch, sibling_busy=False)
    monkeypatch.setattr(cr, "fd_holders", lambda card: [777] if card == 0 else [])
    assert cr.reset_after_kill(0, say=lambda *a: None) == cr.FAILED


def test_no_tt_smi_is_a_failure_rather_than_a_silent_pass(monkeypatch):
    monkeypatch.setattr(cr, "_tt_smi", lambda: None)
    monkeypatch.setattr(cr, "fd_holders", lambda card: [])
    assert cr.reset_after_kill(0, say=lambda *a: None) == cr.FAILED
