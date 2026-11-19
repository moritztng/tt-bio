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
PINNED_TO = cr.pinned_to    # the real scan, before the autouse fixture below stubs it


@pytest.fixture(autouse=True)
def _p300c_by_default(monkeypatch):
    # Pin the board to p300c so the pair tests do not depend on which host runs them.
    monkeypatch.setattr(cr, "subsystem_device", lambda card: "0x0044")


@pytest.fixture(autouse=True)
def _no_pinned_process(monkeypatch):
    # The real environ scan would see whatever runs on the test host (a gate leg pinned to card 1).
    monkeypatch.setattr(cr, "pinned_to", lambda card: [])


@pytest.mark.parametrize("card,sibling", [(0, 1), (1, 0), (2, 3), (3, 2)])
def test_a_reset_takes_the_whole_board_pair_so_every_card_knows_its_sibling(card, sibling):
    assert cr.board_sibling(card) == sibling


@pytest.mark.parametrize("sub", [None, "0x9999"])
def test_an_unrecognised_board_keeps_the_pair_guard(monkeypatch, sub):
    monkeypatch.setattr(cr, "subsystem_device", lambda card: sub)
    assert cr.board_sibling(0) == 1


def test_a_p150a_has_no_sibling_so_a_busy_neighbour_does_not_block_its_reset(monkeypatch):
    # qb1 2026-09-28: card 1 busy, card 0 wedged, the p300c pairs refused a reset of a
    # single-chip board and stopped the gate.
    calls = _spy(monkeypatch, sibling_busy=True)
    monkeypatch.setattr(cr, "subsystem_device", lambda card: "0x0040")
    assert cr.board_sibling(0) is None
    assert cr.reset_after_kill(0, say=lambda *a: None) == cr.OK
    assert calls == [["/fake/tt-smi", "-r", "0"]]


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


def test_a_leg_pinned_to_the_sibling_but_not_yet_open_blocks_the_reset(monkeypatch):
    # qb2 2026-11-06: ux:boltzgen was 19 s into its imports on card 2, no fd yet, when the pair reset killed it.
    calls = _spy(monkeypatch, sibling_busy=False)
    monkeypatch.setattr(cr, "pinned_to", lambda card: [5151] if card == 1 else [])
    assert cr.reset_after_kill(0, say=lambda *a: None) == cr.REFUSED
    assert calls == []


def test_the_pin_scan_reads_tt_visible_devices_of_python_processes(tmp_path, monkeypatch):
    def proc(pid, argv0, env):
        d = tmp_path / str(pid)
        d.mkdir()
        (d / "cmdline").write_bytes(argv0 + b"\0-m\0tt_bio.main\0")
        (d / "environ").write_bytes(b"\0".join(env) + b"\0")
    proc(11, b"/venv/bin/python3", [b"HOME=/h", b"TT_VISIBLE_DEVICES=2"])
    proc(12, b"/venv/bin/python", [b"TT_VISIBLE_DEVICES=0,1"])
    proc(13, b"/bin/bash", [b"TT_VISIBLE_DEVICES=1"])        # an idle shell does not hold the pair
    proc(14, b"/venv/bin/python", [b"TT_VISIBLE_DEVICES=12"])  # card 12, not card 1 or 2
    (tmp_path / "self").mkdir()
    real = cr.Path
    monkeypatch.setattr(cr, "Path", lambda p: tmp_path if p == "/proc" else real(p))
    assert PINNED_TO(2) == [11]
    assert PINNED_TO(1) == [12]
    assert PINNED_TO(3) == []
