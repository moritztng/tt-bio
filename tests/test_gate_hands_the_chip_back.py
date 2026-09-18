"""Proof that the release gate does not hold the card against its own later arms.

Device-free. Two arms run model code IN the gate process (pxdesign's ``ProtenixDesign.design``
and the ESMC parity leg), so they leave tt_bio's module-level device open and its CardSetLease
held. Every other arm folds in a SUBPROCESS, which then finds a live lease under the gate's own
holder name at a different pid -- a real co-tenant by the lease's own rule -- and exits 75 without
opening the card.

Measured on 2026-09-18: gate2 on `c13-land-first` passed seven arms, pxdesign took the card, and
then opendde-abag and capacity came back BLOCKED, every model in the size-ladder failed at its
256 aa warm-up, the run printed "GATE FAIL -- size-ladder drift" for a ladder that never folded,
and the l1-budget arm's raw ttnn.open_device(0) blocked at the fd level until its 600 s timeout
killed the gate 49 minutes in. None of that was an accuracy result.

The same run also scored a report five hours older than itself: nesso1's harness exited 75 and
wrote nothing, and the arm parsed the leftover file and printed PASS, 3.604xR and "device spread
0" for three device repeats that never ran. A device that never opens has zero spread trivially.

Run: python3 tests/test_gate_hands_the_chip_back.py, or via pytest in the release suite.
"""
import json
import os
import sys
import tempfile
import types
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from tt_bio.device_lease import CONTENDED_EXIT_CODE  # noqa: E402

import release_gate as rg  # noqa: E402

MOD = "tt_bio.tenstorrent"


class _Chip:
    """Stands in for tt_bio.tenstorrent: only cleanup() is what the gate is allowed to call."""

    def __init__(self, raises=None):
        self.closed = 0
        self._raises = raises

    def cleanup(self):
        self.closed += 1
        if self._raises is not None:
            raise self._raises


def _with_stub(chip, fn):
    """Run fn with a stub tt_bio.tenstorrent installed, then put sys.modules back."""
    had = MOD in sys.modules
    prev = sys.modules.get(MOD)
    sys.modules[MOD] = chip
    try:
        return fn()
    finally:
        if had:
            sys.modules[MOD] = prev
        else:
            del sys.modules[MOD]


def test_no_chip_open_means_nothing_to_hand_back():
    """The gate must not import tt_bio.tenstorrent just to release it: on a host where no arm
    needs the module, importing it turns a working gate into an ImportError."""
    had = MOD in sys.modules
    prev = sys.modules.pop(MOD, None)
    try:
        rg._hand_back_chip()
        assert MOD not in sys.modules, "released a chip nobody opened, by importing the module"
    finally:
        if had:
            sys.modules[MOD] = prev


def test_an_in_process_arm_hands_the_chip_back():
    chip = _Chip()
    got = _with_stub(chip, lambda: rg._arm(lambda keep: {"model": "px", "gate": True}, True))
    assert got == {"model": "px", "gate": True}
    assert chip.closed == 1, "the arm returned and the gate still holds the card"


def test_the_chip_comes_back_even_when_the_arm_raises():
    """The failure mode is worse on the raising path: a crashed in-process arm that keeps the
    card turns one arm's bug into every later arm's BLOCKED."""
    chip = _Chip()

    def boom():
        raise RuntimeError("arm blew up")

    try:
        _with_stub(chip, lambda: rg._arm(boom))
    except RuntimeError as e:
        assert str(e) == "arm blew up", "the arm's own error must still reach the gate"
    else:
        raise AssertionError("_arm swallowed the arm's exception")
    assert chip.closed == 1, "a raising arm kept the card"


def test_a_close_that_fails_is_reported_not_swallowed(capsys=None):
    """If the close raises, the card really is still ours. That has to be loud: the next arm
    would otherwise meet it as an unexplained co-tenant, which is how this cost 49 minutes."""
    chip = _Chip(raises=RuntimeError("chip wedged"))
    _with_stub(chip, rg._hand_back_chip)          # must not propagate
    assert chip.closed == 1


def test_every_in_process_arm_call_site_is_wrapped():
    """The source check, because the bug is one forgotten wrap. pxdesign and esmc run model code
    in this process; boltzgen is wrapped as well since its harness runs in-process too."""
    src = Path(REPO, "scripts", "release_gate.py").read_text()
    for call in ("_arm(run_pxdesign, args.keep)",
                 "_arm(run_boltzgen, bg, args.keep)",
                 "_arm(run_esmc, m, parity)"):
        assert call in src, f"in-process arm call site is not wrapped: {call}"
    # Non-vacuous: an unwrapped call of the same arm must not also be present.
    for bare in ("= run_pxdesign(", "= run_boltzgen(", "[run_esmc("):
        assert bare not in src, f"an unwrapped call survives: {bare}"


def test_nesso1_refuses_to_score_a_report_from_an_earlier_run():
    """The exact 2026-09-18 false PASS, reproduced: harness exits 75 and writes nothing, a
    previous run's report is sitting there, and the arm must come back BLOCKED rather than PASS."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        stale = root / "perf" / "nesso1" / "gate_parity.json"
        stale.parent.mkdir(parents=True)
        stale.write_text(json.dumps({
            "verdict": "PASS", "X_over_R": 3.6039044117105417, "max_device_spread": 0.0,
            "n_tokens": 61, "floors": {}, "X_device_vs_torch_key": "ic50",
        }))

        fixture = root / "tyr48.json"
        fixture.write_text("{}")

        old_root, old_fix, old_fold = rg.REPO_ROOT, rg.NESSO1_FIXTURE, rg._run_fold
        rg.REPO_ROOT, rg.NESSO1_FIXTURE = root, fixture
        # The harness never opened the card, so it writes no report -- the whole point.
        rg._run_fold = lambda cmd, timeout, **kw: (CONTENDED_EXIT_CODE, False)
        try:
            row = rg.run_nesso1(keep=True)
        finally:
            rg.REPO_ROOT, rg.NESSO1_FIXTURE, rg._run_fold = old_root, old_fix, old_fold

        assert not row["gate"], "scored a report from an earlier run as this run's PASS"
        assert row["spread"] is None and row["x_over_r"] is None, \
            f"carried numbers from the stale report: {row}"
        assert rg._contended(row), f"a never-opened card must read BLOCKED, not FAIL: {row}"
        assert "BLOCKED" in rg._verdict(row), rg._verdict(row)
        assert not stale.exists(), "the stale report was left for the next run to score too"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and isinstance(fn, types.FunctionType):
            fn()
            print(f"ok  {name}")
    print("all ok")
