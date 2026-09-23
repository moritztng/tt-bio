"""perf/mgxdesign/report.py: which killed rungs are measurements.

ladder.py ends every rung it stops with SIGKILL, so rc -9 is on the rows that matter most: a rung
killed at a device throw is the ceiling, one killed at the budget is INCONCLUSIVE, and only a kill
the ladder did not sign (a driver restart) or a contention refusal bounds nothing."""
import importlib.util
import pathlib

_SPEC = importlib.util.spec_from_file_location(
    "mgxdesign_report", pathlib.Path(__file__).resolve().parents[1] / "perf/mgxdesign/report.py")
report = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(report)


def test_ladder_throw_kill_is_a_measurement():
    r = {"rc": -9, "tail": "Out of Memory ...\n\nFATAL: ended at the throw, not at the rung budget\n"}
    assert report.ran(r) and not report.timed_out(r)


def test_budget_kill_is_kept_and_inconclusive():
    r = {"rc": -9, "tail": "... (assert.hpp:104)\n\nTIMEOUT\n"}
    assert report.ran(r) and report.timed_out(r)


def test_unsigned_kill_and_contention_are_dropped():
    assert not report.ran({"rc": -15, "tail": "Designing 1 spec(s)"})
    assert not report.ran({"rc": 75, "tail": "device contention, nothing ran: card 17 ..."})
    assert report.ran({"rc": 0, "tail": "done"})
