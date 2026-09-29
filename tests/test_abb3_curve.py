"""Reading the leg's history back: the torn lines, the replayed steps, and the rate that projects.

Three properties, each one a thing the 2026-09-19 host resets did to this run's own files:

1. a NUL-torn line is recovered rather than dropped, because dropping it turns a resume into an
   apparent jump;
2. the steps a resume replays are compared against their first copy, which is a bit-exactness
   check on every restart the leg takes, on data the run already wrote;
3. the cadence excludes the restart gaps and the realized rate charges them, so the step count
   at the cap is quoted as the band it is rather than as the optimistic half of it.

No device: all of it is reading files.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tt_bio.train.history import read_rows

REPO = Path(__file__).resolve().parents[1]
CURVE = REPO / "scripts" / "abb3_port" / "curve.py"


def _row(step, t, loss=5.2, digest=None):
    return {"step": step, "t": t, "wall": 11.3, "loss": loss, "lr": 5e-4, "grad_norm": 0.1,
            "digest": digest or f"d{step:08d}", "loss_terms": {"fape_backbone": 0.5}}


def _write(path: Path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _curve(out, *extra):
    p = subprocess.run([sys.executable, str(CURVE), "--out", str(out), "--json", *extra],
                       capture_output=True, text=True, cwd=str(REPO))
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


def test_a_nul_torn_line_is_recovered_with_its_record_intact(tmp_path):
    """ext4 journalled the append's size across the crash and not its data, and the next
    process appended into the same block. Line 694 of both rank files, 2026-09-19."""
    path = tmp_path / "h.jsonl"
    _write(path, [_row(682, 1000.0), _row(683, 1011.0)])
    path.write_text(path.read_text() + "\x00" * 697 + json.dumps(_row(684, 1022.0)) + "\n")
    rows = read_rows(path)
    assert [r["step"] for r in rows] == [682, 683, 684]
    assert rows[-1]["digest"] == "d00000684"


def test_a_last_line_cut_off_mid_write_is_dropped_and_the_rest_is_readable(tmp_path):
    path = tmp_path / "h.jsonl"
    _write(path, [_row(10, 1000.0), _row(11, 1011.0)])
    path.write_text(path.read_text() + '{"step": 12, "t": 10')
    assert [r["step"] for r in read_rows(path)] == [10, 11]


def test_a_missing_file_reads_as_nothing_rather_than_raising(tmp_path):
    assert read_rows(tmp_path / "nope.jsonl") == []


def test_the_replayed_steps_are_compared_against_their_first_copy(tmp_path):
    """A resume replays from the checkpoint, so those steps are in the file twice. Identical
    copies are the proof that a reset does not change the result."""
    rows = [_row(683, 1000.0), _row(684, 1011.0), _row(685, 1022.0),
            _row(684, 1400.0), _row(685, 1411.0), _row(686, 1422.0)]
    _write(tmp_path / "history-rank0.jsonl", rows)
    state = _curve(tmp_path)
    assert state["replay"]["replayed_steps"] == 2
    assert state["replay"]["disagreements"] == []
    assert state["distinct_steps"] == 4 and state["last_step"] == 686


def test_a_replayed_step_that_came_back_different_is_named_and_not_averaged_away(tmp_path):
    """The one disagreement this leg has is deliberate: step 552 was re-run after the learning
    rate was repaired, so its loss is bit-identical and the digest after the update is not."""
    rows = [_row(551, 1000.0), _row(552, 1011.0, loss=5.232, digest="9b9ea537"),
            _row(552, 1400.0, loss=5.232, digest="1266e7c3"), _row(553, 1411.0)]
    _write(tmp_path / "history-rank0.jsonl", rows)
    state = _curve(tmp_path)
    d = state["replay"]["disagreements"]
    assert len(d) == 1 and d[0]["step"] == 552
    assert d[0]["first"]["loss"] == d[0]["again"]["loss"], "the forward ran on the same weights"
    assert d[0]["first"]["digest"] != d[0]["again"]["digest"]


def test_the_cadence_excludes_the_restart_gap_and_the_realized_rate_charges_it(tmp_path):
    """Two rates, because they answer different questions. Quoting only the clean one projects
    a step count the leg will not reach; quoting only the realized one hides the step time."""
    rows = [_row(i, 1000.0 + i * 10.0) for i in range(1, 21)]
    # A reset: the run dies at 20, comes back at 16, and the gap is 20 minutes.
    rows += [_row(i, 1000.0 + 200.0 + 1200.0 + (i - 15) * 10.0) for i in range(16, 31)]
    _write(tmp_path / "history-rank0.jsonl", rows)
    state = _curve(tmp_path)
    cad = state["cadence_seconds"]
    assert cad["mean"] == 10.0 and cad["gap_count"] == 1
    assert cad["gap_seconds"] > 1200.0
    # 30 steps minus the first, over the whole span including the 20 minutes nobody computed.
    assert state["realized_seconds_per_step"] > 10.0
    assert state["restarts"] == [{"died_at": 20, "resumed_at": 16, "steps_redone": 5,
                                  "seconds_lost": 1210.0}]


def test_an_empty_run_directory_is_the_readers_failure_and_not_a_curve(tmp_path):
    p = subprocess.run([sys.executable, str(CURVE), "--out", str(tmp_path)],
                       capture_output=True, text=True, cwd=str(REPO))
    assert p.returncode == 2


def test_the_binned_curve_names_the_stage_timers_as_seconds(tmp_path):
    """`loss_terms` is per-stage WALL CLOCK, and the only loss value in a row is `loss`.

    `abodybuilder3_step.py:102` says so, but the values -- 0.50, 0.52, 0.12, 0.21, 0.10 --
    are entirely plausible as loss components, and they were printing in the same unlabelled
    table as `loss` and `grad_norm`. A component breakdown built off them looks right and is a
    breakdown of timings. The `_s` suffix is what stops the cap write-up shipping one.
    """
    _write(tmp_path / "history-rank0.jsonl",
           [_row(i, 1000.0 + i * 10.0, loss=6.0 - i * 0.01) for i in range(1, 101)])
    state = _curve(tmp_path, "--bins", "4")
    assert len(state["curve"]) == 4
    assert [b["n"] for b in state["curve"]] == [25, 25, 25, 25]
    assert state["curve"][0]["loss"] > state["curve"][-1]["loss"]
    assert state["curve"][0]["fape_backbone_s"] == 0.5
    assert "fape_backbone" not in state["curve"][0], (
        "an unsuffixed name is the one a reader mistakes for a loss component")


# ------------------------------------------------ the pre-registered degeneracy bar

BAR = REPO / "perf" / "train_i_run" / "degeneracy_bar.json"


def _bar() -> dict:
    return json.loads(BAR.read_text())


def _leg(path: Path, last: int, loss: float, fbb: float):
    """A history sitting at one point, long enough to fill the bar's window."""
    _write(path, [dict(_row(k, 1000.0 + 12.0 * k, loss=loss), final_output_backbone=fbb,
                       fape=2.54, supervised_chi=0.5) for k in range(1, last + 1)])


def test_the_bar_is_written_before_the_reading_and_says_what_it_is_not():
    """A bar chosen by someone who can see the curve is not a bar, so this one is a file."""
    bar = _bar()
    assert bar["check_steps"] == [1935, 3870] and bar["window"] == 200
    assert set(bar["terms"]) == {"final_output_backbone", "loss"}
    for term, refs in bar["terms"].items():
        assert refs["leg2_step_75_174"] < refs["void_leg"], (
            f"{term}: the leg-2 reference is not below the void one, so the bar would be "
            "satisfied by sitting still")
    assert "not an accuracy bar" in bar["_not_an_accuracy_bar"].lower() or \
        "says nothing about CDR-H3" in bar["_not_an_accuracy_bar"]


def test_a_run_parked_at_the_void_coordinates_fails_the_bar(tmp_path):
    """The failure this whole check exists for, and the campaign has had it once already."""
    bar = _bar()
    _leg(tmp_path / "history-rank0.jsonl", 1935,
         loss=bar["terms"]["loss"]["void_leg"],
         fbb=bar["terms"]["final_output_backbone"]["void_leg"])
    got = _curve(tmp_path)["degeneracy"]
    at = {e["step"]: e for e in got}
    assert at[1935]["verdict"] == "FAIL"
    assert at[3870]["verdict"] == "NOT-REACHED"
    against = at[1935]["terms"]["final_output_backbone"]["against"]
    assert against["void_leg"]["ok"] is False and against["leg2_step_75_174"]["ok"] is False


def test_a_run_below_both_references_passes_it(tmp_path):
    """The control. Same file, same window, only the coordinates moved."""
    bar = _bar()
    _leg(tmp_path / "history-rank0.jsonl", 1935,
         loss=bar["terms"]["loss"]["leg2_step_75_174"] - 0.5,
         fbb=bar["terms"]["final_output_backbone"]["leg2_step_75_174"] - 0.5)
    at = {e["step"]: e for e in _curve(tmp_path)["degeneracy"]}
    assert at[1935]["verdict"] == "PASS"


def test_a_run_that_has_not_reached_the_check_step_reports_that_and_not_a_pass(tmp_path):
    """A bar that reads PASS because the run has not got there yet is worse than no bar."""
    _leg(tmp_path / "history-rank0.jsonl", 300, loss=1.0, fbb=1.0)
    at = {e["step"]: e for e in _curve(tmp_path)["degeneracy"]}
    assert at[1935]["verdict"] == "NOT-REACHED" and at[1935]["last_step"] == 300


def test_the_binned_curve_carries_the_loss_components_the_docstring_promises(tmp_path):
    """`a total that falls while one term climbs is a different result` -- and until this pass
    the only per-term columns in the report were the stage TIMERS. `fape` is flat at 2.54 on
    this leg while `final_output_backbone` falls, which is exactly the shape that sentence is
    about, and it was not on the page."""
    _leg(tmp_path / "history-rank0.jsonl", 40, loss=4.7, fbb=3.3)
    row = _curve(tmp_path)["curve"][0]
    for k in ("fape", "supervised_chi", "final_output_backbone"):
        assert k in row and row[k] is not None, f"{k} is a loss and is missing from the curve"
        assert k + "_s" not in row, f"{k} is not a stage timer and must not carry the _s suffix"
