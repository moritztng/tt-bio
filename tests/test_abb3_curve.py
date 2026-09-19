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
