"""The 5-day cap and the thing that watches it.

Ask 9115 granted the ABodyBuilder3 reproduction one board pair for 5 days of WALL CLOCK. Two
properties carry that grant and neither is obvious from reading the run loop, so both are
pinned here: the cap has to survive the ~9-47 process deaths the run is expected to take, and
a run that quietly restarted from step 1 instead of resuming has to be visible as an incident
rather than as a healthy loss curve.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tt_bio.train import deadline

REPO = Path(__file__).resolve().parents[1]
HEARTBEAT = REPO / "scripts" / "abb3_port" / "heartbeat.py"


def test_the_grant_is_anchored_once_and_a_restart_cannot_extend_it(tmp_path):
    """The failure this prevents: 47 watchdog restarts each granted a fresh 5 days."""
    first = deadline.resolve(tmp_path, days=5.0, now=1000.0)
    assert first["deadline"] == 1000.0 + 5 * 86400.0
    # The restart re-reads --days 5 off its own command line. It must not move the instant.
    again = deadline.resolve(tmp_path, days=5.0, now=1000.0 + 3 * 86400.0)
    assert again["deadline"] == first["deadline"]
    assert deadline.remaining(tmp_path, now=1000.0 + 3 * 86400.0) == pytest.approx(2 * 86400.0)
    assert deadline.remaining(tmp_path, now=1000.0 + 9 * 86400.0) == 0.0


def test_a_first_launch_without_a_grant_refuses_rather_than_guessing(tmp_path):
    with pytest.raises(ValueError, match="how long the grant is"):
        deadline.resolve(tmp_path)
    assert deadline.read(tmp_path) is None
    assert deadline.remaining(tmp_path) == 0.0


def _history(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


#: The run's own schedule, as `abb3_run` writes it: 8,395 structures at global batch 64 with
#: the short final batch kept is 132 steps per epoch, and the rest is `params.yaml`.
SPEC = {"scheduler": "CosineAnnealingWarmRestarts", "lr": 5e-4, "T_0": 50, "T_mult": 1,
        "eta_min": 0.0, "steps_per_epoch": 132}


def _lr(step):
    from tt_bio.train.abb3_run import CosineRestartsByStep
    return CosineRestartsByStep.load(SPEC).set_step(step)


def _row(step, t, digest="ab" * 16, lr=None):
    return {"step": step, "t": t, "wall": 11.3, "loss": 1.0, "digest": digest,
            "lr": _lr(step) if lr is None else lr, "grad_norm": 0.42}


def _run(out, *extra):
    # A healthy run records its schedule, so the fixture does too unless a test is about its
    # absence.
    spec = Path(out) / "schedule.json"
    if not spec.is_file():
        spec.write_text(json.dumps(SPEC))
    p = subprocess.run([sys.executable, str(HEARTBEAT), "--out", str(out), "--world", "1",
                        "--json", *extra], capture_output=True, text=True, cwd=str(REPO))
    return p.returncode, json.loads(p.stdout)


def test_a_restart_that_did_not_resume_is_an_incident(tmp_path):
    """The step counter dropping back to 1 with checkpoints on disk is the silent failure.

    It is silent precisely because the loss curve after it looks perfectly healthy -- it is a
    healthy curve, for a model that threw away three days of training.
    """
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    (ckpts / "step-000005000.safetensors").write_text("x")
    _history(tmp_path / "history-rank0.jsonl",
             [_row(4999, 1000.0), _row(5000, 1011.0), _row(1, 1200.0), _row(2, 1211.0)])
    (tmp_path / "supervisor.pid").write_text("1\n")
    code, state = _run(tmp_path, "--stall-minutes", "1e9")
    assert code == 1
    assert state["verdict"] == "INCIDENT"
    assert "did not resume" in state["incident"]
    assert state["resumes"][0]["verdict"] == "RESTARTED-FROM-SCRATCH"


def test_a_real_resume_off_a_checkpoint_is_healthy_and_counted(tmp_path):
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    (ckpts / "step-000005000.safetensors").write_text("x")
    # Died at 5120, 120 steps past the last checkpoint, and came back on 5001.
    _history(tmp_path / "history-rank0.jsonl",
             [_row(5119, 1000.0), _row(5120, 1011.0), _row(5001, 1200.0), _row(5002, 1211.0)])
    (tmp_path / "supervisor.pid").write_text("1\n")
    code, state = _run(tmp_path, "--stall-minutes", "1e9")
    assert code == 0 and state["verdict"] == "LIVE"
    assert state["resumes_survived"] == 1
    assert state["resumes"][0] == {"died_at": 5120, "resumed_at": 5001, "lost_steps": 120,
                                   "from_checkpoint": True, "verdict": "RESUMED"}
    assert state["last_step"] == 5002


def test_a_death_on_a_checkpoint_step_is_still_counted_from_the_supervisor_log(tmp_path):
    """The one restart the history cannot show: it leaves a perfectly monotonic step counter."""
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    (ckpts / "step-000005000.safetensors").write_text("x")
    _history(tmp_path / "history-rank0.jsonl",
             [_row(5000, 1000.0), _row(5001, 1200.0), _row(5002, 1211.0)])
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "supervisor.log").write_text(
        "[sup] launching 2 rank(s) on chips [0, 1] (restart 0)\n"
        "[sup] launching 2 rank(s) on chips [0, 1] (restart 1)\n")
    (tmp_path / "supervisor.pid").write_text("1\n")
    code, state = _run(tmp_path, "--stall-minutes", "1e9")
    assert code == 0 and state["verdict"] == "LIVE"
    assert state["resumes"] == [] and state["restarts_logged"] == 1
    assert state["resumes_survived"] == 1


def test_a_learning_rate_off_the_recipe_schedule_is_an_incident(tmp_path):
    """The defect this run actually shipped with: a constant lr behind a healthy loss curve.

    The run held 5e-4 for its whole length because nothing wrapped the optimizer -- 1.96x
    upstream's mean over a full schedule. Nothing in a loss curve says so, and the history
    carried no lr to compare, so the check is worth only as much as its ability to fail.
    """
    _history(tmp_path / "history-rank0.jsonl",
             [_row(6000, 1000.0), _row(6001, 1011.0, lr=5e-4)])
    (tmp_path / "supervisor.pid").write_text("1\n")
    code, state = _run(tmp_path, "--stall-minutes", "1e9")
    assert code == 1 and state["verdict"] == "INCIDENT"
    assert "away from the schedule" in state["incident"]
    assert state["lr"]["worst_step"] == 6001


def test_a_history_without_a_learning_rate_cannot_be_checked_and_says_so(tmp_path):
    rows = [_row(10, 1000.0), _row(11, 1011.0)]
    for r in rows:
        r.pop("lr")
    _history(tmp_path / "history-rank0.jsonl", rows)
    (tmp_path / "supervisor.pid").write_text("1\n")
    code, state = _run(tmp_path, "--stall-minutes", "1e9")
    assert code == 1 and state["verdict"] == "INCIDENT"
    assert "logged no lr" in state["incident"]


def test_a_stalled_step_counter_is_an_incident_and_not_a_quiet_night(tmp_path):
    _history(tmp_path / "history-rank0.jsonl", [_row(10, 1000.0), _row(11, 1011.0)])
    (tmp_path / "supervisor.pid").write_text("1\n")
    code, state = _run(tmp_path, "--stall-minutes", "15")
    assert code == 1 and state["verdict"] == "INCIDENT"
    assert "no step logged" in state["incident"]


def test_a_truncated_last_line_does_not_break_the_reader(tmp_path):
    """A SIGKILL lands mid-write. Refusing to read the history then is the wrong failure."""
    path = tmp_path / "history-rank0.jsonl"
    _history(path, [_row(10, 1000.0), _row(11, 1011.0)])
    path.write_text(path.read_text() + '{"step": 12, "t": 10')
    (tmp_path / "supervisor.pid").write_text("1\n")
    _, state = _run(tmp_path, "--stall-minutes", "1e9")
    assert state["last_step"] == 11


def test_the_projection_pays_for_the_gap_a_reset_leaves(tmp_path):
    """Rate is wall clock between steps, so a 10-minute restart gap slows the projection.

    Summing the rows' own `wall` would report the device's step time and project a run that
    never stops, which is the number that would make a dying run look on schedule.
    """
    deadline.resolve(tmp_path, days=5.0)
    rows = [_row(i, 1000.0 + i * 10.0) for i in range(1, 11)]
    rows += [_row(i, 1000.0 + 100.0 + 600.0 + (i - 10) * 10.0) for i in range(11, 21)]
    _history(tmp_path / "history-rank0.jsonl", rows)
    (tmp_path / "supervisor.pid").write_text("1\n")
    _, state = _run(tmp_path, "--stall-minutes", "1e9")
    # 19 intervals over 790 s of wall clock, which includes the 600 s the restart cost.
    # Summing the rows' own `wall` would give 190 s and project a dying run as on schedule.
    assert state["step_seconds"] == pytest.approx(790.0 / 19.0, rel=0.01)


def test_a_dead_supervisor_is_an_incident_even_while_the_history_looks_fresh(tmp_path):
    _history(tmp_path / "history-rank0.jsonl", [_row(10, 1000.0)])
    (tmp_path / "supervisor.pid").write_text("999999999\n")
    code, state = _run(tmp_path, "--stall-minutes", "1e9")
    assert code == 1 and "supervisor is gone" in state["incident"]


def test_the_cap_being_reached_is_a_verdict_and_not_an_incident(tmp_path):
    deadline.resolve(tmp_path, days=5.0, now=0.0)
    _history(tmp_path / "history-rank0.jsonl", [_row(10, 1000.0)])
    code, state = _run(tmp_path)
    assert code == 0 and state["verdict"] == "CAP-REACHED"


def test_a_grant_with_no_run_in_it_yet_reads_cleanly(tmp_path):
    """The state between anchoring the deadline and the first step logged."""
    deadline.resolve(tmp_path, days=5.0)
    code, state = _run(tmp_path)
    assert code == 0 and state["verdict"] == "NOT-STARTED"
    assert state["remaining_hours"] == pytest.approx(120.0, abs=0.1)


# ---------------------------------------------------------- the cross-chip restart order

def _sup():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "sup", REPO / "scripts" / "abb3_port" / "supervise.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Args:
    def __init__(self, chips, after=""):
        self.chips, self.chips_after_restart = chips, after


def test_the_rank_to_chip_order_only_changes_when_it_is_asked_to():
    order = _sup().chip_order
    plain = _Args("0,1")
    assert order(plain, 0) == [0, 1] and order(plain, 7) == [0, 1]
    swap = _Args("0,1", "1,0")
    assert order(swap, 0) == [0, 1], "the first launch is never the swapped one"
    assert order(swap, 1) == [1, 0] and order(swap, 9) == [1, 0]


def test_a_restart_order_that_is_not_a_reordering_of_the_grant_is_refused():
    """The grant is the pair. A restart onto a chip outside it takes a co-tenant's card."""
    with pytest.raises(SystemExit, match="not a reordering"):
        _sup().chip_order(_Args("0,1", "1,2"), 1)


# ----------------------------------------------------- what the 2026-09-19 host resets taught

def _hb():
    """The heartbeat as a module, so the wrapper can be tested without an interpreter that
    happens to be missing a dependency."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("hb", HEARTBEAT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_resume_is_still_a_resume_when_the_first_row_seen_is_past_the_checkpoint(tmp_path):
    """The predicate that latched this instrument at INCIDENT on a healthy run.

    qb2 reset at 04:29Z, the run resumed from checkpoint 683 and the first row the detector
    compared was 685. The old test was `after - 1 in checkpoints`, so 684 was not a checkpoint
    and it recorded RESTARTED-FROM-SCRATCH on an append-only record. A resume replays from
    checkpoint + 1; the first row a reader sees can be later.
    """
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    (ckpts / "step-000000683.safetensors").write_text("x")
    _history(tmp_path / "history-rank0.jsonl",
             [_row(691, 1000.0), _row(692, 1011.0), _row(685, 1200.0), _row(686, 1211.0)])
    (tmp_path / "supervisor.pid").write_text("1\n")
    code, state = _run(tmp_path, "--stall-minutes", "1e9")
    assert code == 0 and state["verdict"] == "LIVE"
    assert state["resumes"][0]["verdict"] == "RESUMED"


def test_a_jump_back_that_lands_before_every_checkpoint_is_named_rather_than_excused(tmp_path):
    """Widening the predicate must not make it unfalsifiable. This is the negative control."""
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    (ckpts / "step-000005000.safetensors").write_text("x")
    _history(tmp_path / "history-rank0.jsonl",
             [_row(5119, 1000.0), _row(5120, 1011.0), _row(300, 1200.0), _row(301, 1211.0)])
    (tmp_path / "supervisor.pid").write_text("1\n")
    code, state = _run(tmp_path, "--stall-minutes", "1e9")
    assert code == 1 and state["verdict"] == "INCIDENT"
    assert state["resumes"][0]["verdict"] == "UNEXPLAINED-JUMP-BACK"
    # And the message says where it came back, rather than the hardcoded "step 1" the old one
    # printed while its own data said 685.
    assert "step 300" in state["incident"] and "step 1," not in state["incident"]


def test_the_line_a_host_reset_tore_is_recovered_and_not_dropped(tmp_path):
    """ext4 journalled the append's size across the crash but not its data.

    Line 694 of both rank files came back as ~697 NUL bytes followed by a complete step-684
    record. Dropping it costs more than a step: the reader then sees 692 -> 685 and the resume
    detector has to decide about a jump it was never shown the middle of.
    """
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    (ckpts / "step-000000683.safetensors").write_text("x")
    path = tmp_path / "history-rank0.jsonl"
    _history(path, [_row(691, 1000.0), _row(692, 1011.0)])
    torn = "\x00" * 697 + json.dumps(_row(684, 1200.0))
    path.write_text(path.read_text() + torn + "\n"
                    + json.dumps(_row(685, 1211.0)) + "\n")
    (tmp_path / "supervisor.pid").write_text("1\n")
    code, state = _run(tmp_path, "--stall-minutes", "1e9")
    assert code == 0 and state["verdict"] == "LIVE"
    assert state["resumes"][0] == {"died_at": 692, "resumed_at": 684, "lost_steps": 9,
                                   "from_checkpoint": True, "verdict": "RESUMED"}


def test_a_relaunched_supervisor_counts_as_a_restart(tmp_path):
    """A host reset kills the supervisor too, and its replacement logs `(restart 0)`.

    Counting only the lines the supervisor numbered above zero read zero restarts across two
    reboots in 25 minutes.
    """
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "supervisor.log").write_text(
        "[sup] launching 2 rank(s) on chips [2, 3] (restart 0)\n"
        "[sup] launching 2 rank(s) on chips [2, 3] (restart 0)\n"
        "[sup] launching 2 rank(s) on chips [2, 3] (restart 0)\n")
    _history(tmp_path / "history-rank0.jsonl", [_row(10, 1000.0)])
    (tmp_path / "supervisor.pid").write_text("1\n")
    code, state = _run(tmp_path, "--stall-minutes", "1e9")
    assert code == 0 and state["restarts_logged"] == 2


def test_the_instruments_own_failure_exits_2_so_1_keeps_meaning_act():
    """It died on `ModuleNotFoundError: numpy` and exited 1, the code the runbook reserves for
    an incident. A dependency this script is missing says nothing about the run."""
    hb = _hb()

    def boom():
        raise ModuleNotFoundError("No module named 'numpy'")

    hb.main = boom
    assert hb._cli() == 2


def test_being_called_wrong_is_also_the_instrument_failing_and_not_the_run():
    p = subprocess.run([sys.executable, str(HEARTBEAT)], capture_output=True, text=True,
                       cwd=str(REPO))
    assert p.returncode == 2
