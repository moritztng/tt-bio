#!/usr/bin/env python3
"""An arm says how it ended, and it says it on the exit that crashed.

`accept_s3` passed all five design stages, died in `predict_validation_ensemble`, and its
wrapper stamped `arm exit 0` over the crash: the shell echoed `$?` after a `while kill -0 $PID`
loop, which is the status of the kill that finally failed, always 0. Every row that reads an arm
stamp to decide whether a trajectory completed read a clean exit for a dead trajectory.

Two halves, both checked here, because either one alone still lies:
  * the wrapper takes the status from `wait $PID`, not from a probe loop;
  * the arm writes its own terminal status into `arm_stamp.json`, so the signal survives a
    launcher that is detached, or absent.

Card-free and about 20 s: `run_campaign` is stubbed, so nothing folds.

    /home/ttuser/bcx_e2e_venv/bin/python3 -m pytest perf/bcx_predictor/test_arm_status.py -x -q
"""
import json
import pathlib
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(HERE), str(ROOT), str(ROOT / "perf" / "bcx_afgrad"), str(ROOT / "perf" / "bcx_stack")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _run(tmp_path, monkeypatch, outcome):
    """Run run_arm.main() with BindCraft 2's campaign stubbed, return the stamp it left."""
    import bc2_state as B  # noqa: F401  -- puts BindCraft 2 on the path
    import run_arm
    from bindcraft import campaign

    project = tmp_path / "arm"
    seen = {}

    def stub(settings, folder, **kw):
        # The launch-time stamp is already on disk here, and it has to say the arm is running:
        # a stamp with no status at all is how a crashed arm reads as "never started".
        seen["at_launch"] = json.loads((pathlib.Path(folder) / "arm_stamp.json").read_text())
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(campaign, "run_campaign", stub)
    monkeypatch.setattr(sys, "argv", ["run_arm.py", "--arm", "reference", "--trajectories", "1",
                                      "--out", str(project)])
    raised = None
    try:
        run_arm.main()
    except BaseException as exc:          # noqa: BLE001 -- the point is that it re-raises
        raised = exc
    return seen["at_launch"], json.loads((project / "arm_stamp.json").read_text()), raised


def test_running_then_completed(tmp_path, monkeypatch):
    launch, stamp, raised = _run(tmp_path, monkeypatch, 1)
    assert raised is None
    assert launch["status"] == "running"
    assert stamp["status"] == "completed"
    assert stamp["trajectories_run"] == 1
    assert stamp["ended_utc"] and stamp["wall_seconds"] >= 0


def test_a_crash_is_stamped_as_a_crash(tmp_path, monkeypatch):
    err = KeyError("'model_1_ptm' is not in the pool")
    launch, stamp, raised = _run(tmp_path, monkeypatch, err)
    assert isinstance(raised, KeyError)          # the arm still fails, loudly
    assert launch["status"] == "running"
    assert stamp["status"] == "crashed"
    assert "model_1_ptm" in stamp["error"]
    assert "trajectories_run" not in stamp       # nothing completed, so nothing is claimed
    assert stamp["ended_utc"]


def test_an_interrupt_is_not_left_running(tmp_path, monkeypatch):
    # SIGINT is how this fleet stops a detached arm. `except Exception` would miss it and leave
    # the stamp reading "running" forever, which is the same false signal pointing the other way.
    _, stamp, raised = _run(tmp_path, monkeypatch, KeyboardInterrupt())
    assert isinstance(raised, KeyboardInterrupt)
    assert stamp["status"] == "crashed"
    assert "KeyboardInterrupt" in stamp["error"]


@pytest.mark.parametrize("construct,expected", [
    # The defect, kept executable so the fix is graded against it rather than described.
    ("sh -c 'exit 7' & PID=$!; while kill -0 $PID 2>/dev/null; do sleep 0.05; done; echo $?", 0),
    ("sh -c 'exit 7' & PID=$!; wait $PID; echo $?", 7),
])
def test_wait_reports_the_arm_and_a_probe_loop_does_not(construct, expected):
    out = subprocess.run(["bash", "-c", construct], capture_output=True, text=True, timeout=30)
    assert int(out.stdout.strip().splitlines()[-1]) == expected


def test_the_launcher_takes_its_status_from_wait():
    src = (HERE.parent / "bcx_mutate" / "launch_route_arm.sh").read_text()
    assert "wait $PID\nRC=$?" in src
    assert 'echo "arm exit $RC' in src
