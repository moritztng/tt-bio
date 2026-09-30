"""What a BindCraft 2 campaign remembers when it is interrupted and restarted.

A design campaign runs for hours, so a researcher will interrupt one: a box goes down, a lease
expires, they hit Ctrl-C to change a filter. BindCraft 2 supports carrying on -- `resume: true`,
without which preflight refuses a folder that already holds output -- and the accounting behind
it is `bindcraft.campaign_output.CampaignProgress`: a `.campaign_state.json` under the project
folder holding `trajectories`, `accepted` and the recipe hashes already attempted, behind an
`flock` on the project directory, written `.partial` then `os.replace` so a kill cannot truncate
it. When that file is absent the counts are recovered from the campaign's own tables.

These tests pin what a resumed campaign actually does, card-free, because the accounting is pure
host code and a defect in it is worth finding without spending five hours of chip time. They are
characterisation, not aspiration: where the behaviour is surprising the test says so in its name
and asserts the surprising thing, so that a change upstream shows up here as a failure to read
rather than as a silent difference in what a campaign counts.
"""
import json
import pathlib

import pytest

campaign_output = pytest.importorskip(
    "bindcraft.campaign_output",
    reason="BindCraft 2 is not importable here; run with BCX_BC2 on PYTHONPATH")

CampaignProgress = campaign_output.CampaignProgress
TRAJECTORY_STAGE = campaign_output.TRAJECTORY_STAGE
append_campaign_metrics = campaign_output.append_campaign_metrics
stage_table = campaign_output.stage_table
accepted_table = campaign_output.accepted_table
csv_row_count = campaign_output.csv_row_count


def completed_trajectory(project: str, number: int, identity: str) -> None:
    """A trajectory row, written the way `run_campaign_arm` writes one when it finishes."""
    append_campaign_metrics(stage_table(project, TRAJECTORY_STAGE),
                            {"design": f"design_{number}", "trajectory": number,
                             "length": 146, "hash": identity, "terminated": ""})


def accepted_design(project: str, name: str, identity: str) -> None:
    append_campaign_metrics(accepted_table(project),
                            {"design": name, "length": 146, "hash": identity})


def state_file(project: str) -> pathlib.Path:
    return pathlib.Path(project) / ".campaign_state.json"


def test_an_interrupted_trajectory_is_charged_to_the_budget(tmp_path):
    """The budget is spent when a trajectory is CLAIMED, not when it produces a row.

    `claim_trajectory` increments the count before the trajectory runs and the row is appended
    only when it finishes, so a campaign killed mid-trajectory resumes having paid for work it
    has no row for. A researcher who asks for three trajectories and is interrupted once gets
    three claims and two rows.
    """
    project = str(tmp_path / "campaign")
    progress = CampaignProgress(project, requested_designs=500, max_trajectories=3)

    assert progress.claim_trajectory() == (1, 0)
    assert progress.claim_recipe("hash1") is True
    completed_trajectory(project, 1, "hash1")

    # Trajectory 2 is claimed and its recipe recorded, then the process dies: no row.
    assert progress.claim_trajectory() == (2, 0)
    assert progress.claim_recipe("hash2") is True

    # The restart. A fresh object on the same folder is what a resumed campaign builds.
    resumed = CampaignProgress(project, requested_designs=500, max_trajectories=3)
    assert resumed.claim_trajectory() == (3, 0), "a resumed campaign carries on rather than restarting"
    assert resumed.claim_recipe("hash3") is True
    completed_trajectory(project, 3, "hash3")

    assert resumed.claim_trajectory() is None, "the budget of 3 is spent"
    accepted, trajectories = resumed.campaign_status()
    assert (accepted, trajectories) == (0, 3)
    assert csv_row_count(stage_table(project, TRAJECTORY_STAGE)) == 2, (
        "two rows for three charged trajectories: the interrupted one is paid for and not retried")


def test_the_two_resume_paths_disagree_about_an_interrupted_trajectory(tmp_path):
    """Resuming with the state file and resuming without it give different trajectory numbers.

    The state file remembers the interrupted claim; recovery from the tables cannot see it,
    because an interrupted trajectory wrote no row. So a researcher who deletes
    `.campaign_state.json` -- a file nothing tells them about -- gets one more trajectory than
    one who keeps it, and the same trajectory number is used twice across the two runs.
    """
    project = str(tmp_path / "campaign")
    pathlib.Path(project).mkdir(parents=True)
    completed_trajectory(project, 1, "hash1")
    state_file(project).write_text(json.dumps(
        {"trajectories": 2, "accepted": 0, "attempted": ["hash1", "hash2"],
         "rejections": {"terminated": {"completed": 1}, "failed_filters": {},
                        "candidates_scored": 0, "candidates_rejected": 0}}))

    with_state = CampaignProgress(project, requested_designs=500, max_trajectories=10)
    assert with_state.claim_trajectory() == (3, 0)

    state_file(project).unlink()
    without_state = CampaignProgress(project, requested_designs=500, max_trajectories=10)
    assert without_state.claim_trajectory() == (2, 0), (
        "recovered from the tables, the interrupted trajectory 2 is claimed a second time")


def test_a_resumed_campaign_does_not_rerun_a_recipe_it_already_designed(tmp_path):
    """Recovery from the tables still refuses a recipe that has a row, so no design is repeated.

    This is the half that works: the recipe hashes come from the trajectory table itself, so a
    resumed campaign that has lost its state file still declines every trajectory it completed.
    """
    project = str(tmp_path / "campaign")
    pathlib.Path(project).mkdir(parents=True)
    completed_trajectory(project, 1, "hash1")
    completed_trajectory(project, 2, "hash2")
    assert not state_file(project).exists()

    resumed = CampaignProgress(project, requested_designs=500, max_trajectories=10)
    assert resumed.claim_recipe("hash1") is False
    assert resumed.claim_recipe("hash2") is False
    assert resumed.claim_recipe("hash3") is True


def test_a_resumed_campaign_counts_the_designs_it_already_accepted(tmp_path):
    """Accepted designs are recovered from the accepted table, so the stop condition holds.

    The campaign stops on acceptance as well as on budget, and a resumed campaign must not hand
    back a second set of designs because it forgot the first. With one accepted row and one
    design requested, a resumed campaign claims no trajectory at all.
    """
    project = str(tmp_path / "campaign")
    pathlib.Path(project).mkdir(parents=True)
    completed_trajectory(project, 1, "hash1")
    accepted_design(project, "design_1_seq0", "hash1")

    resumed = CampaignProgress(project, requested_designs=1, max_trajectories=10)
    assert resumed.claim_trajectory() is None, (
        "the design it already accepted is the one that was asked for, so it stops")
    assert resumed.campaign_status() == (1, 1)

    wants_more = CampaignProgress(project, requested_designs=2, max_trajectories=10)
    assert wants_more.claim_trajectory() == (2, 1), "asked for two, it carries on from one"


def test_the_budget_stops_a_campaign_that_has_accepted_nothing(tmp_path):
    """A campaign whose filters accept nothing stops at `max_trajectories` rather than running
    forever, and it stops without needing the state file."""
    project = str(tmp_path / "campaign")
    pathlib.Path(project).mkdir(parents=True)
    for number in range(1, 5):
        completed_trajectory(project, number, f"hash{number}")
    state_file(project).unlink(missing_ok=True)

    progress = CampaignProgress(project, requested_designs=500, max_trajectories=4)
    assert progress.claim_trajectory() is None
    assert progress.campaign_status() == (0, 4)
