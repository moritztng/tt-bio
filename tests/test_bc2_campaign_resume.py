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


def test_a_fresh_folder_says_nothing_about_resumption(tmp_path):
    from tt_bio import bindcraft2

    assert bindcraft2.print_resumption(str(tmp_path / "fresh")) == ""


def test_resumption_names_the_trajectories_the_interruption_cost(tmp_path, capsys):
    """The report a researcher gets when they restart an interrupted campaign.

    It has to say three numbers -- charged, in the table, accepted -- and then the consequence,
    because the consequence is the part nothing told them: a campaign asked for 24 trajectories
    and interrupted twice delivers 22.
    """
    from tt_bio import bindcraft2

    project = str(tmp_path / "campaign")
    pathlib.Path(project).mkdir(parents=True)
    for number in (1, 2, 3):
        completed_trajectory(project, number, f"hash{number}")
    accepted_design(project, "design_1_seq0", "hash1")
    state_file(project).write_text(json.dumps(
        {"trajectories": 5, "accepted": 1,
         "attempted": ["hash1", "hash2", "hash3", "hash4", "hash5"]}))

    line = bindcraft2.print_resumption(project, max_trajectories=24)
    assert "5 trajectories already charged" in line
    assert "3 in the trajectory table" in line
    assert "1 accepted" in line
    assert "2 claimed trajectories never finished" in line
    assert "runs 2 fewer than you asked for" in line
    assert "max_trajectories to 26" in line
    assert line in capsys.readouterr().out


def test_resumption_without_a_state_file_says_which_side_it_read(tmp_path):
    """Recovered from the tables, the report says so and says what that costs: the interrupted
    trajectory's number is claimed again, though its recipe is still declined."""
    from tt_bio import bindcraft2

    project = str(tmp_path / "campaign")
    pathlib.Path(project).mkdir(parents=True)
    completed_trajectory(project, 1, "hash1")

    line = bindcraft2.print_resumption(project, max_trajectories=10)
    assert "`.campaign_state.json` is absent" in line
    assert "1 trajectory already charged" in line
    assert "no design is repeated" in line


def test_an_unreadable_state_file_does_not_stop_the_campaign(tmp_path):
    """A truncated state file is read as absent rather than raised out of the campaign's first
    second. `locked_progress` writes it atomically, so this should not happen, but a full disk
    or a hard reset is not something to hand a researcher a traceback for."""
    from tt_bio import bindcraft2

    project = str(tmp_path / "campaign")
    pathlib.Path(project).mkdir(parents=True)
    completed_trajectory(project, 1, "hash1")
    state_file(project).write_text('{"trajectories": 2, "accep')

    line = bindcraft2.print_resumption(project, max_trajectories=10)
    assert "absent or unreadable" in line


def test_concurrent_threads_claim_unique_trajectory_numbers(tmp_path):
    """The claim that tt-bio's interleave rests on, tested where it is cheap to test.

    `trajectories_per_card > 1` runs N trajectories as THREADS in one process, each taking its
    own number out of the same `.campaign_state.json` (`tt_bio.bindcraft2.run_campaign`). The
    lock under it is `flock` on the project directory, taken on a freshly opened fd per call.
    `flock` is held on the open file description rather than the process, so two threads with
    their own fds do exclude each other -- but that is a property of the syscall, not something
    the code says, and a campaign that hands two trajectories the same number writes two designs
    into one folder. So: eight threads, two hundred claims, every number exactly once.
    """
    import threading

    project = str(tmp_path / "campaign")
    progress = CampaignProgress(project, requested_designs=10_000, max_trajectories=200)
    claimed, lock = [], threading.Lock()

    def claim_until_empty():
        while True:
            got = progress.claim_trajectory()
            if got is None:
                return
            with lock:
                claimed.append(got[0])

    threads = [threading.Thread(target=claim_until_empty) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    assert not any(thread.is_alive() for thread in threads), "a claim deadlocked"
    assert sorted(claimed) == list(range(1, 201))


def test_concurrent_processes_do_not_lose_an_accepted_design(tmp_path):
    """Two campaigns in one project folder, which is what a researcher does by accident.

    BindCraft 2's own design workers are separate processes sharing a folder, so the same path
    covers a second campaign started against a folder that already has one running. An accepted
    design is counted with a read-modify-write of the state file, and a lost update there means
    a campaign hands back fewer designs than it wrote, or stops late.
    """
    import multiprocessing

    project = str(tmp_path / "campaign")
    CampaignProgress(project, requested_designs=10_000).campaign_status()

    def record(count):
        progress = CampaignProgress(project, requested_designs=10_000)
        for _ in range(count):
            progress.record_accepted_design()

    context = multiprocessing.get_context("fork")
    workers = [context.Process(target=record, args=(20,)) for _ in range(5)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=120)
    assert all(worker.exitcode == 0 for worker in workers), (
        f"a worker did not finish cleanly: {[w.exitcode for w in workers]}")
    accepted, _trajectories = CampaignProgress(project, requested_designs=10_000).campaign_status()
    assert accepted == 100, "five processes, twenty designs each, none lost to a lost update"
