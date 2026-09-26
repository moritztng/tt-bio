"""`round_ab` must refuse a project directory a campaign has already finished in.

`run_campaign` resumes from `.campaign_state.json`. Pointing a second A/B at the same `--out`
therefore runs ZERO rounds and exits 0: the campaign prints "1 trajectories ran and none were
accepted", the harness collects an empty summary and empty lever counts, and the card goes back
having measured nothing. That happened at 04:28Z on 2026-09-26, on a card window this row had
waited six minutes for.
"""
import sys

import pytest

sys.path.insert(0, "perf/bcx_bwbytes")
sys.path.insert(0, ".")


def test_a_fresh_directory_is_accepted(tmp_path):
    import round_ab
    round_ab.require_fresh_project(str(tmp_path))


@pytest.mark.parametrize("name", [".campaign_state.json", "1_Trajectories"])
def test_either_marker_of_a_previous_campaign_refuses(tmp_path, name):
    """Both markers, because a run killed before it wrote its state still leaves the trajectory
    directory, and that is enough for `run_campaign` to consider the work done."""
    import round_ab
    p = tmp_path / name
    p.mkdir() if name.endswith("Trajectories") else p.write_text("{}")
    with pytest.raises(SystemExit, match="already carries campaign state"):
        round_ab.require_fresh_project(str(tmp_path))


def test_reuse_project_is_the_explicit_way_past_it(tmp_path):
    """The control. A guard with no override gets deleted the first time someone means it."""
    import round_ab
    (tmp_path / ".campaign_state.json").write_text("{}")
    round_ab.require_fresh_project(str(tmp_path), reuse=True)


def test_the_guard_runs_before_anything_expensive():
    """It must refuse in a second, not after the weight load and the JAX compile -- the whole
    value is handing the card back straight away. Pinned by position: the call sits between the
    mkdir and the settings read."""
    import inspect

    import round_ab
    src = inspect.getsource(round_ab.main)
    i_mk = src.index("mkdir(parents=True")
    i_guard = src.index("require_fresh_project(")
    i_model = src.index("device_model()")
    assert i_mk < i_guard < i_model
