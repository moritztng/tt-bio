"""`perf/b2p_soak/census.py`: the instrument that has to catch a bad resume.

The soak's claim about a box lost mid-campaign is that every accepted design survives the
restart and nothing is charged twice. That claim is only as good as the diff behind it, so the
diff is tested against a folder broken on purpose: a design whose structure was rewritten, a
design that vanished, a trajectory number charged again, a recipe designed twice. An instrument
that says RESUME CLEAN about a broken folder would let a real defect through as a pass.

Card-free and dependency-free: `census.py` is stdlib only, so this runs anywhere.
"""
import csv
import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CENSUS = ROOT / "perf" / "b2p_soak" / "census.py"


def load_census():
    spec = importlib.util.spec_from_file_location("b2p_census", CENSUS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


census = load_census()


def write_table(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as table:
        writer = csv.DictWriter(table, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


#: Where the two layouts BindCraft 2 has written keep their tables. A project started before the
#: stage folders existed has `trajectories.csv` and `accepted.csv` at its root; every campaign
#: started since keeps them under the stage folders, and the accepted designs live in the RANK
#: stage table rather than a file of their own (`campaign_output.stage_table`/`accepted_table`).
#: The census is read against BOTH, because a fresh campaign -- the soak's own -- is the second.
LAYOUTS = {
    "legacy": ("trajectories.csv", "accepted.csv"),
    "stages": ("1_Trajectories/!_Trajectories.csv", "3_Ranked/!_Ranked.csv"),
}


def campaign(tmp_path, trajectories: int, accepted: list[str], charged: int | None = None,
             layout: str = "legacy"):
    """A project folder shaped like one BindCraft 2 writes."""
    project = tmp_path / "project"
    project.mkdir(parents=True, exist_ok=True)
    trajectory_table, accepted_table = LAYOUTS[layout]
    write_table(project / trajectory_table,
                [{"design": f"design_{n}", "trajectory": str(n), "hash": f"hash{n}",
                  "terminated": ""} for n in range(1, trajectories + 1)])
    if accepted:
        write_table(project / accepted_table,
                    [{"design": name, "hash": f"hash{index + 1}"}
                     for index, name in enumerate(accepted)])
        rank = project / "3_Ranked"
        rank.mkdir(exist_ok=True)
        for name in accepted:
            (rank / f"{name}.cif").write_text(f"structure of {name}\n")
    (project / ".campaign_state.json").write_text(json.dumps(
        {"trajectories": charged if charged is not None else trajectories,
         "accepted": len(accepted), "attempted": [f"hash{n}" for n in range(1, trajectories + 1)]}))
    return project


def test_a_campaign_that_carried_on_cleanly_is_clean(tmp_path):
    project = campaign(tmp_path, trajectories=3, accepted=["design_1_seq0"])
    before = census.census(project)
    # The resume runs three more trajectories and accepts one more design.
    project = campaign(tmp_path, trajectories=6,
                       accepted=["design_1_seq0", "design_4_seq0"])
    broken, moved = census.compare(before, census.census(project))
    assert broken == []
    assert "trajectory rows 3 -> 6" in moved
    assert "accepted designs 1 -> 2" in moved


def test_a_rewritten_accepted_structure_is_caught(tmp_path):
    project = campaign(tmp_path, trajectories=2, accepted=["design_1_seq0"])
    before = census.census(project)
    (project / "3_Ranked" / "design_1_seq0.cif").write_text("a different structure\n")
    broken, _moved = census.compare(before, census.census(project))
    assert any("was rewritten" in line for line in broken)


def test_an_accepted_design_that_disappears_is_caught(tmp_path):
    project = campaign(tmp_path, trajectories=2, accepted=["design_1_seq0", "design_2_seq0"])
    before = census.census(project)
    project = campaign(tmp_path, trajectories=2, accepted=["design_1_seq0"])
    (project / "3_Ranked" / "design_2_seq0.cif").unlink(missing_ok=True)
    broken, _moved = census.compare(before, census.census(project))
    assert any("design_2_seq0 is gone after the resume" in line for line in broken)


def test_a_structure_lost_while_its_row_survives_is_caught(tmp_path):
    project = campaign(tmp_path, trajectories=2, accepted=["design_1_seq0"])
    before = census.census(project)
    (project / "3_Ranked" / "design_1_seq0.cif").unlink()
    broken, _moved = census.compare(before, census.census(project))
    assert any("lost every structure file" in line for line in broken)


def test_a_trajectory_number_charged_twice_is_caught(tmp_path):
    project = campaign(tmp_path, trajectories=2, accepted=[])
    before = census.census(project)
    with (project / "trajectories.csv").open("a", newline="") as table:
        table.write("design_2,2,hash9,\n")
    broken, _moved = census.compare(before, census.census(project))
    assert any("trajectory numbers written twice: ['2']" in line for line in broken)


def test_a_recipe_designed_twice_is_caught(tmp_path):
    project = campaign(tmp_path, trajectories=2, accepted=[])
    before = census.census(project)
    with (project / "trajectories.csv").open("a", newline="") as table:
        table.write("design_3,3,hash1,\n")
    broken, _moved = census.compare(before, census.census(project))
    assert any("recipe designed twice: ['hash1']" in line for line in broken)


def test_a_trajectory_row_lost_in_the_restart_is_caught(tmp_path):
    project = campaign(tmp_path, trajectories=3, accepted=[])
    before = census.census(project)
    project = campaign(tmp_path, trajectories=2, accepted=[])
    broken, _moved = census.compare(before, census.census(project))
    assert any("present before the interruption is gone" in line for line in broken)


def test_the_census_reads_the_state_the_campaign_charged(tmp_path):
    """The charged count is the number the interruption evidence turns on, so it is recorded
    verbatim rather than recomputed from the rows."""
    project = campaign(tmp_path, trajectories=3, accepted=[], charged=5)
    taken = census.census(project)
    assert taken["state"]["trajectories"] == 5
    assert len(taken["trajectories"]) == 3


def test_an_unreadable_state_file_does_not_stop_the_census(tmp_path):
    project = campaign(tmp_path, trajectories=1, accepted=[])
    (project / ".campaign_state.json").write_text('{"trajectories": 1, "accep')
    assert census.census(project)["state"] == "unreadable"


def test_a_missing_folder_exits_two_rather_than_raising(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["census.py", str(tmp_path / "nothing")])
    assert census.main() == 2


def test_the_exit_code_is_the_verdict(tmp_path, monkeypatch, capsys):
    """A shell driving the drill reads the exit code, so it must be non-zero exactly when the
    resume broke something."""
    project = campaign(tmp_path, trajectories=2, accepted=["design_1_seq0"])
    before_path = tmp_path / "before.json"
    monkeypatch.setattr("sys.argv", ["census.py", str(project), "--out", str(before_path)])
    assert census.main() == 0

    monkeypatch.setattr("sys.argv", ["census.py", str(project), "--against", str(before_path)])
    assert census.main() == 0
    assert "RESUME CLEAN" in capsys.readouterr().out

    (project / "3_Ranked" / "design_1_seq0.cif").write_text("rewritten\n")
    monkeypatch.setattr("sys.argv", ["census.py", str(project), "--against", str(before_path)])
    assert census.main() == 1
    assert "RESUME BROKE" in capsys.readouterr().out


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_both_layouts_bindcraft_has_written_are_read(tmp_path, layout):
    """The tables a fresh campaign writes, not only the ones a fixture invented.

    Measured on the live soak folder on dev Galaxy .108, 2026-09-30: `3_Ranked/!_Ranked.csv` is
    where acceptances go ("rewritten as each design is accepted", says the campaign's own
    header), and an earlier census that looked for `accepted.csv` and `!_Accepted.csv` found
    neither. It reported `accepted 0` and would have compared two empty lists across the
    interruption and printed RESUME CLEAN -- an instrument green about nothing.
    """
    project = campaign(tmp_path, trajectories=2, accepted=["design_1_seq0"], layout=layout)
    taken = census.census(project)
    assert [row["trajectory"] for row in taken["trajectories"]] == ["1", "2"]
    assert [design["design"] for design in taken["accepted"]] == ["design_1_seq0"]
    assert taken["accepted"][0]["structures"], "the structure written for an accepted design"


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_a_rewritten_structure_is_caught_in_either_layout(tmp_path, layout):
    project = campaign(tmp_path, trajectories=2, accepted=["design_1_seq0"], layout=layout)
    before = census.census(project)
    (project / "3_Ranked" / "design_1_seq0.cif").write_text("a different structure\n")
    broken, _moved = census.compare(before, census.census(project))
    assert any("was rewritten" in line for line in broken)
