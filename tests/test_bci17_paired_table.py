"""The #17 paired table parses BindCraft 2's stage lines, including the ones no arm has hit yet.

The live arms have only printed `passed`. The rejection forms are what the issue is ABOUT -- a
trajectory that reaches `anneal` and dies at `harden` -- so they are the forms most likely to be
wrong when they first matter, and least likely to be noticed: a missed rejection reads as a blank
cell, which the table renders as "not reached yet".

The fixtures are the exact strings `bindcraft/campaign_log.py` builds, not paraphrases.
"""
import importlib.util
import pathlib

import pytest

MODULE = pathlib.Path(__file__).resolve().parents[1] / "perf" / "bci_accept" / "paired_table.py"


def load():
    spec = importlib.util.spec_from_file_location("paired_table", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ARM = """BindCraft 2 v1.0.1

=== trajectory 1 | design_l60_aaaa | accepted 0/3 ===
  passed screen design stage  i_pTM=0.85
  passed refine design stage  i_pTM=0.84  pLDDT=0.90
  passed anneal design stage  i_pTM=0.89
  rejected at harden design stage  i_pTM=0.14  due to [i_pTM, i_pDAE]
  trajectory rejected  i_pTM=0.14  due to [i_pTM]

=== trajectory 2 | design_l60_bbbb | accepted 0/3 ===
  passed screen design stage  i_pTM=0.86
  passed anneal design stage  i_pTM=0.83
  binder hallucination successful

campaign done: 1 accepted design(s) after 2 trajectory, ranked by i_pDAE
"""


@pytest.fixture
def arm(tmp_path):
    path = tmp_path / "arm.log"
    path.write_text(ARM, encoding="utf-8")
    return load().read_arm(str(path))


def test_a_rejection_is_not_read_as_a_blank(arm):
    """The failure this test exists for: a harden rejection rendering as 'not reached yet'."""
    verdict, metrics = arm["trajectories"][1]["stages"]["harden"]
    assert verdict == "REJECTED"
    assert metrics["i_pTM"] == "0.14"


def test_the_rejection_reason_is_not_parsed_as_a_metric(arm):
    """`due to [i_pTM, i_pDAE]` names the filters that failed. Those are metric-shaped words and a
    looser regex reads them back as readings with no value."""
    assert set(arm["trajectories"][1]["stages"]["harden"][1]) == {"i_pTM"}


def test_passed_stages_keep_every_reported_confidence(arm):
    verdict, metrics = arm["trajectories"][1]["stages"]["refine"]
    assert verdict == "passed"
    assert metrics == {"i_pTM": "0.84", "pLDDT": "0.90"}


def test_the_final_rejection_is_its_own_row(arm):
    assert arm["trajectories"][1]["stages"]["final"][0] == "REJECTED"


def test_trajectories_are_keyed_by_number_so_the_arms_pair(arm):
    """Names carry the arm's settings hash and differ between arms; numbers do not."""
    assert sorted(arm["trajectories"]) == [1, 2]
    assert arm["trajectories"][1]["design"] == "design_l60_aaaa"


def test_a_skipped_stage_stays_absent_rather_than_shifting_the_rows(arm):
    """Trajectory 2 never ran `refine`. The table must not slide `anneal` up into its row."""
    assert "refine" not in arm["trajectories"][2]["stages"]
    assert arm["trajectories"][2]["stages"]["anneal"][1]["i_pTM"] == "0.83"


def test_the_accepted_count_comes_from_the_campaign_line(arm):
    assert (arm["accepted"], arm["requested"]) == (1, 2)


def test_a_running_campaign_reports_no_count_rather_than_zero(tmp_path):
    """A campaign still running has no `campaign done` line. Reporting 0 accepted there would be
    a fabricated result, and 0 is exactly the number issue #17 is about."""
    path = tmp_path / "partial.log"
    path.write_text(ARM.split("campaign done")[0], encoding="utf-8")
    assert load().read_arm(str(path))["accepted"] is None


#: The dropout-off arm on pc, which chose its own interleaving width from free host memory before
#: `--trajectories-per-card 1` was pinned. Both trajectories' stage lines land under the LAST
#: header, so `screen` appears twice and trajectory 1's readings are invisible. This is the exact
#: shape of the real log; filing these under trajectory 2 is the silent failure the table must not
#: have.
INTERLEAVED = """BindCraft 2 v1.0.1
[tt_bio.bindcraft2] 2 design trajectories on this card: 2 of them at 192 tokens hold about 8 GB.

=== trajectory 1 | design_l60_cccc | accepted 0/3 ===

=== trajectory 2 | design_l60_dddd | accepted 0/3 ===
  passed screen design stage  i_pTM=0.85
  passed screen design stage  i_pTM=0.87
  passed refine design stage  i_pTM=0.83
  passed refine design stage  i_pTM=0.8
"""


def test_interleaved_arm_is_not_paired(tmp_path):
    """An arm that interleaved trajectories reports ! rather than a number under a guessed row."""
    module = load()
    log = tmp_path / "interleaved.log"
    log.write_text(INTERLEAVED, encoding="utf-8")
    arm = module.read_arm(str(log))

    assert arm["width"] == 2
    assert arm["collisions"] == 2, "screen and refine each reported twice"
    why = module.unpairable(arm)
    assert why and "interleaved" in why
    # Every cell is withheld, including the stage that really was reached.
    assert module.cell(arm, 2, "screen", "i_pTM") == "!"
    assert module.cell(arm, 1, "screen", "i_pTM") == "!"


def test_pinned_arm_is_still_paired(tmp_path):
    """The control: width 1 parses exactly as before, so the refusal is not blanket."""
    module = load()
    log = tmp_path / "pinned.log"
    log.write_text("[tt_bio.bindcraft2] 1 design trajectory on this card.\n" + ARM,
                   encoding="utf-8")
    arm = module.read_arm(str(log))

    assert arm["width"] == 1
    assert arm["collisions"] == 0
    assert module.unpairable(arm) is None
    assert module.cell(arm, 1, "screen", "i_pTM") == "0.85"
    assert module.cell(arm, 1, "harden", "i_pTM") == "0.14*"
