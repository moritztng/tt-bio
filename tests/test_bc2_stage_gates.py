"""The per-stage gates a trajectory dies on, said out loud before a card is opened.

A campaign announces its acceptance filters and nothing about the thresholds that end a
trajectory halfway through, which is how a soak leg spends two of its first four trajectories on
a bar the log names but never states. See `bindcraft2.stage_gates`.
"""
import pytest

from tt_bio.bindcraft2 import STAGE_GATES, print_stage_gates, stage_gates

PDL1 = {                       # the resolved pdl1 campaign settings, read on .108 2026-09-30
    "min_plddt_screen": 0.6, "min_plddt_refine": 0.6, "min_plddt_anneal": 0.65,
    "min_plddt_harden": 0.65, "min_iptm_harden": 0.5,
    "min_plddt_mutate": 0.6, "min_iptm_mutate": 0.5,
}


def test_every_gate_in_the_settings_is_named_with_its_threshold():
    line = stage_gates(PDL1)
    for stage, key, _ in STAGE_GATES:
        assert key in line, f"{key} is a gate this campaign runs and the line does not name it"
    assert "refine pLDDT >= 0.6 (min_plddt_refine)" in line


def test_the_gate_the_live_leg_died_on_is_the_one_a_reader_could_not_find():
    # trajectory 2: "rejected at refine design stage i_pTM=0.79 pLDDT=0.59 due to [pLDDT]"
    line = stage_gates(PDL1)
    assert "min_plddt_refine" in line and "0.6" in line


def test_it_says_the_trajectory_is_charged_and_counted_as_terminated():
    line = stage_gates(PDL1)
    assert "charged against the budget" in line
    assert "terminated" in line


def test_a_campaign_without_stage_gates_says_nothing():
    assert stage_gates({"number_of_final_designs": 10}) == ""


def test_a_malformed_threshold_is_skipped_rather_than_crashing_the_campaign():
    line = stage_gates({"min_plddt_refine": "tight", "min_iptm_harden": 0.5})
    assert "min_plddt_refine" not in line
    assert "min_iptm_harden" in line


def test_a_gate_is_named_once_even_though_a_stage_judges_both_metrics():
    line = stage_gates(PDL1)
    assert line.count("min_iptm_harden") == 1
    assert line.count("harden") == 2       # its pLDDT gate and its i_pTM gate, one each


@pytest.mark.parametrize("threshold", [0.0, 1, 0.65])
def test_a_threshold_is_printed_without_trailing_zeros(threshold):
    assert f">= {threshold:g} (min_plddt_screen)" in stage_gates({"min_plddt_screen": threshold})


def test_printing_returns_the_same_line_it_printed(capsys):
    line = print_stage_gates(PDL1)
    assert line and line in capsys.readouterr().out


def test_printing_nothing_prints_nothing(capsys):
    assert print_stage_gates({}) == ""
    assert capsys.readouterr().out == ""
