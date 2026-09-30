"""The per-stage gates a trajectory dies on, said out loud before a card is opened.

A campaign announces its acceptance filters and nothing about the thresholds that end a
trajectory halfway through, which is how a soak leg spends two of its first four trajectories on
a bar the log names but never states. See `bindcraft2.stage_gates`.
"""
import pytest

from tt_bio.bindcraft2 import print_stage_gates, stage_gates

PDL1 = {                       # the resolved pdl1 campaign settings, read on .108 2026-09-30
    "min_plddt_screen": 0.6, "min_plddt_refine": 0.6, "min_plddt_anneal": 0.65,
    "min_plddt_harden": 0.65, "min_iptm_harden": 0.5, "min_iptm_anneal": 0.5,
    "min_plddt_mutate": 0.6, "min_iptm_mutate": 0.5,
    "min_plddt_final": 0.7, "min_iptm_final": 0.7, "min_monomer_plddt_final": 0.7,
}


def test_every_gate_in_the_settings_is_named_with_its_threshold():
    line = stage_gates(PDL1)
    for key in PDL1:
        assert key in line, f"{key} is a gate this campaign runs and the line does not name it"
    assert "refine pLDDT >= 0.6 (min_plddt_refine)" in line


def test_the_gates_are_read_from_the_settings_so_the_list_cannot_go_stale():
    # The first version of this carried a seven-entry tuple while pdl1 resolved eleven gates,
    # so it silently told a researcher that min_iptm_final and min_iptm_anneal did not exist.
    line = stage_gates(PDL1)
    assert "min_iptm_final" in line and "min_iptm_anneal" in line
    assert "monomer pLDDT >= 0.7 (min_monomer_plddt_final)" in line
    invented = stage_gates({"min_plddt_invented_stage": 0.42})
    assert "invented_stage pLDDT >= 0.42" in invented


def test_the_gates_are_printed_in_the_order_the_stages_run():
    line = stage_gates(PDL1)
    positions = [line.index(s) for s in ("screen ", "refine ", "anneal ", "harden ", "mutate ")]
    assert positions == sorted(positions)
    assert line.index("mutate ") < line.index("final ")


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
    assert line.count("harden pLDDT >=") == 1   # its pLDDT gate and its i_pTM gate,
    assert line.count("harden i_pTM >=") == 1   # one entry each, neither swallowing the other


@pytest.mark.parametrize("threshold", [0.0, 1, 0.65])
def test_a_threshold_is_printed_without_trailing_zeros(threshold):
    assert f">= {threshold:g} (min_plddt_screen)" in stage_gates({"min_plddt_screen": threshold})


def test_printing_returns_the_same_line_it_printed(capsys):
    line = print_stage_gates(PDL1)
    assert line and line in capsys.readouterr().out


def test_printing_nothing_prints_nothing(capsys):
    assert print_stage_gates({}) == ""
    assert capsys.readouterr().out == ""


def test_the_line_does_not_claim_the_filters_line_is_above_it():
    # It prints with the stop conditions, before upstream's banner: in the live bh2c campaign log
    # this line landed at 37 and `filters ...` at 44, so "above" was simply false.
    line = stage_gates(PDL1)
    assert "above" not in line
    assert "`filters` line" in line
