"""Controls for the ledger extractor. CPU only; no device, no model, no network.

Every test here is a case where the extractor must REFUSE rather than guess, plus the arithmetic
controls on the clock model. Run with: python3 -m pytest perf/c10_lever_corpus/tests -q
"""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import corpus  # noqa: E402
import clock_sensitivity as cs  # noqa: E402

REPO = corpus.REPO
STATE = corpus.DEFAULT_STATE_ROOT


def mkrow(**kw):
    row = {
        "name": "probe",
        "op_class": "test",
        "deletes": "nothing",
        "status": "superseded",
        "clock": "unrecorded",
        "ratio": 1.01,
        "ratio_text": "1.01000x",
        "ratio_scope": "fold",
        "evidence": [{"path": "repo:perf/c10_lever_corpus/tests/fixture.md",
                      "quote": "a probe lever worth 1.01000x on the fold"}],
        "absent_symbol": "TT_BIO_DEFINITELY_NOT_A_REAL_FLAG",
    }
    row.update(kw)
    return row


def refuse(row):
    ledger, refusals = corpus.build([row], REPO, STATE)
    return ledger, refusals


# --------------------------------------------------------------------- the happy path exists
def test_a_grounded_row_is_accepted():
    ledger, refusals = refuse(mkrow())
    assert refusals == [], refusals
    assert len(ledger) == 1
    assert ledger[0]["clock"] == "unrecorded"
    assert ledger[0]["implied_mcycles"] is None
    assert ledger[0]["clock_band"]["spread_pct"] == pytest.approx(9.0, abs=0.2)


# --------------------------------------------------------------------- refusals, one per rule
def test_missing_evidence_file_is_refused():
    _, r = refuse(mkrow(evidence=[{"path": "state:no/such/file.md", "quote": "x"}]))
    assert "evidence file missing" in r[0]["reasons"][0]


def test_quote_not_in_source_is_refused():
    _, r = refuse(mkrow(evidence=[{
        "path": "repo:perf/c10_lever_corpus/tests/fixture.md",
        "quote": "a probe lever worth 1.99000x on the fold"}]))
    assert any("quote not found verbatim" in x for x in r[0]["reasons"])


def test_ratio_the_source_does_not_say_is_refused():
    """The number has to be in the row's own quote. This is the anti-invention rule."""
    _, r = refuse(mkrow(ratio=1.05, ratio_text="1.05000x"))
    assert any("does not appear in the row's own quotes" in x for x in r[0]["reasons"])


def test_ratio_field_disagreeing_with_its_text_is_refused():
    _, r = refuse(mkrow(ratio=1.02))
    assert any("disagrees with ratio_text" in x for x in r[0]["reasons"])


def test_a_guessed_clock_is_refused():
    """A clock is attested or it is 'unrecorded'. There is no inferred clock."""
    _, r = refuse(mkrow(clock=1350))
    assert any("no clock_evidence" in x for x in r[0]["reasons"])


def test_clock_evidence_that_does_not_name_the_mhz_is_refused():
    _, r = refuse(mkrow(clock=1350, clock_evidence={
        "path": "repo:perf/c10_lever_corpus/tests/fixture.md",
        "quote": "a probe lever worth 1.01000x on the fold"}))
    assert any("does not contain 1350" in x for x in r[0]["reasons"])


def test_an_attested_clock_is_accepted_and_priced_in_cycles():
    ledger, r = refuse(mkrow(clock=1350, clock_evidence={
        "path": "repo:perf/c10_lever_corpus/tests/fixture.md",
        "quote": "sampled during the fold at 1350 MHz, min = max"}))
    assert r == [], r
    assert ledger[0]["clock_band"] is None
    assert ledger[0]["implied_mcycles"] == pytest.approx(190.8, abs=0.5)


def test_a_bare_float_clock_is_refused():
    _, r = refuse(mkrow(clock=1350.0))
    assert any("must be an int MHz or the literal" in x for x in r[0]["reasons"])


def test_a_contested_record_is_refused_rather_than_arbitrated():
    _, r = refuse(mkrow(contested=True))
    assert any("CONTESTED" in x for x in r[0]["reasons"])


def test_status_is_checked_against_the_tree_not_against_prose():
    """TT_BIO_UNFUSED_SILU defaults False in tt_bio/, so calling it shipped-on must fail."""
    _, r = refuse(mkrow(flag="TT_BIO_UNFUSED_SILU", status="shipped-on"))
    assert any("defaults False in tt_bio/" in x for x in r[0]["reasons"])


def test_a_flag_absent_from_the_tree_cannot_be_called_shipped():
    _, r = refuse(mkrow(flag="TT_BIO_ATOM_L1", status="shipped-on"))
    assert any("is not in tt_bio/" in x for x in r[0]["reasons"])


def test_unshipped_claim_needs_a_symbol_check():
    _, r = refuse(mkrow(status="unmerged", absent_symbol=None))
    assert any("needs absent_symbol or present_symbol" in x for x in r[0]["reasons"])


def test_absent_symbol_that_is_actually_present_is_refused():
    _, r = refuse(mkrow(status="unmerged", absent_symbol="TT_BIO_ATOM_SHIFT_GATHER"))
    assert any("IS present in tt_bio/" in x for x in r[0]["reasons"])


def test_unflagged_shipped_row_needs_a_line_of_tt_bio():
    _, r = refuse(mkrow(status="shipped-on", flag=None))
    assert any("no status_evidence" in x for x in r[0]["reasons"])


def test_unknown_scope_is_refused():
    _, r = refuse(mkrow(ratio_scope="vibes"))
    assert any("ratio_scope" in x for x in r[0]["reasons"])


def test_missing_field_is_refused():
    row = mkrow()
    del row["op_class"]
    _, r = refuse(row)
    assert any("missing required field" in x for x in r[0]["reasons"])


def test_duplicate_names_are_refused():
    _, r = corpus.build([mkrow(), mkrow()], REPO, STATE)
    assert any("duplicate lever name" in x for reasons in r for x in reasons["reasons"])


# --------------------------------------------------------------------- the status oracle
def test_flag_oracle_reads_real_defaults_out_of_the_tree():
    d = corpus.flag_defaults(REPO)
    assert d["TT_BIO_ATOM_SHIFT_GATHER"] is True
    assert d["TT_BIO_UNFUSED_SILU"] is False
    assert d["TT_BIO_TRIMUL_GP_BANK_SPLIT"] is True     # via a module constant
    assert d["TT_BIO_TRIATT_FUSED_QKVGB"] is True       # via the "1" if CONST else "0" form
    assert d["TT_BIO_TRIATT_GATE_EPILOGUE"] is False    # constant is False
    assert "TT_BIO_ATOM_L1" not in d


# --------------------------------------------------------------------- the clock arithmetic
def test_the_planning_fit_reproduces_its_own_endpoints():
    assert corpus.fold_seconds_at(1350) == pytest.approx(14.275, abs=0.01)
    assert corpus.fold_seconds_at(800) == pytest.approx(22.095, abs=0.01)


def test_a_cycle_lever_reads_smaller_at_burst_not_larger():
    """The campaign's opening hypothesis has the wrong sign for the byte/arithmetic class."""
    at800 = cs.cycle_lever_ratio(0.03, 800)
    at1350 = cs.cycle_lever_ratio(0.03, 1350)
    assert at1350 < at800
    assert (at1350 - 1) / (at800 - 1) == pytest.approx(0.915, abs=0.01)


def test_a_fixed_cost_lever_reads_much_larger_at_burst():
    at800 = cs.fixed_lever_ratio(0.5, 800)
    at1350 = cs.fixed_lever_ratio(0.5, 1350)
    assert (at1350 - 1) / (at800 - 1) == pytest.approx(1.568, abs=0.01)


def test_implied_cycles_are_only_defined_for_a_fold_scope():
    assert corpus.implied_mcycles(1.05, 1350, "step") is None
    assert corpus.implied_mcycles(1.05, 1350, "block") is None
    assert corpus.implied_mcycles(1.05, 1350, "fold") > 0


# --------------------------------------------------------------------- the real ledger
def test_the_shipped_ledger_builds_and_nothing_silently_vanishes():
    src = json.loads((HERE.parent / "ledger_src.json").read_text())
    ledger, refusals = corpus.build(src, REPO, STATE)
    assert len(ledger) + len(refusals) == len(src)
    assert {r["name"] for r in ledger} | {r["name"] for r in refusals} == {r["name"] for r in src}


def test_not_one_row_in_the_corpus_carries_a_recorded_clock():
    """The finding this row exists to establish. If this ever fails, the ledger found a clock."""
    src = json.loads((HERE.parent / "ledger_src.json").read_text())
    ledger, _ = corpus.build(src, REPO, STATE)
    assert [r["name"] for r in ledger if isinstance(r["clock"], int)] == []
