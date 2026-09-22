"""A memory wall belongs to a model, not to a token count.

`plan()` used to hold one flat `FORWARD_OOM` table and apply it to whatever
`tt-bio finetune --model X` was given, and its two entries were Protenix-v2's. That was wrong
in both directions on a documented command: `--model openfold3 --tokens 512` was REFUSED on
Protenix-v2's number for a crop OpenFold3 is measured to run, and 544/576/640/768 came back
UNMEASURED for OpenFold3 when `of3t-crop768` measured all four to refuse.

These tests are the two directions, plus the rule that keeps them fixed: a model with no entry
borrows nobody's wall.
"""
import pytest

from tt_bio.train.dryrun import (FORWARD_OOM_BY_MODEL, LARGEST_MEASURED_TO_FIT, UNMEASURED,
                                 plan)


def test_openfold3_is_not_refused_at_512_on_protenix_v2s_measurement():
    # The first direction. of3t-crop768 measured 512 to RUN on OpenFold3, so a refusal here is
    # a working configuration turned away on a number from a different model.
    assert plan(tokens=512, model="openfold3").verdict != "refused"
    assert plan(tokens=512, model="protenix-v2").verdict == "refused"


@pytest.mark.parametrize("tokens", [544, 576, 640, 768])
def test_openfold3_is_refused_at_every_crop_measured_to_refuse(tokens):
    # The second direction. All four were measured to refuse on Blackhole at 1350 MHz sampled
    # during; UNMEASURED here would let a user start a run we have watched run out of memory.
    p = plan(tokens=tokens, model="openfold3")
    assert p.verdict == "refused", p.why
    assert "openfold3" in p.why
    assert p.sources and "of3t-crop768" in p.sources[0]


def test_a_refusal_names_the_model_it_was_measured_on():
    # The defect was invisible because the message did not say whose measurement it was.
    why = plan(tokens=384, model="protenix-v2").why
    assert "protenix-v2" in why
    assert "no other model's wall is applied" in why


def test_an_unmeasured_model_borrows_nobody_elses_wall():
    # Protenix-v2 refuses 384 and 512; OpenFold3 refuses 576. A model with no table must take
    # neither, because borrowing a neighbour's wall is what produced the defect.
    for tokens in (384, 512, 576, 640, 768):
        assert plan(tokens=tokens, model="a-model-we-have-not-measured").verdict == UNMEASURED
        assert plan(tokens=tokens, model=None).verdict == UNMEASURED


def test_no_two_models_share_a_table_object():
    # A shared dict would let one row's measurement silently become another's.
    tables = list(FORWARD_OOM_BY_MODEL.values())
    assert len({id(t) for t in tables}) == len(tables)


def test_a_measured_fit_is_reported_even_when_the_step_time_is_not():
    # UNMEASURED must not throw away the fit we do have: saying "no measurement" for a crop we
    # have watched complete is the flattering-to-nobody direction.
    p = plan(tokens=512, model="openfold3")
    assert p.verdict == UNMEASURED
    assert "memory fits up to 512 aa" in p.why
    assert any("of3t-crop768" in s for s in p.sources)


def test_the_fit_note_does_not_appear_above_the_measured_fit():
    # 544 is the first crop that refuses, so nothing may claim a fit at or above it.
    largest, _ = LARGEST_MEASURED_TO_FIT["openfold3"]
    assert largest == 512
    assert "memory fits" not in (plan(tokens=576, model="openfold3").why or "")


def test_every_table_entry_carries_the_row_that_measured_it():
    # A wall with no source is a number nobody can re-check.
    for model, table in FORWARD_OOM_BY_MODEL.items():
        for tokens, entry in table.items():
            assert len(entry) == 3, (model, tokens)
            assert isinstance(entry[2], str) and entry[2].strip(), (model, tokens)
