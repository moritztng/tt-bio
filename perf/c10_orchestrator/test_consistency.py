#!/usr/bin/env python3
"""Controls for consistency.py — including that it reds on the exact rot it was built to catch."""
import pytest

import consistency as C


def test_the_corpus_is_currently_clean():
    r = C.check()
    assert r["clean"], f"retired numbers quoted bare: {r['problems']}"


def test_it_actually_reads_every_readme_in_the_tree():
    r = C.check()
    assert r["docs_checked"] >= 15
    names = {p.name for p in C._docs()}
    assert names == {"README.md"}


def test_it_reds_on_the_real_instance_it_was_built_for(tmp_path, monkeypatch):
    """grid_evidence/ presented the 67.59 TFLOP/s cube as a Blackhole rate with no mention that it
    implies 799 MHz. Recreate that document and the checker must flag it."""
    d = tmp_path / "grid_evidence"
    d.mkdir()
    (d / "README.md").write_text(
        "# What Blackhole does say\n\n| arm | rate |\n|---|---|\n"
        "| dense cube matmul | 67.59 TFLOP/s |\n\n"
        "Measured on the right architecture, one session, one card.\n")
    monkeypatch.setattr(C, "HERE", tmp_path)
    r = C.check()
    assert not r["clean"]
    assert any(p["number"] == "67.59" for p in r["problems"])


def test_adding_the_retirement_anywhere_in_the_document_clears_it(tmp_path, monkeypatch):
    d = tmp_path / "grid_evidence"
    d.mkdir()
    (d / "README.md").write_text(
        "# What Blackhole does say\n\n| dense cube matmul | 67.59 TFLOP/s |\n\n"
        "That session was throttled: 67.59 implies a chip at 799 MHz, the clock floor.\n")
    monkeypatch.setattr(C, "HERE", tmp_path)
    assert C.check()["clean"]


@pytest.mark.parametrize("num", sorted(C.RETIRED))
def test_every_tracked_retired_number_is_caught_when_quoted_bare(tmp_path, monkeypatch, num):
    d = tmp_path / "somewhere"
    d.mkdir()
    (d / "README.md").write_text(f"# A new note\n\nThe figure is {num} and it is fine.\n")
    monkeypatch.setattr(C, "HERE", tmp_path)
    r = C.check()
    assert any(p["number"] == num for p in r["problems"]), f"{num} not caught"


def test_a_document_not_mentioning_a_retired_number_is_never_flagged(tmp_path, monkeypatch):
    d = tmp_path / "clean"
    d.mkdir()
    (d / "README.md").write_text("# Nothing retired here\n\nF is 3.9830 s at a pinned 1350 MHz.\n")
    monkeypatch.setattr(C, "HERE", tmp_path)
    assert C.check()["clean"]


def test_distance_reporting_does_not_fail_the_run(tmp_path, monkeypatch):
    """A retirement 200 lines from its number is honest but easy to miss: reported, not failed."""
    d = tmp_path / "far"
    d.mkdir()
    body = "\n".join(["filler"] * 200)
    (d / "README.md").write_text(f"The floor is 15.031 s.\n{body}\nThat floor is suspect.\n")
    monkeypatch.setattr(C, "HERE", tmp_path)
    r = C.check()
    assert r["clean"]
    assert r["retirement_far_from_its_number"], "a 200-line gap should be reported"


def test_the_numbers_of_record_are_the_measured_ones():
    assert "3.9830" in C.OF_RECORD["F at 512 aa"]
    assert "14665.0" in C.OF_RECORD["W at 512 aa"]
    assert "14.8813" in C.OF_RECORD["baseline at 512 aa"]
    assert "1350" in C.OF_RECORD["pinned clock"]


def test_the_retired_list_covers_what_the_campaign_actually_retired():
    assert set(C.RETIRED) == {"15.031", "2.309", "67.59", "3,301", "8,220"}
    for num, (markers, what) in C.RETIRED.items():
        assert markers and what, f"{num} has no markers or no explanation"
