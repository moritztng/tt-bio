"""The ladder's wall classifier, against the message tt-metal actually prints.

Written because the first version of the regex matched nothing: the refusal WRAPS after
"across 12 banks," and the pattern assumed one line. A classifier that silently matches
nothing is worse than none -- every wall reads as a plain ERROR, and a ladder that ran for
hours comes back with no classification at all. Every sample below is a real message copied
from a run log, and the last two are the negative control: an unrelated error and a timeout
must not be reported as memory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "perf" / "whceil"))
from ladder import classify  # noqa: E402

WRAPPED = """       Out of Memory: Not enough space to allocate 1833828352 B DRAM buffer across 12 banks,
       where each bank needs to store 152819200 B"""
NO_PER_BANK = "TT_FATAL: Out of Memory: Not enough space to allocate 524288 B DRAM buffer across 12 banks"
L1 = ("TT_FATAL: Out of Memory: Not enough space to allocate 134217728 B L1 buffer across 130 "
      "banks,\n where each bank needs to store 1032448 B")


def test_the_wrapped_dram_refusal_is_read_whole():
    verdict, d = classify(WRAPPED, 1, False)
    assert verdict == "OOM_DRAM"
    assert (d["request_bytes"], d["banks"], d["per_bank_bytes"]) == (1833828352, 12, 152819200)
    assert d["per_bank_reported"]


def test_a_refusal_without_the_per_bank_clause_still_classifies():
    verdict, d = classify(NO_PER_BANK, 1, False)
    assert verdict == "OOM_DRAM"
    assert d["request_bytes"] == 524288 and d["banks"] == 12
    assert not d["per_bank_reported"], "a derived share must not claim to be the reported one"


def test_l1_and_dram_are_not_the_same_wall():
    assert classify(L1, 1, False)[0] == "OOM_L1"
    assert classify(L1, 1, False)[1]["banks"] == 130


@pytest.mark.parametrize("text", [
    "ImportError: libXrender.so.1: cannot open shared object file: No such file or directory",
    "tt_bio.size_limits.SizeTooLargeError: 'x.yaml' has 1024 residues",
    "",
])
def test_a_failure_that_is_not_a_refusal_is_not_reported_as_one(text):
    assert classify(text, 1, False)[0] == "ERROR"


def test_a_timeout_is_a_runtime_wall_not_a_memory_one():
    assert classify(WRAPPED, 124, True)[0] == "TIMEOUT", (
        "a run killed by the clock is a runtime wall even if a refusal appeared earlier in "
        "its log; OpenDDE's 896 is exactly this distinction")


def test_a_rerun_cannot_inherit_the_previous_attempts_structure(tmp_path, monkeypatch):
    """The PASS predicate is 'a structure file exists' and the out dir is keyed by
    (model, rung), so a rerun of a rung that passed once and now fails would read the old
    .cif and report PASS. A relaunched ladder is a rerun by construction, so this is the
    check that must be able to fail."""
    import ladder as L

    out_root = tmp_path / "runs"
    stale = out_root / "m_r"
    stale.mkdir(parents=True)
    (stale / "old.cif").write_text("stale structure from a previous attempt")

    class _P:
        returncode, stdout, stderr = 1, "", "boom"

    monkeypatch.setattr(L.subprocess, "run", lambda *a, **k: _P())
    row = L.run_rung("m", tmp_path / "r.yaml", 0, out_root, 60, {}, [])
    assert row["verdict"] != "PASS", "the stale structure was read as this run's output"
    assert not (stale / "old.cif").exists()


# tt-metal puts the allocator's state LAST, in a parenthetical. It is the only part of the
# refusal that distinguishes the brief's two wall classes, so these three cases are the point
# of the classifier, not a detail of it. Bank size below is a real Wormhole Galaxy bank:
# 1071480832 B, ~1.02 GiB, twelve of them against Blackhole's eight of 3.984 GiB.
_HEAD = ("Out of Memory: Not enough space to allocate {req} B DRAM buffer across 12 banks,\n"
         "  where each bank needs to store {per} B, but bank size is 1071480832 B "
         "(allocated: {alloc} B, free: {free} B, largest free block: {run} B)")


def test_a_share_that_does_not_fit_an_empty_bank_is_one_oversized_tensor():
    t = _HEAD.format(req=34359738368, per=2863311530, alloc=0, free=1071480832, run=1071480832)
    assert classify(t, 1, False)[1]["wall_kind"] == "ONE_OVERSIZED_TENSOR"


def test_a_share_that_would_fit_an_empty_bank_but_not_this_one_is_residency():
    t = _HEAD.format(req=2424307712, per=202027008, alloc=1000000000, free=71480832, run=40000000)
    assert classify(t, 1, False)[1]["wall_kind"] == "CUMULATIVE_RESIDENCY"


def test_enough_free_but_not_in_one_run_is_fragmentation_not_residency():
    """Different fix -- compaction rather than holding less -- so it is a different name."""
    t = _HEAD.format(req=1200000000, per=100000000, alloc=300000000, free=771480832, run=44520544)
    assert classify(t, 1, False)[1]["wall_kind"] == "FRAGMENTATION"


def test_a_refusal_with_no_allocator_state_says_so_rather_than_guessing():
    assert classify(NO_PER_BANK, 1, False)[1]["wall_kind"] == "UNCLASSIFIED"
