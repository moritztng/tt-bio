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
