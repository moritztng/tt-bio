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


def test_a_pass_reports_the_refusals_the_engine_recovered_from(tmp_path, monkeypatch):
    """A refusal the engine narrowed past is not a wall, but it is the only visible evidence
    that the reactive narrowing ran at all -- and a PASS row hides it otherwise."""
    import ladder as L

    out_root = tmp_path / "runs"

    class _P:
        returncode = 0
        stdout = ("Out of Memory: Not enough space to allocate 2147483648 B DRAM buffer "
                  "across 12 banks\nretrying narrower\n")
        stderr = ""

    def _fake_run(*a, **k):
        (out_root / "m_r").mkdir(parents=True, exist_ok=True)
        (out_root / "m_r" / "x.cif").write_text("folded")
        return _P()

    monkeypatch.setattr(L.subprocess, "run", _fake_run)
    row = L.run_rung("m", tmp_path / "r.yaml", 0, out_root, 60, {}, [])
    assert row["verdict"] == "PASS"
    assert row["refusals_recovered"] == 1
    assert row["largest_recovered_bytes"] == 2147483648


def test_each_attempt_writes_its_own_log(tmp_path, monkeypatch):
    """Two attempts at the same rung must not share a log name: the collector reads the log to
    classify a wall, and a shared name let a later run's recovered refusal be reported as an
    earlier run's cause of death."""
    import ladder as L

    class _P:
        returncode, stdout, stderr = 1, "", "boom"

    monkeypatch.setattr(L.subprocess, "run", lambda *a, **k: _P())
    monkeypatch.setattr(L.time, "strftime", lambda f, t=None: "000001")
    a = L.run_rung("m", tmp_path / "r.yaml", 0, tmp_path / "runs", 60, {}, [])
    monkeypatch.setattr(L.time, "strftime", lambda f, t=None: "000002")
    b = L.run_rung("m", tmp_path / "r.yaml", 0, tmp_path / "runs", 60, {}, [])
    assert a["log"] != b["log"] and Path(a["log"]).is_file() and Path(b["log"]).is_file()


CLASH = ("TT_THROW: Statically allocated circular buffers in program 188 clash with L1 buffers "
         "on core range [(x=0,y=0) - (x=7,y=8)]. L1 buffer allocated at 319488 and static "
         "circular buffer region ends at 541984 (assert.hpp:104)")


def test_an_l1_circular_buffer_clash_is_its_own_wall_not_a_dram_one():
    """It is L1, not DRAM, and a clash rather than a shortage -- it carries no bank numbers, so
    the DRAM rule can never classify it. It is also the non-monotone class: OpenDDE folds 544,
    throws at 576 and folds 608, which is why a cap is the size below the FIRST failure."""
    verdict, d = classify(CLASH, 1, False)
    assert verdict == "CLASH_L1"
    assert d["wall_kind"] == "L1_CB_CLASH"
    assert (d["program"], d["l1_buffer_at"], d["cb_region_ends"]) == (188, 319488, 541984)


def test_a_dram_refusal_in_the_same_log_wins_over_an_earlier_clash():
    """The clash was recovered from; the refusal is what the run died on."""
    assert classify(CLASH + "\n" + WRAPPED, 1, False)[0] == "OOM_DRAM"


OVERSIZE = ("TT_THROW: Statically allocated circular buffers on core range "
            "[(x=0,y=0) - (x=7,y=8)] grow to 2471200 B which is beyond max L1 size of 1499136 B")


def test_a_cb_region_that_does_not_fit_l1_is_an_oversized_tensor_not_a_clash():
    """Different failure, different fix: the region does not collide with a neighbour, it does
    not fit at all. Seen on OpenDDE's 1088 rung. Folding it into the clash class would say the
    wall moves with what else is resident, and this one does not."""
    verdict, d = classify(OVERSIZE, 1, False)
    assert verdict == "OVERSIZE_L1"
    assert d["wall_kind"] == "ONE_OVERSIZED_TENSOR_L1"
    assert (d["request_bytes"], d["l1_size_bytes"]) == (2471200, 1499136)
    assert classify(CLASH, 1, False)[0] == "CLASH_L1", "the two L1 messages must not collapse"


def test_fragmentation_on_a_nearly_full_chip_is_named_apart():
    """Same refusal, different fix. Compaction can only hand back what is free, so at 98 %
    occupancy it buys a few MiB and the lever is residency instead. Calling both plain
    FRAGMENTATION would point the next person at the wrong one."""
    full = _HEAD.format(req=117051392, per=9754282, alloc=1052000000, free=21741792, run=7025459)
    verdict, d = classify(full, 1, False)
    assert verdict == "OOM_DRAM"
    assert d["wall_kind"] == "FRAGMENTATION_ON_FULL_CHIP"
    assert d["occupancy_pct"] == 98.2   # 1052000000 of the 1071480832 B bank

    roomy = _HEAD.format(req=2424307712, per=202027008, alloc=856253280,
                         free=217488512, run=178813056)
    assert classify(roomy, 1, False)[1]["wall_kind"] == "FRAGMENTATION"
