"""`scripts/ladder_preread.py` must reach the same verdict as the arm it reads for.

The pre-read exists because the size-ladder arm takes ~3 h and prints its verdict only at
the end, so a run 40 minutes from done has already decided most of its rows with no way to
read them. That is only worth having if a row it calls PASS is a row the arm calls PASS, so
the two things that can drift are pinned here: the census -> lever translation (the pre-read
carries its own copy, because `_run_census_fold` does it inline around a subprocess it
cannot run off-device) and the staleness rule (a work dir holds the PREVIOUS run's rungs for
every model the live one has not reached, and scoring those as current is the one way this
tool can lie).

Host-only: no device, no folds, no network.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pre():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    return _load("ladder_preread_under_test", REPO_ROOT / "scripts" / "ladder_preread.py")


CENSUS = {
    "grid": "11x10",
    "rows": [
        {"flag": "SERVED", "resolved": "True", "served": 10, "declined": 0, "how": "stats"},
        {"flag": "PARTLY", "resolved": "True", "served": 3, "declined": 7, "how": "stats",
         "rejects": {"m_tiles=64": 7}},
        {"flag": "DARK", "resolved": "True", "served": 0, "declined": 0, "how": "stats"},
        {"flag": "ABSENT", "resolved": "not-imported", "served": None, "declined": None,
         "how": "stats"},
    ],
}


def test_the_preread_translates_a_census_exactly_as_the_arm_does(pre):
    """Both readers see one artifact; a divergence here is a divergence in every verdict."""
    got = pre._census_levers(CENSUS)
    assert got["SERVED"]["frac"] == 1.0
    assert got["PARTLY"]["frac"] == 0.3
    assert got["PARTLY"]["rejects"] == {"m_tiles=64": 7}
    # never reached reads as dark (0.0), never as unknown -- the arm fails a dark-and-ON
    # lever with no exemption, and None would silently skip that rule
    assert got["DARK"]["frac"] == 0.0
    # not-imported stays None: absent is not dark, and conflating them reds every arm
    # running a build that does not import the module
    assert got["ABSENT"]["frac"] is None
    assert "rejects" not in got["SERVED"]


def _rung(work, model, rung, tag, runtime, census=None):
    (work / f"census_{model}-{rung}-{tag}.json").write_text(json.dumps(census or CENSUS))
    out = work / f"out_{model}-{rung}-{tag}" / f"{model}_results_x"
    out.mkdir(parents=True)
    (out / "results.json").write_text(
        json.dumps([{"id": "x", "status": "ok", "runtime_s": runtime}]))


def test_a_rung_left_by_an_earlier_run_is_dropped_by_since(pre, tmp_path):
    """The arm walks its models in order, so a live run's work dir is half last run's.

    openbind's rows sat in the work dir from a 15:2xZ run for the whole of a 19:56Z one and
    would have been reported as that run's, with the previous verdict attached.
    """
    work = tmp_path / "work"
    work.mkdir()
    _rung(work, "boltz2", 256, "rep0", 4.2)
    _rung(work, "boltz2", 512, "rep0", 11.0)
    old = work / "census_boltz2-256-rep0.json"
    import os
    os.utime(old, (1000, 1000))
    fresh = pre._measure_from_disk(work, "boltz2", (256, 512), since=2000)
    assert "256" not in fresh["runtime_s"] and 256 in fresh["pending"]
    assert fresh["runtime_s"]["512"] == 11.0
    everything = pre._measure_from_disk(work, "boltz2", (256, 512), since=0)
    assert everything["runtime_s"] == {"256": 4.2, "512": 11.0}
    # the span is what makes a mixed read visible in the table rather than silent
    assert everything["span"][0] < everything["span"][1]


def test_a_rung_that_has_not_folded_yet_is_pending_not_missing(pre, tmp_path):
    """A live ladder scored as a finished one reports holes as baseline gaps.

    `_size_ladder_compare` says "rung not recorded in the baseline" for a rung it cannot
    find on either side, so a run read mid-model must not hand it rungs that simply have
    not happened. They come back as `pending` and the caller prints them.
    """
    work = tmp_path / "work"
    work.mkdir()
    _rung(work, "opendde", 256, "rep0", 21.0)
    meas = pre._measure_from_disk(work, "opendde", (256, 512, 768), since=0)
    assert meas["pending"] == [512, 768]
    assert meas["runtime_s"] == {"256": 21.0}
    assert meas["refused"] == {}


def test_the_warmup_fold_is_read_for_its_refusal_and_never_for_its_timing(pre, tmp_path):
    """The arm discards the first fold at each rung; a refusal is still a recorded cell."""
    work = tmp_path / "work"
    work.mkdir()
    _rung(work, "rf3", 256, "warmup", 99.9)
    _rung(work, "rf3", 256, "rep0", 21.0)
    (work / "rf3-1088-warmup.log").write_text(
        "tt_bio.size_limits.SizeTooLargeError: rf3 supports up to 1095 tokens, got 1200\n")
    meas = pre._measure_from_disk(work, "rf3", (256, 1088), since=0)
    assert meas["runtime_s"] == {"256": 21.0}, "the warm-up must not enter the median"
    assert "1088" in meas["refused"]
