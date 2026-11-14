"""A size-ladder timing is only scored against a baseline timed at the same host-thread cap.

Device-free and torch-free. On 2026-10-10 the sharded release gate ran three uncapped ladder legs
on qb2 at once (16 cores, every fold sizing its pools to all 16): boltz2's 256 aa rung read 12.9 s
against a 4.2 s baseline timed alone, 1024 aa reproduced at 0.97x, and the exponents failed on CPU
contention rather than on the build. The gate now caps each ladder fold, so a baseline timed at
another cap must be reported as such instead of being scored.

Run: python3 tests/test_size_ladder_host_threads.py, or via pytest.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import release_gate as rg

# The 2026-10-10 rel-1b36c8906 qb2 card 1 reading against the committed p300c baseline.
BASE = {"256": 4.2, "512": 10.9, "640": 17.4, "768": 30.2, "896": 41.8, "1024": 48.6}
RUN = {"256": 12.9, "512": 19.7, "640": 25.3, "768": 27.8, "896": 36.9, "1024": 47.3}
BASE_MODEL = {"runtime_s": BASE, "levers": {r: {} for r in BASE}, "grid": "11x10",
              "exponents": {"256->512": {"k": 1.376, "tol": 0.5},
                            "512->768": {"k": 2.513, "tol": 0.5}}}
MEAS = {"runtime_s": RUN, "levers": {r: {} for r in RUN}, "grid": "11x10"}


def _baseline(tmp: Path, frag_threads, mono_threads=None) -> Path:
    path = tmp / "size_ladder_baseline.json"
    path.write_text(json.dumps({"cards": {"p300c": {"host_threads": mono_threads,
                                                    "models": {}}}}))
    (tmp / "size_ladder_baseline.d").mkdir()
    for model, threads in frag_threads.items():
        (tmp / "size_ladder_baseline.d" / f"{model}.json").write_text(json.dumps(
            {"cards": {"p300c": {"host_threads": threads, "models": {model: BASE_MODEL}}}}))
    return path


def test_a_cap_mismatch_is_a_finding_and_the_exponents_are_not_scored():
    out = rg._size_ladder_compare(BASE_MODEL, MEAS, "boltz2", [int(r) for r in BASE],
                                  "baseline timed at host threads all, this run at 5")
    assert not out["gate"], out
    joined = " | ".join(out["findings"])
    assert "host threads all, this run at 5" in joined, joined
    assert "exponent" not in joined, joined
    assert out["exponents"] == {}, out


def test_the_same_numbers_without_a_mismatch_still_fail_on_the_exponents():
    """The control: the reading itself is red, so the cap check is what changes the finding."""
    out = rg._size_ladder_compare(BASE_MODEL, MEAS, "boltz2", [int(r) for r in BASE])
    assert not out["gate"], out
    assert any("exponent 1.38 -> 0.61" in f for f in out["findings"]), out["findings"]


def test_the_cap_is_read_from_the_models_own_fragment(monkeypatch):
    """The merged view keeps the first card stamp it meets; boltz2 at 5 and nesso1 uncapped must
    each be judged on their own."""
    with tempfile.TemporaryDirectory() as d:
        path = _baseline(Path(d), {"boltz2": 5, "nesso1": None})
        monkeypatch.setattr(rg, "HOST_THREADS", 5)
        assert rg._size_ladder_threads_differ(path, "p300c", "boltz2") is None
        why = rg._size_ladder_threads_differ(path, "p300c", "nesso1")
        assert why and "host threads all, this run at 5" in why, why
        assert "--host-threads 5" in why, why
        monkeypatch.setattr(rg, "HOST_THREADS", None)
        assert rg._size_ladder_threads_differ(path, "p300c", "nesso1") is None
        assert "this run at all" in rg._size_ladder_threads_differ(path, "p300c", "boltz2")


def test_a_model_with_no_rows_on_this_card_is_left_to_the_missing_model_check(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        path = _baseline(Path(d), {"boltz2": 5})
        monkeypatch.setattr(rg, "HOST_THREADS", 5)
        assert rg._size_ladder_threads_differ(path, "p300c", "rf3") is None
        assert rg._size_ladder_threads_differ(path, "p150a", "boltz2") is None


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
