"""A size-ladder rung is timed against the last release on the same chip, not against a recorded cell.

Device-free and torch-free. Single folds on the gate pools are bimodal: across the 14 gates of
2026-10-14..20 about 30 % of them read 1.1-3.4x the clean level, on every host. Judged one draw
against a quiet-box record, nearly every train gate failed a ladder leg on that noise. The replay
below runs the real _size_ladder_ab loop over the 15 ladder legs those gates failed on timing,
with the recorded slow reading as the first candidate fold and every other fold drawn from the
recorded noise, and then over the same legs with a 1.3x slowdown planted on one rung.

Run: python3 tests/test_size_ladder_ab.py (prints the replay table), or via pytest.
"""
import os
import random
import sys
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import release_gate as rg

#: Single-fold time over its cell's clean level (the 25th percentile of that model/host/rung across
#: the 14 gates), 1455 readings at 256/512/768 aa on g107/g108/g110/g114/qb1/qb2, as 39 quantiles.
#: The lowest 2.5 % is dropped: those are code speedups between gates, which no co-tenant causes.
NOISE = [0.915, 0.955, 0.971, 0.977, 0.983, 0.989, 0.995, 1.0, 1.0, 1.0, 1.0, 1.0, 1.004, 1.008,
         1.011, 1.017, 1.023, 1.027, 1.033, 1.04, 1.047, 1.055, 1.061, 1.073, 1.08, 1.088, 1.099,
         1.117, 1.135, 1.159, 1.19, 1.229, 1.252, 1.311, 1.374, 1.474, 1.618, 1.956, 3.756]

#: Every ladder leg that failed on timing in rel-f62c3b67a..rel-53e9a469a, as run/base at 256, 512
#: and 768 aa from its `per-rung base->run` line.
FAILED = {
    "rel-53e9a469a esmfold2 bh": (0.78, 0.53, 0.85),
    "rel-53e9a469a esmfold2 wh": (1.04, 1.06, 0.84),
    "rel-53e9a469a protenix-v2 bh": (0.99, 1.74, 1.43),
    "rel-ea58ac531 openbind bh": (1.08, 1.14, 0.86),
    "rel-ea58ac531 esmfold2 bh": (1.34, 0.85, 0.92),
    "rel-ea58ac531 openbind wh": (1.42, 1.40, 1.11),
    "rel-ea58ac531 openfold3 wh": (1.79, 1.46, 1.15),
    "rel-ea58ac531 rf3 wh": (1.44, 1.19, 1.47),
    "rel-938d50505 protenix-v1 bh": (1.20, 1.12, 0.88),
    "rel-938d50505 openbind wh": (2.52, 1.55, 1.08),
    "rel-938d50505 openfold3 wh": (1.57, 1.64, 1.13),
    "rel-938d50505 rf3 wh": (2.24, 1.66, 1.12),
    "rel-fc773b53a openfold3 bh": (1.12, 1.04, 0.82),
    "rel-fc773b53a protenix-v1 bh": (0.67, 0.82, 1.05),
    "rel-f62c3b67a protenix-v1 bh": (1.12, 0.84, 1.06),
}
RUNGS = (256, 512, 768)


def _fake_folds(monkeypatch, first: dict, slow: dict, rng: random.Random, calls=None):
    """_run_census_fold with recorded noise: `first[rung]` is the candidate's first fold there,
    `slow[rung]` multiplies every candidate fold (the planted regression)."""
    seen = set()

    def fold(model, rung, workdir, tag, need_runtime=True, tree=None):
        side = "ref" if tree else "cand"
        if calls is not None:
            calls.append((rung, tag, side))
        if side == "cand" and rung not in seen and rung in first:
            seen.add(rung)
            t = first[rung]
        else:
            t = rng.choice(NOISE)
        return {"runtime_s": 10.0 * t * (slow.get(rung, 1.0) if side == "cand" else 1.0)}
    monkeypatch.setattr(rg, "_run_census_fold", fold)


def _leg_fails(monkeypatch, first, slow, rng) -> bool:
    _fake_folds(monkeypatch, first, slow, rng)
    ab = rg._size_ladder_ab("openbind", list(RUNGS), Path("/nonexistent"), Path("/r/trees/v0"))["ab"]
    return bool(rg._size_ladder_ab_findings("openbind", ab))


def replay(monkeypatch, sims: int = 400, seed: int = 0) -> dict:
    rng = random.Random(seed)
    out = {}
    for leg, cells in FAILED.items():
        # Below 1.0 the record was slower than this code's clean level, which no fold can beat.
        first = {r: max(1.0, c) for r, c in zip(RUNGS, cells)}
        noise = sum(_leg_fails(monkeypatch, first, {}, rng) for _ in range(sims)) / sims
        caught = min(sum(_leg_fails(monkeypatch, first, {r: 1.3}, rng) for _ in range(sims)) / sims
                     for r in RUNGS)
        out[leg] = (noise, caught)
    return out


def test_the_recorded_noise_fails_clear_and_a_planted_1_3x_still_fails(monkeypatch):
    res = replay(monkeypatch)
    noise = sum(n for n, _ in res.values()) / len(res)
    worst_catch = min(c for _, c in res.values())
    assert noise < 0.03, res
    assert worst_catch > 0.95, res


def test_a_1_3x_slowdown_reads_slow_at_every_fold_count():
    cell = {"cand": [13.0, 13.1, 26.0], "ref": [10.0, 20.0, 10.1]}
    assert rg._size_ladder_ab_findings("rf3", {"256": cell})


def test_a_bimodal_draw_on_either_side_is_not_a_finding():
    # nesso1's perf reps on g110 in rel-938d50505 read 19, 38, 38 s in one session.
    cell = {"cand": [38.0, 19.0, 38.0], "ref": [19.2, 38.0, 19.0]}
    assert not rg._size_ladder_ab_findings("nesso1", {"512": cell})


def test_folds_interleave_abba_after_a_reference_warmup_and_extend_only_while_slow(monkeypatch):
    calls = []
    _fake_folds(monkeypatch, {}, {512: 1.3}, random.Random(1), calls)
    ab = rg._size_ladder_ab("openbind", [256, 512, 640, 768], Path("/x"), Path("/r/t"))["ab"]
    assert sorted(ab, key=int) == ["256", "512", "768"], ab
    first = [c for c in calls if c[0] == 512]
    assert first[0] == (512, "abwarm", "ref"), first
    assert [s for _, _, s in first[1:5]] == ["cand", "ref", "ref", "cand"], first
    assert len(ab["512"]["cand"]) == rg.SIZE_LADDER_AB_MAX_REPS, ab["512"]  # never settles: 1.3x
    assert len(ab["256"]["cand"]) >= rg.SIZE_LADDER_AB_REPS


def test_a_reference_that_cannot_fold_the_model_is_reported_not_judged(monkeypatch):
    monkeypatch.setattr(rg, "_run_census_fold", lambda *a, tree=None, **k:
                        {"error": "no such model"} if tree else {"runtime_s": 1.0})
    got = rg._size_ladder_ab("openbind", [256, 512, 768], Path("/x"), Path("/r/trees/v0.20.1"))
    assert "error" in got and "v0.20.1" in got["error"], got


def test_with_an_ab_the_recorded_exponents_are_reported_not_judged():
    """rel-938d50505 openbind wh: 256 9.0->22.7 s failed the exponent band against the record."""
    base = {"256": 9.0, "512": 20.4, "768": 43.6}
    run = {"256": 22.7, "512": 31.6, "768": 47.3}
    base_model = {"runtime_s": base, "levers": {r: {} for r in base},
                  "exponents": {"256->512": {"k": 1.18, "tol": 0.5}, "512->768": {"k": 1.87, "tol": 0.5}}}
    meas = {"runtime_s": run, "levers": {r: {} for r in run}}
    old = rg._size_ladder_compare(base_model, meas, "openbind", [256, 512, 768])
    assert not old["gate"] and any("exponent" in f for f in old["findings"]), old
    clean = {r: {"cand": [10.0, 10.2, 10.1], "ref": [10.1, 10.0, 10.3]} for r in base}
    new = rg._size_ladder_compare(base_model, meas, "openbind", [256, 512, 768], ab=clean)
    assert new["gate"], new
    assert new["exponents"]["256->512"] == old["exponents"]["256->512"]
    slow = {**clean, "512": {"cand": [13.0, 13.2, 13.1], "ref": [10.0, 10.1, 10.0]}}
    red = rg._size_ladder_compare(base_model, meas, "openbind", [256, 512, 768], ab=slow)
    assert red["findings"] == [f for f in red["findings"] if "openbind/512 A/B: 1.30x" in f], red


class _MP:
    def setattr(self, obj, name, value, raising=True):
        setattr(obj, name, value)


if __name__ == "__main__":
    res = replay(_MP(), sims=2000)
    print(f"{'leg':34} {'noise FAIL rate':>16} {'1.3x caught (worst rung)':>26}")
    for leg, (n, c) in res.items():
        print(f"{leg:34} {n:16.1%} {c:26.1%}")
    print(f"{'mean':34} {sum(n for n, _ in res.values()) / len(res):16.1%} "
          f"{sum(c for _, c in res.values()) / len(res):26.1%}")
