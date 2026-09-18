"""A size-ladder exponent finding must say which rung moved.

Device-free and torch-free. On 2026-09-18 a release gate came back with two red ladder rows --
boltz2 "256->512: exponent 1.41 -> 0.75" and openbind "1.60 -> 2.14" -- and the whole of both was
the 256 aa rung, while every rung from 512 aa up reproduced the baseline inside 2 %. The change
under test was a 512 aa lever, so the arm's red rows pointed away from the regime being changed.
Nothing in the finding said so, and recovering it meant reading per-rung absolutes out of the
baseline fragment dir by hand.

Run: python3 tests/test_size_ladder_rung_localisation.py, or via pytest.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import release_gate as rg

# The real 2026-09-18 p300c numbers: baseline from docs/size_ladder_baseline.d/, run from the
# gate's own size-ladder table. Kept verbatim so this stays a regression test, not a mock.
B2_BASE = {"256": 4.1, "512": 10.9, "640": 17.2, "768": 25.6, "896": 37.8, "1024": 48.4}
B2_RUN = {"256": 6.5, "512": 10.9, "640": 17.2, "768": 25.5, "896": 38.0, "1024": 47.7}
OB_BASE = {"256": 9.3, "512": 28.2, "640": 47.4, "768": 76.0, "896": 117.5, "1024": 163.4}
OB_RUN = {"256": 6.5, "512": 28.7, "640": 46.2, "768": 76.4, "896": 119.8, "1024": 165.3}


def test_it_localises_the_real_boltz2_failure_to_the_256_rung():
    note = rg._size_ladder_rung_localisation("boltz2", B2_BASE, B2_RUN)
    assert "256 4.1->6.5 1.59x" in note, note
    assert "only 256 aa moved" in note, note
    for held in ("512", "640", "768", "896", "1024"):
        assert held in note, (held, note)


def test_it_localises_the_real_openbind_failure_to_the_same_rung_the_other_way():
    """The two failures moved in OPPOSITE directions, 1.59x and 0.70x, and both were one rung."""
    note = rg._size_ladder_rung_localisation("openbind", OB_BASE, OB_RUN)
    assert "256 9.3->6.5 0.70x" in note, note
    assert "only 256 aa moved" in note, note


def test_a_whole_ladder_shift_is_not_reported_as_one_rung():
    """The negative control: if everything moved, saying "only X moved" would be a lie."""
    run = {k: v * 1.3 for k, v in B2_BASE.items()}
    note = rg._size_ladder_rung_localisation("boltz2", B2_BASE, run)
    assert "every rung moved" in note, note
    assert "only" not in note, note


def test_no_rung_moving_indicts_the_tolerance_not_the_build():
    """An exponent can drift while every rung reproduces -- then the band is the problem."""
    note = rg._size_ladder_rung_localisation("boltz2", B2_BASE, dict(B2_BASE))
    assert "no rung moved" in note, note
    assert "tolerance is tighter" in note, note


def test_a_baseline_with_no_absolutes_asks_to_be_re_recorded():
    note = rg._size_ladder_rung_localisation("boltz2", {}, B2_RUN)
    assert "no per-rung baseline absolutes" in note, note
    assert "re-record" in note, note


def test_a_rung_the_run_skipped_is_not_invented():
    note = rg._size_ladder_rung_localisation("boltz2", B2_BASE, {"256": 6.5, "512": 10.9})
    assert "640" not in note, note
    assert "256 4.1->6.5" in note, note


def test_the_drift_finding_carries_the_localisation():
    """Integration: the check must ATTACH it, not merely be able to compute it."""
    base_model = {
        "runtime_s": B2_BASE,
        "exponents": {"256->512": {"k": 1.411, "tol": 0.5}},
        "levers": {r: {} for r in B2_BASE},
        "grid": "11x10",
    }
    meas = {"runtime_s": B2_RUN, "levers": {r: {} for r in B2_RUN}, "grid": "11x10"}
    out = rg._size_ladder_compare(base_model, meas, "boltz2", [int(r) for r in B2_BASE])
    assert not out["gate"], out
    joined = " | ".join(out["findings"])
    assert "exponent 1.41 -> 0.75" in joined, joined
    assert "only 256 aa moved" in joined, joined


def test_a_clean_run_gets_no_localisation_noise():
    base_model = {
        "runtime_s": B2_BASE,
        "exponents": {"256->512": {"k": 1.411, "tol": 0.5}},
        "levers": {r: {} for r in B2_BASE},
        "grid": "11x10",
    }
    meas = {"runtime_s": dict(B2_BASE), "levers": {r: {} for r in B2_BASE}, "grid": "11x10"}
    out = rg._size_ladder_compare(base_model, meas, "boltz2", [int(r) for r in B2_BASE])
    assert out["gate"], out
    assert not any("per-rung" in f for f in out["findings"]), out["findings"]


# The real p300c boltz2 cell: 4.1 s at 256, 10.9 s at 512, k 1.411, tol 0.50, reps 1.
B2_BASE_MODEL = {"runtime_s": B2_BASE, "reps": 1,
                 "exponents": {"256->512": {"k": 1.411, "tol": 0.5},
                               "512->768": {"k": 2.106, "tol": 0.5}}}


def test_the_denominator_rung_fragility_is_quantified():
    """+-0.50 on 256->512 is worth only 1.42 s on a 4.1 s rung, so one host event trips it."""
    note = rg._size_ladder_denominator_fragility("boltz2", "256->512", B2_BASE_MODEL, B2_BASE)
    assert note, "the boltz2 case is the whole reason this exists"
    assert "1.42 s" in note, note        # tol * t1 * ln(2) = 0.50 * 4.1 * 0.693
    assert "1.35x" in note, note         # (4.1 + 1.42) / 4.1
    assert "SINGLE-SHOT" in note, note
    assert "512 aa, not here" in note, "say sigma was characterised at the WRONG rung"
    assert "MINIMA" in note, "name the remedy, not just the problem"


def test_only_the_denominator_rung_gets_the_note():
    """512->768 divides by 10.9 s, an order less sensitive -- not this failure class."""
    assert rg._size_ladder_denominator_fragility(
        "boltz2", "512->768", B2_BASE_MODEL, B2_BASE) is None


def test_a_model_already_carrying_a_median_is_not_this_class():
    """The negative control: reps>1 means the median-of-3 escape already fired."""
    withreps = dict(B2_BASE_MODEL, reps=3)
    assert rg._size_ladder_denominator_fragility(
        "boltz2", "256->512", withreps, B2_BASE) is None


def test_a_missing_tolerance_or_time_does_not_crash():
    assert rg._size_ladder_denominator_fragility(
        "boltz2", "256->512", {"runtime_s": {}, "reps": 1}, {}) is None
    assert rg._size_ladder_denominator_fragility(
        "boltz2", "not-an-interval", B2_BASE_MODEL, B2_BASE) is None


def test_the_drift_finding_carries_the_fragility_note():
    """Integration: it must be ATTACHED to the failing check, not merely computable."""
    base_model = dict(B2_BASE_MODEL, levers={r: {} for r in B2_BASE}, grid="11x10")
    base_model["exponents"] = {"256->512": {"k": 1.411, "tol": 0.5}}
    meas = {"runtime_s": B2_RUN, "levers": {r: {} for r in B2_RUN}, "grid": "11x10"}
    out = rg._size_ladder_compare(base_model, meas, "boltz2", [int(r) for r in B2_BASE])
    joined = " | ".join(out["findings"])
    assert not out["gate"], out
    assert "only 256 aa moved" in joined, joined          # the localisation
    assert "SINGLE-SHOT" in joined, joined                # and why that rung cannot carry it
    assert "Third sighting" in joined, joined


def test_a_clean_run_carries_no_fragility_note():
    base_model = dict(B2_BASE_MODEL, levers={r: {} for r in B2_BASE}, grid="11x10")
    base_model["exponents"] = {"256->512": {"k": 1.411, "tol": 0.5}}
    meas = {"runtime_s": dict(B2_BASE), "levers": {r: {} for r in B2_BASE}, "grid": "11x10"}
    out = rg._size_ladder_compare(base_model, meas, "boltz2", [int(r) for r in B2_BASE])
    assert out["gate"], out
    assert not any("SINGLE-SHOT" in f for f in out["findings"]), out["findings"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
