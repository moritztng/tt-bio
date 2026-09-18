"""An exponent has two rungs and they do not share a noise level.

Device-free. The size-ladder band used to propagate the MIDDLE rung's sigma into every interval,
i.e. it assumed the lowest rung is as quiet as the rung sigma was measured at. Measured over all
52 recorded p300c cells against a zero-contention gate (perf/c13_land/gate.log, f2b62c7bc):
geomean ratio 0.968, median 0.978, 50 of 52 inside [0.78, 1.05] -- and BOTH outliers are the
256 aa rung, on opposite sides. boltz2 baseline 4.1 s vs 6.5 s measured (1.585) and openbind
baseline 9.3 s vs 6.5 s measured (0.699). Those two were the only models that failed the arm.

boltz2's two modes are 0.66 apart in exponent against a +/-0.50 band, so no single-shot
re-record fixes it: whichever mode you write down, the other one fails the gate.

Run: python3 tests/test_size_ladder_per_rung_sigma.py, or via pytest in the release suite.
"""
import math
import os
import sys
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import release_gate as rg  # noqa: E402

# boltz2's real recorded p300c ladder, and its real 512 aa sigma.
RT = {"256": 4.1, "512": 10.9, "640": 17.2, "768": 25.6, "896": 37.8, "1024": 48.4}
SIGMA_512 = 0.065


def _tol(block, interval):
    return block["exponents"][interval]["tol"]


def test_no_per_rung_sigma_reproduces_the_old_band_exactly():
    """Every baseline on disk predates this and carries no per-rung sigma. None of them may
    move: with s1 == s2 the new formula IS the old one."""
    block, skip = rg._size_ladder_exponent_block("boltz2", RT, SIGMA_512)
    assert skip is None, skip
    for interval in block["exponents"]:
        n1, n2 = (int(x) for x in interval.split("->"))
        old = max(rg.SIZE_LADDER_EXP_TOL_FLOOR,
                  3 * math.sqrt(2) * SIGMA_512 / math.log(n2 / n1))
        assert abs(_tol(block, interval) - round(old, 3)) < 1e-9, \
            f"{interval}: {_tol(block, interval)} != legacy {old}"
    assert "sigma_runtime" not in block, "invented a per-rung sigma nobody measured"


def test_recorded_exponents_never_move():
    """The k values are the reference the whole fleet gates against. This change is allowed to
    widen a band; it is not allowed to shift a k."""
    a, _ = rg._size_ladder_exponent_block("boltz2", RT, SIGMA_512)
    b, _ = rg._size_ladder_exponent_block("boltz2", RT, SIGMA_512,
                                          {"256": 0.22, "512": 0.02, "640": 0.02,
                                           "768": 0.02, "896": 0.02, "1024": 0.02})
    assert {i: e["k"] for i, e in a["exponents"].items()} == \
           {i: e["k"] for i, e in b["exponents"].items()}, "a recorded exponent moved"


def test_a_noisy_low_rung_widens_only_its_own_interval():
    """The 256 rung must not inflate 512->768. That was the whole defect, in reverse."""
    quiet = {"256": 0.02, "512": 0.02, "640": 0.02, "768": 0.02, "896": 0.02, "1024": 0.02}
    noisy = dict(quiet, **{"256": 0.12})
    a, _ = rg._size_ladder_exponent_block("boltz2", RT, SIGMA_512, quiet)
    b, _ = rg._size_ladder_exponent_block("boltz2", RT, SIGMA_512, noisy)
    assert _tol(b, "256->512") > _tol(a, "256->512"), "a noisy 256 rung did not widen its band"
    for interval in a["exponents"]:
        if interval != "256->512":
            assert _tol(b, interval) == _tol(a, interval), \
                f"{interval} moved because a DIFFERENT rung was noisy"


def test_boltz2s_real_bimodal_256_rung_gets_an_honest_band_and_stops_failing_on_noise():
    """The measured case, end to end. 4.1 and 6.5 s are the two observed modes of that cell, so
    sigma/mean over the pair is 0.320. The honest band on 256->512 is then +/-1.389 instead of
    the +/-0.50 floor, the observed 0.66 two-mode gap sits INSIDE it, and 512->768 is untouched
    at +/-0.50. The recorded k does not move."""
    import statistics
    ts = [4.1, 6.5]
    s256 = statistics.stdev(ts) / statistics.mean(ts)
    assert round(s256, 3) == 0.320, s256

    block, skip = rg._size_ladder_exponent_block(
        "boltz2", RT, SIGMA_512,
        {"256": round(s256, 4), "512": 0.02, "640": 0.02, "768": 0.02,
         "896": 0.02, "1024": 0.02})
    assert skip is None, skip
    assert round(_tol(block, "256->512"), 3) == 1.389, _tol(block, "256->512")
    assert _tol(block, "512->768") == 0.5, "a quiet interval was widened by a noisy neighbour"
    assert block["exponents"]["256->512"]["k"] == 1.411, "the recorded exponent moved"

    k = lambda t1, t2: math.log(t2 / t1) / math.log(2)
    gap = abs(k(4.1, 10.9) - k(6.5, 10.9))
    assert gap < _tol(block, "256->512"), \
        f"the two-mode gap {gap:.2f} still fails its own honest band"
    assert gap > rg.SIZE_LADDER_EXP_TOL_FLOOR, \
        "control: this gap must fail the OLD floor, or the test proves nothing"
    assert block["sigma_runtime"]["256"] == round(s256, 4), "per-rung sigma not recorded"


def test_a_rung_noisier_still_trips_the_cliff_and_names_which_rung():
    """The cliff must still fire, and its message must name the rung that blew the band -- the
    old message quoted the MIDDLE rung's sigma, which said nothing about the culprit."""
    block, skip = rg._size_ladder_exponent_block(
        "boltz2", RT, SIGMA_512,
        {"256": 0.40, "512": 0.02, "640": 0.02, "768": 0.02, "896": 0.02, "1024": 0.02})
    assert block is None and skip, "gated an interval whose own noise exceeds the cliff"
    assert "256->512" in skip and "256 aa" in skip and "%" in skip, skip


def test_the_gap_between_boltz2s_two_modes_exceeds_the_floor_either_way_round():
    """Why re-recording cannot fix it, as arithmetic rather than as an opinion."""
    k = lambda t1, t2: math.log(t2 / t1) / math.log(2)
    fast, slow = k(4.1, 10.9), k(6.5, 10.9)
    assert abs(fast - slow) > rg.SIZE_LADDER_EXP_TOL_FLOOR, (fast, slow)
    assert round(abs(fast - slow), 2) == 0.66, round(abs(fast - slow), 2)


def test_measure_returns_a_sigma_for_every_repeated_rung():
    """The per-rung sigmas must come from folds that happened, not be invented at record time."""
    src = open(os.path.join(REPO, "scripts", "release_gate.py")).read()
    assert 'sigmas[str(rung)] = round(statistics.stdev(ts) / statistics.mean(ts), 4)' in src
    assert '"sigmas": sigmas' in src, "the measurement never hands the per-rung sigmas up"
    assert src.count('meas.get("sigmas")') == 3, \
        "an _size_ladder_exponent_block call site still drops the per-rung sigmas"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and isinstance(fn, types.FunctionType):
            fn()
            print(f"ok  {name}")
    print("all ok")
