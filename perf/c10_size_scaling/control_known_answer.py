"""Known-answer calibration of the ladder reducer, run BEFORE any fold.

Four campaigns in this corpus were derailed by an uncalibrated instrument, so every function that
turns seconds into a scaling exponent is fed synthetic timings built from an EXACT A and p and has
to hand both back. The controls that matter most are the refusals: a single clock must not yield a
work term, and a wrong exponent must come back wrong rather than be rounded into agreement.
"""
from __future__ import annotations
import math, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
from sizefit import work_two_clock, leg_exponent, size_independent_floor

CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn)); return fn
    return deco


def amplitude(p, W512=14665.0):
    """A such that W(512) = A*512**p equals the measured 512 aa work term, so the synthetic ladder
    sits at the real one's magnitude and the injected noise is the real A/A noise."""
    return W512 / 512.0 ** p


def synth(N, p, F0, Fslope, f, A=None):
    """A fold that is exactly T = F + W/f with W = A*N**p. Seconds, from an exact ladder."""
    A = amplitude(p) if A is None else A
    return (F0 + Fslope * N) + (A * N ** p) / f


@case("exact A and p recovered from two clocks at every rung")
def _():
    p, F0, Fslope = 0.6300, -0.881, 0.0095
    A = amplitude(p)
    worst_W, worst_F, worst_p = 0.0, 0.0, 0.0
    W = {}
    for N in (384, 512, 640, 768):
        r = work_two_clock(1350, synth(N, p, F0, Fslope, 1350), 0.0,
                           800, synth(N, p, F0, Fslope, 800), 0.0)
        W[N] = r["W_Mcycles"]
        worst_W = max(worst_W, abs(r["W_Mcycles"] - A * N ** p) / (A * N ** p))
        worst_F = max(worst_F, abs(r["F_s"] - (F0 + Fslope * N)))
    for a, b in ((384, 512), (512, 640), (640, 768)):
        worst_p = max(worst_p, abs(leg_exponent(a, W[a], 0.0, b, W[b], 0.0)["p_work"] - p))
    assert worst_W < 1e-12, worst_W
    assert worst_F < 1e-9, worst_F
    assert worst_p < 1e-12, worst_p
    return f"W recovered to {worst_W:.1e} relative, F to {worst_F:.1e} s, p to {worst_p:.1e} absolute"


@case("the 1350/800 gains are c10-fixed-cost's, 2.4545 and -1.4545")
def _():
    r = work_two_clock(1350, 14.9306, 0.0, 800, 22.3574, 0.0)
    g = r["dF_dt_gain"]
    assert abs(g[0] - 2.454545) < 1e-5 and abs(g[1] + 1.454545) < 1e-5, g
    # and reproduces that row's own published endpoint F to the digit it published
    assert abs(r["F_s"] - 4.1280) < 5e-4, r["F_s"]
    return (f"gains {g[0]:.4f} / {g[1]:.4f}; F from c10-fixed-cost's own 512 aa medians "
            f"= {r['F_s']:.4f} s against its published 4.1280 s")


@case("ONE clock cannot produce a work term: refused, not approximated")
def _():
    for f in (1350, 800):
        try:
            work_two_clock(f, 14.9, 0.0, f, 14.9, 0.0)
        except ValueError as e:
            assert "DISTINCT clocks" in str(e), e
        else:
            raise AssertionError(f"single clock at {f} MHz was NOT refused")
    return "work_two_clock(f, ..., f, ...) raises at both arms"


@case("ONE size cannot produce an exponent, and a non-positive work term cannot either")
def _():
    for args, want in (((512, 1.0, 0.0, 512, 2.0, 0.0), "DISTINCT sizes"),
                       ((512, 0.0, 0.0, 768, 1.0, 0.0), "non-positive"),
                       ((512, 1.0, 0.0, 768, -3.0, 0.0), "non-positive")):
        try:
            leg_exponent(*args)
        except ValueError as e:
            assert want in str(e), (args, e)
        else:
            raise AssertionError(f"{args} was NOT refused")
    return "identical sizes and non-positive work both raise"


@case("negative control: a planted WRONG exponent comes back wrong")
def _():
    F0, Fslope = -0.881, 0.0095
    W = {N: work_two_clock(1350, synth(N, 1.85, F0, Fslope, 1350), 0.0,
                           800, synth(N, 1.85, F0, Fslope, 800), 0.0)["W_Mcycles"]
         for N in (512, 768)}
    got = leg_exponent(512, W[512], 0.0, 768, W[768], 0.0)["p_work"]
    assert abs(got - 1.85) < 1e-12, got
    assert abs(got - 0.63) > 1.0, got
    return f"a ladder built at p=1.85 reads back 1.850000, not 0.63; the check is not pinned to the expected answer"


@case("standard error propagates: se_p brackets the truth under the measured A/A floor")
def _():
    import random
    p, F0, Fslope = 0.6300, -0.881, 0.0095
    sd = {1350: 0.049, 800: 0.123}          # c10-bare-baseline and c10-fixed-cost, 512 aa
    rng = random.Random(0); n = 5; trials = 4000; inside = 0; widths = []
    for _ in range(trials):
        W, se = {}, {}
        for N in (512, 768):
            med, sem = {}, {}
            for f in (1350, 800):
                xs = sorted(synth(N, p, F0, Fslope, f) + rng.gauss(0, sd[f]) for _ in range(n))
                med[f] = xs[n // 2]
                s = math.sqrt(sum((x - sum(xs) / n) ** 2 for x in xs) / (n - 1))
                sem[f] = 1.2533 * s / math.sqrt(n)
            r = work_two_clock(1350, med[1350], sem[1350], 800, med[800], sem[800])
            W[N], se[N] = r["W_Mcycles"], r["se_W_Mcycles"]
        leg = leg_exponent(512, W[512], se[512], 768, W[768], se[768])
        widths.append(leg["se_p_work"])
        if abs(leg["p_work"] - p) <= 2 * leg["se_p_work"]:
            inside += 1
    cov = inside / trials; med_se = sorted(widths)[trials // 2]
    assert 0.85 <= cov <= 0.99, cov
    assert med_se < 0.15, med_se
    return (f"2-sigma coverage {cov:.3f} over {trials} synthetic sessions, median se_p {med_se:.4f}, "
            f"inside the 0.15 quotability bar")


@case("the two hypotheses are separated at this precision")
def _():
    F0, Fslope = -0.881, 0.0095
    out = {}
    for p in (0.63, 1.85):
        W = {N: work_two_clock(1350, synth(N, p, F0, Fslope, 1350), 0.0,
                               800, synth(N, p, F0, Fslope, 800), 0.0)["W_Mcycles"]
             for N in (512, 768)}
        # se_W = 1963.64 * hypot(se_1350, se_800) at the pre-registered 5 folds per cell
        se = 1963.6364 * math.hypot(1.2533 * 0.049 / math.sqrt(5), 1.2533 * 0.123 / math.sqrt(5))
        leg = leg_exponent(512, W[512], se, 768, W[768], se)
        out[p] = (leg["p_work"], leg["se_p_work"])
    sep = abs(out[1.85][0] - out[0.63][0]) / math.hypot(out[1.85][1], out[0.63][1])
    assert sep > 10, sep
    return (f"p=0.63 reads {out[0.63][0]:.4f}+-{out[0.63][1]:.4f}, p=1.85 reads "
            f"{out[1.85][0]:.4f}+-{out[1.85][1]:.4f}: {sep:.1f} sigma apart")


@case("the non-negative-mixture floor is exact on a planted mixture")
def _():
    K, L, n1 = 8220.0, 6445.0, 512
    W = lambda N: K + L * (N / n1)
    for n2 in (640, 768):
        got = size_independent_floor(n1, W(n1), n2, W(n2))
        assert abs(got["floor_Mcycles"] - K) < 1e-9, got
        assert got["implies_size_independent_term"]
    # and a super-linear ladder must report NO implied size-independent term
    sup = size_independent_floor(512, 14665.0, 768, 14665.0 * 1.5 ** 1.85)
    assert sup["floor_Mcycles"] < 0 and not sup["implies_size_independent_term"], sup
    return (f"K recovered exactly from a planted K={K:.0f}+L*(N/512) mixture; a p=1.85 ladder returns "
            f"{sup['floor_Mcycles']:.0f} Mcycles, correctly refusing to imply a fixed term")


@case("seconds convert to cycles at the clock they were measured at")
def _():
    assert abs(work_two_clock(1350, 10.0, 0.0, 800, 10.0, 0.0)["W_Mcycles"]) < 1e-9
    r = work_two_clock(1350, 14.9306, 0.0, 800, 22.3574, 0.0)
    assert abs(r["W_Mcycles"] - (22.3574 - 14.9306) * 1963.6364) < 1e-3, r["W_Mcycles"]
    return (f"a clock-flat fold yields 0 Mcycles of clock-scaled work; c10-fixed-cost's 512 aa "
            f"medians yield {r['W_Mcycles']:.1f} Mcycles against its fitted 14665.0 +- 121.1")


def main():
    bad = 0
    for name, fn in CASES:
        try:
            print(f"PASS  {name}\n        {fn()}")
        except BaseException as e:
            bad += 1; print(f"FAIL  {name}\n        {e!r}")
    print(f"\n{len(CASES) - bad}/{len(CASES)} known-answer controls pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
