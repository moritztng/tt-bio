"""Two-clock work/fixed separation per size, and the per-LEG scaling exponent.

The campaign's currency is device work in Mcycles, not seconds: on this fixture a second is worth
1350 Mcycles at burst and 800 at the pinned floor, so a size ladder measured at one clock cannot
say anything about work at all. Every function here therefore REFUSES a single clock rather than
silently returning a number, and every leg is computed on its own two neighbouring sizes so a
changing exponent stays visible. A global fit over the whole ladder would average a rising
exponent into a flat one, which is exactly the confound this row exists to separate.
"""
from __future__ import annotations
import math


def work_two_clock(f_hi, t_hi, se_hi, f_lo, t_lo, se_lo):
    """T = F + W/f at two pinned clocks. f in MHz, t in s; W comes out in Mcycles.

    F = t_hi * g_hi + t_lo * g_lo with g_hi = (1/f_lo)/(1/f_lo - 1/f_hi) and g_lo = -(1/f_hi)/(...).
    At 1350/800 those gains are 2.4545 and -1.4545, which is c10-fixed-cost's closed form unchanged.
    """
    if f_hi == f_lo:
        raise ValueError("a work term needs two DISTINCT clocks: one clock cannot separate F from W")
    u, v = 1.0 / f_hi, 1.0 / f_lo
    den = v - u
    F = (t_hi * v - t_lo * u) / den
    W = (t_hi - t_lo) / (u - v)
    g_hi, g_lo = v / den, -u / den
    return {"clocks_MHz": [f_hi, f_lo], "F_s": F, "W_Mcycles": W,
            "se_F_s": math.hypot(g_hi * se_hi, g_lo * se_lo),
            "se_W_Mcycles": math.hypot(se_hi / (u - v), se_lo / (u - v)),
            "dF_dt_gain": [g_hi, g_lo], "dW_dt_gain": [1.0 / (u - v), -1.0 / (u - v)]}


def leg_exponent(n1, W1, se1, n2, W2, se2):
    """p such that W scales as N**p between two adjacent ladder sizes, with its standard error.

    p = ln(W2/W1) / ln(n2/n1); se_p = hypot(se2/W2, se1/W1) / ln(n2/n1), the first-order
    propagation of the two work errors through the log. Reported per leg, never pooled.
    """
    if n1 == n2:
        raise ValueError("an exponent needs two DISTINCT sizes")
    if W1 <= 0 or W2 <= 0:
        raise ValueError(f"non-positive work term ({W1}, {W2}): an exponent is undefined")
    ln_ratio = math.log(n2 / n1)
    p = math.log(W2 / W1) / ln_ratio
    se_p = math.hypot(se2 / W2, se1 / W1) / abs(ln_ratio)
    return {"sizes_aa": [n1, n2], "token_ratio": n2 / n1, "W_Mcycles": [W1, W2],
            "work_ratio": W2 / W1, "se_work_ratio": (W2 / W1) * math.hypot(se2 / W2, se1 / W1),
            "p_work": p, "se_p_work": se_p,
            "p_minus_1_sigma": (p - 1.0) / se_p if se_p else None}


def size_independent_floor(n1, W1, n2, W2):
    """Non-negative-mixture floor on the part of W that does not grow with the target.

    Write W(N) = K + L*(N/n1) with K, L >= 0, the weakest assumption that still admits the
    at-least-linear work the fold provably contains. Two sizes pin K exactly:
    K = (W1*r - W2) / (r - 1) with r = n2/n1. K < 0 means the ladder grew FASTER than linearly
    over this leg and no size-independent term is implied at all.
    """
    r = n2 / n1
    K = (W1 * r - W2) / (r - 1.0)
    return {"floor_Mcycles": K, "floor_pct_of_W1": 100.0 * K / W1,
            "implies_size_independent_term": K > 0,
            "model": "W(N) = K + L*(N/n1), K,L >= 0; K<0 means super-linear growth on this leg"}
