"""Q6 -- what would it actually take to get there by sampling alone?

Fits the measured k = 1..256 curves with two families and reports both:

    power    y(N) = a - b * N^(-alpha)      (saturating, has a finite ceiling a)
    log      y(N) = c + d * log2(N)         (never saturates)

and extrapolates each to the N that reaches 80% of targets at DockQ >= 0.23 and >= 0.49.
Every extrapolated N beyond 256 is an EXTRAPOLATION PAST THE MEASURED RANGE and is
labelled as such; where the power fit's own ceiling sits below the target the honest
answer is "unreachable by sampling", not a number.

Cost conversion uses the same fleet per-sample card-seconds as Q4.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

import core
import q4_pareto
from core import THRESHOLDS

TARGET_FRAC = 0.80


def _power(n, a, b, alpha):
    return a - b * np.power(n, -alpha)


def _log(n, c, d):
    return c + d * np.log2(n)


def fit_both(y: np.ndarray, ceiling: float = 1.0) -> dict:
    """`ceiling` bounds the saturating fit's asymptote (1.0 for a fraction or a DockQ).

    Unbounded, the power family is UNIDENTIFIABLE over k <= 256: it walks alpha to 0 and
    the asymptote to hundreds, i.e. it degenerates into the log fit. Bounding the
    asymptote to what the metric can physically reach makes the saturating alternative a
    real alternative, and `degenerate` records when it still collapses.
    """
    n = np.arange(1, len(y) + 1, dtype=float)
    out = {}
    try:
        p, _ = curve_fit(
            _power, n, y, p0=[min(ceiling, y[-1] * 1.3), max(y[-1] - y[0], 1e-3), 0.3],
            bounds=([y[-1], 1e-6, 1e-3], [ceiling, 10.0, 3.0]), maxfev=60000)
        out["power"] = {"a": float(p[0]), "b": float(p[1]), "alpha": float(p[2]),
                        "rmse": float(np.sqrt(np.mean((_power(n, *p) - y) ** 2))),
                        "degenerate": bool(p[2] < 5e-3 or p[0] >= ceiling - 1e-6)}
    except Exception:
        out["power"] = None
    q = np.polyfit(np.log2(n), y, 1)
    out["log"] = {"c": float(q[1]), "d": float(q[0]),
                  "rmse": float(np.sqrt(np.mean((_log(n, q[1], q[0]) - y) ** 2)))}
    return out


def solve(fit: dict, level: float) -> dict:
    """Smallest N reaching `level` under each fit; None where the fit cannot reach it."""
    out = {}
    p = fit.get("power")
    if p and not p["degenerate"] and p["a"] > level and p["b"] > 0:
        out["power_n"] = float(((p["a"] - level) / p["b"]) ** (-1.0 / p["alpha"]))
    else:
        out["power_n"] = None
        out["power_ceiling"] = p["a"] if p else None
        out["power_degenerate"] = bool(p["degenerate"]) if p else None
    lg = fit["log"]
    out["log_n"] = float(2 ** ((level - lg["c"]) / lg["d"])) if lg["d"] > 0 else None
    return out


def run() -> dict:
    cost = q4_pareto.fleet_cost()
    res = {}
    for m in core.MODELS:
        pl = core.pools(m)
        targets = sorted(pl)
        oracle = np.array([core.oracle_curves(pl[t]) for t in targets]).mean(0)
        curves = {"oracle_dockq": oracle}
        for name, cut in THRESHOLDS:
            curves[f"frac_{name}"] = np.array([
                core.curve(np.argsort(v := pl[t].dockq.to_numpy(), kind="stable"),
                           (v >= cut).astype(float))
                for t in targets
            ]).mean(0)
        fits = {k: fit_both(v) for k, v in curves.items()}
        # oracle DockQ and threshold fractions are both bounded by 1.0.
        s_per_sample = float(np.median([cost[m][t] for t in targets if t in cost.get(m, {})]))
        need = {}
        for name, _ in THRESHOLDS[:2]:
            key = f"frac_{name}"
            sol = solve(fits[key], TARGET_FRAC)
            for k in ("power_n", "log_n"):
                n = sol.get(k)
                sol[k.replace("_n", "_card_h_per_target")] = (
                    None if n is None else n * s_per_sample / 3600.0)
            sol["measured_at_256"] = float(curves[key][-1])
            need[name] = sol
        res[m] = {
            "n_targets": len(targets),
            "s_per_sample": s_per_sample,
            "fits": fits,
            "measured": {k: {"at_1": float(v[0]), "at_16": float(v[15]),
                             "at_256": float(v[-1])} for k, v in curves.items()},
            "n_for_80pct": need,
        }
    return {"target_frac": TARGET_FRAC, "per_model": res}


if __name__ == "__main__":
    r = run()
    for m, a in r["per_model"].items():
        f = a["fits"]["oracle_dockq"]["power"]
        print(f"\n== {m}  ({a['n_targets']} targets, {a['s_per_sample']:.1f} card-s/sample)")
        print(f"  oracle: power ceiling a={f['a']:.3f} alpha={f['alpha']:.3f}"
              f" rmse={f['rmse']:.4f} degenerate={f['degenerate']}"
              f" | log rmse={a['fits']['oracle_dockq']['log']['rmse']:.4f}")
        for name, s in a["n_for_80pct"].items():
            pn = s["power_n"]
            pw = (f"{pn:.3g} ({s['power_card_h_per_target']:.3g} card-h/tgt)" if pn
                  else f"unreachable (ceiling {s['power_ceiling']:.3f})")
            print(f"  80% at DockQ>={dict(THRESHOLDS)[name]}: measured@256 "
                  f"{s['measured_at_256']:.3f} | saturating N={pw} | log-linear N="
                  f"{s['log_n']:.3g} ({s['log_card_h_per_target']:.3g} card-h/tgt)")
