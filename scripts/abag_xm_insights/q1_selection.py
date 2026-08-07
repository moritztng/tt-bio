"""Q1 -- sampling scales, selection does not.

For every model we build, from the N=256 pools, the exact expected curves

    oracle(k)  = E[best DockQ among k uniformly drawn samples]
    user(k)    = E[DockQ of the sample the model's own selector picks out of those k]
    random     = oracle(1) = user(1) = the per-target mean DockQ

and read three things off them:

    selection efficiency  SE(k) = (user(k) - random) / (oracle(k) - random)
                          -- the share of the ceiling that sampling unlocks which
                             actually reaches the user. Decays with k.
    effective N           N_eff = the k at which the ORACLE curve equals user(256)
                          -- "sampling 256 times and trusting confidence delivers what a
                             perfect selector would have got from N_eff samples".
    threshold fractions   P(delivered DockQ >= 0.23 / 0.49 / 0.80), oracle and user.
"""

from __future__ import annotations

import numpy as np

import core
from core import THRESHOLDS, TOP_RUNG

# k grid shipped to the site (log-ish); the internal curves are exact at every k.
KGRID = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]


def _per_target(model, targets):
    """(nt, 256) oracle / user DockQ curves plus the threshold indicator curves."""
    pl = core.pools(model)
    oracle = np.empty((len(targets), TOP_RUNG))
    user = np.empty_like(oracle)
    thr = {name: {"oracle": np.empty_like(oracle), "user": np.empty_like(oracle)}
           for name, _ in THRESHOLDS}
    for i, t in enumerate(targets):
        p = pl[t]
        d = p.dockq.to_numpy()
        o_ord = np.argsort(d, kind="stable")
        u_ord = core.rank_order(p.selector.to_numpy(), d)
        oracle[i] = core.curve(o_ord, d)
        user[i] = core.curve(u_ord, d)
        for name, cut in THRESHOLDS:
            hit = (d >= cut).astype(float)
            thr[name]["oracle"][i] = core.curve(o_ord, hit)
            thr[name]["user"][i] = core.curve(u_ord, hit)
    return oracle, user, thr


def _effective_n(oracle_curves: np.ndarray, level: np.ndarray) -> np.ndarray:
    """Vectorised inverse of a monotone-increasing curve, one level per row."""
    reached = oracle_curves >= level[:, None]
    i = reached.argmax(axis=1)
    i = np.where(reached.any(axis=1), i, oracle_curves.shape[1] - 1)
    prev = np.maximum(i - 1, 0)
    lo = np.take_along_axis(oracle_curves, prev[:, None], 1).ravel()
    hi = np.take_along_axis(oracle_curves, i[:, None], 1).ravel()
    frac = np.where(hi > lo, (level - lo) / np.where(hi > lo, hi - lo, 1.0), 0.0)
    return np.where(i == 0, 1.0, prev + 1 + np.clip(frac, 0, 1))


def analyse(model: str, targets: list) -> dict:
    oracle, user, thr = _per_target(model, targets)
    b_or, b_us = core.boot_means(oracle), core.boot_means(user)
    m_or, m_us = oracle.mean(0), user.mean(0)
    rnd, b_rnd = m_or[0], b_or[:, 0]

    # SE is undefined at k=1, where oracle == user == random by construction.
    with np.errstate(divide="ignore", invalid="ignore"):
        se = (m_us - rnd) / (m_or - rnd)
        b_se = (b_us - b_rnd[:, None]) / (b_or - b_rnd[:, None])
    se[0] = np.nan
    b_se[:, 0] = np.nan
    neff = _effective_n(m_or[None, :], np.array([m_us[-1]]))[0]
    b_neff = _effective_n(b_or, b_us[:, -1])

    gi = [k - 1 for k in KGRID]
    out = {
        "model": model,
        "n_targets": len(targets),
        "k_grid": KGRID,
        "random_baseline": core.ci_of(b_rnd, rnd),
        "oracle": core.ci_of(b_or[:, gi], m_or[gi]),
        "user": core.ci_of(b_us[:, gi], m_us[gi]),
        "selection_efficiency": core.ci_of(b_se[:, gi[1:]], se[gi[1:]]),
        "selection_efficiency_k": KGRID[1:],
        "effective_n": core.ci_of(b_neff, neff),
        "gap_256": core.ci_of(b_or[:, -1] - b_us[:, -1], m_or[-1] - m_us[-1]),
        "user_gain_16_to_256": core.ci_of(b_us[:, -1] - b_us[:, 15], m_us[-1] - m_us[15]),
        "oracle_gain_16_to_256": core.ci_of(b_or[:, -1] - b_or[:, 15], m_or[-1] - m_or[15]),
        "thresholds": {},
    }
    for name, cut in THRESHOLDS:
        o, u = thr[name]["oracle"], thr[name]["user"]
        bo, bu = core.boot_means(o), core.boot_means(u)
        out["thresholds"][name] = {
            "cut": cut,
            "oracle": core.ci_of(bo[:, gi], o.mean(0)[gi]),
            "user": core.ci_of(bu[:, gi], u.mean(0)[gi]),
            "effective_n": core.ci_of(
                _effective_n(bo, bu[:, -1]),
                _effective_n(o.mean(0)[None, :], np.array([u.mean(0)[-1]]))[0],
            ),
        }
    return out


def run() -> dict:
    per_model = {m: analyse(m, sorted(core.pools(m))) for m in core.MODELS}
    common = core.common_targets(core.MODELS)
    return {
        "per_model": per_model,
        "common_targets": common,
        "per_model_common": {m: analyse(m, common) for m in core.MODELS},
    }


if __name__ == "__main__":
    r = run()
    for m in core.MODELS:
        a = r["per_model"][m]
        g = dict(zip(a["k_grid"], a["oracle"]["mean"]))
        u = dict(zip(a["k_grid"], a["user"]["mean"]))
        se = dict(zip(a["selection_efficiency_k"], a["selection_efficiency"]["mean"]))
        print(f"\n== {m}  ({a['n_targets']} targets)")
        print(f"  random={a['random_baseline']['mean']:.4f}")
        print("  k        16      64     256")
        print(f"  oracle  {g[16]:.4f}  {g[64]:.4f}  {g[256]:.4f}")
        print(f"  user    {u[16]:.4f}  {u[64]:.4f}  {u[256]:.4f}")
        print(f"  SE      {se[16]:.3f}   {se[64]:.3f}   {se[256]:.3f}")
        print(f"  N_eff   {core.fmt(a['effective_n'], 2)}")
        print(f"  gap@256 {core.fmt(a['gap_256'])}")
        print(f"  user gain 16->256 {core.fmt(a['user_gain_16_to_256'])}"
              f"   oracle gain {core.fmt(a['oracle_gain_16_to_256'])}")
