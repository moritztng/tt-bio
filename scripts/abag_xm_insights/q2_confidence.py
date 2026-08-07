"""Q2 -- why confidence cannot rank.

Two correlations, deliberately separated:

    within-target   Spearman(confidence, DockQ) over one target's 256 samples, then
                    summarised across targets. "Does the score know WHICH SAMPLE is right?"
    across-target   Spearman(target-mean confidence, target-mean DockQ) over targets.
                    "Does the score know WHICH TARGET is easy?"

Run for every confidence flavour in the parquets, including iptm -- the one the field
actually uses for interfaces. Stratified by target difficulty to check whether the failure
is uniform or concentrated on hard targets.

Also carries the honest control on the `selector` column (is the "user" pick really the
model's own shipped selector?) and the replication of finding F1.
"""

from __future__ import annotations

import numpy as np

import core
from core import FLAVORS


def flavors(model: str) -> list:
    return core.available(model, tuple(FLAVORS))


def selector_control(model: str) -> dict:
    """`selector` must be a copy of one shipped confidence flavour, not a fitted or
    leaked quantity, and must not simply reproduce the file order."""
    pl = core.pools(model)
    match = {f: 0 for f in flavors(model) if f != "selector"}
    rank_rho = []
    for p in pl.values():
        s = p.selector.to_numpy()
        for f in match:
            if np.allclose(s, p[f].to_numpy(), equal_nan=True):
                match[f] += 1
        rank_rho.append(core.spearman(p["rank"].to_numpy(), s))
    n = len(pl)
    return {
        "n_targets": n,
        "selector_equals": {f: c / n for f, c in match.items()},
        "spearman_selector_vs_file_rank": float(np.nanmedian(rank_rho)),
    }


def analyse(model: str) -> dict:
    pl = core.pools(model)
    targets = sorted(pl)
    fl = flavors(model)
    within = {f: np.array([core.spearman(pl[t][f].to_numpy(), pl[t].dockq.to_numpy())
                           for t in targets]) for f in fl}
    tmean = {f: np.array([np.nanmean(pl[t][f].to_numpy()) for t in targets]) for f in fl}
    dmean = np.array([pl[t].dockq.to_numpy().mean() for t in targets])
    doracle = np.array([pl[t].dockq.to_numpy().max() for t in targets])

    out = {"model": model, "n_targets": len(targets), "flavor_names": fl, "flavors": {}}
    for f in fl:
        w = within[f]
        ok = np.isfinite(w)
        out["flavors"][f] = {
            "within_target": {
                "median": float(np.nanmedian(w)),
                "mean": core.paired_bootstrap(np.nan_to_num(w, nan=0.0)),
                "q25": float(np.nanpercentile(w, 25)),
                "q75": float(np.nanpercentile(w, 75)),
                "frac_above_0_3": float((w[ok] > 0.3).mean()),
                "n_finite": int(ok.sum()),
            },
            "across_target_mean_dockq": core.spearman(tmean[f], dmean),
            "across_target_oracle_dockq": core.spearman(tmean[f], doracle),
        }

    # Is the failure concentrated? Quartiles of target difficulty (oracle DockQ at N=256).
    q = np.quantile(doracle, [0.25, 0.5, 0.75])
    bins = np.digitize(doracle, q)
    out["difficulty_strata"] = [
        {
            "stratum": ["Q1 hardest", "Q2", "Q3", "Q4 easiest"][b],
            "n_targets": int((bins == b).sum()),
            "oracle_dockq_mean": float(doracle[bins == b].mean()),
            "within_target_rho_median": {
                f: float(np.nanmedian(within[f][bins == b])) for f in fl
            },
        }
        for b in range(4)
    ]
    out["selector_control"] = selector_control(model)
    return out


def run() -> dict:
    return {m: analyse(m) for m in core.MODELS}


if __name__ == "__main__":
    r = run()
    for m in core.MODELS:
        a = r[m]
        print(f"\n== {m}  ({a['n_targets']} targets)")
        print(f"  {'flavor':<17}{'within med':>11}{'within mean':>13}{'frac>0.3':>10}"
              f"{'across(mean)':>14}{'across(oracle)':>15}")
        for f in a["flavor_names"]:
            d = a["flavors"][f]
            print(f"  {f:<17}{d['within_target']['median']:>11.3f}"
                  f"{d['within_target']['mean']['mean']:>13.3f}"
                  f"{d['within_target']['frac_above_0_3']:>10.3f}"
                  f"{d['across_target_mean_dockq']:>14.3f}"
                  f"{d['across_target_oracle_dockq']:>15.3f}")
        c = a["selector_control"]
        print(f"  control: selector==  {c['selector_equals']}  "
              f"rho(selector, file rank)={c['spearman_selector_vs_file_rank']:.3f}")
        for s in a["difficulty_strata"]:
            print(f"    {s['stratum']:<12} n={s['n_targets']:>3} oracle={s['oracle_dockq_mean']:.3f}"
                  f"  rho_med selector={s['within_target_rho_median']['selector']:+.3f}")
