"""Q3 -- the epitope wall.

Decomposes every (model, target) into three states, using the complete DockQ labels and
the predicted-vs-native epitope overlap (`epitope_jaccard`, EJ):

    never-finds-site        max EJ over the labelled pool  <  EJ*
    right-site-wrong-pose   max EJ >= EJ*  but  max DockQ < 0.23
    solved                  max DockQ >= 0.23

EJ* is read off the data, not chosen: the per-sample EJ distribution is bimodal and EJ* is
the trough between the modes.

Then two growth curves, to test whether depth buys site DISCOVERY or only pose refinement:
P(at least one of k samples lands on the right site) against P(at least one of k samples
is an acceptable pose).

Label coverage: DockQ is complete for every model at N=256. EJ is complete for boltz2 and
opendde-abag; for protenix-v2 and esmfold2 the epitope scorer ran on a subset of the
64-sample chunks, so those two are analysed at the deepest chunk-aligned depth that still
covers most targets, and are never quoted at N=256. Partial EJ coverage can only
UNDER-count site discovery, so "never finds the site" is a conservative count for them.
"""

from __future__ import annotations

import numpy as np

import core

ACCEPTABLE = 0.23
MIN_TARGETS_FOR_CURVE = 100


def ej_pools(model: str) -> dict:
    """Per-target sub-pools carrying an epitope label, in chunk-then-rank order.

    EJ missingness is chunk-aligned (whole 64-sample chunks were or were not scored), so a
    prefix of this ordering is always a union of complete folds, never a confidence-biased
    slice of one.
    """
    out = {}
    for t, p in core.pools(model).items():
        q = p[p.epitope_jaccard.notna()].sort_values(["chunk", "rank"])
        if len(q) >= 16:
            out[t] = q.reset_index(drop=True)
    return out


def ej_threshold(model: str, bins: int = 60) -> dict:
    """Trough of the bimodal per-sample EJ distribution, searched over the interior."""
    v = np.concatenate([p.epitope_jaccard.to_numpy() for p in ej_pools(model).values()])
    hist, edges = np.histogram(v, bins=bins, range=(0.0, 1.0))
    mid = 0.5 * (edges[:-1] + edges[1:])
    smooth = np.convolve(hist, np.ones(5) / 5, mode="same")
    lo, hi = np.searchsorted(mid, 0.10), np.searchsorted(mid, 0.70)
    return {
        "n_samples": int(len(v)),
        "hist_mid": mid.tolist(),
        "hist_count": hist.tolist(),
        "trough": float(mid[int(lo + np.argmin(smooth[lo:hi]))]),
    }


def curve_depth(ej: dict) -> int:
    """Deepest chunk-aligned depth still covering MIN_TARGETS_FOR_CURVE targets."""
    depths = np.array([len(p) for p in ej.values()])
    for d in (256, 192, 128, 64):
        if (depths >= d).sum() >= MIN_TARGETS_FOR_CURVE:
            return d
    return 64


def analyse(model: str, ej_star: float) -> dict:
    dq = core.pools(model)
    ej = ej_pools(model)
    targets = sorted(ej)
    max_ej = np.array([ej[t].epitope_jaccard.max() for t in targets])
    max_dq = np.array([dq[t].dockq.max() for t in targets])  # complete 256-sample pool

    state = np.where(max_dq >= ACCEPTABLE, "solved",
                     np.where(max_ej >= ej_star, "right_site_wrong_pose", "never_finds_site"))
    solved, unsolved = state == "solved", state != "solved"
    fails = int(unsolved.sum())

    f2 = {
        "n_targets": len(targets),
        "n_unsolved": fails,
        "frac_failures_that_never_find_site":
            float((state == "never_finds_site").sum() / fails) if fails else None,
        "max_ej_median_unsolved": float(np.median(max_ej[unsolved])) if unsolved.any() else None,
        "max_ej_median_solved": float(np.median(max_ej[solved])) if solved.any() else None,
    }

    # Growth curves at a chunk-aligned depth that keeps the target set wide.
    D = curve_depth(ej)
    full = [t for t in targets if len(ej[t]) >= D]
    site_hit = np.array([core.curve(np.argsort(v := ej[t].head(D).epitope_jaccard.to_numpy(),
                                               kind="stable"), (v >= ej_star).astype(float))
                         for t in full])
    dq_hit = np.array([core.curve(np.argsort(v := dq[t].head(D).dockq.to_numpy(), kind="stable"),
                                  (v >= ACCEPTABLE).astype(float)) for t in full])
    ej_c = np.array([core.oracle_curves(ej[t].head(D), "epitope_jaccard") for t in full])
    dq_c = np.array([core.oracle_curves(dq[t].head(D), "dockq") for t in full])

    ks = [k for k in (1, 2, 4, 8, 16, 32, 64, 128, 256) if k <= D]
    gi = [k - 1 for k in ks]

    def rel(c):
        m = c.mean(0)[gi]
        return ((m - m[0]) / (m[-1] - m[0])).tolist()

    return {
        "model": model,
        "ej_star": ej_star,
        "curve_depth": D,
        "n_targets_in_curve": len(full),
        "ej_labels_complete": bool(D == core.TOP_RUNG and len(full) == len(targets)),
        "f2": f2,
        "states": {s: int((state == s).sum()) for s in
                   ["solved", "right_site_wrong_pose", "never_finds_site"]},
        "k_grid": ks,
        "p_finds_site": core.ci_of(core.boot_means(site_hit)[:, gi], site_hit.mean(0)[gi]),
        "p_acceptable_pose": core.ci_of(core.boot_means(dq_hit)[:, gi], dq_hit.mean(0)[gi]),
        "max_ej_curve": core.ci_of(core.boot_means(ej_c)[:, gi], ej_c.mean(0)[gi]),
        "max_dockq_curve": core.ci_of(core.boot_means(dq_c)[:, gi], dq_c.mean(0)[gi]),
        "max_ej_relative_gain": rel(ej_c),
        "max_dockq_relative_gain": rel(dq_c),
        "per_target": [
            {"target": t, "max_ej": float(max_ej[i]), "max_dockq": float(max_dq[i]),
             "state": str(state[i]), "ej_depth": int(len(ej[t]))}
            for i, t in enumerate(targets)
        ],
    }


def cross_model(results: dict) -> dict:
    """Do different models fail on the SAME targets, and at the same stage?"""
    per = {m: {r["target"]: r for r in results[m]["per_target"]} for m in results}
    common = sorted(set.intersection(*[set(v) for v in per.values()]))
    fail = {m: {t for t in common if per[m][t]["state"] != "solved"} for m in per}
    nosite = {m: {t for t in common if per[m][t]["state"] == "never_finds_site"} for m in per}
    ms = list(per)
    pair = {}
    for i, a in enumerate(ms):
        for b in ms[i + 1:]:
            u, v = fail[a], fail[b]
            pair[f"{a}|{b}"] = {
                "failure_jaccard": len(u & v) / len(u | v) if u | v else None,
                "nosite_jaccard": (len(nosite[a] & nosite[b]) / len(nosite[a] | nosite[b])
                                   if nosite[a] | nosite[b] else None),
            }
    all_fail = set.intersection(*fail.values())
    return {
        "n_common_targets": len(common),
        "failed_by_all_four": len(all_fail),
        "solved_by_at_least_one": len(common) - len(all_fail),
        "best_single_model": max((len(common) - len(fail[m]), m) for m in ms)[1],
        "per_model_solved": {m: len(common) - len(fail[m]) for m in ms},
        "pairwise": pair,
    }


def run() -> dict:
    thr = {m: ej_threshold(m) for m in core.MODELS}
    # One EJ* shared by every model -- the median of the per-model troughs -- so the
    # three-state decomposition is comparable across models instead of each using its own.
    ej_star = float(np.median([thr[m]["trough"] for m in core.MODELS]))
    res = {m: analyse(m, ej_star) for m in core.MODELS}
    return {"ej_star": ej_star, "ej_hist": thr, "per_model": res,
            "cross_model": cross_model(res)}


if __name__ == "__main__":
    r = run()
    print("EJ* =", round(r["ej_star"], 3),
          " per-model troughs:", {m: round(r["ej_hist"][m]["trough"], 3) for m in core.MODELS})
    for m in core.MODELS:
        a, f = r["per_model"][m], r["per_model"][m]["f2"]
        print(f"\n== {m}  EJ-complete={a['ej_labels_complete']}  curve depth={a['curve_depth']}"
              f" on {a['n_targets_in_curve']} targets")
        print(f"  states {a['states']}   ({f['n_targets']} targets)")
        print(f"  F2: {f['n_unsolved']} unsolved; {f['frac_failures_that_never_find_site']:.2f}"
              f" of failures never find the site;"
              f" max-EJ median unsolved {f['max_ej_median_unsolved']:.3f}"
              f" vs solved {f['max_ej_median_solved']:.3f}")
        print(f"  k             {a['k_grid']}")
        print(f"  P(find site)  {[round(x,3) for x in a['p_finds_site']['mean']]}")
        print(f"  P(good pose)  {[round(x,3) for x in a['p_acceptable_pose']['mean']]}")
        print(f"  rel maxEJ     {[round(x,2) for x in a['max_ej_relative_gain']]}")
        print(f"  rel maxDockQ  {[round(x,2) for x in a['max_dockq_relative_gain']]}")
    c = r["cross_model"]
    print(f"\ncross-model on {c['n_common_targets']} common targets:"
          f" solved by >=1 model {c['solved_by_at_least_one']},"
          f" failed by all four {c['failed_by_all_four']}")
    print("  per-model solved:", c["per_model_solved"])
    print("  pairwise:", {k: round(v["failure_jaccard"], 3) for k, v in c["pairwise"].items()})
