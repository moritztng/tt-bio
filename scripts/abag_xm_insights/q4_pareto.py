"""Q4 -- diversify or deepen, at matched compute.

Cost comes from the fleet records (`galaxy/fleet_results.jsonl`), which the DATASHEET
names as the cost authority: seconds of one Wormhole Galaxy chip per completed 64-sample
chunk, per model and per target.

  NOTE: the `wall_s` column in the packaged sample parquets is NOT a per-model cost --
  it is byte-identical across all four models for the same (target, chunk) and takes ~90
  distinct minute-quantised values. It is a packaging artifact. Never cost from it.

Given a per-target budget in card-hours, a strategy assigns each model a sample count
n_m = min(256, budget_share / cost_m(target)). For that allocation we compute EXACTLY,
per target:

    oracle    E[max DockQ over the union of the n_m draws]
              via P(union max <= v) = prod_m C(c_m(v), n_m) / C(N_m, n_m)
    solved    P(union max >= threshold), same product
    delivered E[DockQ of the sample with the highest confidence across the whole union]
              -- exact, but the cross-model confidence comparison is UNCALIBRATED, which
              is the point: it is reported to show what a naive user actually gets.

The pre-declared comparison is single-model-deep vs an even four-way split. No subset
search feeds that claim; the full 15-subset frontier is reported separately as
descriptive, post-hoc.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

import core
from core import THRESHOLDS

FLEET = Path.home() / "abag_xm/deepn/galaxy/fleet_results.jsonl"
CHUNK = 64
BUDGETS_CARD_H = [0.02, 0.04, 0.08, 0.15, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5]


def fleet_cost() -> dict:
    """card-seconds per sample, per (model, target); median over that target's chunks."""
    per = {}
    for line in FLEET.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue  # two malformed lines in the merged fleet log
        if r.get("rung") == 256 and r.get("rc") == 0 and r.get("cifs") == CHUNK:
            per.setdefault(r["model"], {}).setdefault(r["target"], []).append(r["seconds"])
    return {m: {t: float(np.median(v)) / CHUNK for t, v in d.items()} for m, d in per.items()}


class TargetPools:
    """Sorted per-model DockQ / confidence arrays for one target, plus exact union math."""

    def __init__(self, model_pools: dict):
        self.models = sorted(model_pools)
        self.dq = {m: np.sort(p.dockq.to_numpy()) for m, p in model_pools.items()}
        self.N = {m: len(self.dq[m]) for m in self.models}
        self.conf_order = {
            m: core.rank_order(p.selector.to_numpy(), p.dockq.to_numpy())
            for m, p in model_pools.items()
        }
        self.conf = {m: p.selector.to_numpy()[self.conf_order[m]] for m, p in model_pools.items()}
        self.conf_dq = {m: p.dockq.to_numpy()[self.conf_order[m]] for m, p in model_pools.items()}
        self.grid = np.unique(np.concatenate([self.dq[m] for m in self.models]))

    def _log_cdf(self, m: int, n: int, counts: np.ndarray) -> np.ndarray:
        """log P(max of n draws from model m's pool <= v) at each grid point."""
        if n <= 0:
            return np.zeros_like(counts, dtype=float)  # no draws -> max is -inf, CDF = 1
        w = core.topk_weights(self.N[m])  # reuse the cached weight table
        del w
        return _log_ratio(counts, self.N[m], n)

    def union(self, alloc: dict) -> dict:
        active = [m for m in self.models if alloc.get(m, 0) > 0]
        if not active:
            return {"oracle": float("nan"),
                    "solved": {name: float("nan") for name, _ in THRESHOLDS},
                    "delivered": float("nan")}
        g = self.grid
        log_cdf = np.zeros_like(g)
        for m in active:
            c = np.searchsorted(self.dq[m], g, side="right")
            log_cdf = log_cdf + _log_ratio(c, self.N[m], alloc[m])
        cdf = np.exp(log_cdf)
        oracle = g[-1] - float(np.sum(np.diff(g) * cdf[:-1]))
        solved = {}
        for name, cut in THRESHOLDS:
            i = np.searchsorted(g, cut, side="left") - 1
            solved[name] = 1.0 if i < 0 else float(1.0 - cdf[i])

        # delivered: exact E[DockQ of the global max-confidence pick]
        total = 0.0
        for m in active:
            k = alloc[m]
            w = core.topk_weights(self.N[m])[k - 1]
            others = np.zeros(self.N[m])
            for m2 in active:
                if m2 == m:
                    continue
                c = np.searchsorted(self.conf[m2], self.conf[m], side="left")
                others = others + _log_ratio(c, self.N[m2], alloc[m2])
            total += float(np.sum(w * np.exp(others) * self.conf_dq[m]))
        return {"oracle": oracle, "solved": solved, "delivered": total}


def _log_ratio(counts: np.ndarray, N: int, n: int) -> np.ndarray:
    """log[ C(counts, n) / C(N, n) ] = log P(all n draws fall at or below the cut)."""
    from scipy.special import gammaln

    c = np.asarray(counts, dtype=float)
    ok = c >= n
    out = np.full(c.shape, -np.inf)
    cc = np.where(ok, c, n)
    out = np.where(
        ok,
        (gammaln(cc + 1) - gammaln(cc - n + 1)) - (gammaln(N + 1) - gammaln(N - n + 1)),
        -np.inf,
    )
    return out


def allocations(subset: tuple, budget_h: float, cost: dict, target: str) -> dict:
    share = budget_h * 3600.0 / len(subset)
    return {m: int(min(core.TOP_RUNG, share // cost[m][target])) for m in subset}


def run() -> dict:
    cost = fleet_cost()
    targets = [t for t in core.common_targets(core.MODELS)
               if all(t in cost.get(m, {}) for m in core.MODELS)]
    pools = {t: TargetPools({m: core.pools(m)[t] for m in core.MODELS}) for t in targets}

    subsets = [s for r in range(1, 5) for s in combinations(core.MODELS, r)]
    out = {"n_targets": len(targets), "budgets_card_h": BUDGETS_CARD_H,
           "cost_per_sample_s": {m: float(np.median(list(cost[m].values()))) for m in core.MODELS},
           "rung256_card_h": {m: float(sum(cost[m][t] for t in targets) * core.TOP_RUNG / 3600)
                              for m in core.MODELS},
           "strategies": {}}
    for s in subsets:
        name = "+".join(m.replace("-abag", "").replace("-v2", "") for m in s)
        rows = {"subset": list(s), "oracle": [], "delivered": [], "mean_n": [],
                "solved": {n: [] for n, _ in THRESHOLDS}}
        for b in BUDGETS_CARD_H:
            per_t = np.array([
                (lambda u: [u["oracle"], u["delivered"]] +
                           [u["solved"][n] for n, _ in THRESHOLDS])(
                    pools[t].union(allocations(s, b, cost, t)))
                for t in targets
            ])
            bm = core.boot_means(per_t)
            pt = np.nanmean(per_t, axis=0)
            rows["oracle"].append(core.ci_of(bm[:, 0], pt[0]))
            rows["delivered"].append(core.ci_of(bm[:, 1], pt[1]))
            for i, (n, _) in enumerate(THRESHOLDS):
                rows["solved"][n].append(core.ci_of(bm[:, 2 + i], pt[2 + i]))
            rows["mean_n"].append(float(np.mean([
                sum(allocations(s, b, cost, t).values()) for t in targets])))
        out["strategies"][name] = rows
    return out


if __name__ == "__main__":
    r = run()
    print(f"targets={r['n_targets']}  cost s/sample={ {k: round(v,1) for k,v in r['cost_per_sample_s'].items()} }")
    print(f"full N=256 rung card-h: { {k: round(v,1) for k,v in r['rung256_card_h'].items()} }")
    keys = ["boltz2", "opendde", "protenix", "esmfold2", "boltz2+opendde+protenix+esmfold2"]
    for metric in ("oracle", "delivered"):
        print(f"\n-- {metric} mean DockQ vs budget (card-h/target)")
        print("  strategy".ljust(34) + "".join(f"{b:>8}" for b in r["budgets_card_h"]))
        for k in keys:
            v = r["strategies"][k][metric]
            print(f"  {k:<32}" + "".join(f"{d['mean']:>8.3f}" for d in v))
    print("\n-- P(DockQ >= 0.23)")
    print("  strategy".ljust(34) + "".join(f"{b:>8}" for b in r["budgets_card_h"]))
    for k in keys:
        v = r["strategies"][k]["solved"]["acceptable"]
        print(f"  {k:<32}" + "".join(f"{d['mean']:>8.3f}" for d in v))
