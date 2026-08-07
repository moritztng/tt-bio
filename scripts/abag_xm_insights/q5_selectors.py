"""Q5 (arm 1) -- can anything a user already has beat the shipped selector?

This is the arm that needs no structures: every candidate here is built only from the
confidence numbers a model already returns alongside its own samples, so a user could
apply it today, at inference time, with no ground truth.

The candidate list is FIXED IN ADVANCE and every candidate is reported, whether it wins or
loses. No search over combinations feeds the verdict -- with six candidates on four models a
single nominal 95% interval is not multiplicity-corrected, which is exactly why the
pre-declared bar is a majority rule rather than "any interval excludes zero":

    A candidate BEATS the baseline only if its delivered mean DockQ exceeds the shipped
    selector's, with the paired-bootstrap CI on the DIFFERENCE excluding zero, on the same
    target set, for at least 2 of the 4 models.

Prior: a cross-model consensus-confidence pilot on this panel was a null result. Expect
null; report null as a result.

The remaining arm -- within-pool STRUCTURAL agreement (rank samples by closeness to the
pool's modal pose) -- needs the CIF files and is not implemented here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import core

BASELINE = "selector"
EVAL_KS = [16, 64, 256]


def rank_mean(pool: pd.DataFrame, cols: list) -> np.ndarray:
    """Mean of within-pool ranks -- scale-free, so flavours on different scales combine."""
    return np.mean([pool[c].rank().to_numpy() for c in cols], axis=0)


def candidates(model: str) -> dict:
    """Fixed, pre-declared. Keys absent for a model simply do not apply to it."""
    have = core.available(model, tuple(core.FLAVORS))
    out = {}
    for f in ("ptm", "iptm", "complex_plddt"):
        if f in have:
            out[f] = (lambda p, f=f: p[f].to_numpy())
    pairs = [c for c in ("iptm", "complex_plddt") if c in have]
    if len(pairs) == 2:
        out["rank_mean(iptm, plddt)"] = lambda p: rank_mean(p, pairs)
    trio = [c for c in ("ptm", "iptm", "complex_plddt") if c in have]
    if len(trio) == 3:
        out["rank_mean(ptm, iptm, plddt)"] = lambda p: rank_mean(p, trio)
    allf = [c for c in have if c != BASELINE] + [BASELINE]
    if len(allf) >= 3:
        out["rank_mean(all flavours)"] = lambda p: rank_mean(p, allf)
    return out


def analyse(model: str) -> dict:
    pl = core.pools(model)
    targets = sorted(pl)
    gi = [k - 1 for k in EVAL_KS]

    base = np.array([core.selector_curves(pl[t], BASELINE) for t in targets])
    oracle = np.array([core.oracle_curves(pl[t]) for t in targets])
    out = {"model": model, "n_targets": len(targets), "k_grid": EVAL_KS,
           "baseline": core.ci_of(core.boot_means(base)[:, gi], base.mean(0)[gi]),
           "oracle": core.ci_of(core.boot_means(oracle)[:, gi], oracle.mean(0)[gi]),
           "candidates": {}}

    for name, fn in candidates(model).items():
        cur = np.empty_like(base)
        for i, t in enumerate(targets):
            p = pl[t]
            d = p.dockq.to_numpy()
            cur[i] = core.curve(core.rank_order(fn(p), d), d)
        diff = cur - base
        bd = core.boot_means(diff)
        out["candidates"][name] = {
            "delivered": core.ci_of(core.boot_means(cur)[:, gi], cur.mean(0)[gi]),
            "vs_baseline": core.ci_of(bd[:, gi], diff.mean(0)[gi]),
            "beats_at_256": bool(diff.mean(0)[-1] > 0
                                 and not core.crosses_zero(core.ci_of(bd[:, -1], diff.mean(0)[-1]))),
        }
    return out


def run() -> dict:
    per = {m: analyse(m) for m in core.MODELS}
    names = sorted({n for m in per for n in per[m]["candidates"]})
    verdict = {}
    for n in names:
        wins = [m for m in core.MODELS
                if per[m]["candidates"].get(n, {}).get("beats_at_256")]
        verdict[n] = {"models_beating_baseline": wins, "n_models_tested":
                      sum(1 for m in core.MODELS if n in per[m]["candidates"]),
                      "clears_bar": len(wins) >= 2}
    return {
        "pre_declared_bar": "delivered mean DockQ above the shipped selector with a paired "
                            "bootstrap CI on the difference excluding zero, for >= 2 of 4 models",
        "structural_consensus_arm": "not implemented -- needs the CIF pools",
        "per_model": per,
        "verdict": verdict,
        "any_candidate_clears_bar": any(v["clears_bar"] for v in verdict.values()),
    }


if __name__ == "__main__":
    r = run()
    for m in core.MODELS:
        a = r["per_model"][m]
        print(f"\n== {m}  ({a['n_targets']} targets)   baseline@256 "
              f"{a['baseline']['mean'][-1]:.4f}   oracle@256 {a['oracle']['mean'][-1]:.4f}")
        for n, c in a["candidates"].items():
            d = {k: v[-1] for k, v in c["vs_baseline"].items()}
            flag = "BEATS" if c["beats_at_256"] else ("worse" if d["mean"] < 0 else "n.s.")
            print(f"   {n:<28} delivered {c['delivered']['mean'][-1]:.4f}  "
                  f"vs baseline {d['mean']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]  {flag}")
    print("\nverdict:")
    for n, v in r["verdict"].items():
        print(f"   {n:<28} beats on {len(v['models_beating_baseline'])}/{v['n_models_tested']} "
              f"models -> {'CLEARS BAR' if v['clears_bar'] else 'does not clear'}"
              f"  {v['models_beating_baseline']}")
    print("\nANY CANDIDATE CLEARS THE PRE-DECLARED BAR:", r["any_candidate_clears_bar"])
