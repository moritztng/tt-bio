"""Q7 -- the antibody-specific angle.

Two questions an antibody engineer actually asks:

  1. Does deep sampling buy CDR-H3 loop accuracy the way it buys interface accuracy?
  2. Is the interface-best sample also the loop-best sample -- can one sample give you both?

Measured as: the oracle (best-of-k) curve for cdr_h3_rmsd alongside the DockQ one, the
within-target rank correlation between DockQ and -cdr_h3_rmsd, and the penalty you pay in
H3 RMSD by taking the DockQ-oracle sample instead of the H3-oracle sample.

CDR-H3 labels share the chunk-aligned coverage gap of the epitope labels, so protenix-v2
and esmfold2 run at their honest reduced depth.
"""

from __future__ import annotations

import numpy as np

import core

KS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def h3_pools(model: str) -> dict:
    out = {}
    for t, p in core.pools(model).items():
        q = p[p.cdr_h3_rmsd.notna() & p.dockq.notna()].sort_values(["chunk", "rank"])
        if len(q) >= 16:
            out[t] = q.reset_index(drop=True)
    return out


def analyse(model: str) -> dict:
    pl = h3_pools(model)
    depths = np.array([len(p) for p in pl.values()])
    D = next((d for d in (256, 192, 128, 64) if (depths >= d).sum() >= 100), 64)
    targets = [t for t in sorted(pl) if len(pl[t]) >= D]
    ks = [k for k in KS if k <= D]
    gi = [k - 1 for k in ks]

    h3_best, dq_best, rho, penalty = [], [], [], []
    for t in targets:
        p = pl[t].head(D)
        d = p.dockq.to_numpy()
        h = p.cdr_h3_rmsd.to_numpy()
        # best-of-k on H3 means MINIMUM rmsd -> run the engine on -h and flip back.
        h3_best.append(-core.curve(np.argsort(-h, kind="stable"), -h))
        dq_best.append(core.curve(np.argsort(d, kind="stable"), d))
        rho.append(core.spearman(d, -h))
        penalty.append(h[int(np.argmax(d))] - h.min())

    h3_best = np.array(h3_best)
    dq_best = np.array(dq_best)
    rho = np.array(rho)
    penalty = np.array(penalty)
    return {
        "model": model,
        "depth": D,
        "n_targets": len(targets),
        "k_grid": ks,
        "h3_oracle_rmsd": core.ci_of(core.boot_means(h3_best)[:, gi], h3_best.mean(0)[gi]),
        "dockq_oracle": core.ci_of(core.boot_means(dq_best)[:, gi], dq_best.mean(0)[gi]),
        "within_target_rho_dockq_vs_h3": {
            "median": float(np.nanmedian(rho)),
            "mean": core.paired_bootstrap(np.nan_to_num(rho, nan=0.0)),
            "frac_above_0_5": float(np.mean(rho[np.isfinite(rho)] > 0.5)),
        },
        "h3_penalty_of_dockq_pick": {
            "median_angstrom": float(np.median(penalty)),
            "mean": core.paired_bootstrap(penalty),
        },
    }


def run() -> dict:
    return {m: analyse(m) for m in core.MODELS}


if __name__ == "__main__":
    r = run()
    for m, a in r.items():
        print(f"\n== {m}  depth={a['depth']} on {a['n_targets']} targets")
        print(f"  k              {a['k_grid']}")
        print(f"  H3 oracle RMSD {[round(x, 2) for x in a['h3_oracle_rmsd']['mean']]}")
        print(f"  DockQ oracle   {[round(x, 3) for x in a['dockq_oracle']['mean']]}")
        w = a["within_target_rho_dockq_vs_h3"]
        print(f"  rho(DockQ, -H3) median {w['median']:+.3f}  frac>0.5 {w['frac_above_0_5']:.3f}")
        p = a["h3_penalty_of_dockq_pick"]
        print(f"  H3 penalty of taking the DockQ-best sample: median {p['median_angstrom']:.2f} A,"
              f" mean {core.fmt(p['mean'], 2)}")
