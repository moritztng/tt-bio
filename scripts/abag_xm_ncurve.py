#!/usr/bin/env python3
"""AbAg-XM N-curve figure (closeout spec 2.10): success vs sampling budget N.

The budget-N protocol: generate N samples, rank by confidence, pick the top-1.
Estimated per fold as the mean over 200 without-replacement subsamples of N
sample-ids (the `rank` column used purely as a sample id; per-fold seed
crc32(target|gen|N)) of the indicator [quantity over the subsample >= thr]:

  ranked  = DockQ of the within-subsample argmax ranking_score (top-1 pick)
  oracle  = max DockQ within the subsample
  random  = DockQ of a uniform pick within the subsample
  ranked_deeprank = DockQ of the within-subsample argmax deeprank_ab

WHY not the prefix estimator (rank < n): `rank` is confidence-ordered in all
492 folds, so a prefix estimator makes ranked@N exactly flat BY CONSTRUCTION
(the top-1 is rank 0 at every n). That artifact is why the session headline
read ranked@5 == ranked@50; the subsample estimator is the honest curve.

N in {1,2,4,8,16,32,50}; 95% cluster-bootstrap CI (resample the 161 scorable
targets, B=1,000, seed 20260729). Self-tests (exit 1 on failure):
  * ranked@1 == random@1 within Monte-Carlo noise (both are the per-sample mean)
  * ranked@50 == plain rank-0 success EXACTLY (the full pool is the N=50 subsample)
  * the estimator-free rank-0 success at 0.23 for opendde-abag reproduces the
    harness-validation number 107/161 = 66.5% (their paper: 66.4%)

    python3 scripts/abag_xm_ncurve.py [--csv ranker_scores.csv] [--out_dir docs]

Outputs docs/abag-xm-ncurve.csv, docs/abag-xm-ncurve.{svg,png} (DockQ>=0.23),
docs/abag-xm-ncurve-hq.{svg,png} (DockQ>=0.8).
"""
import argparse
import sys
import zlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GENS = ("opendde-abag", "protenix-v2", "boltz2")
NS = (1, 2, 4, 8, 16, 32, 50)
THRESHOLDS = (0.23, 0.8)
N_SUB = 200
SEED = 20260729
LINES = ("ranked", "oracle", "random", "ranked_deeprank")
# Their published anchors (OpenDDE paper, 2026ARK-AB): acceptable ranked/oracle
# success 66.4/80.1%; high-quality ranked ~35% at 1 seed rising to ~38% at large budget.
ARK_ACC_RANKED, ARK_ACC_ORACLE = 66.4, 80.1
ARK_HQ_RANKED_LO, ARK_HQ_RANKED_HI = 35.0, 38.0


def per_fold_curves(g, target, gen, n_sub):
    """One fold's success probabilities: dict line -> (len(NS),) array."""
    dockq = g.sort_values("rank").dockq.to_numpy()
    score = g.sort_values("rank").ranking_score.to_numpy()
    dr = g.sort_values("rank").deeprank_ab.to_numpy()
    out = {(thr, ln): np.zeros(len(NS)) for thr in THRESHOLDS for ln in LINES}
    for ni, N in enumerate(NS):
        rs = np.random.RandomState(
            zlib.crc32(f"{target}|{gen}|{N}".encode()) & 0x7fffffff)
        perms = np.array([rs.permutation(50) for _ in range(n_sub)])[:, :N]
        d_sub = dockq[perms]
        picks = {
            "ranked": d_sub[np.arange(n_sub), score[perms].argmax(axis=1)],
            "oracle": d_sub.max(axis=1),
            "random": d_sub[np.arange(n_sub), rs.randint(0, N, n_sub)],
            "ranked_deeprank": d_sub[np.arange(n_sub), dr[perms].argmax(axis=1)],
        }
        for thr in THRESHOLDS:
            for ln in LINES:
                out[(thr, ln)][ni] = (picks[ln] >= thr).mean()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(Path.home() / ".coworker" / "state"
                                         / "abag-xm-closeout" / "ranker_scores.csv"))
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out_dir", default=str(ROOT / "docs"))
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    scorable = df.groupby("target")["dockq"].apply(lambda s: s.notna().any())
    targets = sorted(scorable[scorable].index)
    df = df[df.target.isin(targets)]
    assert len(targets) == 161 and len(df) == 161 * 3 * 50

    # Per-fold curves -> per-generator arrays (161, len(NS)) per (thr, line).
    curves = {(gen, thr, ln): np.zeros((len(targets), len(NS)))
              for gen in GENS for thr in THRESHOLDS for ln in LINES}
    rank0_hit = {(gen, thr): np.zeros(len(targets)) for gen in GENS for thr in THRESHOLDS}
    for gen in GENS:
        sub = df[df.gen == gen]
        for i, (t, g) in enumerate(sub.groupby("target")):
            fc = per_fold_curves(g, t, gen, N_SUB)
            for thr in THRESHOLDS:
                for ln in LINES:
                    curves[(gen, thr, ln)][i] = fc[(thr, ln)]
                g50 = g.sort_values("rank")
                rank0_hit[(gen, thr)][i] = float(g50.dockq.iloc[0] >= thr)

    # ---- self-tests -------------------------------------------------------------
    fails = []
    for gen in GENS:
        for thr in THRESHOLDS:
            r1 = curves[(gen, thr, "ranked")][:, 0].mean()
            q1 = curves[(gen, thr, "random")][:, 0].mean()
            # Monte-Carlo noise on 200 draws of a p~0.1-0.7 indicator: se ~ 0.03 max
            if abs(r1 - q1) > 0.035:
                fails.append(f"{gen} thr={thr}: ranked@1 {r1:.4f} != random@1 {q1:.4f}")
            r50 = curves[(gen, thr, "ranked")][:, -1]
            if not np.allclose(r50, rank0_hit[(gen, thr)], atol=0.0):
                fails.append(f"{gen} thr={thr}: ranked@50 != plain rank-0 success exactly")
    op = rank0_hit[("opendde-abag", 0.23)].sum()
    if op != 107:
        fails.append(f"opendde rank-0 acceptable success {int(op)}/161 != 107/161 "
                     f"(the 66.5% harness-validation number)")
    if fails:
        for f in fails:
            print("FAIL:", f)
        sys.exit(1)
    print(f"self-tests pass: ranked@1 == random@1 (MC noise), ranked@50 == rank-0 "
          f"exactly, opendde rank-0 acceptable = {int(op)}/161 = {op/161:.1%} "
          f"(paper: 66.4%)")

    # ---- cluster bootstrap CIs (resample targets) --------------------------------
    B = a.boot
    idx = np.random.RandomState(SEED).randint(0, len(targets), size=(B, len(targets)))
    rows, bands = [], {}
    for gen in GENS:
        for thr in THRESHOLDS:
            for ln in LINES:
                arr = curves[(gen, thr, ln)]
                mean = arr.mean(axis=0)
                boot = arr[idx].mean(axis=1)                    # (B, len(NS))
                lo = np.percentile(boot, 2.5, axis=0)
                hi = np.percentile(boot, 97.5, axis=0)
                bands[(gen, thr, ln)] = (mean, lo, hi)
                for ni, N in enumerate(NS):
                    rows.append(dict(generator=gen, threshold=thr, N=N, line=ln,
                                     success=mean[ni], lo=lo[ni], hi=hi[ni]))
    rep = pd.DataFrame(rows)
    out_csv = Path(a.out_dir) / "abag-xm-ncurve.csv"
    rep.to_csv(out_csv, index=False)

    # ---- figures ------------------------------------------------------------------
    style = {"ranked": ("C0", "-", "ranked (native ranking_score)"),
             "oracle": ("C2", "-", "oracle"),
             "random": ("C7", "--", "random"),
             "ranked_deeprank": ("C1", "-", "ranked (DeepRank-Ab)")}
    for thr, name, title in ((0.23, "abag-xm-ncurve", "CAPRI-acceptable (DockQ >= 0.23)"),
                             (0.8, "abag-xm-ncurve-hq", "high-quality (DockQ >= 0.8)")):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
        for ax, gen in zip(axes, GENS):
            for ln in LINES:
                mean, lo, hi = bands[(gen, thr, ln)]
                c, ls, label = style[ln]
                ax.plot(NS, 100 * mean, ls, color=c, label=label,
                        marker="o", ms=3)
                ax.fill_between(NS, 100 * lo, 100 * hi, color=c, alpha=0.15, lw=0)
            if gen == "opendde-abag":
                if thr == 0.23:
                    ax.axhline(ARK_ACC_RANKED, color="0.4", ls=":", lw=1)
                    ax.axhline(ARK_ACC_ORACLE, color="0.7", ls=":", lw=1)
                    ax.text(1.1, ARK_ACC_RANKED + 1, "OpenDDE paper ranked 66.4%",
                            fontsize=7, color="0.4")
                    ax.text(1.1, ARK_ACC_ORACLE + 1, "OpenDDE paper oracle 80.1%",
                            fontsize=7, color="0.7")
                else:
                    ax.axhspan(ARK_HQ_RANKED_LO, ARK_HQ_RANKED_HI, color="0.4",
                               alpha=0.15, lw=0)
                    ax.text(1.1, ARK_HQ_RANKED_HI + 1,
                            "OpenDDE paper ranked ~35-38% (Fig. 5)", fontsize=7,
                            color="0.4")
            ax.set_xscale("log", base=2)
            ax.set_xticks(list(NS))
            ax.set_xticklabels([str(n) for n in NS])
            ax.set_title(gen, fontsize=10)
            ax.set_xlabel("sampling budget N (top-1 pick from N samples)")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("success over 161 scorable targets (%)")
        axes[0].legend(fontsize=7, loc="lower right")
        fig.suptitle(f"AbAg-XM test-time scaling, {title} -- subsample estimator, "
                     f"95% target-bootstrap bands", fontsize=11)
        fig.tight_layout()
        for ext in ("svg", "png"):
            fig.savefig(Path(a.out_dir) / f"{name}.{ext}", dpi=150)
        plt.close(fig)
    print(f"wrote {out_csv} and docs/abag-xm-ncurve{{,-hq}}.{{svg,png}}")

    # Console table for the writeup (ranked/oracle/random per gen, 0.23).
    for thr in THRESHOLDS:
        print(f"\n== DockQ >= {thr} ==")
        for gen in GENS:
            line = "  ".join(
                f"{ln}@1 {bands[(gen, thr, ln)][0][0]:.1%} -> "
                f"@50 {bands[(gen, thr, ln)][0][-1]:.1%}"
                for ln in ("ranked", "oracle", "random"))
            print(f"  {gen:13s} {line}")


if __name__ == "__main__":
    main()
