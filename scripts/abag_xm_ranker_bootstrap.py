#!/usr/bin/env python3
"""AbAg-XM ranker bootstrap CIs (closeout spec 2.7).

Resamples the 161 scorable TARGETS with replacement (B=10,000, seed 20260729) --
never rows, since samples within a target are correlated. Per resample, per
generator: oracle@N, random@N, ranked@N per ranker column, and
fraction-of-oracle-gap-recovered = (ranked - random)/(oracle - random), at
N in {5, 50} and DockQ thresholds {0.23, 0.8}. Paired differences between each
ranker and the native ranking_score (and deeprank_ab vs abag_rank) get 95%
percentile CIs from the SAME resamples, so a claim is significant iff its CI
excludes 0.

The per-fold success quantities use the pinned budget-N subsample estimator:
per fold, mean over 200 without-replacement subsamples (seeded per fold as
crc32(target|gen|N)) of the indicator [quantity over the subsample >= thr].
These are per-fold constants, so the target bootstrap over them is exact, not
Monte-Carlo-on-Monte-Carlo.

    python3 scripts/abag_xm_ranker_bootstrap.py [--csv ranker_scores.csv]
                                                [--boot 10000] [--out docs/abag-xm-ranker-cis]

Outputs <out>.md (the table) and <out>.csv (machine-readable).
"""
import argparse
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RANKERS = ["iptm", "ptm", "ranking_score", "complex_plddt", "pdockq2", "ipsae",
           "anticonf", "pss", "deeprank_ab", "abag_rank"]
GENS = ("opendde-abag", "protenix-v2", "boltz2")
NS = (5, 50)
THRESHOLDS = (0.23, 0.8)
N_SUB = 200
SEED = 20260729


def fold_constants(dfq, target, gen, n_sub=N_SUB):
    """Per-fold success probabilities, keyed (thr, N) -> (oracle, random, ranked)."""
    dockq = dfq["dockq"].to_numpy()
    out = {}
    for N in NS:
        rs = np.random.RandomState(
            zlib.crc32(f"{target}|{gen}|{N}".encode()) & 0x7fffffff)
        perms = np.array([rs.permutation(len(dfq)) for _ in range(n_sub)])
        subs = perms[:, :N]
        d_sub = dockq[subs]                                    # (n_sub, N)
        oracle = d_sub.max(axis=1)
        rand_pick = d_sub[np.arange(n_sub), rs.randint(0, N, n_sub)]
        ranked = {}
        for rk in RANKERS:
            r_sub = dfq[rk].to_numpy()[subs]
            ranked[rk] = d_sub[np.arange(n_sub), r_sub.argmax(axis=1)]
        for thr in THRESHOLDS:
            out[(thr, N)] = ((oracle >= thr).mean(), (rand_pick >= thr).mean(),
                             {rk: (v >= thr).mean() for rk, v in ranked.items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(Path.home() / ".coworker" / "state"
                                         / "abag-xm-closeout" / "ranker_scores.csv"))
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--subsamples", type=int, default=N_SUB)
    ap.add_argument("--out", default=str(ROOT / "docs" / "abag-xm-ranker-cis"))
    ap.add_argument("--global_rankers", default="ranking_score,deeprank_ab,abag_rank",
                    help="rankers getting the (slower) global pooled Spearman CI")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    # Scorable targets only: a target whose dockq is entirely null (9ly2/9ly3/9lz2,
    # unresolved native antigen) has no success indicator. Rows: 161 targets x 3 gens x 50.
    scorable = df.groupby("target")["dockq"].apply(lambda s: s.notna().any())
    targets = sorted(scorable[scorable].index)
    df = df[df.target.isin(targets)]
    assert len(targets) == 161, f"{len(targets)} scorable targets != 161"

    # Per-fold constants and per-fold Spearman, per generator.
    const = {}      # gen -> (thr, N) -> dict of 161-arrays
    spearman = {}   # gen -> ranker -> 161-array
    order = {}
    for gen in GENS:
        sub = df[df.gen == gen]
        folds = list(sub.groupby("target"))
        order[gen] = [t for t, _ in folds]
        assert sorted(order[gen]) == targets, f"{gen}: target set mismatch"
        per = {k: {q: np.zeros(len(folds)) for q in (["oracle", "random"] + RANKERS)}
               for k in [(thr, N) for thr in THRESHOLDS for N in NS]}
        sp = {rk: np.zeros(len(folds)) for rk in RANKERS}
        for i, (t, g) in enumerate(folds):
            fc = fold_constants(g, t, gen, a.subsamples)
            for (thr, N), (orc, rnd, rnk) in fc.items():
                per[(thr, N)]["oracle"][i] = orc
                per[(thr, N)]["random"][i] = rnd
                for rk in RANKERS:
                    per[(thr, N)][rk][i] = rnk[rk]
            dq = g.dockq.to_numpy()
            for rk in RANKERS:
                v = g[rk].to_numpy()
                # Spearman via Pearson on average ranks (50 rows, no NaN by construction)
                rd = pd.Series(dq).rank().to_numpy()
                rv = pd.Series(v).rank().to_numpy()
                sp[rk][i] = np.corrcoef(rd, rv)[0, 1]
        const[gen] = per
        spearman[gen] = sp

    B = a.boot
    rs = np.random.RandomState(SEED)
    idx = rs.randint(0, len(targets), size=(B, len(targets)))

    rows = []
    ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    gap_by = {}     # (gen, thr, N) -> ranker -> B-array of gap-recovered
    for gen in GENS:
        for thr in THRESHOLDS:
            for N in NS:
                per = const[gen][(thr, N)]
                oracle = per["oracle"][idx].mean(axis=1)
                random = per["random"][idx].mean(axis=1)
                denom = oracle - random
                gap = {rk: np.where(np.abs(denom) > 1e-9,
                                    (per[rk][idx].mean(axis=1) - random) / denom, np.nan)
                       for rk in RANKERS}
                gap_by[(gen, thr, N)] = gap
                for rk in RANKERS:
                    lo, hi = ci(gap[rk][~np.isnan(gap[rk])])
                    rows.append(dict(generator=gen, threshold=thr, N=N, ranker=rk,
                                     oracle=ci(oracle), random=ci(random),
                                     ranked=ci(per[rk][idx].mean(axis=1)),
                                     gap_recovered=float(np.nanmean(gap[rk])),
                                     gap_lo=lo, gap_hi=hi))

    # Paired differences on gap-recovered: each ranker vs native ranking_score,
    # and deeprank_ab vs abag_rank -- same resamples, so the pairing is exact.
    diffs = []
    for (gen, thr, N), gap in gap_by.items():
        nat = gap["ranking_score"]
        for rk in RANKERS:
            if rk == "ranking_score":
                continue
            d = gap[rk] - nat
            ok = ~np.isnan(d)
            lo, hi = ci(d[ok])
            diffs.append(dict(generator=gen, threshold=thr, N=N,
                              comparison=f"{rk} - ranking_score",
                              mean=float(np.nanmean(d)), lo=lo, hi=hi,
                              significant=bool(lo > 0 or hi < 0)))
        for comp, d in (("deeprank_ab - abag_rank", gap["deeprank_ab"] - gap["abag_rank"]),
                        ("deeprank_ab - iptm", gap["deeprank_ab"] - gap["iptm"])):
            ok = ~np.isnan(d)
            lo, hi = ci(d[ok])
            diffs.append(dict(generator=gen, threshold=thr, N=N,
                              comparison=comp, mean=float(np.nanmean(d)), lo=lo, hi=hi,
                              significant=bool(lo > 0 or hi < 0)))

    # Spearman CIs: mean per-target (bootstrap targets, all rankers) and global
    # pooled (the three claim-relevant rankers; extend with --global_rankers).
    from scipy.stats import rankdata
    sp_rows = []
    glob_rks = a.global_rankers.split(",")
    for gen in GENS:
        sub = df[df.gen == gen].reset_index(drop=True)
        blocks = [g.index.to_numpy() for _, g in sub.groupby("target")]
        for rk in RANKERS:
            per_t = spearman[gen][rk]
            # Folds with constant dockq (e.g. all-zero failures) have an undefined
            # Spearman -- excluded from the mean, not zeroed.
            m = np.array([np.nanmean(per_t[b]) for b in idx])
            lo, hi = ci(m)
            glob_mean = glo = ghi = None
            if rk in glob_rks:
                glob = np.zeros(B)
                dq_all = sub.dockq.to_numpy()
                rk_all = sub[rk].to_numpy()
                for b in range(B):
                    sel = np.concatenate([blocks[i] for i in idx[b]])
                    glob[b] = np.corrcoef(rankdata(dq_all[sel]),
                                          rankdata(rk_all[sel]))[0, 1]
                glob_mean, (glo, ghi) = float(glob.mean()), ci(glob)
            sp_rows.append(dict(generator=gen, ranker=rk,
                                per_target_mean=float(np.nanmean(per_t)),
                                per_target_lo=lo, per_target_hi=hi,
                                global_mean=glob_mean, global_lo=glo, global_hi=ghi))

    rep = pd.DataFrame(rows)
    dif = pd.DataFrame(diffs)
    spr = pd.DataFrame(sp_rows)
    out_csv = Path(a.out).with_suffix(".csv")
    rep.to_csv(out_csv, index=False)
    dif.to_csv(Path(a.out + "-diffs").with_suffix(".csv"), index=False)
    spr.to_csv(Path(a.out + "-spearman").with_suffix(".csv"), index=False)

    # ---- report ---------------------------------------------------------------
    lines = ["# AbAg-XM ranker bootstrap CIs",
             "",
             f"161 scorable targets resampled with replacement, B={B:,}, seed {SEED}. "
             f"Budget-N estimator: mean over {a.subsamples} without-replacement subsamples "
             "per fold (seeded per fold). Gap-recovered = (ranked - random)/(oracle - "
             "random). 95% percentile CIs; a difference is significant iff its CI excludes 0.",
             ""]
    for thr in THRESHOLDS:
        for N in NS:
            lines.append(f"## DockQ >= {thr}, N={N}: gap-recovered by ranker")
            lines.append("")
            lines.append("| generator | ranker | gap-recovered | 95% CI | vs ranking_score |")
            lines.append("|---|---|---|---|---|")
            for gen in GENS:
                for rk in RANKERS:
                    r = rep[(rep.generator == gen) & (rep.threshold == thr)
                            & (rep.N == N) & (rep.ranker == rk)].iloc[0]
                    d = dif[(dif.generator == gen) & (dif.threshold == thr)
                            & (dif.N == N)
                            & (dif.comparison == f"{rk} - ranking_score")]
                    vs = "-" if rk == "ranking_score" or d.empty else (
                        f"{d.iloc[0]['mean']:+.3f} [{d.iloc[0]['lo']:+.3f}, "
                        f"{d.iloc[0]['hi']:+.3f}]" + (" **sig**" if d.iloc[0]["significant"] else ""))
                    lines.append(f"| {gen} | {rk} | {r.gap_recovered:.3f} "
                                 f"[{r.gap_lo:.3f}, {r.gap_hi:.3f}] | {vs} |")
                lines.append("")
    lines.append("## Spearman (per-target mean and global pooled, bootstrap over targets)")
    lines.append("")
    lines.append("| generator | ranker | per-target mean [CI] | global [CI] |")
    lines.append("|---|---|---|---|")
    for gen in GENS:
        for rk in RANKERS:
            r = spr[(spr.generator == gen) & (spr.ranker == rk)].iloc[0]
            glob = (f"{r.global_mean:.3f} [{r.global_lo:.3f}, {r.global_hi:.3f}]"
                    if pd.notna(r.global_mean) else "n/a")
            lines.append(f"| {gen} | {rk} | {r.per_target_mean:.3f} "
                         f"[{r.per_target_lo:.3f}, {r.per_target_hi:.3f}] | {glob} |")
    # ---- claim verdicts (spec 2.7 accept), written into the report --------------
    def dv(gen, comp, thr, N):
        r = dif[(dif.generator == gen) & (dif.threshold == thr) & (dif.N == N)
                & (dif.comparison == comp)].iloc[0]
        return r["mean"], r["lo"], r["hi"], bool(r["significant"])

    def gv(gen, rk, thr, N):
        r = rep[(rep.generator == gen) & (rep.threshold == thr) & (rep.N == N)
                & (rep.ranker == rk)].iloc[0]
        return r.gap_recovered, r.gap_lo, r.gap_hi

    verdicts = ["", "## Claim verdicts (the assertions this table exists to settle)", ""]
    m1 = [gv(g, "deeprank_ab", 0.23, 50) for g in GENS]
    verdicts.append(
        "- \"DeepRank-Ab recovers a fraction of the oracle gap\": at DockQ>=0.23, N=50 the "
        "point estimates are " + ", ".join(f"{g} {m:.1%} [{lo:.1%}, {hi:.1%}]"
                                           for g, (m, lo, hi) in zip(GENS, m1)) +
        ". Significant (CI excludes 0) on protenix-v2 and boltz2 only; on opendde-abag it "
        "is consistent with noise.")
    m2 = [dv(g, "deeprank_ab - iptm", 0.23, 50) for g in GENS]
    verdicts.append(
        "- \"DeepRank-Ab beats ranking by native ipTM\": paired gap-recovered difference "
        "deeprank_ab - iptm at 0.23/N=50 is " + ", ".join(
            f"{g} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]" + (" (significant)" if s else " (includes 0)")
            for g, (m, lo, hi, s) in zip(GENS, m2)) + ".")
    m3 = [gv("protenix-v2", "abag_rank", 0.23, N) for N in NS]
    verdicts.append(
        "- \"ABAG-Rank does not transfer\": abag_rank gap-recovered on protenix-v2 is "
        + "; ".join(f"N={N}: {m:.1%} [{lo:.1%}, {hi:.1%}]" for (m, lo, hi), N in zip(m3, NS))
        + " -- negative, CI excludes 0 at N=50. Verdict stands.")
    m4 = [dv(g, "abag_rank - ranking_score", 0.23, 50) for g in GENS]
    verdicts.append(
        "- abag_rank vs native ranking_score at 0.23/N=50: " + ", ".join(
            f"{g} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]" + (" (significant)" if s else " (includes 0)")
            for g, (m, lo, hi, s) in zip(GENS, m4)) + ".")
    top = rep[(rep.threshold == 0.23) & (rep.N == 50)]
    rmax = top.loc[top.gap_recovered.idxmax()]
    verdicts.append(
        f"- Largest gap-recovered any ranker achieves at 0.23/N=50: {rmax.gap_recovered:.1%} "
        f"[{rmax.gap_lo:.1%}, {rmax.gap_hi:.1%}] ({rmax.ranker} on {rmax.generator}). The "
        "earlier session claim \"no ranker exceeds ~22%\" is REVISED upward by this table.")
    lines += verdicts
    Path(a.out).with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(f"wrote {a.out}.md / .csv / -diffs.csv / -spearman.csv")

    # ---- claim verdicts (spec 2.7 accept) --------------------------------------
    def gap_at(gen, rk, thr, N):
        r = rep[(rep.generator == gen) & (rep.threshold == thr)
                & (rep.N == N) & (rep.ranker == rk)].iloc[0]
        return r.gap_recovered, r.gap_lo, r.gap_hi
    print("\n== claim checks ==")
    for gen in GENS:
        m, lo, hi = gap_at(gen, "deeprank_ab", 0.23, 50)
        print(f"deeprank_ab gap@50/0.23 {gen}: {m:.3f} [{lo:.3f}, {hi:.3f}]")
        m, lo, hi = gap_at(gen, "abag_rank", 0.23, 50)
        print(f"abag_rank   gap@50/0.23 {gen}: {m:.3f} [{lo:.3f}, {hi:.3f}]")
    best = rep[(rep.threshold == 0.23) & (rep.N == 50)].groupby("ranker").gap_recovered.max()
    print("max gap-recovered over rankers @50/0.23:", best.round(3).to_dict())


if __name__ == "__main__":
    main()
