#!/usr/bin/env python3
"""AbAg-XM structural spread (closeout spec 2.9): does a failing ensemble
sample several wrong basins, or collapse onto one wrong pose?

Zero new structure compute: reuses the per-fold pairwise DockQ/TM matrix and
PSS already in the labels JSONs. Per fold: mean pairwise TM (convergence),
sd of pairwise TM, PSS. Failing = oracle@50 (max sample DockQ) < 0.23.
Per generator: failing vs succeeding distributions of mean pairwise TM,
Mann-Whitney + Cliff's delta, and a failure-class split:

  failing + LOW mean pairwise TM  -> samples disagree structurally: several
                                     wrong basins (trunk-limited; more seeds
                                     MIGHT help)
  failing + HIGH mean pairwise TM -> collapsed onto one wrong pose
                                     (converged-wrong; seeds won't help,
                                     trunk diversity would)

DockQ sd is compressed near zero and cannot make this distinction (measured
0.0129/0.0305 failing/succeeding) -- the TM column is the point.

    python3 scripts/abag_xm_structural_spread.py --labels_dir ~/abag_xm/tier_a/labels \
        --out_dir docs

Outputs docs/abag-xm-structural-spread.{md,csv} (per-fold stats + summary).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

GEN_PREFIX = {"opendde_abag": "opendde-abag", "protenix_v2": "protenix-v2",
              "boltz2": "boltz2"}


def _cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_dir", default=str(Path.home() / "abag_xm/tier_a/labels"))
    ap.add_argument("--out_dir", default="docs")
    a = ap.parse_args()

    rows = []
    for p in sorted(Path(a.labels_dir).glob("*.json")):
        gen = target = None
        for prefix, g in GEN_PREFIX.items():
            if p.stem.startswith(prefix + "_"):
                gen, target = g, p.stem[len(prefix) + 1:]
                break
        if gen is None:
            continue
        d = json.loads(p.read_text())
        pm = d.get("pairwise_matrix") or {}
        mat = pm.get("matrix") or []
        tm = np.array([e["tm"] for e in mat if e.get("tm") is not None])
        dockq = np.array([(s.get("dockq") or {}).get("dockq", np.nan)
                          for s in d["samples"]], dtype=float)
        oracle = np.nanmax(dockq) if (~np.isnan(dockq)).any() else np.nan
        rows.append({"target": d["target"], "gen": gen,
                     "mean_tm": float(tm.mean()) if len(tm) else np.nan,
                     "sd_tm": float(tm.std()) if len(tm) else np.nan,
                     "pss": pm.get("PSS"),
                     "oracle50_dockq": oracle,
                     "failing": bool(oracle < 0.23) if not np.isnan(oracle) else None})
    df = pd.DataFrame(rows)
    out_csv = Path(a.out_dir) / "abag-xm-structural-spread.csv"
    df.to_csv(out_csv, index=False)

    from scipy.stats import mannwhitneyu
    md = ["# AbAg-XM structural spread (spec 2.9)", "",
          "Per-fold structural convergence of the 50-sample ensemble (mean pairwise "
          "TM over all 1225 sample pairs; sd in parens) split by success. Failing = "
          "no sample at DockQ >= 0.23. A LOW mean pairwise TM means the samples "
          "disagree structurally (several basins); HIGH means they converged on one "
          "pose. The 3 unscorable targets (9ly2/9ly3/9lz2) are excluded.", ""]
    summary = []
    md.append("| generator | n fail | n succ | median mean-TM fail (sd) | "
              "median mean-TM succ (sd) | MWU p | Cliff's d |")
    md.append("|---|---|---|---|---|---|---|")
    for gen in ("opendde-abag", "protenix-v2", "boltz2"):
        g = df[df.gen == gen].dropna(subset=["mean_tm"])
        f, s = g[g.failing == True], g[g.failing == False]
        if not len(f) or not len(s):
            continue
        u = mannwhitneyu(f.mean_tm, s.mean_tm, alternative="two-sided")
        cd = _cliffs_delta(f.mean_tm, s.mean_tm)
        med = g.mean_tm.median()
        multi = f[f.mean_tm < med]
        conv = f[f.mean_tm >= med]
        summary.append({"gen": gen, "n_fail": len(f), "n_succ": len(s),
                        "med_tm_fail": f.mean_tm.median(),
                        "med_tm_succ": s.mean_tm.median(),
                        "mwu_p": u.pvalue, "cliffs_delta": cd,
                        "fail_multibasin": len(multi), "fail_converged": len(conv)})
        md.append(f"| {gen} | {len(f)} | {len(s)} | "
                  f"{f.mean_tm.median():.3f} ({f.sd_tm.median():.3f}) | "
                  f"{s.mean_tm.median():.3f} ({s.sd_tm.median():.3f}) | "
                  f"{u.pvalue:.2e} | {cd:+.2f} |")
    md.append("")
    md.append("## Failure class per generator (split at the generator median "
              "mean-TM)")
    md.append("")
    md.append("| generator | multi-basin wrong (seeds might help) | "
              "converged-wrong (seeds won't help) | dominant class |")
    md.append("|---|---|---|---|")
    for r in summary:
        dom = ("multi-basin" if r["fail_multibasin"] > r["fail_converged"]
               else "converged-wrong")
        md.append(f"| {r['gen']} | {r['fail_multibasin']} | "
                  f"{r['fail_converged']} | {dom} |")
    md.append("")
    for r in summary:
        verdict = ("failing ensembles mostly collapse onto one wrong pose"
                   if r["fail_converged"] >= r["fail_multibasin"]
                   else "failing ensembles mostly sample several wrong basins")
        md.append(f"- {r['gen']}: {verdict} "
                  f"({r['fail_converged']}/{r['n_fail']} converged-wrong, "
                  f"{r['fail_multibasin']}/{r['n_fail']} multi-basin).")
    md.append("")
    out_md = Path(a.out_dir) / "abag-xm-structural-spread.md"
    out_md.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"wrote {out_md} and {out_csv}")


if __name__ == "__main__":
    main()
