"""PHASE 0 cross-hardware sanity check: WH-Galaxy p2 (N=16, seed 42) vs BH-QB tier_a (N=50, seed 42).

Pre-registered design (state/abag-xm-deepn-saturation-fullpanel.md, 2026-08-02):
structures on both sides are stored by confidence rank, not draw order, and samples within a fold
are i.i.d., so galaxy-16 is tested as "one plausible 16-draw from the BH distribution": each
statistic of the galaxy 16 is compared against the bootstrap distribution of the same statistic
over 16-subsets of tier_a's 50 (joint (confidence, dockq) pairs). The within-hardware floor is the
identical test with the 16-draw taken from a disjoint subset of tier_a itself.

Verdict rule (per model x statistic): tail_cross (fraction of targets two-sided extreme at 0.05)
must be <= 3x tail_within, and the cross-hardware median |z| <= 2x the within-hardware median |z|.
A systematic one-direction bias fraction > 0.75 of extremes is reported separately.

Run with the tt-bio env python on qb1 (pandas + numpy). CPU-only.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

MODELS = {  # campaign dir name -> (galaxy parquet suffix, galaxy model name)
    "opendde_abag": ("", "opendde-abag"),
    "protenix_v2": ("_protenix", "protenix-v2"),
    "boltz2": ("_boltz2", "boltz2"),
    "esmfold2": ("_esmfold2", "esmfold2"),
}
THR = (0.23, 0.49, 0.80)
B_NULL = 2000   # 16-subset null draws per target
B_WITHIN = 200  # within-hardware floor draws per target (inner null b=100)
ALPHA = 0.05


def load_galaxy(p0: pathlib.Path, suffix: str, model: str) -> pd.DataFrame:
    df = pd.read_parquet(p0 / f"abag_xm_scaling{suffix}_samples.parquet")
    df = df[(df.status == "ok") & (df.n_samples == 16)]
    out = {}
    for t, g in df.groupby("target"):
        g = g.sort_values("rank")
        out[t] = {
            "conf": g.confidence_score.to_numpy(float),
            "dockq": g.global_dockq.to_numpy(float),  # may be all-NaN (no GT)
        }
    return out


def load_tiera(p0: pathlib.Path, tier_a: pathlib.Path, md: str) -> dict:
    dq = {}
    tsv = p0 / f"tiera_{md}_dockq.tsv"
    if tsv.exists():
        for line in tsv.read_text().splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) < 3 or parts[2].startswith("ERR"):
                continue
            t, r, v = parts[0], int(parts[1]), float(parts[2])
            dq.setdefault(t, {})[r] = v
    out = {}
    for rj in sorted(tier_a.glob(f"{md}/*_results_*/results.json")):
        t = rj.parent.name.split("_results_")[-1]
        try:
            runs = json.loads(rj.read_text())[0].get("all_runs", [])
        except Exception:
            continue
        conf = {int(r["rank"]): float(r["confidence_score"]) for r in runs}
        d = dq.get(t, {})
        ranks = sorted(set(conf) & set(d))
        if t not in out:
            out[t] = {"conf_full": conf, "dockq_full": d}
    return out


def stats_of(conf: np.ndarray, dockq: np.ndarray | None) -> dict[str, float]:
    s = {"conf_mean": float(np.mean(conf)), "conf_std": float(np.std(conf))}
    if dockq is not None and np.isfinite(dockq).all():
        order = np.argsort(-conf)
        s.update({
            "dq_mean": float(np.mean(dockq)),
            "dq_q90": float(np.quantile(dockq, 0.9)),
            "dq_max": float(np.max(dockq)),
            "dq_f23": float(np.mean(dockq >= THR[0])),
            "dq_f49": float(np.mean(dockq >= THR[1])),
            "dq_f80": float(np.mean(dockq >= THR[2])),
            "user": float(dockq[order[0]]),
            "oracle": float(np.max(dockq)),
        })
        if len(dockq) >= 8 and np.std(conf) > 0 and np.std(dockq) > 0:
            from scipy.stats import spearmanr  # noqa
            s["spearman"] = float(spearmanr(conf, dockq).statistic)
    return s


def subset_null(conf: np.ndarray, dockq: np.ndarray | None, rng, b=B_NULL):
    n = len(conf)
    acc: dict[str, list] = {}
    for _ in range(b):
        i = rng.choice(n, 16, replace=False)
        st = stats_of(conf[i], dockq[i] if dockq is not None else None)
        for k, v in st.items():
            acc.setdefault(k, []).append(v)
    return {k: np.array(v) for k, v in acc.items()}


def p_value(null: np.ndarray, x: float) -> float:
    lo = float(np.mean(null <= x))
    hi = float(np.mean(null >= x))
    return min(1.0, 2.0 * min(lo, hi))


def z_of(p: float) -> float:
    from scipy.stats import norm
    return float(norm.isf(min(max(p, 1e-12), 1 - 1e-12) / 2))


def compare(gal, ta, seed) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    for t in sorted(set(gal) & set(ta)):
        g = gal[t]
        conf_full = ta[t]["conf_full"]
        dq_full = ta[t]["dockq_full"]
        ranks = sorted(set(conf_full) & (set(dq_full) if dq_full else set(conf_full)))
        if len(ranks) < 32:
            continue
        conf50 = np.array([conf_full[r] for r in ranks])
        dq50 = np.array([dq_full[r] for r in ranks]) if dq_full else None
        has_dq = dq50 is not None and np.isfinite(g["dockq"]).all()
        null = subset_null(conf50, dq50 if has_dq else None, rng)
        gst = stats_of(g["conf"], g["dockq"] if has_dq else None)
        row = {"target": t}
        for k, gv in gst.items():
            if k not in null:
                continue
            row[f"p_{k}"] = p_value(null[k], gv)
            row[f"dir_{k}"] = int(gv > np.median(null[k]))  # 1 = galaxy above null median
        # within-hardware floor: independent 16-subset vs null of the remaining 34
        wrows: dict[str, list] = {}
        for _ in range(B_WITHIN):
            i = rng.choice(len(ranks), 32, replace=False)
            a, b = i[:16], i[16:]
            nul = subset_null(conf50[b], dq50[b] if has_dq else None, rng, b=100)
            wst = stats_of(conf50[a], dq50[a] if has_dq else None)
            for k, v in wst.items():
                if k in nul:
                    wrows.setdefault(k, []).append(p_value(nul[k], v))
        for k, v in wrows.items():
            row[f"w_{k}"] = float(np.mean(np.array(v) <= ALPHA))
        rows.append(row)
    return rows


def summarize(model: str, rows: list[dict]) -> dict:
    keys = [k for k in rows[0] if k.startswith("p_")]
    out = {"model": model, "n_targets": len(rows), "stats": {}}
    for k in keys:
        name = k[2:]
        p = np.array([r[k] for r in rows if k in r and f"w_{name}" in r])
        w = np.array([r[f"w_{name}"] for r in rows if k in r and f"w_{name}" in r])
        if len(p) == 0:
            continue
        dirs = np.array([r[f"dir_{name}"] for r in rows if k in r and f"w_{name}" in r])
        extreme = p <= ALPHA
        bias = float(dirs[extreme].mean()) if extreme.any() else float("nan")
        out["stats"][name] = {
            "n": int(len(p)),
            "tail_cross": float(extreme.mean()),
            "tail_within": float(w.mean()),
            "median_abs_z_cross": float(np.median([z_of(x) for x in p])),
            "bias_frac_above": bias,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p0", default="/home/ttuser/abag_xm/deepn/phase0", type=pathlib.Path)
    ap.add_argument("--tier_a", default="/home/ttuser/abag_xm/tier_a", type=pathlib.Path)
    ap.add_argument("--out", default=None, type=pathlib.Path)
    args = ap.parse_args()
    args.out = args.out or args.p0 / "phase0_report.json"

    report = {"design": "galaxy-16 as one plausible 16-draw from the tier_a-50 BH distribution; "
                        f"B_NULL={B_NULL}, B_WITHIN={B_WITHIN}, alpha={ALPHA}",
              "models": []}
    for md, (suffix, _gname) in MODELS.items():
        gal = load_galaxy(args.p0, suffix, _gname)
        ta = load_tiera(args.p0, args.tier_a, md)
        rows = compare(gal, ta, seed=20260802)
        s = summarize(md, rows)
        report["models"].append(s)
        print(f"\n=== {md}: {s['n_targets']} paired targets ===")
        print(f"{'stat':<12} {'n':>4} {'tail_cross':>10} {'tail_within':>11} "
              f"{'med|z|_cross':>13} {'bias_frac':>9}")
        for name, d in s["stats"].items():
            print(f"{name:<12} {d['n']:>4} {d['tail_cross']:>10.3f} {d['tail_within']:>11.3f} "
                  f"{d['median_abs_z_cross']:>13.2f} {d['bias_frac_above']:>9.2f}")
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
