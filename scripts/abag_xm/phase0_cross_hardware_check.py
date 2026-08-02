"""PHASE 0 cross-hardware sanity check: WH-Galaxy p2 (N=16, seed 42) vs BH-QB tier_a (N=50, seed 42).

Pre-registered design (state/abag-xm-deepn-saturation-fullpanel.md, 2026-08-02):
structures on both sides are stored by confidence rank, not draw order, and samples within a fold
are i.i.d., so galaxy-16 is tested as "one plausible 16-draw from the BH distribution": each
statistic of the galaxy 16 is compared against the bootstrap distribution of the same statistic
over 16-subsets of tier_a's 50 (joint (confidence, dockq) pairs). The within-hardware floor is the
distribution of |stat(A)-stat(B)| over disjoint 16+16 partitions of the same tier_a 50.

v2 fixes (2026-08-02, same day as v1):
- v1 within-floor built its null by drawing 16-of-16 (degenerate, zero variance) -> tail_within
  was ~1.0 by construction. Now direct disjoint-partition differences as pre-registered.
- v1 summarized stat keys from the first row only -> every DockQ stat silently dropped whenever
  the alphabetically-first target lacked GT. Now the union over rows.
- v1 p-value floor 1/(2B) hit a z clamp of 7.13 and hid magnitude. Now (k+1)/(B+1) correction;
  magnitude is reported as |delta| relative to the within floor, not z of a clamped p.
- esmfold2 confidence like-for-like: galaxy parquet confidence_score is pLDDT (30b33111); tier_a
  now uses its all_runs plddt for esmfold2 instead of the old composite confidence_score.

Verdict rule (per model x statistic): exceedance rate (fraction of targets with |delta_cross|
above the target-specific within-floor q95) <= 0.15 (3x the alpha=0.05 construction rate) and
median |delta_cross| <= 2x median within-floor |delta|. Bias fraction reported separately.

Run with python3 on qb1 (pandas + numpy + scipy). CPU-only.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

MODELS = {
    "opendde_abag": ("", "opendde-abag"),
    "protenix_v2": ("_protenix", "protenix-v2"),
    "boltz2": ("_boltz2", "boltz2"),
    "esmfold2": ("_esmfold2", "esmfold2"),
}
THR = (0.23, 0.49, 0.80)
B_NULL = 2000   # 16-subset null draws per target
B_WITHIN = 2000 # disjoint 16+16 partitions per target
ALPHA = 0.05


def load_galaxy(p0: pathlib.Path, suffix: str) -> dict:
    df = pd.read_parquet(p0 / f"abag_xm_scaling{suffix}_samples.parquet")
    df = df[(df.status == "ok") & (df.n_samples == 16)]
    out = {}
    for t, g in df.groupby("target"):
        g = g.sort_values("rank")
        out[t] = {
            "conf": g.confidence_score.to_numpy(float),
            "dockq": g.global_dockq.to_numpy(float),
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
            dq.setdefault(parts[0], {})[int(parts[1])] = float(parts[2])
    out = {}
    conf_key = "plddt" if md == "esmfold2" else "confidence_score"
    for rj in sorted(tier_a.glob(f"{md}/*_results_*/results.json")):
        t = rj.parent.name.split("_results_")[-1]
        try:
            runs = json.loads(rj.read_text())[0].get("all_runs", [])
        except Exception:
            continue
        conf = {}
        for r in runs:
            v = r.get(conf_key)
            if v is not None and np.isfinite(float(v)):
                conf[int(r["rank"])] = float(v)
        if conf:
            out[t] = {"conf_full": conf, "dockq_full": dq.get(t, {})}
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
            from scipy.stats import spearmanr
            s["spearman"] = float(spearmanr(conf, dockq).statistic)
    return s


def p_value(null: np.ndarray, x: float) -> float:
    b = len(null)
    lo = float((np.sum(null <= x) + 1) / (b + 1))
    hi = float((np.sum(null >= x) + 1) / (b + 1))
    return min(1.0, 2.0 * min(lo, hi))


def compare(gal: dict, ta: dict, seed: int):
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    pooled_within: dict[str, list] = {}
    for t in sorted(set(gal) & set(ta)):
        g = gal[t]
        conf_full = ta[t]["conf_full"]
        dq_full = ta[t]["dockq_full"]
        ranks = sorted(set(conf_full) & (set(dq_full) if dq_full else set(conf_full)))
        if len(ranks) < 32 or not np.isfinite(g["conf"]).all():
            continue
        conf50 = np.array([conf_full[r] for r in ranks])
        dq50 = np.array([dq_full[r] for r in ranks]) if dq_full else None
        has_dq = dq50 is not None and np.isfinite(g["dockq"]).all()
        dqn = dq50 if has_dq else None
        n = len(ranks)

        null_acc: dict[str, list] = {}
        for _ in range(B_NULL):
            i = rng.choice(n, 16, replace=False)
            for k, v in stats_of(conf50[i], dqn[i] if dqn is not None else None).items():
                null_acc.setdefault(k, []).append(v)
        null = {k: np.array(v) for k, v in null_acc.items()}

        within_acc: dict[str, list] = {}
        for _ in range(B_WITHIN):
            i = rng.choice(n, 32, replace=False)
            sa = stats_of(conf50[i[:16]], dqn[i[:16]] if dqn is not None else None)
            sb = stats_of(conf50[i[16:]], dqn[i[16:]] if dqn is not None else None)
            for k, v in sa.items():
                if k in sb:
                    within_acc.setdefault(k, []).append(abs(v - sb[k]))
        within = {k: np.array(v) for k, v in within_acc.items()}

        gst = stats_of(g["conf"], g["dockq"] if has_dq else None)
        row = {"target": t}
        for k, gv in gst.items():
            if k not in null or k not in within:
                continue
            row[f"delta_{k}"] = gv - float(np.median(null[k]))
            row[f"p_{k}"] = p_value(null[k], gv)
            row[f"dir_{k}"] = int(gv > np.median(null[k]))
            row[f"q95w_{k}"] = float(np.quantile(within[k], 0.95))
            row[f"medw_{k}"] = float(np.median(within[k]))
            pooled_within.setdefault(k, []).append(within[k])
        rows.append(row)
    pooled = {k: np.concatenate(v) for k, v in pooled_within.items()}
    return rows, pooled


def summarize(model: str, rows: list[dict], pooled: dict[str, np.ndarray]) -> dict:
    keys = sorted({k[2:] for r in rows for k in r if k.startswith("p_")})
    out = {"model": model, "n_targets": len(rows), "stats": {}}
    for name in keys:
        sel = [r for r in rows if f"p_{name}" in r]
        p = np.array([r[f"p_{name}"] for r in sel])
        dc = np.abs([r[f"delta_{name}"] for r in sel])
        q95 = np.array([r[f"q95w_{name}"] for r in sel])
        medw = np.array([r[f"medw_{name}"] for r in sel])
        dirs = np.array([r[f"dir_{name}"] for r in sel])
        extreme = p <= ALPHA
        exceed = dc > q95
        out["stats"][name] = {
            "n": int(len(sel)),
            "tail_cross": float(extreme.mean()),
            "exceed_q95_within": float(exceed.mean()),
            "med_abs_delta_cross": float(np.median(dc)),
            "med_abs_within": float(np.median(medw)),
            "ratio_med": float(np.median(dc) / max(np.median(medw), 1e-12)),
            "pooled_within_q95": float(np.quantile(pooled[name], 0.95)) if name in pooled else None,
            "bias_frac_above": float(dirs[extreme].mean()) if extreme.any() else float("nan"),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p0", default="/home/ttuser/abag_xm/deepn/phase0", type=pathlib.Path)
    ap.add_argument("--tier_a", default="/home/ttuser/abag_xm/tier_a", type=pathlib.Path)
    ap.add_argument("--out", default=None, type=pathlib.Path)
    args = ap.parse_args()
    args.out = args.out or args.p0 / "phase0_report_v2.json"

    report = {"design": "v2: galaxy-16 vs 16-subset null of tier_a-50; within floor = disjoint "
                        "16+16 partition |delta| distribution; B_NULL=%d, B_WITHIN=%d, alpha=%.2f"
                        % (B_NULL, B_WITHIN, ALPHA),
              "models": []}
    for md, (suffix, _gname) in MODELS.items():
        gal = load_galaxy(args.p0, suffix)
        ta = load_tiera(args.p0, args.tier_a, md)
        rows, pooled = compare(gal, ta, seed=20260802)
        s = summarize(md, rows, pooled)
        report["models"].append(s)
        print(f"\n=== {md}: {s['n_targets']} paired targets ===")
        print(f"{'stat':<10} {'n':>4} {'tail':>6} {'exceed':>7} {'med|dX|':>9} {'med|dW|':>9} "
              f"{'ratio':>6} {'bias':>6}")
        for name, d in s["stats"].items():
            print(f"{name:<10} {d['n']:>4} {d['tail_cross']:>6.3f} {d['exceed_q95_within']:>7.3f} "
                  f"{d['med_abs_delta_cross']:>9.4f} {d['med_abs_within']:>9.4f} "
                  f"{d['ratio_med']:>6.2f} {d['bias_frac_above']:>6.2f}")
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
