#!/usr/bin/env python3
"""AbAg-XM deep-N campaign: protenix-v2 / esmfold2 N=64 cross-hardware gate.

Pre-registered rule (state doc abag-xm-deepn-saturation-fullpanel, PHASE 2):
the PHASE 0 divergent-small verdict for protenix-v2/esmfold2 was attributed (mps-chunk
numerics / seed-variance class) at N=16. Before either model folds its first full-panel
galaxy rung beyond the pilot, its N=64 galaxy overlay must show the offset GONE within
deep-N noise. The deep-N noise floor: disjoint 64+64 partitions of the pooled 128
qb1+galaxy samples per target -- exactly how much two same-size same-model arms differ
under pure seed noise. A model is LICENSED for the galaxy panel iff over the pilot
targets: ratio_med = med|cross delta| / med(within-floor) <= 2 AND the fraction of
targets whose |delta| exceeds their floor's q95 <= 0.33 (PHASE 0 bars). Otherwise STOP:
px/esm panel folds halt and the verdict is recorded.

Pools: qb1 = ~/abag_xm/deepn/<prefix>/<t>_n64 (deepn arm); galaxy =
~/abag_xm/deepn/galaxy/<prefix>/<t>_n64 (harvested via scripts/abag_xm/p25_harvest.sh).
DockQ flavor: ARK interface DockQ on BOTH arms (same labeler) -- unlike the PHASE 0
N=16 overlay, no global-vs-interface caveat applies.

Runs on qb1:  python3 scripts/abag_xm/phase0_n64_gate.py [--model protenix-v2]
"""
import argparse
import importlib.util
import json
import pathlib

import numpy as np

_WT = pathlib.Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location("deepn", _WT / "abag_xm_deepn_analysis.py")
deepn = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(deepn)

BASE = pathlib.Path.home() / "abag_xm" / "deepn"
GATE_MODELS = ("protenix-v2", "esmfold2")
B_NULL = 2000
ALPHA = 0.05
LICENSE = {"ratio_med_max": 2.0, "exceed_q95_max": 0.33}


def load_arm(root: pathlib.Path, model: str, rung: int) -> dict:
    prefix, _md, sel = deepn.MODELS[model]
    out = {}
    mdir = root / prefix
    if not mdir.is_dir():
        return out
    for out_dir in sorted(mdir.glob(f"*_n{rung}")):
        if not out_dir.is_dir() or "_c" in out_dir.name:
            continue
        t = out_dir.name.split("_n")[0]
        pool = deepn.pool_fold(out_dir / f"{prefix}_results_{t}" / "results.json",
                               out_dir / "labels.json", sel)
        if pool:
            out[t] = pool
    return out


def stats_of(pool) -> dict:
    conf = np.array([c for c, _d in pool])
    dq = np.array([d for _c, d in pool])
    return {"oracle": float(dq.max()), "user": float(dq[int(np.argmax(conf))]),
            "dq_mean": float(dq.mean())}


def gate(model: str, qb1: dict, gal: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    for t in sorted(set(qb1) & set(gal)):
        a, b = qb1[t], gal[t]
        n = min(len(a), len(b))
        if n < 32:
            continue
        pooled = a[:n] + b[:n] if len(a) >= len(b) else b[:n] + a[:n]
        # deterministic arm pairing: qb1 first n, galaxy first n (rank order = seed order)
        sa, sb = stats_of(a[:n]), stats_of(b[:n])
        null_acc = {}
        vals = np.array([d for _c, d in pooled])
        confs = np.array([c for c, _d in pooled])
        m = len(pooled)
        for _ in range(B_NULL):
            i = rng.choice(m, 2 * n, replace=False)
            ia, ib = i[:n], i[n:]
            pa = [(confs[j], vals[j]) for j in ia]
            pb = [(confs[j], vals[j]) for j in ib]
            x, y = stats_of(pa), stats_of(pb)
            for k in x:
                null_acc.setdefault(k, []).append(abs(x[k] - y[k]))
        row = {"target": t, "n": n}
        for k in sa:
            null = np.array(null_acc[k])
            d = abs(sa[k] - sb[k])
            row[f"delta_{k}"] = d
            row[f"sign_{k}"] = float(np.sign(sb[k] - sa[k]))
            row[f"p_{k}"] = float((np.sum(null >= d) + 1) / (len(null) + 1))
            row[f"medw_{k}"] = float(np.median(null))
            row[f"q95w_{k}"] = float(np.quantile(null, 0.95))
        rows.append(row)
    out = {"model": model, "n_targets": len(rows), "stats": {}}
    for k in ("oracle", "user", "dq_mean"):
        sel = [r for r in rows if f"delta_{k}" in r]
        if not sel:
            continue
        dc = np.array([r[f"delta_{k}"] for r in sel])
        medw = np.array([r[f"medw_{k}"] for r in sel])
        q95 = np.array([r[f"q95w_{k}"] for r in sel])
        sgn = np.array([r[f"sign_{k}"] for r in sel])
        p = np.array([r[f"p_{k}"] for r in sel])
        out["stats"][k] = {
            "n": len(sel),
            "med_abs_delta_cross": float(np.median(dc)),
            "med_within_floor": float(np.median(medw)),
            "ratio_med": float(np.median(dc) / max(np.median(medw), 1e-12)),
            "exceed_q95": float(np.mean(dc > q95)),
            "tail_p_le_alpha": float(np.mean(p <= ALPHA)),
            "bias_frac_galaxy_low": float(np.mean(sgn < 0)),
        }
    o = out["stats"].get("oracle", {})
    verdict = ("LICENSED" if o
               and o["ratio_med"] <= LICENSE["ratio_med_max"]
               and o["exceed_q95"] <= LICENSE["exceed_q95_max"]
               else "STOP")
    out["verdict"] = verdict
    out["license_bars"] = LICENSE
    out["rows"] = rows
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, choices=sorted(GATE_MODELS))
    ap.add_argument("--base", default=BASE, type=pathlib.Path)
    ap.add_argument("--rung", type=int, default=64)
    ap.add_argument("--out", default=None, type=pathlib.Path)
    args = ap.parse_args()
    args.out = args.out or args.base / "phase0" / f"n64_gate_report.json"
    models = [args.model] if args.model else list(GATE_MODELS)
    report = {"design": "qb1-N64 vs galaxy-N64 same (model,target); null = disjoint n+n "
                        "partitions of pooled arms (B=%d); license bars %s" % (B_NULL, LICENSE),
              "rung": args.rung, "models": {}}
    for md in models:
        qb1 = load_arm(args.base, md, args.rung)
        gal = load_arm(args.base / "galaxy", md, args.rung)
        common = sorted(set(qb1) & set(gal))
        print(f"\n=== {md}: qb1 {len(qb1)} targets, galaxy {len(gal)}, paired {len(common)} ===")
        if len(common) < 7:
            print(f"  INCOMPLETE: need >=7 paired targets, have {len(common)} {common}")
            report["models"][md] = {"error": "incomplete", "paired": common}
            continue
        g = gate(md, qb1, gal, seed=20260803)
        report["models"][md] = g
        for k, s in g["stats"].items():
            print(f"  {k:>8}: med|cross|={s['med_abs_delta_cross']:.4f} "
                  f"floor={s['med_within_floor']:.4f} ratio={s['ratio_med']:.2f} "
                  f"exceed_q95={s['exceed_q95']:.2f} tail={s['tail_p_le_alpha']:.2f} "
                  f"gal-low-frac={s['bias_frac_galaxy_low']:.2f}")
        print(f"  VERDICT: {g['verdict']}  (bars ratio<=2, exceed<=0.33 on oracle)")
        for r in g["rows"]:
            print(f"    {r['target']}: d_oracle={r['delta_oracle']:.4f} "
                  f"p={r['p_oracle']:.3f} q95w={r['q95w_oracle']:.4f} "
                  f"d_user={r['delta_user']:.4f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
