#!/usr/bin/env python3
"""AbAg-XM deep-N analysis core (state doc abag-xm-deepn-saturation-fullpanel, PHASE 3).

Assembles per-(model, target, N) sample pools and computes the saturation curves:
oracle (max DockQ over the pool) and user (DockQ at the pool's argmax of the model's own
selector -- confidence_score, pLDDT for esmfold2) at DockQ thresholds 0.23/0.49/0.80.

Pools merge every chunk of a (target, rung): the deep-N user sees all N samples and picks by
confidence, so chunk-boundary ranks are re-derived over the pool, never taken per-chunk.

Data arms (this script: deepn + tier_a; overlays bolt on later):
  deepn   ~/abag_xm/deepn/<model>/<target>_n<N>[_c<j>]/  results.json all_runs + labels.json
  tier_a  ~/abag_xm/tier_a/<model_dir>/<prefix>_results_<target>/results.json + labels dir

Writes deepn/analysis_curves.json and prints the curve table. CPU-only.
"""
import argparse, json, sys
from pathlib import Path

BASE = Path.home() / "abag_xm" / "deepn"
TIER_A = Path.home() / "abag_xm" / "tier_a"
THR = (0.23, 0.49, 0.80)
THR_KEY = {t: str(t).replace(".", "") for t in THR}
MODELS = {"opendde-abag": ("opendde", "opendde_abag", "confidence_score"),
          "protenix-v2": ("protenix", "protenix_v2", "confidence_score"),
          "boltz2": ("boltz2", "boltz2", "confidence_score"),
          "esmfold2": ("esmfold2", "esmfold2", "plddt")}


def pool_fold(results_json: Path, labels_json: Path, sel_key: str):
    """One fold -> list of (selector, dockq) joined by rank. None on any gap."""
    try:
        runs = json.loads(results_json.read_text())[0].get("all_runs", [])
        labs = json.loads(labels_json.read_text()).get("samples", [])
    except Exception:
        return None
    conf = {}
    for r in runs:
        v = r.get(sel_key)
        if v is not None:
            conf[int(r["rank"])] = float(v)
    dockq = {}
    for s in labs:
        d = s.get("dockq")
        if isinstance(d, dict) and d.get("dockq") is not None:
            dockq[int(s["rank"])] = float(d["dockq"])
    ranks = sorted(set(conf) & set(dockq))
    if not ranks:
        return None
    return [(conf[r], dockq[r]) for r in ranks]


def deepn_pools(model: str):
    """(target, rung) -> pooled [(sel, dockq)] across chunks, plus wall_s."""
    prefix, _md, sel = MODELS[model]
    out = {}
    mdir = BASE / prefix
    if not mdir.is_dir():
        return out
    walls = {}
    pj = BASE / "progress.jsonl"
    if pj.exists():
        for line in pj.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("model") == model and r.get("status") == "ok":
                k = (r["target"], r["rung"])
                walls[k] = walls.get(k, 0.0) + r["wall_s"]
    for out_dir in sorted(mdir.iterdir()):
        if not out_dir.is_dir():
            continue
        name = out_dir.name  # <target>_n<N>[_c<j>]
        try:
            t, rest = name.split("_n")
            rung = int(rest.split("_c")[0])
        except ValueError:
            continue
        pool = pool_fold(out_dir / f"{prefix}_results_{t}" / "results.json",
                         out_dir / "labels.json", sel)
        if pool is None:
            continue
        k = (t, rung)
        out.setdefault(k, []).extend(pool)
    for k in out:
        out[k] = {"pool": out[k], "wall_s": walls.get(k)}
    return out


def tiera_pools(model: str):
    prefix, md, sel = MODELS[model]
    out = {}
    lab_dir = TIER_A / "labels"
    for rj in sorted(TIER_A.glob(f"{md}/{prefix}_results_*/results.json")):
        t = rj.parent.name.split("_results_")[-1]
        pool = pool_fold(rj, lab_dir / f"{md}_{t}.json", sel)
        if pool:
            out[(t, 50)] = {"pool": pool, "wall_s": None}
    return out


def curve_points(pools):
    """Aggregate a {(target, N): {pool}} map into per-N curve points."""
    by_n = {}
    for (t, n), d in pools.items():
        pool = d["pool"]
        oracle = max(v for _c, v in pool)
        user = max(pool, key=lambda x: x[0])[1]
        by_n.setdefault(n, []).append({"target": t, "oracle": oracle, "user": user,
                                       "wall_s": d["wall_s"]})
    pts = {}
    for n, rows in sorted(by_n.items()):
        nt = len(rows)
        pts[n] = {"n_targets": nt,
                  "oracle_mean": sum(r["oracle"] for r in rows) / nt,
                  "user_mean": sum(r["user"] for r in rows) / nt,
                  **{f"oracle_ge_{THR_KEY[t]}": sum(1 for r in rows if r["oracle"] >= t) / nt
                     for t in THR},
                  **{f"user_ge_{THR_KEY[t]}": sum(1 for r in rows if r["user"] >= t) / nt
                     for t in THR},
                  "card_h": sum(r["wall_s"] for r in rows if r["wall_s"]) / 3600}
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, choices=sorted(MODELS))
    ap.add_argument("--out", default=str(BASE / "analysis_curves.json"))
    a = ap.parse_args()
    models = [a.model] if a.model else sorted(MODELS)
    report = {}
    for model in models:
        pools = tiera_pools(model) | deepn_pools(model)
        pts = curve_points(pools)
        report[model] = pts
        print(f"\n=== {model} ===")
        print(f"{'N':>5} {'nt':>4} {'oracle':>7} {'user':>7} "
              + " ".join(f"{'o>='+str(t):>8} {'u>='+str(t):>8}" for t in THR) + f" {'card-h':>8}")
        for n, p in pts.items():
            row = " ".join(f"{p['oracle_ge_' + THR_KEY[t]]:>8.3f} "
                           f"{p['user_ge_' + THR_KEY[t]]:>8.3f}" for t in THR)
            print(f"{n:>5} {p['n_targets']:>4} {p['oracle_mean']:>7.4f} {p['user_mean']:>7.4f} "
                  + row + f" {p['card_h']:>8.1f}")
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
