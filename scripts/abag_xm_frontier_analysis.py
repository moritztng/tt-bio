#!/usr/bin/env python3
"""AbAg-XM seeds-vs-samples frontier analysis (state doc §5, exact math, no simulation).

Inputs (all measured):
  Arm A labels   ~/abag_xm/frontier/A/<T>/labels.json                 (200 samples/target)
  Arm B labels   ~/abag_xm/frontier/B/<T>_seed<j>/labels.json         (10 samples x 20/target)
  Arm B pools    ~/abag_xm/frontier/B_pool/<T>/labels.json            (200 pooled, matrix+basin)
  progress       ~/abag_xm/frontier/progress.jsonl + progress_qb2.jsonl
  slab labels    ~/abag_xm/tier_a/labels/opendde_abag_<T>.json        (50, continuity only)

Outputs ~/abag_xm/frontier/analysis.json + a markdown §7 body on stdout.

Oracle (exact hypergeometric, threshold th): target with S successes in N samples
contributes 1 - C(N-S, m)/C(N, m) at structure count m; mean over 12 targets.
Arm A grid m in {1,2,4,8,16,32,50,64,100,128,200} (50 for slab continuity).
Arm B: w seed-blocks (of 20) with >= 1 success; oracle at k seeds (m = 10k) is
1 - C(20-w, k)/C(20, k), k in {1,2,4,5,10,20}.
Pooled B: Arm-A-style hypergeometric on the 200 pooled samples (equivalence test,
bootstrap 95% CI of pooledB - A over the 12 targets, rng seed 42, 10k reps).
"""
import json, math, random, statistics
from pathlib import Path

BASE = Path.home() / "abag_xm" / "frontier"
SLAB = Path.home() / "abag_xm" / "tier_a" / "labels"
TARGETS = ["9q6y", "9tmp", "9gei", "9fte", "9wpm", "9qrv",
           "9ma0", "9q6z", "9j4c", "9uoi", "9m8l", "9ldx"]
FAIL = {"9q6y", "9tmp", "9gei", "9fte", "9wpm", "9qrv"}
THRESHOLDS = [0.23, 0.49, 0.80]
GRID_A = [1, 2, 4, 8, 16, 32, 50, 64, 100, 128, 200]
GRID_B_K = [1, 2, 4, 5, 10, 20]
BOOT_REPS = 10000


def comb(n, k):
    return math.comb(n, k) if 0 <= k <= n else 0


def hyper_oracle(n, s, m):
    return 1.0 - comb(n - s, m) / comb(n, m) if n > 0 else 0.0


def dockq_list(labels_path):
    d = json.loads(labels_path.read_text())
    out = []
    for s in d["samples"]:
        v = s.get("dockq")
        if isinstance(v, dict) and v.get("dockq") is not None:
            out.append(float(v["dockq"]))
        else:
            out.append(None)
    return out, d


def load_arm_a():
    per = {}
    for t in TARGETS:
        p = BASE / "A" / t / "labels.json"
        if p.exists():
            dq, d = dockq_list(p)
            per[t] = (dq, d)
    return per


def load_arm_b_folds():
    per = {t: {} for t in TARGETS}
    for t in TARGETS:
        for j in range(20):
            p = BASE / "B" / f"{t}_seed{j}" / "labels.json"
            if p.exists():
                dq, d = dockq_list(p)
                per[t][j] = (dq, d)
    return per


def load_pools():
    per = {}
    for t in TARGETS:
        p = BASE / "B_pool" / t / "labels.json"
        if p.exists():
            dq, d = dockq_list(p)
            per[t] = (dq, d)
    return per


def mean_oracle(per, grid, thr):
    """per: {target: dockq list}. Returns {m: (mean, [per-target values])}."""
    out = {}
    for m in grid:
        vals = []
        for t in TARGETS:
            dq = [x for x in per.get(t, []) if x is not None]
            s = sum(1 for x in dq if x >= thr)
            vals.append(hyper_oracle(len(dq), s, m))
        out[m] = (statistics.mean(vals), vals)
    return out


def arm_b_seed_oracle(folds, thr):
    out = {}
    for k in GRID_B_K:
        vals = []
        for t in TARGETS:
            blocks = folds.get(t, {})
            w = sum(1 for j, (dq, _) in blocks.items()
                    if any(x is not None and x >= thr for x in dq))
            vals.append(hyper_oracle(20, w, k))
        out[10 * k] = (statistics.mean(vals), vals)
    return out


def boot_ci(diff_fn, reps=BOOT_REPS):
    """diff_fn(targets_subset) -> float; returns (point, lo, hi)."""
    rng = random.Random(42)
    point = diff_fn(TARGETS)
    boot = sorted(diff_fn([rng.choice(TARGETS) for _ in TARGETS]) for _ in range(reps))
    return point, boot[int(0.025 * reps)], boot[int(0.975 * reps) - 1]


def equivalence(a_labels, b_pool, thr, m):
    def diff(ts):
        ds = []
        for t in ts:
            da = [x for x in a_labels.get(t, []) if x is not None]
            db = [x for x in b_pool.get(t, []) if x is not None]
            sa = sum(1 for x in da if x >= thr)
            sb = sum(1 for x in db if x >= thr)
            ds.append(hyper_oracle(len(db), sb, m) - hyper_oracle(len(da), sa, m))
        return statistics.mean(ds)
    return boot_ci(diff)


def progress_records():
    recs = []
    for name in ("progress.jsonl", "progress_qb2.jsonl"):
        p = BASE / name
        if p.exists():
            for line in p.read_text().splitlines():
                try:
                    recs.append(json.loads(line))
                except Exception:
                    pass
    return recs


def cost_table(recs):
    """Per-arm card-seconds from ok records; two-point fixed/marginal split."""
    ok = [r for r in recs if r.get("status") == "ok"]
    a = [r for r in ok if r["arm"] == "A"]
    b = [r for r in ok if r["arm"] == "B"]
    out = {"A_card_s": sum(r["wall_s"] for r in a), "B_card_s": sum(r["wall_s"] for r in b),
           "A_folds": len(a), "B_folds": len(b)}
    if a and b:
        wa = statistics.mean(r["wall_s"] for r in a)
        wb = statistics.mean(r["wall_s"] for r in b)
        marg = (wa - wb) / 190.0
        out.update(A_mean_wall=wa, B_mean_wall=wb, marginal_s_per_sample=marg,
                   fixed_s=wb - 10 * marg)
        by_host = {}
        for r in ok:
            by_host.setdefault(r["host"], []).append(r["wall_s"])
        out["wall_by_host"] = {h: round(statistics.mean(v), 1) for h, v in by_host.items()}
    return out


def spread_table(a_labels, pools, thr=0.23):
    """Per-target structural spread at m=200 per arm: PSS, basin clusters, dockq sd."""
    rows = []
    for t in TARGETS:
        row = {"target": t, "bucket": "fail" if t in FAIL else "succeed"}
        for arm, src in (("A", a_labels), ("Bp", pools)):
            if t not in src:
                continue
            dq, d = src[t] if isinstance(src[t], tuple) else (src[t], None)
            vals = [x for x in dq if x is not None]
            pm = (d or {}).get("pairwise_matrix", {})
            bc = (d or {}).get("basin_clust", {})
            row[f"{arm}_PSS"] = pm.get("PSS")
            row[f"{arm}_clusters"] = bc.get("n_clusters")
            row[f"{arm}_dockq_sd"] = round(statistics.pstdev(vals), 4) if len(vals) > 1 else None
            row[f"{arm}_S"] = sum(1 for x in vals if x >= thr)
        rows.append(row)
    return rows


# Frozen slab s50 counts (state doc §4): measured and verified by p1 from the full
# 161-target label set on qb1. qb1 has been unreachable since 2026-07-30 ~02:30 UTC
# and 4 of the 12 targets' slab label files live only there (9q6y, 9gei, 9qrv, 9q6z).
# At thr 0.23 substituting the frozen count is mathematically identical to the
# file-derived one (the hypergeometric uses only S and N, and p1 recomputed these
# exact S values from the files). At other thresholds no frozen counts exist, so
# slab continuity is omitted while any file is unreachable.
FROZEN_S50 = {"9q6y": 0, "9tmp": 0, "9gei": 0, "9fte": 0, "9wpm": 0, "9qrv": 0,
              "9ma0": 1, "9q6z": 1, "9j4c": 7, "9uoi": 7, "9m8l": 48, "9ldx": 47}


def slab_continuity(thr):
    missing = [t for t in TARGETS if not (SLAB / f"opendde_abag_{t}.json").exists()]
    if missing and thr != 0.23:
        return {"omitted": f"slab labels unreachable for {missing} (qb1 down); "
                           f"continuity is exact only at thr 0.23 via FROZEN_S50"}
    per = {}
    for t in TARGETS:
        p = SLAB / f"opendde_abag_{t}.json"
        if p.exists():
            dq, _ = dockq_list(p)
            per[t] = dq
        else:
            s = FROZEN_S50[t]
            per[t] = [1.0] * s + [0.0] * (50 - s)
    grid = [1, 2, 4, 8, 16, 32, 50]
    return {m: round(mean_oracle(per, [m], thr)[m][0], 4) for m in grid}


def main():
    a_labels = {t: dq for t, (dq, _) in load_arm_a().items()}
    b_folds = load_arm_b_folds()
    pools = {t: dq for t, (dq, _) in load_pools().items()}
    b_pool_flat = pools
    recs = progress_records()
    res = {"n_A_labeled": len(a_labels),
           "n_B_targets_complete": sum(1 for t in TARGETS if len(b_folds[t]) == 20),
           "n_B_pools": len(pools), "cost": cost_table(recs)}

    for thr in THRESHOLDS:
        key = f"thr{thr}"
        res[key] = {
            "A_oracle": {m: round(v, 4) for m, (v, _) in mean_oracle(a_labels, GRID_A, thr).items()},
            "B_seed_oracle": {m: round(v, 4) for m, (v, _) in arm_b_seed_oracle(b_folds, thr).items()},
            "slab_oracle": slab_continuity(thr),
        }
        if pools:
            res[key]["Bpool_oracle"] = {
                m: round(v, 4) for m, (v, _) in mean_oracle(b_pool_flat, GRID_A, thr).items()}
            res[key]["equiv_Bpool_minus_A"] = {
                str(m): [round(x, 4) for x in equivalence(a_labels, b_pool_flat, thr, m)]
                for m in (50, 200)}

    if a_labels and pools:
        res["spread"] = spread_table(
            {t: (dq, d) for t, (dq, d) in load_arm_a().items()},
            {t: (dq, d) for t, (dq, d) in load_pools().items()})

    # ranked (top-1 by confidence rank): Arm A rank-0 dockq; B pool re-ranked by ptm.
    ranked = {}
    for thr in (0.23,):
        a_top1, bp_top1 = [], []
        for t in TARGETS:
            pa = BASE / "A" / t / "labels.json"
            if pa.exists():
                d = json.loads(pa.read_text())
                r0 = next((s for s in d["samples"] if s["rank"] == 0), None)
                if r0 and isinstance(r0.get("dockq"), dict):
                    a_top1.append(1.0 if (r0["dockq"].get("dockq") or 0) >= thr else 0.0)
            pp = BASE / "B_pool" / t / "labels.json"
            if pp.exists():
                d = json.loads(pp.read_text())
                conf_cache = {}
                best = None
                for s in d["samples"]:
                    v = s.get("dockq")
                    dq = v.get("dockq") if isinstance(v, dict) else None
                    j = s.get("seed_j")
                    if j not in conf_cache:
                        conf_cache[j] = {}
                        rj_path = (BASE / "B" / f"{t}_seed{j}" /
                                   f"opendde_results_{t}" / "results.json")
                        if rj_path.exists():
                            try:
                                rj = json.loads(rj_path.read_text())
                                rj = rj[0] if isinstance(rj, list) else rj
                                conf_cache[j] = {r.get("rank"): r.get("confidence_score", r.get("ptm"))
                                                 for r in rj.get("all_runs", [])}
                            except Exception:
                                pass
                    conf = conf_cache.get(j, {}).get(s["rank"])
                    if dq is not None and conf is not None and (best is None or conf > best[0]):
                        best = (conf, dq)
                if best:
                    bp_top1.append(1.0 if best[1] >= thr else 0.0)
        ranked["A_top1"] = round(statistics.mean(a_top1), 4) if a_top1 else None
        ranked["Bpool_top1"] = round(statistics.mean(bp_top1), 4) if bp_top1 else None
    res["ranked"] = ranked

    (BASE / "analysis.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
