"""Paired statistics for the MOL1 quality grade (rules: state/mol1/quality/PLAN.md, written before any number).

Input: rows {arm, case, set, seed, <metric>: float} where a structural metric is the top-ranked sample's score for
that (case, seed) and a confidence is the mean over its samples. Arms: "gpu" (seeds S), "gpu_self" (seeds S'),
"tt_bh", "tt_wh" (seeds S). Every comparison is per target: mean over its seeds, then a percentile bootstrap over
targets (20,000 resamples, fixed RNG).

    verdict(rows, test="tt_bh")  ->  {"verdict": "PASS"|"FAIL"|"INSUFFICIENT", "metrics": {...}, "per_set": {...}}

A metric passes when the 95 % interval of mean(test - gpu) lies inside the A/A interval of mean(gpu_self - gpu)
widened by its own half-width (equivalence at the seed floor), and the success-rate difference interval contains 0.
Calibration: Spearman(confidence, structure) of the test arm inside the GPU's bootstrap interval.
"""
from collections import defaultdict

import numpy as np

# metric -> (higher is better, success threshold or None); FoldBench success definitions
METRICS = {"dockq": (True, 0.23), "lddt": (True, None), "lddt_pli": (True, 0.8), "lrmsd": (False, 2.0),
           "plddt": (True, None), "iptm": (True, None), "ptm": (True, None), "ranking_score": (True, None)}
CALIBRATION = (("plddt", "lddt"), ("iptm", "dockq"))
N_BOOT, SEED = 20000, 0
MIN_SEEDS = 4


def per_target(rows, arm, metric):
    """{case: mean over seeds} for one arm; cases with fewer than MIN_SEEDS finite values are left out."""
    acc = defaultdict(list)
    for r in rows:
        if r["arm"] == arm and r.get(metric) is not None:
            acc[r["case"]].append(r[metric])
    return {c: float(np.mean(v)) for c, v in acc.items() if len(v) >= MIN_SEEDS and np.all(np.isfinite(v))}


def success(rows, arm, case_set=None):
    """{case: 0/1} FoldBench success of the top-ranked prediction over the whole seed budget (max ranking_score)."""
    best = {}
    for r in rows:
        if r["arm"] != arm or (case_set and r["case"] not in case_set):
            continue
        if r["case"] not in best or r.get("ranking_score", -1e9) > best[r["case"]].get("ranking_score", -1e9):
            best[r["case"]] = r
    out = {}
    for c, r in best.items():
        if r.get("lrmsd") is not None and r.get("lddt_pli") is not None:
            out[c] = float(r["lrmsd"] < 2.0 and r["lddt_pli"] > 0.8)
        elif r.get("dockq") is not None:
            out[c] = float(r["dockq"] >= 0.23)
    return out


def boot_mean(x, rng):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired(a, b, rng):
    """mean(a - b) over shared targets with its 95 % interval, or None if they share fewer than 5 targets."""
    keys = sorted(set(a) & set(b))
    if len(keys) < 5:
        return None
    m, lo, hi = boot_mean([a[k] - b[k] for k in keys], rng)
    return {"mean": m, "ci95": [lo, hi], "n": len(keys)}


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def boot_spearman(conf, struct, rng):
    keys = sorted(set(conf) & set(struct))
    if len(keys) < 10:
        return None
    x, y = np.array([conf[k] for k in keys]), np.array([struct[k] for k in keys])
    s = [spearman(x[i], y[i]) for i in rng.integers(0, len(keys), size=(2000, len(keys)))]
    return {"rho": spearman(x, y), "ci95": [float(np.nanpercentile(s, 2.5)), float(np.nanpercentile(s, 97.5))],
            "n": len(keys)}


def equivalent(test, floor):
    """test interval inside the floor interval widened by the floor's own half-width."""
    lo, hi = floor["ci95"]
    w = (hi - lo) / 2
    return lo - w <= test["ci95"][0] and test["ci95"][1] <= hi + w


def verdict(rows, test, base="gpu", self_arm="gpu_self", cases=None):
    if cases is not None:
        rows = [r for r in rows if r["case"] in cases]
    rng = np.random.default_rng(SEED)
    out, ok, insufficient = {}, True, False
    for m in METRICS:
        b, t, s = (per_target(rows, a, m) for a in (base, test, self_arm))
        if not b:
            continue
        d, f = paired(t, b, rng), paired(s, b, rng)
        if d is None or f is None:
            insufficient = True
            out[m] = {"test_minus_base": d, "seed_floor": f, "pass": None}
            continue
        missing = sorted(set(b) - set(t))
        p = equivalent(d, f) and not missing
        out[m] = {"test_minus_base": d, "seed_floor": f, "missing": missing, "pass": p}
        ok &= p
    sb, st, ss = (success(rows, a) for a in (base, test, self_arm))
    if sb:
        keys = sorted(set(sb) & set(st))
        d = boot_mean([st[k] - sb[k] for k in keys], rng) if len(keys) >= 5 else None
        f = boot_mean([ss[k] - sb[k] for k in sorted(set(sb) & set(ss))], rng) if len(set(sb) & set(ss)) >= 5 else None
        rate = {a: float(np.mean(list(v.values()))) for a, v in ((base, sb), (test, st), (self_arm, ss)) if v}
        p = d is not None and d[1] <= 0 <= d[2]
        out["success_rate"] = {"rate": rate, "test_minus_base": d and {"mean": d[0], "ci95": [d[1], d[2]]},
                               "seed_floor": f and {"mean": f[0], "ci95": [f[1], f[2]]}, "pass": p}
        ok &= p
        insufficient |= d is None
    for conf, struct in CALIBRATION:
        cb = boot_spearman(per_target(rows, base, conf), per_target(rows, base, struct), rng)
        ct = boot_spearman(per_target(rows, test, conf), per_target(rows, test, struct), rng)
        if cb and ct:
            p = cb["ci95"][0] <= ct["rho"] <= cb["ci95"][1]
            out[f"calibration_{conf}_vs_{struct}"] = {"base": cb, "test": ct, "pass": p}
            ok &= p
    v = "INSUFFICIENT" if insufficient else ("PASS" if ok else "FAIL")
    return {"verdict": v, "metrics": out}
