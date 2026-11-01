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
METRICS = {"dockq": (True, 0.23), "lddt": (True, None), "lddt_pli": (True, 0.8), "lig_rmsd": (False, 2.0),
           "plddt": (True, None), "iptm": (True, None), "ptm": (True, None), "ranking_score": (True, None)}
CONFIDENCE = ("plddt", "iptm", "ptm", "ranking_score")
CALIBRATION = (("plddt", "lddt"), ("iptm", "dockq"))
N_BOOT, SEED = 20000, 0
MIN_SEEDS = 4


def per_target(rows, arm, metric, min_seeds=MIN_SEEDS):
    """{case: mean over seeds} for one arm; cases with fewer than min_seeds finite values are left out."""
    acc = defaultdict(list)
    for r in rows:
        if r["arm"] == arm and r.get(metric) is not None:
            acc[r["case"]].append(r[metric])
    return {c: float(np.mean(v)) for c, v in acc.items() if len(v) >= min_seeds and np.all(np.isfinite(v))}


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
        if r.get("lig_rmsd") is not None and r.get("lddt_pli") is not None:
            out[c] = float(r["lig_rmsd"] < 2.0 and r["lddt_pli"] > 0.8)
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


def verdict(rows, test, base="gpu", self_arm="gpu_self", cases=None, conf_split=(), seeds=None):
    """conf_split: case ids ("<id>[:recipe]") whose CONFIDENCE metrics are left out (upstream's own fp32 and bf16
    disagree there, so the GPU is no reference); their structure metrics still count.
    seeds: grade a test arm that ran only these seeds against the base on the same seeds and a floor of the self arm
    on as many of its own seeds per case (lowest first), so all three means average the same number of folds."""
    if cases is not None:
        rows = [r for r in rows if r["case"] in cases]
    min_seeds = MIN_SEEDS
    if seeds:
        own = defaultdict(set)
        for r in rows:
            if r["arm"] == self_arm:
                own[r["case"]].add(r["seed"])
        own = {c: set(sorted(v)[:len(seeds)]) for c, v in own.items()}
        rows = [r for r in rows if (r["seed"] in own.get(r["case"], ()) if r["arm"] == self_arm else
                                    r["seed"] in seeds if r["arm"] in (base, test) else False)]
        min_seeds = len(seeds)
    rng = np.random.default_rng(SEED)
    out, ok, insufficient = {}, True, False

    def targets(arm, m):
        x = per_target(rows, arm, m, min_seeds)
        return {c: v for c, v in x.items() if c.split("|")[0] not in conf_split} if m in CONFIDENCE else x

    for m in METRICS:
        b, t, s = (targets(a, m) for a in (base, test, self_arm))
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
        cb = boot_spearman(targets(base, conf), targets(base, struct), rng)
        ct = boot_spearman(targets(test, conf), targets(test, struct), rng)
        if cb and ct:
            p = cb["ci95"][0] <= ct["rho"] <= cb["ci95"][1]
            out[f"calibration_{conf}_vs_{struct}"] = {"base": cb, "test": ct, "pass": p}
            ok &= p
    v = "INSUFFICIENT" if insufficient else ("PASS" if ok else "FAIL")
    return {"verdict": v, "metrics": out}


def self_test(rows, base="gpu", other="gpu_self"):
    """Upstream against itself on other seeds: every metric's mean difference interval and the success-rate
    difference interval must contain 0 (RFD4's first self-test failed this at too few steps)."""
    rng = np.random.default_rng(SEED)
    out, ok = {}, True
    for m in METRICS:
        d = paired(per_target(rows, other, m), per_target(rows, base, m), rng)
        if d:
            out[m] = dict(d, pass_=d["ci95"][0] <= 0 <= d["ci95"][1])
            ok &= out[m]["pass_"]
    sb, so = success(rows, base), success(rows, other)
    keys = sorted(set(sb) & set(so))
    if len(keys) >= 5:
        m, lo, hi = boot_mean([so[k] - sb[k] for k in keys], rng)
        out["success_rate"] = {"mean": m, "ci95": [lo, hi], "n": len(keys), "pass_": lo <= 0 <= hi}
        ok &= out["success_rate"]["pass_"]
    return {"verdict": ("PASS" if ok else "FAIL") if out else "INSUFFICIENT", "metrics": out}
