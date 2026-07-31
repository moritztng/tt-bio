#!/usr/bin/env python3
"""AbAg-XM oracle-saturation-depth analysis (state doc abag-xm-oracle-saturation-depth §4).

Inputs (all measured):
  labels    ~/abag_xm/saturation/<model_dir>/<T>[_c<j>]/labels.json  (per-sample DockQ)
  conf      <out_dir>/<prefix>_results_<T>/results.json [0].all_runs (rank -> confidence_score)
  progress  ~/abag_xm/saturation/progress.jsonl + progress_qb2.jsonl (per-job wall_s)

Outputs ~/abag_xm/saturation/analysis.json + a markdown §7 body on stdout.

Oracle (exact hypergeometric, threshold th): target with S successes in N=1000 samples
contributes 1 - C(N-S, m)/C(N, m) at sample count m; mean over targets. Grid:
{1,2,4,8,16,32,50,64,100,128,200,256,400,512,640,800,1000}. Bootstrap 95 pct CI over
targets (10k resamples, rng 7). Knee criterion (pre-registered §4): saturation at m* =
smallest m with per-doubling gain < 1.0 pp over [m*, 2m*] AND the next doubling's
bootstrap CI lower bound < 1.0 pp; if the 800->1000 gain exceeds that, the verdict is
"no saturation found by N=1000".

Ranked top-1 vs N: per (target, m), R=1000 subsamples without replacement (deterministic
rng "12345|target|m"); success = DockQ of the subsample's max-confidence sample >= th;
fraction over reps, mean over targets. m=1000 is the full ensemble (single rep).

Cost: per model, fixed + marginal least-squares over ok progress records; per-target
marginal from measured total walls; cost_t(m) = fixed + m*marg_t. Marginal-oracle table:
dO/dc (pp per 1000 card-s) per grid interval, and O at budgets {5,10,20,40,80} card-ks
per target.
"""
import json, math, random, statistics
from functools import lru_cache
from pathlib import Path

BASE = Path.home() / "abag_xm" / "saturation"
TARGETS = ["9q6y", "9tmp", "9gei", "9fte", "9wpm", "9qrv", "9ma0", "9q6z", "9uoi",
           "9m8l", "9ldx", "9nl0", "9l9y", "9mnu", "9gfr", "9zen"]
CONT11 = TARGETS[:11]  # frontier continuity panel (9j4c excluded by design, §3)
THRESHOLDS = [0.23, 0.49, 0.80]
GRID = [1, 2, 4, 8, 16, 32, 50, 64, 100, 128, 200, 256, 400, 512, 640, 800, 1000]
MODELS = {"opendde": "opendde", "protenix": "protenix", "boltz2": "boltz2"}  # dir -> prefix
BOOT_REPS = 10000
RANKED_REPS = 1000
KNEE_PP = 1.0  # pre-registered per-doubling gain threshold, percentage points
# §0 label-derived Arm-A oracle @0.23 (n=11), the G3 continuity reference.
S0_CONT = {50: 42.8, 100: 53.4, 200: 63.6}
TRUNK_FIXED_S = 342.0  # measured trunk-pass cost (frontier cost fit); not refitted here


def comb(n, k):
    return math.comb(n, k) if 0 <= k <= n else 0


@lru_cache(maxsize=None)
def hyper_oracle(n, s, m):
    """P(at least one success among m draws without replacement).

    Cached: the bootstrap resamples targets, not samples, so it asks for the same
    (n, s, m) tens of thousands of times, and each miss is a ratio of ~300-digit
    binomials. Pure function of its arguments, so the cache cannot change a result.
    """
    return 1.0 - comb(n - s, m) / comb(n, m) if n > 0 else 0.0


def load_target(model_dir, prefix, target):
    """Merged (dockq, conf) sample lists over the target's chunk dirs. None on missing/
    incomplete labeling — G4: never quote a target with any sample dockq None."""
    chunks = sorted(BASE.glob(f"{model_dir}/{target}_c*")) or [BASE / model_dir / target]
    if not all(c.is_dir() for c in chunks):
        return None
    out = []
    for cdir in chunks:
        lp = cdir / "labels.json"
        rp = cdir / f"{prefix}_results_{target}" / "results.json"
        if not lp.exists():
            return None
        d = json.loads(lp.read_text())
        conf = {}
        if rp.exists():
            try:
                rj = json.loads(rp.read_text())
                rj = rj[0] if isinstance(rj, list) else rj
                conf = {r.get("rank"): r.get("confidence_score")
                        for r in rj.get("all_runs", [])}
            except Exception:
                pass
        for s in d["samples"]:
            v = s.get("dockq")
            dq = v.get("dockq") if isinstance(v, dict) else None
            if dq is None:
                return None
            out.append((float(dq), conf.get(s.get("rank"))))
    return out


def load_model(model_dir):
    prefix = MODELS[model_dir]
    per = {}
    for t in TARGETS:
        samples = load_target(model_dir, prefix, t)
        if samples is not None:
            per[t] = samples
    return per


_SUCC = {}


def successes(per, t, thr):
    key = (id(per), t, thr)
    if key not in _SUCC:
        dq = [x[0] for x in per[t]]
        _SUCC[key] = (len(dq), sum(1 for v in dq if v >= thr))
    return _SUCC[key]


def mean_oracle(per, targets, m, thr):
    vals = [hyper_oracle(*successes(per, t, thr), m) for t in targets]
    return statistics.mean(vals), vals


def boot_ci(fn, targets, rng_seed=7, reps=BOOT_REPS, clusters=None):
    """Bootstrap over targets, or over CLUSTERS of targets when given.

    Two targets that fold a byte-identical input (9q6y/9q6z) share every prediction, so
    resampling them independently counts one piece of evidence twice and narrows the CI.
    Passing `clusters` (a list of target lists) resamples whole clusters instead.
    """
    rng = random.Random(rng_seed)
    point = fn(targets)
    units = clusters if clusters is not None else [[t] for t in targets]
    boot = []
    for _ in range(reps):
        draw = []
        for _ in range(len(units)):
            draw += rng.choice(units)
        boot.append(fn(draw))
    boot.sort()
    return point, boot[int(0.025 * reps)], boot[int(0.975 * reps) - 1]


def clusters_of(labeled, groups):
    """Partition `labeled` into evidence units: each duplicate group is one unit."""
    dup = {t: tuple(g) for g in groups for t in g}
    seen, units = set(), []
    for t in labeled:
        key = dup.get(t, (t,))
        if key not in seen:
            seen.add(key)
            units.append(list(key))
    return units


def oracle_block(per, thr, groups=()):
    targets = sorted(per)
    curve, per_target = {}, {}
    for m in GRID:
        mean, vals = mean_oracle(per, targets, m, thr)
        curve[m] = mean
        per_target[m] = dict(zip(targets, vals))
    ci = {m: boot_ci(lambda ts, m=m: mean_oracle(per, ts, m, thr)[0], targets)[1:]
          for m in GRID}
    out = {"mean": {m: round(curve[m], 4) for m in GRID},
           "ci95": {m: [round(x, 4) for x in ci[m]] for m in GRID},
           "per_target": {m: {t: round(v, 4) for t, v in per_target[m].items()}
                          for m in GRID},
           "n_targets": len(targets)}
    if groups:
        # Duplicate-input-aware variants, so §7 can quote either treatment without a re-run.
        units = clusters_of(targets, groups)
        reps_ = [u[0] for u in units]
        out["ci95_clustered"] = {
            m: [round(x, 4) for x in
                boot_ci(lambda ts, m=m: mean_oracle(per, ts, m, thr)[0], targets,
                        clusters=units)[1:]]
            for m in GRID}
        out["mean_distinct_inputs"] = {
            m: round(mean_oracle(per, reps_, m, thr)[0], 4) for m in GRID}
        out["n_distinct_inputs"] = len(units)
    return out


def doubling_gains(curve_mean):
    """Exact-doubling pairs present in the grid -> (m, 2m, gain_pp)."""
    out = []
    ms = sorted(curve_mean)
    for m in ms:
        if 2 * m in curve_mean:
            out.append((m, 2 * m, 100.0 * (curve_mean[2 * m] - curve_mean[m])))
    return out


def knee_verdict(per, thr):
    """Pre-registered criterion (§4). Returns dict with m* or no-saturation statement."""
    targets = sorted(per)
    means = {m: mean_oracle(per, targets, m, thr)[0] for m in GRID}
    pairs = doubling_gains(means)
    # bootstrap CI of each doubling's gain over target resamples
    def gain_fn(ts, m):
        a = mean_oracle(per, ts, m, thr)[0]
        b = mean_oracle(per, ts, 2 * m, thr)[0]
        return 100.0 * (b - a)
    gain_ci = {m: boot_ci(lambda ts, m=m: gain_fn(ts, m), targets)
               for m, _, _ in pairs}
    mstar = None
    for m, m2, g in pairs:
        nxt = next(((a, b, gg) for a, b, gg in pairs if a == m2), None)
        if g < KNEE_PP and nxt is not None and gain_ci[m2][1] < KNEE_PP:
            mstar = m
            break
    # Grid-edge hole in the pre-registered criterion: confirming m* needs the NEXT doubling
    # [2m, 4m] to exist on the grid, and the grid stops at 1000. So the last confirmable
    # candidate is m=200 (confirmed by [400, 800]); m=256 and m=400 can satisfy the
    # gain < 1 pp test yet never be confirmable. Reporting those as "no saturation" would be
    # false when the measured curve has visibly flattened, so they get their own state: the
    # smallest m whose own doubling gain is under the bar but whose confirmation is off-grid.
    edge = None
    if mstar is None:
        edge = next((m for m, m2, g in pairs
                     if g < KNEE_PP
                     and not any(a == m2 for a, _, _ in pairs)), None)
    last = means[GRID[-1]] - means[GRID[-2]]
    last_pp = 100.0 * last
    return {"m_star": mstar,
            "m_star_edge_unconfirmable": edge,
            "last_confirmable_m": max((m for m, m2, _ in pairs
                                       if any(a == m2 for a, _, _ in pairs)), default=None),
            "doubling_gains_pp": [[m, m2, round(g, 2)] for m, m2, g in pairs],
            "gain_ci95_pp": {m: [round(x, 2) for x in gain_ci[m]] for m, _, _ in pairs},
            "final_interval": [GRID[-2], GRID[-1], round(last_pp, 2)],
            "final_interval_per_doubling_pp": round(
                last_pp / math.log2(GRID[-1] / GRID[-2]), 2),
            "verdict": (
                f"saturation at m*={mstar}" if mstar is not None else
                (f"flattens at m={edge} but the pre-registered confirmation is off-grid "
                 f"(needs the {2 * edge}->{4 * edge} doubling; grid stops at {GRID[-1]}); "
                 f"final interval {GRID[-2]}->{GRID[-1]} gain {last_pp:.2f} pp"
                 if edge is not None else
                 f"no saturation found by N=1000 (final interval "
                 f"{GRID[-2]}->{GRID[-1]} gain {last_pp:.2f} pp)"))}


def ranked_block(per, thr):
    targets = sorted(per)
    res = {}
    for m in GRID:
        probs = []
        for t in targets:
            samples = per[t]
            n = len(samples)
            if m >= n:
                scored = [x for x in samples if x[1] is not None]
                if not scored:
                    continue
                best = max(scored, key=lambda x: x[1])
                probs.append(1.0 if best[0] >= thr else 0.0)
                continue
            rng = random.Random(f"12345|{t}|{m}")
            hits = 0
            valid = 0
            for _ in range(RANKED_REPS):
                idx = rng.sample(range(n), m)
                sub = [samples[i] for i in idx if samples[i][1] is not None]
                if not sub:
                    continue
                valid += 1
                best = max(sub, key=lambda x: x[1])
                hits += 1.0 if best[0] >= thr else 0.0
            probs.append(hits / valid if valid else 0.0)
        res[m] = round(statistics.mean(probs), 4)
    return {"mean": res, "n_targets": len(targets), "reps": RANKED_REPS}


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


def cost_fit(recs, model_dir):
    """Per-target cost of 1000 samples, from measured walls and a measured trunk cost.

    Deliberately NOT a regression. Two fits were tried and both are unsound here:
      * wall ~ n_samples ACROSS targets is confounded — only the 716-residue targets were
        chunked to 500, so sample count tracks target size and the slope collapses (it
        produced a 8234 s intercept on qb2 and a NEGATIVE marginal for 9zen);
      * a two-point solve against the frontier's 200-sample walls is invalid across
        campaigns — 9tmp cost 3005 s at 200 samples and 20309 s at 1000, i.e. 6.8x the
        wall for 5x the samples, because the two ran at different host contention. It
        implies a negative fixed cost.
    So take the trunk pass as the measured constant it is (TRUNK_FIXED_S, the frontier's own
    cost fit) and derive each target's marginal from its measured total. The trunk is
    342 s against 4k-21k s totals, so the marginal is insensitive to it; a chunked target
    pays the trunk once per invocation.
    """
    model_id = {"opendde": "opendde-abag", "protenix": "protenix-v2",
                "boltz2": "boltz2"}[model_dir]
    # Campaign folds only. The G2 determinism smoke wrote three 10-sample opendde records
    # (and any future smoke would too); they are ok-status records of the right model, so
    # without this filter they enter the model and inflate 9zen to 4 "invocations".
    ok = [r for r in recs if r.get("status") == "ok" and r.get("model") == model_id
          and r.get("n_samples", 0) >= 500 and "/smoke/" not in (r.get("out_dir") or "")]
    seen, dedup = set(), []
    for r in ok:  # a record can arrive twice if a progress file is mirrored twice
        key = (r.get("host"), r.get("out_dir"), r.get("ts"))
        if key not in seen:
            seen.add(key)
            dedup.append(r)
    ok = dedup
    if not ok:
        return None
    per_t, hosts = {}, {}
    for t in TARGETS:
        rs = [r for r in ok if r["target"] == t]
        if not rs or sum(r["n_samples"] for r in rs) != 1000:
            continue  # partial target: no cost claim until all its samples are in
        total = sum(r["wall_s"] for r in rs)
        marg = (total - TRUNK_FIXED_S * len(rs)) / 1000.0
        per_t[t] = {"wall_s": round(total, 1), "invocations": len(rs),
                    "marg_s_per_sample": round(marg, 3),
                    "hosts": sorted({r.get("host") for r in rs})}
        for h in per_t[t]["hosts"]:
            hosts.setdefault(h, []).append(marg)
    if not per_t:
        return None
    margs = [v["marg_s_per_sample"] for v in per_t.values()]
    return {"fixed_s": TRUNK_FIXED_S,
            "fixed_s_source": "frontier measured cost fit (not refitted here)",
            "marginal_s_per_sample": round(statistics.mean(margs), 3),
            "marginal_range": [round(min(margs), 3), round(max(margs), 3)],
            "n_records": len(ok), "n_targets_costed": len(per_t),
            "total_card_s": round(sum(r["wall_s"] for r in ok), 1),
            "per_target": per_t,
            "by_host_mean_marginal": {h: round(statistics.mean(v), 3)
                                      for h, v in sorted(hosts.items())},
            "negative_marginals": [t for t, v in per_t.items()
                                   if v["marg_s_per_sample"] <= 0]}


def cost_block(per, fit, thr):
    if not fit:
        return None
    targets = [t for t in sorted(per) if t in fit["per_target"]]
    if not targets:
        return None
    fixed = fit["fixed_s"]

    def cost_mean(m):
        return statistics.mean(fixed + m * fit["per_target"][t]["marg_s_per_sample"]
                               for t in targets)

    curve = {m: mean_oracle(per, targets, m, thr)[0] for m in GRID}
    intervals = []
    for a, b in zip(GRID, GRID[1:]):
        dc = cost_mean(b) - cost_mean(a)
        do = 100.0 * (curve[b] - curve[a])
        intervals.append({"m": [a, b], "dO_pp": round(do, 2),
                          "d_cost_card_s": round(dc, 1),
                          "pp_per_1000_card_s": round(1000.0 * do / dc, 2) if dc else None})
    budgets = {}
    for b_ks in (5, 10, 20, 40, 80):
        B = b_ks * 1000.0
        os_ = []
        for t in targets:
            marg_t = fit["per_target"][t]["marg_s_per_sample"]
            m = int(min(1000, max(0, (B - fixed) // marg_t))) if marg_t > 0 else 0
            dq = [x[0] for x in per[t]]
            s = sum(1 for x in dq if x >= thr)
            os_.append(hyper_oracle(len(dq), s, m))
        budgets[b_ks] = round(statistics.mean(os_), 4)
    return {"cost_at_m": {m: round(cost_mean(m), 1) for m in GRID},
            "intervals": intervals,
            "oracle_at_budget_card_ks": budgets,
            "budget_note": "budget is card-s per target; m(B) = (B - fixed)/marg_t, clamped to 1000"}


def continuity_block(per):
    """G3: saturation-panel O(m) on the 11 continuity targets vs the §0 label-derived
    values (independent draws, same targets/checkpoint)."""
    targets = [t for t in CONT11 if t in per]
    if len(targets) < len(CONT11):
        return {"omitted": f"continuity panel incomplete: {len(targets)}/11 labeled"}
    out = {}
    for m, ref in S0_CONT.items():
        mean, _ = mean_oracle(per, targets, m, 0.23)
        _, lo, hi = boot_ci(lambda ts, m=m: mean_oracle(per, ts, m, 0.23)[0], targets)
        out[m] = {"saturation": round(100 * mean, 1), "ci95": [round(100 * lo, 1), round(100 * hi, 1)],
                  "s0_reference": ref, "agrees": lo * 100 <= ref <= hi * 100 or abs(mean * 100 - ref) < 5}
    return out


def duplicate_groups(labeled):
    """Targets among `labeled` whose fold INPUT is byte-identical (comments stripped).

    9q6y/9q6z are one such pair: same antigen+nanobody, different native. Their oracle
    curves therefore share every prediction and are not independent evidence for the panel
    mean or the bootstrap CI. Reported, not corrected — the treatment is a judgement call
    recorded in the state doc, and silently collapsing them would hide it.
    """
    import hashlib
    yaml_dir = Path(__file__).resolve().parent.parent / "examples" / "abag_xm"
    by_hash = {}
    for t in labeled:
        f = yaml_dir / f"{t}.yaml"
        if not f.exists():
            continue
        body = "\n".join(l for l in f.read_text().splitlines()
                          if not l.strip().startswith("#"))
        by_hash.setdefault(hashlib.md5(body.encode()).hexdigest(), []).append(t)
    return [sorted(v) for v in by_hash.values() if len(v) > 1]


def main():
    recs = progress_records()
    res = {"targets": TARGETS, "grid": GRID, "thresholds": THRESHOLDS}
    models = {}
    for model_dir in MODELS:
        per = load_model(model_dir)
        if not per:
            continue
        fit = cost_fit(recs, model_dir)
        dup = duplicate_groups(sorted(per))
        entry = {"g4_n_targets_labeled": len(per), "labeled_targets": sorted(per),
                 "cost": fit}
        for thr in THRESHOLDS:
            key = f"thr{thr}"
            # Every block is threshold-generic (thr is a parameter), so all three get the
            # full treatment: the deliverable asks for the knee verdict at 0.23/0.49/0.80,
            # and a ranked/budget curve at one threshold cannot answer it for the others.
            entry[key] = {"oracle": oracle_block(per, thr, dup),
                          "ranked_top1": ranked_block(per, thr),
                          "budget": cost_block(per, fit, thr),
                          "knee": knee_verdict(per, thr)}
        entry["duplicate_input_groups"] = dup
        entry["n_distinct_inputs"] = len(per) - sum(len(g) - 1 for g in dup)
        if model_dir == "opendde":
            entry["g3_continuity"] = continuity_block(per)
        models[model_dir] = entry
    res["models"] = models

    # Q2: per-generator depth payoff on the shared labeled subset.
    shared = None
    for md, entry in models.items():
        ts = set(entry["labeled_targets"])
        shared = ts if shared is None else shared & ts
    if shared and len(models) > 1:
        shared = sorted(shared)
        q2 = {"shared_targets": shared, "n_shared": len(shared)}
        for md, entry in models.items():
            per = load_model(md)
            o50 = mean_oracle(per, shared, 50, 0.23)[0]
            o1000 = mean_oracle(per, shared, 1000, 0.23)[0]
            q2[md] = {"oracle_50": round(o50, 4), "oracle_1000": round(o1000, 4),
                      "pp_per_doubling_50_1000": round(
                          100.0 * (o1000 - o50) / math.log2(1000.0 / 50.0), 2)}
        res["q2_generators"] = q2

    (BASE / "analysis.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
