"""Grade an interleaved trajectories-per-card A/B: python3 bc2_concur_grade.py <raw dir>

Each chunk directory holds one bench.py campaign (bench.jsonl's `ev=campaign` line), its leg.txt
(`cpus=`, `seed=`, `N=`) and cpu_sampler.py's cpu.tsv. Chunks pair by (core budget, seed): the A and
B of one pair ran back to back on the same cores and the same campaign seed. Per chunk the metric is
campaign wall net of the predictor build and net of validation, per campaign iteration.

Validation is why a raw campaign wall cannot grade this: each trajectory that passes the screen refolds its MPNN
sequences at ~100 s a refold, and two chunks of a pair do not refold the same number of times (5 to 20 in the
2026-10-22 pc chain), which swamps the effect under test. Refolds are counted from the campaign's own
`2_Refolded/!_Refolded.csv`, one row per refold, and their cost `b` is fitted over the chain by least squares on
wall ~ iterations + refolds + isN2. `s_it_net` = (wall - setup - b*refolds) / iterations is the graded metric;
the raw per-iteration wall is reported beside it. hidden = 1 - s_B / s_A per pair,
median over pairs per budget. A pair is dropped when its two chunks saw other-process load on the
pinned set differing by more than 20 % and by more than 0.5 cores (the floor keeps a 0.1 vs 0.2 core
idle box from counting as a 2x swing).
"""
import csv, json, statistics, sys
from pathlib import Path


def refolds(d):
    """Rows of the campaign's refolded table: one per MPNN sequence refolded and validated."""
    f = next(iter(sorted(d.glob("campaign_s*/2_Refolded/!_Refolded.csv"))), None)
    if f is None:
        return 0
    with open(f) as fh:
        return sum(1 for _ in csv.DictReader(fh))


def fit_refold_cost(C):
    """Least squares of wall on (iterations, refolds, isN2, 1), returning the refold coefficient.

    Falls back to the A legs' own measured validation time per refold when the fit is degenerate."""
    X = [[c["iters"], c["refolds"], float(c["N"] > 1), 1.0] for c in C]
    y = [c["wall"] - (c["setup"] or 0) for c in C]
    beta = _lstsq(X, y)
    if beta is not None and 20.0 <= beta[1] <= 400.0:
        return beta[1], "fit"
    v = [c["validate_s"] / c["refolds"] for c in C if c["N"] == 1 and c["refolds"] and c.get("validate_s")]
    return (statistics.median(v), "validate_s") if v else (0.0, "none")


def _lstsq(X, y):
    """Normal equations with Gauss-Jordan; None if singular or underdetermined."""
    k = len(X[0])
    if len(X) <= k:
        return None
    A = [[sum(r[i] * r[j] for r in X) for j in range(k)] + [sum(r[i] * t for r, t in zip(X, y))] for i in range(k)]
    for i in range(k):
        pv = max(range(i, k), key=lambda r: abs(A[r][i]))
        if abs(A[pv][i]) < 1e-9:
            return None
        A[i], A[pv] = A[pv], A[i]
        A[i] = [v / A[i][i] for v in A[i]]
        for r in range(k):
            if r != i and A[r][i]:
                A[r] = [a - A[r][i] * b for a, b in zip(A[r], A[i])]
    return [A[i][k] for i in range(k)]


def chunk(d):
    meta = dict(kv.split("=", 1) for kv in (d / "leg.txt").read_text().split("\n", 1)[0].split())
    rows = [json.loads(x) for x in open(d / "bench.jsonl") if x.startswith("{")]
    c = next((r for r in rows if r.get("ev") == "campaign"), None)
    if c is None or c.get("err") or not c.get("iterations"):
        return None
    cpu = [l.split("\t") for l in (d / "cpu.tsv").read_text().splitlines()[1:]]
    other = statistics.mean(float(r[4]) for r in cpu) if cpu else None
    clk = next(iter((c.get("aiclk") or {}).values()), {})
    net = c["wall_s"] - (c.get("setup_s") or 0)
    reps = [r for r in rows if r.get("ev") == "rep"]
    return dict(name=d.name, cpus=meta["cpus"], seed=meta["seed"], N=int(meta["N"]), wall=c["wall_s"], setup=c.get("setup_s"),
                refolds=refolds(d), validate_s=sum(r.get("validate_s") or 0 for r in reps) or None,
                design_s=sum(r.get("design_s") or 0 for r in reps) or None, reps=len(reps),
                iters=c["iterations"], s_it_wall=net / c["iterations"], s_it_gross=c["wall_per_iter"], other=other,
                load=c.get("loadavg"), aiclk_med=clk.get("median"), aiclk_min=clk.get("min"), sha=c.get("sha"),
                nval=sum(1 for t in c["trajectories"] if t.get("terminated") is None),
                its=[t["iterations"] for t in c["trajectories"]])


raw = Path(sys.argv[1])
C = [c for d in sorted(raw.iterdir()) if (d / "done").exists() and (c := chunk(d))]
b, b_from = fit_refold_cost(C)
for c in C:
    c["s_it"] = (c["wall"] - (c["setup"] or 0) - b * c["refolds"]) / c["iters"]
out = {"refold_cost_s": round(b, 1), "refold_cost_from": b_from, "metric": "(wall - setup - refold_cost_s*refolds)/iterations",
       "chunks": C, "budgets": {}}
for cpus in sorted({c["cpus"] for c in C}, key=lambda s: int(s.split("-")[-1])):
    pairs = []
    for seed in sorted({c["seed"] for c in C if c["cpus"] == cpus}):
        a = [c for c in C if c["cpus"] == cpus and c["seed"] == seed and c["N"] == 1]
        bs = [c for c in C if c["cpus"] == cpus and c["seed"] == seed and c["N"] > 1]
        if not (a and bs):
            continue
        a, bb = a[0], bs[0]
        d = abs(a["other"] - bb["other"])
        keep = not (d > 0.2 * max(a["other"], bb["other"]) and d > 0.5)
        pairs.append(dict(seed=seed, s_A=round(a["s_it"], 4), s_B=round(bb["s_it"], 4), hidden=round(1 - bb["s_it"] / a["s_it"], 4),
                          s_A_wall=round(a["s_it_wall"], 4), s_B_wall=round(bb["s_it_wall"], 4),
                          hidden_wall=round(1 - bb["s_it_wall"] / a["s_it_wall"], 4),
                          refolds=[a["refolds"], bb["refolds"]],
                          other_A=round(a["other"], 2), other_B=round(bb["other"], 2), kept=keep))
    kept = [p["hidden"] for p in pairs if p["kept"]]
    out["budgets"][cpus] = dict(cores=len(range(int(cpus.split("-")[0]), int(cpus.split("-")[-1]) + 1)), pairs=pairs,
                                hidden_median=statistics.median(kept) if kept else None, n_kept=len(kept),
                                s_A_median=statistics.median(p["s_A"] for p in pairs if p["kept"]) if kept else None,
                                s_B_median=statistics.median(p["s_B"] for p in pairs if p["kept"]) if kept else None)
json.dump(out, sys.stdout, indent=1)
print()
