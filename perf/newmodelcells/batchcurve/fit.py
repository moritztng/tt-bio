#!/usr/bin/env python3
"""Fit per-design seconds at N_step=400 from two cheap step counts, per batch rung.

t_call(b, n) = F(b) + n * P(b): the sampler has no data-dependent control flow, so a call is a
fixed conditioning hoist plus n identical steps. Two step counts determine both terms, and at
b=1 the fit reproduces the independently measured 400-step number to 0.7 %, which is what
licenses the extrapolation for the rungs a 400-step run would cost half an hour each.

    python3 perf/newmodelcells/batchcurve/fit.py perf/newmodelcells/batchcurve --target 400
"""
import argparse, json, pathlib, re, statistics


def warm(rec, key):
    v = [d[key] for d in rec["designs"] if not d["cold"]]
    return statistics.median(v) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--target", type=int, default=400)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    runs = {}
    for p in sorted(pathlib.Path(a.dir).glob("*.json")):
        r = json.loads(p.read_text())
        if "designs" not in r or not r.get("warm_n"):
            continue
        runs.setdefault(r["n_sample_per_call"], {})[r["n_step"]] = (r, p.name)

    rows = []
    for b in sorted(runs):
        steps = sorted(runs[b])
        if len(steps) < 2:
            print(f"b={b}: only step counts {steps}, need two — skipped")
            continue
        n1, n2 = steps[0], steps[-1]
        d1, d2 = warm(runs[b][n1][0], "t_design_s"), warm(runs[b][n2][0], "t_design_s")
        h = statistics.median([warm(runs[b][n][0], "t_feat_s") + warm(runs[b][n][0], "t_write_s")
                               for n in (n1, n2)])
        P = (d2 - d1) / (n2 - n1)
        F = d1 - n1 * P
        call = F + a.target * P + h
        rows.append({"batch": b, "n_step_fit": [n1, n2], "F_s": round(F, 4),
                     "P_s_per_step": round(P, 6), "host_s": round(h, 4),
                     "call_s_at_target": round(call, 3),
                     "s_per_design": round(call / b, 4),
                     "distinct": all(runs[b][n][0].get("all_designs_distinct") for n in steps)})
    base = rows[0]["s_per_design"] if rows else None
    for r in rows:
        r["amortisation_x"] = round(base / r["s_per_design"], 4)
    best = min(rows, key=lambda r: r["s_per_design"]) if rows else None
    out = {"target_n_step": a.target, "rungs": rows,
           "best_batch": best["batch"] if best else None,
           "best_s_per_design": best["s_per_design"] if best else None,
           "best_amortisation_x": best["amortisation_x"] if best else None}
    print(json.dumps(out, indent=1))
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
