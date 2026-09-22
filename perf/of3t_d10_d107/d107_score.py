#!/usr/bin/env python3
"""D107 scored against a FLOAT64 reference, which is the half `of3t-optsem` left open.

`of3t-optsem` closed D107 against upstream's own optimizer at `reference_dtype: float32`
(`perf/of3t_optsem/SUMMARY.json`). Two fp32 arms agreeing says the two rules match to fp32;
it cannot say that the surviving 2.0e-08 is our fp32 master rather than a second, smaller
divergence hiding inside the reference's own rounding. This runs the same pipeline with the
reference in float64 and adds a second, independently written float64 reference so the
reference itself is checked rather than trusted.

Three references, same gradient file, same participation pattern:

  up64   upstream's `PerSampleGradManager` + `torch.optim.Adam`, parameters float64,
         imported from the 0.4.3 checkout by path. Their code, not a reimplementation.
  up32   the same, parameters float32 -- `of3t-optsem`'s reference, kept so the reading
         can be split into "our fp32 master" and "the reference's fp32 rounding".
  np64   an independent float64 Adam written here from their update rule. It exists to
         catch a harness error in `up64`: two implementations that agree to 1e-16 are a
         reference, one implementation is a hope.

One caveat that belongs next to the word float64: upstream's `compute_global_norm` casts
each gradient with `.float()` before norming (`grad_manager.py:50`), so the CLIP COEFFICIENT
is computed in fp32 even when the parameters are float64. `np64` mirrors that exactly rather
than quietly improving on it -- the reference is upstream's rule, not a better one.
"""
import argparse
import json
import pathlib

import numpy as np

CONF = ("conf.w", "conf.b")
LR, BETA1, BETA2, EPS = 1.8e-3, 0.9, 0.95, 1e-8
CLIP_VAL = 10.0


def np64_arm(grads_path):
    """An independent float64 trajectory of upstream's rule.

    Per sample: global norm over the ENABLED gradients only (`_clip_grads` drops
    `disabled_params` before norming, :140-147), clip by `max/max(norm, max)` (:168-170),
    accumulate. Per step: divide by that parameter's OWN participation count, or zero it
    when the count is 0 (:232-239), then a plain Adam step with no weight decay
    (`runner.py:845-850`). The norm and the coefficient are float32, as theirs are.
    """
    data = np.load(grads_path)
    enabled = data["enabled"]
    steps, samples = enabled.shape
    names = sorted(k.split("/", 1)[1] for k in data.files if k.startswith("init/"))
    theta = {n: data[f"init/{n}"].astype(np.float64) for n in names}
    grads = {n: data[f"grad/{n}"] for n in names}
    m = {n: np.zeros_like(theta[n]) for n in names}
    v = {n: np.zeros_like(theta[n]) for n in names}
    b1p = b2p = 1.0
    trace, clipped = [], 0
    for k in range(steps):
        accum = {n: np.zeros_like(theta[n]) for n in names}
        count = {n: 0 for n in names}
        for s in range(samples):
            disabled = set() if enabled[k, s] else set(CONF)
            live = [n for n in names if n not in disabled]
            gnorm = np.float32(np.sqrt(sum(
                float(np.linalg.norm(grads[n][k, s].astype(np.float32))) ** 2
                for n in live)))
            coef = np.float32(CLIP_VAL) / np.maximum(gnorm, np.float32(CLIP_VAL))
            if float(coef) < 1.0:
                clipped += 1
            for n in live:
                accum[n] += grads[n][k, s].astype(np.float64) * np.float64(coef)
                count[n] += 1
        b1p *= BETA1
        b2p *= BETA2
        for n in names:
            g = accum[n] / count[n] if count[n] > 0 else np.zeros_like(theta[n])
            m[n] = BETA1 * m[n] + (1.0 - BETA1) * g
            v[n] = BETA2 * v[n] + (1.0 - BETA2) * (g * g)
            theta[n] = theta[n] - LR * (m[n] / (1.0 - b1p)) / (
                np.sqrt(v[n] / (1.0 - b2p)) + EPS)
        trace.append({"k": k + 1,
                      "participation": {n: int(count[n]) for n in names},
                      "theta": {n: theta[n].tolist() for n in names}})
    return {"reference": "independent float64 Adam (this file)", "dtype": "float64",
            "samples_clipped": clipped, "samples_total": steps * samples, "trace": trace}


def _theta(run, k, n):
    return np.asarray(run["trace"][k]["theta"][n], dtype=np.float64)


def rel(a, b):
    """||a - b|| / ||b||, the per-parameter per-step relative deviation."""
    d = float(np.linalg.norm(a - b))
    s = float(np.linalg.norm(b))
    return d / s if s > 0 else float("nan")


def score(arm, ref, init, label, ref_label):
    names = sorted(arm["trace"][0]["theta"])
    rows = []
    for k in range(len(arm["trace"])):
        for n in names:
            r = _theta(ref, k, n)
            rows.append({
                "k": k + 1, "param": n,
                "participation": int(ref["trace"][k]["participation"][n]),
                "rel": rel(_theta(arm, k, n), r),
                # What a parameter that never moved at all would read at this step. A
                # reading only means something when it is well below its own zero baseline.
                "zero_baseline": rel(init[n], r)})
    worst = max(rows, key=lambda r: r["rel"])
    zero_rows = [r for r in rows if r["participation"] == 0]
    return {"arm": label, "reference": ref_label,
            "worst_rel": worst["rel"], "worst_row": {k: worst[k] for k in
                                                     ("k", "param", "participation")},
            "worst_zero_baseline": worst["zero_baseline"],
            "worst_rel_at_zero_participation": (max(r["rel"] for r in zero_rows)
                                                if zero_rows else None),
            "n_readings": len(rows),
            "n_over_1e-06": sum(1 for r in rows if r["rel"] > 1e-6),
            "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--scenario", default="single")
    a = ap.parse_args()
    d = pathlib.Path(a.outdir)
    sc = a.scenario
    grads = d / f"grads_{sc}.npz"
    init = {k.split("/", 1)[1]: np.load(grads)[k].astype(np.float64)
            for k in np.load(grads).files if k.startswith("init/")}

    up64 = json.load(open(d / f"up64_{sc}.json"))
    up32 = json.load(open(d / f"up32_{sc}.json"))
    np64 = np64_arm(grads)
    json.dump(np64, open(d / f"np64_{sc}.json", "w"))

    out = {"scenario": sc, "bar": "1e-06 relative per parameter per step",
           "grads": str(grads), "upstream_file": up64["reference"],
           "samples_clipped": np64["samples_clipped"],
           "samples_total": np64["samples_total"],
           "zero_participation_steps": [t["k"] for t in up64["trace"]
                                        if t["participation"]["conf.b"] == 0],
           "arms": []}
    # 1. Is the reference a reference? Two independent float64 implementations of the
    #    same rule, on the same values.
    out["reference_selfcheck"] = score(np64, up64, init, "np64 (independent float64)",
                                       "up64 (upstream float64)")
    # 2. How much of any reading is the reference's own fp32 rounding?
    out["reference_fp32_cost"] = score(up32, up64, init, "up32 (upstream float32)",
                                       "up64 (upstream float64)")
    for label in ("ours_pre", "ours_post"):
        run = json.load(open(d / f"{label}_{sc}.json"))
        out["arms"].append(score(run, up64, init, label, "up64 (upstream float64)"))
    # The sharp form of the post-fix result: our arm against upstream's arm AT OUR OWN
    # PRECISION. If the whole residual against float64 is the fp32 arithmetic upstream
    # also does, this reads zero rather than a small number.
    out["arms"].append(score(json.load(open(d / f"ours_post_{sc}.json")), up32, init,
                             "ours_post", "up32 (upstream float32, our own precision)"))
    for arm in [out["reference_selfcheck"], out["reference_fp32_cost"]] + out["arms"]:
        arm.pop("rows")
    json.dump(out, open(d / f"SCORE_{sc}.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
