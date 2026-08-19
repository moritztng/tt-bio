"""Close the FreeBindCraft wall split honestly: no double counting, compile separated from compute.

`parse_fbc_run.py` sums each instrumented stage independently, which is wrong in two ways that
only show up once you have a real run to look at:

  1. `pr_relax` is called from *inside* `predict_binder_complex` (12 of 18 relaxes on the run of
     2026-08-19), so the AF2 validation figure carries OpenMM relax time inside it. Summing both
     stages counts those seconds twice and pushes the "portable inference" share up by a third.
  2. Every trajectory draws a fresh binder length, so JAX recompiles for the new shapes. The first
     call of a stage at a new length costs one to two orders of magnitude more than the next one
     (`Stage 1 logits` ~200 s against 0.6 s/iteration immediately afterwards in the same
     trajectory). Compile is real wall clock and must be reported, but it is not the arithmetic a
     port would move to another device, so it cannot sit inside the compute shares.

This script rebuilds the split from the interval timestamps in `stages.jsonl` (start = t_end - s),
which makes nesting visible instead of invisible, and reports compile and compute separately.

    python perf/freebindcraft/analyze_measured_split.py --run-dir /path/to/out \
        --report /path/to/measured_split.json
"""

import argparse
import collections
import json
import pathlib
import re

AF2_LINE = re.compile(r"\[AF2\] (.+?) in ([0-9.]+)s")
GRADIENT = ["Stage 1 logits", "Additional logits", "Softmax stage", "One-hot stage"]
FORWARD = ["PSSM semigreedy"]
# The two AF2 stages whose first call at a new binder length pays for an XLA compilation. The
# bimodality is stark (0.4-1.5 s against 29-68 s), so any threshold in the gap gives the same
# answer; 20 s is the midpoint of the empty interval on the run this was written against.
COMPILE_THRESHOLD_S = 20.0
INFERENCE_STAGES = ("predict_binder_complex", "predict_binder_alone")


def load_intervals(run_dir):
    rows = [json.loads(l) for l in (run_dir / "stages.jsonl").read_text().splitlines() if l.strip()]
    return [{"stage": r["stage"], "s": r["s"], "t0": r["t_end"] - r["s"], "t1": r["t_end"]} for r in rows]


def nested_seconds(intervals):
    """Seconds of each (inner, outer) stage pair where inner sits wholly inside outer."""
    out = collections.defaultdict(float)
    for i, a in enumerate(intervals):
        for j, b in enumerate(intervals):
            if i == j or b["s"] <= a["s"]:
                continue
            if b["t0"] <= a["t0"] + 1e-6 and a["t1"] <= b["t1"] + 1e-6:
                out[(a["stage"], b["stage"])] += a["s"]
                break
    return out


def union_seconds(intervals):
    merged = []
    for iv in sorted(intervals, key=lambda x: x["t0"]):
        if merged and iv["t0"] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], iv["t1"])
        else:
            merged.append([iv["t0"], iv["t1"]])
    return sum(b - a for a, b in merged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    run_dir = pathlib.Path(args.run_dir)

    iv = load_intervals(run_dir)
    span = max(x["t1"] for x in iv) - min(x["t0"] for x in iv)
    by_stage = collections.defaultdict(float)
    calls = collections.Counter()
    for x in iv:
        by_stage[x["stage"]] += x["s"]
        calls[x["stage"]] += 1

    nest = nested_seconds(iv)
    # Charge nested seconds to the inner stage, which is the one that actually did the work.
    relax_inside_predict = sum(v for (inner, _), v in nest.items() if inner == "pr_relax")
    align_inside_predict = sum(v for (inner, _), v in nest.items() if inner == "align_pdbs")

    # AF2 hallucination decomposition from the verbose timers.
    log = (run_dir / "run.log").read_text(errors="ignore")
    af2 = collections.defaultdict(list)
    for label, secs in AF2_LINE.findall(log):
        af2[label].append(float(secs))
    tot = lambda keys: sum(sum(v) for k, v in af2.items() if any(m in k for m in keys))
    gradient_s = tot(GRADIENT)
    semigreedy_s = tot(FORWARD)

    # Compile inside the gradient loop: the first design_logits call of each trajectory pays for a
    # new-shape compilation. Price the 50 Stage-1 iterations at the per-iteration cost measured by
    # `Additional logits` (25 iterations) in the same trajectory; the excess is first-call overhead.
    stage1 = af2.get("Stage 1 logits", [])
    addl = af2.get("Additional logits", [])
    per_iter = [a / 25.0 for a in addl]
    mean_per_iter = sum(per_iter) / len(per_iter) if per_iter else 0.0
    grad_compile_s = 0.0
    for k, s1 in enumerate(stage1):
        rate = per_iter[k] if k < len(per_iter) else mean_per_iter
        grad_compile_s += max(0.0, s1 - 50.0 * rate)

    # Compile inside the validation inference: bimodal per-call cost, one spike per new length.
    inf_compile_s = 0.0
    inf_compute_s = 0.0
    inf_compile_calls = 0
    for x in iv:
        if x["stage"] not in INFERENCE_STAGES:
            continue
        s = x["s"]
        if x["stage"] == "predict_binder_complex":
            s -= 0.0  # nested relax removed in aggregate below, not per call
        if s > COMPILE_THRESHOLD_S:
            inf_compile_s += s
            inf_compile_calls += 1
        else:
            inf_compute_s += s
    # The nested relax seconds sit inside the big predict_binder_complex calls, so they were just
    # counted into inf_compile_s. Take them back out; they belong to relax.
    inf_compile_s -= relax_inside_predict

    validation_af2_s = by_stage["predict_binder_complex"] + by_stage["predict_binder_alone"] \
        - relax_inside_predict - align_inside_predict
    relax_s = by_stage["pr_relax"]
    scoring_s = by_stage["score_interface"]
    mpnn_s = by_stage["mpnn_gen_sequence"]
    halluc_s = by_stage["binder_hallucination"]
    halluc_other_s = halluc_s - gradient_s - semigreedy_s
    covered = union_seconds(iv)
    uninstrumented_s = span - covered

    pct = lambda x: round(100.0 * x / span, 1)
    rep = {
        "instrumented_span_s": round(span, 1),
        "naive_sum_of_stages_s": round(sum(by_stage.values()), 1),
        "double_counted_s": {
            "pr_relax_inside_predict_binder_complex": round(relax_inside_predict, 1),
            "align_pdbs_inside_predict_binder_alone": round(align_inside_predict, 1),
        },
        "calls": dict(calls),
        "split_s": {
            "hallucination_gradient": round(gradient_s, 1),
            "hallucination_semigreedy_forward": round(semigreedy_s, 1),
            "hallucination_other": round(halluc_other_s, 1),
            "validation_af2_inference": round(validation_af2_s, 1),
            "proteinmpnn": round(mpnn_s, 1),
            "openmm_relax": round(relax_s, 1),
            "scoring": round(scoring_s, 1),
            "uninstrumented_host": round(uninstrumented_s, 1),
        },
        "compile_vs_compute_s": {
            "xla_compile_in_gradient_loop": round(grad_compile_s, 1),
            "xla_compile_in_validation_inference": round(inf_compile_s, 1),
            "validation_inference_compute": round(inf_compute_s, 1),
        },
    }
    rep["split_pct"] = {k: pct(v) for k, v in rep["split_s"].items()}
    rep["closes_to_pct"] = round(sum(rep["split_pct"].values()), 1)
    rep["gradient_pct"] = pct(gradient_s)
    # What a ttnn port could actually take: forward AF2 inference plus ProteinMPNN, compile excluded.
    rep["ttnn_portable_compute_s"] = round(inf_compute_s + mpnn_s, 1)
    rep["ttnn_portable_compute_pct"] = pct(inf_compute_s + mpnn_s)
    rep["ttnn_portable_including_compile_pct"] = pct(validation_af2_s + mpnn_s)

    pathlib.Path(args.report).write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
