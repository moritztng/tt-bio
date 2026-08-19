"""Turn one instrumented FreeBindCraft run into the wall split the feasibility gate asks for.

SUPERSEDED for the split itself by `analyze_measured_split.py`. This script sums each stage
independently, and on a real run `pr_relax` turns out to be called from inside
`predict_binder_complex`, so it counts those seconds twice and reports a negative `unattributed_s`.
Kept because its `[AF2]`-timer decomposition of the hallucination stages is what the analyzer reads.

Inputs are whatever `gpu_fbc_run.sh` left behind: the stage JSONL from the timing shim, the
`--verbose` stdout log, the nvidia-smi memory samples, and the run's own CSVs.

The split that matters is one line: how much of the wall is the AF2 gradient loop, which cannot
move to Tenstorrent, versus how much is ordinary forward inference (the two AF2 validation
predictions and ProteinMPNN), which can. Everything else (OpenMM relax, FreeSASA/sc-rs/FASPR
scoring, host and I/O) is named separately because it is neither.

    python perf/freebindcraft/parse_fbc_run.py --run-dir /work/out --report /work/out/split.json

`gradient` is the four backprop stages of the hallucination loop, taken from the `[AF2]` verbose
timers. `design_forward` is the PSSM-semigreedy stage, which ColabDesign runs with backprop=False.
`hallucination_other` is whatever is left inside `binder_hallucination` after those two: model
construction, input prep, the DSSP beta assessment, the PDB write. It is reported rather than
folded in, so the breakdown closes.
"""

import argparse
import collections
import csv
import json
import pathlib
import re

GRADIENT_MARKERS = ["Stage 1 logits", "Additional logits", "Softmax stage", "One-hot stage"]
FORWARD_MARKERS = ["PSSM semigreedy"]
AF2_LINE = re.compile(r"\[AF2\] (.+?) in ([0-9.]+)s")
RELAX_LINE = re.compile(r"\[OpenMM-Relax\] Completed relax for \S+ in ([0-9.]+)s")
SCORE_LINE = re.compile(r"\[Alt-Score\] Completed scoring for \S+ in ([0-9.]+)s")
TOTAL_LINE = re.compile(r"Script execution for (\d+) trajectories took: (\d+) hours, (\d+) minutes, (\d+) seconds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="directory holding run.log, stages.jsonl, gpumem.csv")
    ap.add_argument("--design-path", help="the run's design_path, for the stats CSVs")
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    log = (run_dir / "run.log").read_text(errors="ignore")

    af2 = collections.defaultdict(float)
    for label, secs in AF2_LINE.findall(log):
        af2[label] += float(secs)
    gradient_s = sum(v for k, v in af2.items() if any(m in k for m in GRADIENT_MARKERS))
    design_forward_s = sum(v for k, v in af2.items() if any(m in k for m in FORWARD_MARKERS))

    relax_s = sum(float(x) for x in RELAX_LINE.findall(log))
    scoring_s = sum(float(x) for x in SCORE_LINE.findall(log))

    stages = collections.defaultdict(float)
    calls = collections.Counter()
    jsonl = run_dir / "stages.jsonl"
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            stages[rec["stage"]] += rec["s"]
            calls[rec["stage"]] += 1

    wall_s = None
    trajectories = None
    m = TOTAL_LINE.search(log)
    if m:
        trajectories = int(m.group(1))
        wall_s = int(m.group(2)) * 3600 + int(m.group(3)) * 60 + int(m.group(4))

    peak_mem_mib = None
    mem_csv = run_dir / "gpumem.csv"
    if mem_csv.exists():
        vals = [int(x) for x in re.findall(r"(\d+) MiB", mem_csv.read_text())]
        if vals:
            peak_mem_mib = max(vals)

    accepted = None
    if args.design_path:
        final_csv = next(pathlib.Path(args.design_path).glob("*final_design_stats.csv"), None)
        if final_csv and final_csv.stat().st_size:
            accepted = sum(1 for _ in csv.DictReader(final_csv.open()))

    hallucination_s = stages.get("binder_hallucination", 0.0)
    validation_s = stages.get("predict_binder_complex", 0.0) + stages.get("predict_binder_alone", 0.0)
    mpnn_s = stages.get("mpnn_gen_sequence", 0.0)

    named = {
        "gradient_loop_s": round(gradient_s, 1),
        "design_forward_semigreedy_s": round(design_forward_s, 1),
        "hallucination_other_s": round(max(0.0, hallucination_s - gradient_s - design_forward_s), 1),
        "validation_af2_inference_s": round(validation_s, 1),
        "proteinmpnn_s": round(mpnn_s, 1),
        "openmm_relax_s": round(relax_s, 1),
        "scoring_s": round(scoring_s, 1),
    }
    accounted = sum(named.values())
    report = {
        "wall_s": wall_s,
        "trajectories": trajectories,
        "accepted_designs": accepted,
        "s_per_accepted_design": round(wall_s / accepted, 1) if (wall_s and accepted) else None,
        "s_per_trajectory": round(wall_s / trajectories, 1) if (wall_s and trajectories) else None,
        "peak_gpu_mem_mib": peak_mem_mib,
        "stage_calls": dict(calls),
        "split_s": named,
        "unattributed_s": round(wall_s - accounted, 1) if wall_s else None,
    }
    if wall_s:
        report["split_pct"] = {k: round(100 * v / wall_s, 1) for k, v in named.items()}
        report["split_pct"]["unattributed"] = round(100 * (wall_s - accounted) / wall_s, 1)
        portable = named["validation_af2_inference_s"] + named["proteinmpnn_s"]
        report["portable_inference_pct"] = round(100 * portable / wall_s, 1)
        report["gradient_pct"] = round(100 * named["gradient_loop_s"] / wall_s, 1)

    pathlib.Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
