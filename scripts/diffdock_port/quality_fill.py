"""Turn the graded run into `state/dfd/QUALITY.json`, with the five PASS conditions applied.

    quality_fill.py --acc ~/scratch/dfd/eval/acc --poses ~/scratch/dfd/eval/poses-all

Reads the grader's `summary.json` / `per_complex.json`, runs `pose_eval.compare` against
`dfd-gpu-fair`'s reference half if `compare.json` is not there yet, cross-checks the numbers with
the shipped `tt_bio.diffdock.evaluate.grade_set` library, and writes QUALITY.json in the shape
`state/dfd/QUALITY.md` fixes. The verdict is computed from the five conditions in that file, not
chosen: a field that was not measured stays null and a null in a gating field is a FAIL, never a
pass by omission.
"""
import argparse
import json
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CO = Path("/home/moritz/.coworker")
REF = CO / "state/dfd/gpu/acc/per_complex-torch2-tf32.json"
QJ = CO / "state/dfd/QUALITY.json"
ROW = CO / "state/dfd/rows/dfd-eval.json"
PB = Path(os.environ.get("DFD_PB", Path.home() / "scratch/dfd/pb/posebusters_benchmark_set"))


def library_crosscheck(poses: Path, ids):
    """Grade the same poses through the library a user gets, as an independent reading.

    `pose_eval.grade` is dfd-gpu-fair's grader and is the gated protocol, run identically on both
    halves. `evaluate.grade_set` is the shipped library. They implement the same arithmetic from
    different code, so a disagreement on top-1 is a defect in one of them, and agreement is the
    cheapest check that the number is the metric it claims to be.
    """
    sys.path.insert(0, str(REPO))
    from tt_bio.diffdock import evaluate
    g = evaluate.grade_set(ids, poses, PB, max_rank=10)
    m = g["metrics"]
    # success_rates' top-k family starts at k=5, so the rank-1 rate this campaign gates on comes
    # from the per-complex records: rmsd[0] is the confidence model's column-0 pose (dfd-ref).
    r1 = [r["rmsd"][0] for r in g["per_complex"] if r.get("scored") and r.get("rmsd")]
    return {
        "n_attempted": g["n_attempted"], "n_scored": g["n_scored"],
        "nan_or_collapse": g["nan_or_collapse"], "n_naive_rmsd": g["n_naive_rmsd"],
        "samples_graded": g["samples_graded"],
        "top1_rmsd_lt2": round(sum(1 for v in r1 if v < 2.0) / len(r1), 4) if r1 else None,
        "median_rmsd": round(statistics.median(r1), 3) if r1 else None,
        "self_intersect_frac": m.get("self_intersect_fraction"),
        "top5_rmsd_lt2": (m["top5_rmsds_below_2"] / 100 if "top5_rmsds_below_2" in m else None),
        "scale_note": "the library's top-k family is a percentage and starts at k=5; the rank-1 "
                      "rate above is counted from its per-complex rmsd[0] as a fraction",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc", required=True)
    ap.add_argument("--poses", required=True)
    ap.add_argument("--no-crosscheck", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the verdict block without writing QUALITY.json or the row")
    ap.add_argument("--min-attempted", type=int, default=428,
                    help="refuse to write a verdict on a partial set: the full PoseBusters set is "
                         "428 complexes and a short denominator would read as a measured result")
    a = ap.parse_args()
    acc, poses = Path(a.acc).expanduser(), Path(a.poses).expanduser()

    summary = json.load(open(acc / "summary.json"))
    per = json.load(open(acc / "per_complex.json"))
    cmp_path = acc / "compare.json"
    if not cmp_path.exists():
        sys.path.insert(0, str(HERE))
        import pose_eval
        pose_eval.compare(str(acc / "per_complex.json"), str(REF), str(cmp_path))
    comp = json.load(open(cmp_path))

    ids = sorted(r["id"] for r in per)
    if len(ids) < a.min_attempted and not a.dry_run:
        sys.exit(f"refusing to write: {len(ids)} complexes graded, --min-attempted "
                 f"{a.min_attempted}. A partial set is not a verdict.")
    cross = None if a.no_crosscheck else library_crosscheck(poses, ids)

    q = json.load(open(QJ))
    pbs = q["sets"]["posebusters"]
    ref_summary = json.load(open(CO / "state/dfd/evidence/gpu-accuracy.json"))

    pbs["n_attempted"] = len(ids)
    pbs["n_scored"] = summary["n_graded"]
    pbs["tt"] = {
        "top1_rmsd_lt2": summary["top1_success_lt2A"],
        "top1_ci95_wilson": summary["top1_success_ci95"],
        "top5_rmsd_lt2": summary["top5_success_lt2A"],
        "median_rmsd": summary["median_rmsd_top1"],
        "percentiles": summary["percentiles_top1"],
        "posebusters_validity": summary["posebusters_validity_rate"],
        "posebusters_physics_validity": summary["posebusters_physics_validity_rate"],
        "steric": {"min_cross_distance": None,
                   "self_intersection_frac": (cross or {}).get("self_intersect_frac")},
        "nan_or_collapse": summary["nonfinite_or_collapse"],
        "network": "host torch HostScoreNetwork, released v1.1 checkpoints",
    }
    pbs["upstream"] = ref_summary
    pbs["paired"] = {
        "n_common_graded": comp["n_common_graded"],
        "delta_top1": comp["top1"]["delta"],
        "ci95": comp["top1"]["ci95_paired_bootstrap"],
        "mcnemar_p": comp["top1"]["mcnemar_p"],
        "mcnemar_discordant": comp["top1"]["mcnemar_discordant"],
        "ks_stat": comp["rmsd_distribution"]["ks_stat"],
        "ks_p": comp["rmsd_distribution"]["ks_p"],
        "validity_delta_ci95": comp["validity"]["delta_ci95_paired"],
        "tt_only_graded": comp["tt_only_graded"],
        "upstream_only_graded": comp["upstream_only_graded"],
        "paired": True,
    }
    pbs["outliers"] = {k: dict(v, why="D-dfd-trace-1 / D-dfd-orch-31: the 32-neighbour cap is "
                                      "deliberately not reproduced, see state/dfd/KNN32.md")
                       for k, v in comp.get("outliers", {}).items()}
    if cross:
        pbs["library_crosscheck"] = cross
    q["paths"] = {
        "tt_runs": str(poses), "upstream_runs": str(REF),
        "script": "scripts/diffdock_port/pose_eval.py (dock/grade/compare) + quality_fill.py",
        "summary": str(acc / "summary.json"), "compare": str(cmp_path),
    }

    # ---- the five conditions of state/dfd/QUALITY.md, applied, not argued
    ci = pbs["paired"]["ci95"]
    cond = {
        "1_top1_paired_ci_contains_or_above_0":
            bool(ci and ci[0] is not None and ci[1] >= 0),
        "2_ks_p_ge_0.05":
            bool(pbs["paired"]["ks_p"] is not None and pbs["paired"]["ks_p"] >= 0.05),
        "3_validity_not_below_by_more_than_ci":
            bool(pbs["paired"]["validity_delta_ci95"]
                 and pbs["paired"]["validity_delta_ci95"][1] is not None
                 and pbs["paired"]["validity_delta_ci95"][1] >= 0),
        "4_no_nan_or_collapse_and_denominator_reported":
            bool(pbs["tt"]["nan_or_collapse"] == 0 and pbs["n_attempted"] and pbs["n_scored"]),
        "5_equivariance_passes":
            q.get("equivariance", {}).get("verdict") == "PASS",
    }
    q["conditions"] = cond
    q["verdict"] = "PASS" if all(cond.values()) else "FAIL"
    q["_note"] = ("Filled by scripts/diffdock_port/quality_fill.py from the graded run. Every "
                  "number here is read from summary.json or compare.json; the verdict is the "
                  "conjunction of the five conditions in QUALITY.md, computed above.")
    if not a.dry_run:
        json.dump(q, open(QJ, "w"), indent=2)

    r = json.load(open(ROW)) if not a.dry_run else {}
    r.update(status="done", branch="wk/dfd-eval",
             n_attempted=pbs["n_attempted"], n_scored=pbs["n_scored"],
             top1_rmsd_lt2=pbs["tt"]["top1_rmsd_lt2"], ks_p=pbs["paired"]["ks_p"],
             paired_ci=pbs["paired"]["ci95"], validity=pbs["tt"]["posebusters_validity"],
             nan_or_collapse=pbs["tt"]["nan_or_collapse"], verdict=q["verdict"])
    if not a.dry_run:
        json.dump(r, open(ROW, "w"), indent=1)
    print(json.dumps({"verdict": q["verdict"], "conditions": cond,
                      "tt_top1": pbs["tt"]["top1_rmsd_lt2"],
                      "upstream_top1": ref_summary.get("top1_success_lt2A"),
                      "delta": pbs["paired"]["delta_top1"], "ci95": ci,
                      "ks_p": pbs["paired"]["ks_p"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
