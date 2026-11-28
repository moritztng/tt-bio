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


RESCOUNT = CO / "state/dfd/evidence/residue-counts.json"
CAP = 3000


NAN_MARK = ("non-finite", "nan", "linalg.svd", "inf")


def nan_failures(failures: dict) -> dict:
    """The subset of refusals that are a NUMERICAL failure, not an envelope refusal.

    A complex that goes non-finite mid-rollout writes no pose, so the grader's own
    `nonfinite_or_collapse` (which reads written coordinates) cannot see it: it would read 0
    while three complexes died of NaN. Condition 4 counts those, so they are read back off the
    refusal reasons the dock logged.
    """
    return {i: why for i, why in failures.items()
            if any(m in (why or "").lower() for m in NAN_MARK)}


def exclusions(per, failures: dict, attempted=()):
    """Every complex that produced no pose, named with its reason and its receptor size.

    Condition 4 of QUALITY.md wants `n_attempted` separate from `n_scored`; a denominator
    that shrinks silently is the failure shape this row exists to catch. So the gap between
    the two is itself enumerated: who, how big, and why, with the upstream side's answer for
    the same complex beside it.
    """
    counts = {}
    if RESCOUNT.exists():
        counts = json.load(open(RESCOUNT)).get("per_complex", {})
    ref = {r["id"]: r for r in json.load(open(REF))}
    out = {}
    # A complex that never reached the grader is not in `per` at all: it has no pose directory
    # to grade. Iterating only over graded rows would drop it from the exclusion table AND from
    # the denominator at once, which is the exact silent-denominator shape condition 4 forbids.
    graded = {r["id"] for r in per}
    rows = list(per) + [{"id": i} for i in sorted(set(attempted) - graded)]
    for r in rows:
        i = r["id"]
        if r.get("rmsd") and r["rmsd"] and r["rmsd"][0] is not None:
            continue
        n = counts.get(i)
        over = isinstance(n, int) and n > CAP
        up = ref.get(i, {})
        reason = failures.get(i) or (
            f"receptor over the {CAP}-residue cap ({n}), refused by upstream "
            f"(process_mols.py:169) and by the port (receptor.py:205)" if over else None)
        out[i] = {
            "residues": n,
            "over_residue_cap": over,
            "reason": reason,
            # A complex the port REFUSED, with the refusal on record, is an envelope result.
            # A complex with no pose and no reason is an unexplained hole: it was never
            # attempted, or it died somewhere that printed nothing. The two must not be
            # summed, because only the first is a property of the model.
            "status": "refused" if reason else "no_pose_no_reason",
            "upstream_also_ungraded": not (up.get("rmsd") and up["rmsd"]
                                           and up["rmsd"][0] is not None),
        }
    unexplained = sorted(k for k, v in out.items() if v["status"] == "no_pose_no_reason")
    same = all(v["upstream_also_ungraded"] for v in out.values())
    note = "These are counted, never dropped: n_attempted stays at the full set. "
    if unexplained:
        note += (f"WARNING: {len(unexplained)} complexes have no pose and no recorded reason. "
                 "That is an unexplained denominator gap, not an envelope result: pass "
                 "--failures (dock_failures.sh) so the refusals are on record, and check the "
                 "pool is complete before reading any rate. ")
    note += ("The same complexes are ungraded upstream, so both halves share one denominator "
             "and the comparison is paired over the same set."
             if same else
             "WARNING: the two sides do NOT exclude the same complexes. Every id with "
             "upstream_also_ungraded false is scored on one side only and the paired "
             "comparison drops it; say so in EVAL.md before reading the verdict.")
    return {
        "n": len(out),
        "n_refused_with_reason": len(out) - len(unexplained),
        "n_no_pose_no_reason": len(unexplained),
        "unexplained": unexplained[:20],
        "cap": CAP,
        "complexes": out,
        "identical_to_upstream": same,
        "note": note,
        "evidence": str(RESCOUNT),
    }


def load_evaluate():
    """The shipped grading library, loaded BY FILE PATH.

    Not `from tt_bio.diffdock import evaluate`: the package __init__ imports the input pipeline,
    which imports torch, and the grading environment deliberately has no torch. evaluate.py itself
    needs numpy, and rdkit/spyrmsd inside the functions.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "dfd_evaluate", REPO / "tt_bio" / "diffdock" / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def library_crosscheck(poses: Path, ids):
    """Grade the same poses through the library a user gets, as an independent reading.

    `pose_eval.grade` is dfd-gpu-fair's grader and is the gated protocol, run identically on both
    halves. `evaluate.grade_set` is the shipped library. They implement the same arithmetic from
    different code, so a disagreement on top-1 is a defect in one of them, and agreement is the
    cheapest check that the number is the metric it claims to be.
    """
    g = load_evaluate().grade_set(ids, poses, PB, max_rank=10)
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


def steric_cross(evaluate, poses: Path, ids):
    """Closest ligand-receptor approach over the set's rank-1 poses, in Angstrom.

    `evaluate.min_cross_distance` is shipped and documented but nothing called it: `grade_set`
    never loads receptor coordinates, so the field sat null while the library had the function.
    QUALITY.md asks for the steric pair, so it is measured here against the same receptor file
    the docking read. Reported as the worst approach over the set plus the median, because one
    number over 420 poses is a minimum and says nothing about the bulk.
    """
    from rdkit import Chem
    vals = []
    for cid in ids:
        rank1 = poses / cid / "rank1.sdf"
        pdb = PB / cid / f"{cid}_protein.pdb"
        if not rank1.exists() or not pdb.exists():
            continue
        mol = Chem.MolFromMolFile(str(rank1), sanitize=False)
        if mol is None or mol.GetNumConformers() == 0:
            continue
        lig = mol.GetConformer().GetPositions()
        rec = [(float(l[30:38]), float(l[38:46]), float(l[46:54]))
               for l in open(pdb)
               if l.startswith("ATOM") and l[76:78].strip() not in ("H", "")]
        if not rec:
            continue
        vals.append(evaluate.min_cross_distance(lig, rec))
    if not vals:
        return None
    return {"min_over_set_a": round(min(vals), 3),
            "median_a": round(statistics.median(vals), 3),
            "n_poses": len(vals),
            "note": "heavy atoms both sides, rank-1 pose vs the crystal receptor as docked"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc", required=True)
    ap.add_argument("--poses", required=True)
    ap.add_argument("--no-crosscheck", action="store_true")
    ap.add_argument("--failures", default=None,
                    help="json map id -> reason, from dock_failures.sh; the per-complex refusal "
                         "reasons live in the dock hosts' chunk logs, which are not pooled")
    ap.add_argument("--redock", default=None,
                    help="json describing the re-dock that replaced earlier failures: tree, host, "
                         "venv and the per-complex count of samples frozen non-finite")
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
    fails = json.load(open(a.failures)) if a.failures and Path(a.failures).exists() else {}
    # ATTEMPTED is graded + refused-with-a-reason, not graded alone. On 2026-11-27 three
    # complexes went non-finite mid-rollout and wrote no pose, so the graded set is 421 of the
    # 428 the run started: reading n_attempted off the graded rows would have both hidden the
    # three failures and let a 421-complex set pass a 428-complex guard by redefining the set.
    # A failure the fixed tree has since docked is no longer a failure: the re-dock replaced it
    # in the pool and the grader scored it. It stays on record under `redocked_after_fix` with
    # its old reason, so the history is visible and the count is the pool's, not the log's.
    scored = {r["id"] for r in per if r.get("rmsd") and r["rmsd"][0] is not None}
    redocked = {i: why for i, why in fails.items() if i in scored}
    fails = {i: why for i, why in fails.items() if i not in scored}
    attempted = sorted(set(ids) | set(fails))
    if len(attempted) < a.min_attempted and not a.dry_run:
        sys.exit(f"refusing to write: {len(attempted)} complexes attempted ({len(ids)} graded, "
                 f"{len(fails)} refused with a reason), --min-attempted {a.min_attempted}. "
                 "A partial set is not a verdict.")
    cross = None if a.no_crosscheck else library_crosscheck(poses, ids)

    q = json.load(open(QJ))
    pbs = q["sets"]["posebusters"]
    ref_summary = json.load(open(CO / "state/dfd/evidence/gpu-accuracy.json"))

    pbs["n_attempted"] = len(attempted)
    pbs["n_scored"] = summary["n_graded"]
    pbs["exclusions"] = exclusions(per, fails, attempted)
    nans = nan_failures(fails)
    pbs["tt"] = {
        "top1_rmsd_lt2": summary["top1_success_lt2A"],
        "top1_ci95_wilson": summary["top1_success_ci95"],
        "top5_rmsd_lt2": summary["top5_success_lt2A"],
        "median_rmsd": summary["median_rmsd_top1"],
        "percentiles": summary["percentiles_top1"],
        "posebusters_validity": summary["posebusters_validity_rate"],
        "posebusters_physics_validity": summary["posebusters_physics_validity_rate"],
        "steric": {"min_cross_distance": steric_cross(load_evaluate(), poses, ids),
                   "self_intersection_frac": (cross or {}).get("self_intersect_frac")},
        # written-pose NaNs the grader can see, PLUS the complexes that died non-finite before
        # any pose was written and so cannot appear in a coordinate scan
        "nan_or_collapse": summary["nonfinite_or_collapse"] + len(nans),
        "nan_in_written_poses": summary["nonfinite_or_collapse"],
        "nan_failed_before_writing": nans,
        "network": "host torch HostScoreNetwork, released v1.1 checkpoints",
    }
    if redocked:
        redock = json.load(open(a.redock)) if a.redock else {}
        pbs["tt"]["redocked_after_fix"] = dict(redock, complexes=redocked)
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
