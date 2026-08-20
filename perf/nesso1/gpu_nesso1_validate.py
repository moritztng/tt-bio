"""Does Nesso-1's output actually rank binders? Answered before any second is recorded.

    python perf/nesso1/gpu_nesso1_validate.py --targets 2 --ligands 30

A fast wrong number is not a reference. Nesso-1 emits no structure, so there is no RMSD to check
and no way to eyeball a bad prediction: the only validity signal is whether the predicted affinity
correlates with measured affinity. That check has to pass before the timing table means anything.

Benchmark: DAVIS (kinase / inhibitor Kd, nM), pulled from Therapeutics Data Commons. It is public,
it needs only a sequence and a SMILES -- exactly Nesso's input format, no MSA, no structure -- and
its Kd values are real measurements, not another model's output.

Metric: WITHIN-TARGET Pearson, which is what the technical report reports (a weighted Pearson
across 25 biochemical assays, 0.44 for Nesso-1 against 0.37 for Boltz-2 and 0.27 for molecular
weight alone). Pooling across targets instead would mostly measure the between-target offset and
flatter or wreck the number for reasons that have nothing to do with ranking compounds, which is
the task. The molecular-weight-only control from the report is computed here too, on the same
ligands, because a correlation that a single RDKit descriptor already explains is not evidence the
model works.

Only non-censored measurements are used. DAVIS reports every non-binder as exactly 10000 nM, so
including them would put most of the mass on one tied value and turn Pearson into a statement about
how many ties there are.
"""

import argparse
import csv
import json
import math
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).parent
RUN = HERE / "gpu_nesso1_run.py"
CENSORED_KD_NM = 10000.0


def pearson(xs, ys) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def spearman(xs, ys) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return pearson(rank(xs), rank(ys))


def yaml_for(seq: str, smiles: str) -> str:
    return ("sequences:\n  - protein:\n      id: A\n      sequence: %s\n"
            "  - ligand:\n      id: B\n      smiles: '%s'\n"
            "properties:\n  - affinity:\n      binder: B\n" % (seq, smiles))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--davis", default="/work/davis.csv")
    ap.add_argument("--targets", type=int, default=2)
    ap.add_argument("--ligands", type=int, default=30, help="per target")
    ap.add_argument("--work", default="/work/validate")
    ap.add_argument("--python", default="/work/v_nesso/bin/python")
    ap.add_argument("--report", default="/work/results/validate.json")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    rows = []
    with open(args.davis, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                y = float(r["Y"])
            except (TypeError, ValueError):
                continue
            if y >= CENSORED_KD_NM or y <= 0:
                continue
            rows.append({"drug_id": r["Drug_ID"], "smiles": r["Drug"],
                         "target_id": r["Target_ID"], "seq": r["Target"], "kd_nM": y})

    by_target: dict[str, list] = {}
    for r in rows:
        by_target.setdefault(r["target_id"], []).append(r)

    def spread(v):
        ls = [math.log10(x["kd_nM"]) for x in v]
        m = sum(ls) / len(ls)
        return math.sqrt(sum((x - m) ** 2 for x in ls) / len(ls))

    ranked = sorted(by_target.items(), key=lambda kv: (len(kv[1]), spread(kv[1])), reverse=True)
    chosen = [t for t in ranked if len(t[1]) >= 8 and spread(t[1]) >= 0.4][:args.targets]
    if not chosen:
        raise SystemExit("no DAVIS target with >=8 non-censored measurements and log-spread >=0.4")

    report = {"benchmark": "DAVIS (TDC), kinase/inhibitor Kd in nM, non-censored only",
              "n_noncensored_pairs_available": len(rows),
              "n_targets_available": len(by_target),
              "metric": ("within-target Pearson of predicted pAffinity (= -affinity_pred_value, "
                         "so higher = stronger, same direction as pKd) against pKd = "
                         "9 - log10(Kd_nM)"),
              "control": "molecular weight alone (RDKit Descriptors.MolWt) vs pKd, same ligands",
              "targets": []}

    from rdkit import Chem
    from rdkit.Chem import Descriptors

    work = pathlib.Path(args.work)
    for target_id, recs in chosen:
        recs = sorted(recs, key=lambda r: r["kd_nM"])
        if len(recs) > args.ligands:                          # keep the Kd range, thin the middle
            idx = [round(i * (len(recs) - 1) / (args.ligands - 1)) for i in range(args.ligands)]
            recs = [recs[i] for i in sorted(set(idx))]
        d = work / target_id.replace("/", "_")
        yml = d / "yaml"
        if yml.exists():
            import shutil
            shutil.rmtree(yml)
        yml.mkdir(parents=True)
        keep = []
        for i, r in enumerate(recs):
            m = Chem.MolFromSmiles(r["smiles"])
            if m is None:
                continue
            r["mw"] = Descriptors.MolWt(m)
            r["heavy"] = m.GetNumHeavyAtoms()
            rid = "l%02d" % i
            r["record_id"] = rid
            (yml / ("%s.yaml" % rid)).write_text(yaml_for(r["seq"], r["smiles"]))
            keep.append(r)

        rp = d / "run.json"
        cmd = [args.python, str(RUN), "--inputs", str(yml), "--out-dir", str(d / "out"),
               "--report", str(rp), "--reps", "1", "--label", "validate_%s" % target_id]
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
        wall = time.time() - t0
        run = json.loads(rp.read_text()) if rp.exists() else {"ok": False, "why": p.stderr[-800:]}

        preds = {}
        for f in sorted((d / "out" / "predictions").rglob("affinity.json")):
            preds[f.parent.name] = json.loads(f.read_text())
        paired = [(r, preds[r["record_id"]]) for r in keep if r["record_id"] in preds]
        pkd = [9.0 - math.log10(r["kd_nM"]) for r, _ in paired]
        pred = [-float(a["affinity_pred_value"]) for _, a in paired]
        mw = [r["mw"] for r, _ in paired]
        prob = [float(a.get("affinity_probability_binary", float("nan"))) for _, a in paired]
        cell = {"target_id": target_id, "seq_len": len(keep[0]["seq"]) if keep else None,
                "n_pairs": len(paired), "wall_s": round(wall, 2),
                "run_ok": run.get("ok"), "run_why": (run.get("why") or "")[:200],
                "kd_nM_range": [min(r["kd_nM"] for r, _ in paired),
                                max(r["kd_nM"] for r, _ in paired)] if paired else None,
                "pearson_pred_vs_pkd": None, "spearman_pred_vs_pkd": None,
                "pearson_mw_vs_pkd": None, "pearson_prob_vs_pkd": None}
        if len(paired) >= 5:
            cell["pearson_pred_vs_pkd"] = round(pearson(pred, pkd), 4)
            cell["spearman_pred_vs_pkd"] = round(spearman(pred, pkd), 4)
            cell["pearson_mw_vs_pkd"] = round(pearson(mw, pkd), 4)
            if not any(x != x for x in prob):
                cell["pearson_prob_vs_pkd"] = round(pearson(prob, pkd), 4)
            cell["pairs"] = [{"record": r["record_id"], "drug": r["drug_id"],
                              "kd_nM": r["kd_nM"], "pkd": round(9 - math.log10(r["kd_nM"]), 3),
                              "pred": round(float(a["affinity_pred_value"]), 4),
                              "mw": round(r["mw"], 1), "heavy": r["heavy"]}
                             for r, a in paired]
        report["targets"].append(cell)
        print("%-12s n=%2d  seq=%4d aa  Pearson(pred,pKd)=%s  Spearman=%s  MW-only=%s  (%.0fs)"
              % (target_id, cell["n_pairs"], cell["seq_len"], cell["pearson_pred_vs_pkd"],
                 cell["spearman_pred_vs_pkd"], cell["pearson_mw_vs_pkd"], wall), flush=True)
        pathlib.Path(args.report).write_text(json.dumps(report, indent=2) + "\n")

    good = [c for c in report["targets"] if c["pearson_pred_vs_pkd"] is not None]
    if good:
        report["mean_pearson"] = round(sum(c["pearson_pred_vs_pkd"] for c in good) / len(good), 4)
        report["mean_pearson_mw_control"] = round(
            sum(c["pearson_mw_vs_pkd"] for c in good) / len(good), 4)
        report["verdict"] = ("VALID: within-target Pearson %.3f beats the MW-only control %.3f"
                             % (report["mean_pearson"], report["mean_pearson_mw_control"])
                             if report["mean_pearson"] > report["mean_pearson_mw_control"]
                             else ("SUSPECT: within-target Pearson %.3f does not beat the MW-only "
                                   "control %.3f" % (report["mean_pearson"],
                                                     report["mean_pearson_mw_control"])))
        print(report["verdict"], flush=True)
    pathlib.Path(args.report).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
