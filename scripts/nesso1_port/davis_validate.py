#!/usr/bin/env python3
"""Does the port rank binders? DAVIS within-target Pearson, on device, same inputs as the GPU arm.

Nesso-1 emits no structure, so there is no RMSD and no way to eyeball a bad prediction. The only
validity signal is whether the predicted affinity correlates with measured affinity, which is why
the GPU reference ran this before recording a single second (perf/nesso1/gpu_nesso1_validate.py:
within-target Pearson 0.636 against a 0.175 molecular-weight control). This is the same protocol
against the same selection, on one Blackhole p150a.

Identical inputs, not merely the same benchmark: the target and ligand selection is a pure function
of davis.csv, so l00..l29 here are the same 30 compounds the GPU arm scored, in the same order.

Metric: WITHIN-target Pearson, which is what the technical report reports. Pooling across targets
would mostly measure the between-target offset. Only non-censored measurements are used -- DAVIS
reports every non-binder as exactly 10000 nM, so including them turns Pearson into a statement about
how many ties there are.

Resumable: one JSON per prediction, and an existing one is not recomputed. A relaunch continues.

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=... <env>/bin/python \
        scripts/nesso1_port/davis_validate.py --target-index 0 --out perf/nesso1
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

CENSORED_KD_NM = 10000.0
N_TARGETS = 2
N_LIGANDS = 30


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


def select(davis: Path) -> list[tuple[str, list[dict]]]:
    """The GPU arm's selection, verbatim in behaviour: a pure function of davis.csv."""
    rows = []
    with open(davis, newline="") as fh:
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
    chosen = [t for t in ranked if len(t[1]) >= 8 and spread(t[1]) >= 0.4][:N_TARGETS]
    if not chosen:
        raise SystemExit("no DAVIS target with >=8 non-censored measurements and log-spread >=0.4")
    out = []
    for target_id, recs in chosen:
        recs = sorted(recs, key=lambda r: r["kd_nM"])
        if len(recs) > N_LIGANDS:                       # keep the Kd range, thin the middle
            idx = [round(i * (len(recs) - 1) / (N_LIGANDS - 1)) for i in range(N_LIGANDS)]
            recs = [recs[i] for i in sorted(set(idx))]
        out.append((target_id, recs))
    return out


def write_yamls(target_id: str, recs: list[dict], yml: Path) -> list[dict]:
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    yml.mkdir(parents=True, exist_ok=True)
    keep = []
    for i, r in enumerate(recs):
        mol = Chem.MolFromSmiles(r["smiles"])
        if mol is None:
            continue
        rid = "l%02d" % i
        (yml / f"{rid}.yaml").write_text(
            "sequences:\n  - protein:\n      id: A\n      sequence: %s\n"
            "  - ligand:\n      id: B\n      smiles: '%s'\n"
            "properties:\n  - affinity:\n      binder: B\n" % (r["seq"], r["smiles"]))
        keep.append({"record": rid, "drug": r["drug_id"], "kd_nM": r["kd_nM"],
                     "pkd": round(9.0 - math.log10(r["kd_nM"]), 4),
                     "mw": Descriptors.MolWt(mol), "heavy": mol.GetNumHeavyAtoms()})
    return keep


def correlate(pairs: list[dict]) -> dict:
    done = [p for p in pairs if p.get("pred") is not None]
    if len(done) < 3:
        return {"n_scored": len(done)}
    pkd = [p["pkd"] for p in done]
    pred = [-p["pred"] for p in done]           # higher = stronger, same direction as pKd
    mw = [p["mw"] for p in done]
    prob = [p["prob"] for p in done if p.get("prob") is not None]
    out = {"n_scored": len(done),
           "pearson_pred_vs_pkd": round(pearson(pred, pkd), 4),
           "spearman_pred_vs_pkd": round(spearman(pred, pkd), 4),
           "pearson_mw_vs_pkd": round(pearson(mw, pkd), 4),
           "spearman_mw_vs_pkd": round(spearman(mw, pkd), 4)}
    if len(prob) == len(done):
        out["pearson_prob_vs_pkd"] = round(pearson(prob, pkd), 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--davis", type=Path,
                    default=Path("~/scratch/nesso1/davis/davis.csv").expanduser())
    ap.add_argument("--target-index", type=int, default=None,
                    help="run one target only (0 or 1); default runs both on this card")
    ap.add_argument("--scratch", type=Path, default=Path("~/scratch/nesso1/davis").expanduser())
    ap.add_argument("--weights", default="recursionpharma/nesso")
    ap.add_argument("--trunk", default="bf16", choices=("bf16", "fp32"),
                    help="fp32 cannot run these targets: 1167 aa asks for more DRAM than a bank has")
    ap.add_argument("--esm-cache", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=REPO / "perf/nesso1")
    args = ap.parse_args()

    from tt_bio.nesso1 import DEFAULT_SEED, REPORTED_SCALARS, Nesso1
    from tt_bio.nesso1_input import CLI_PREDICT_ARGS, collate, prepare

    targets = select(args.davis)
    if args.target_index is not None:
        targets = [targets[args.target_index]]
    print("targets: %s" % [(t, len(r), len(r[0]["seq"])) for t, r in targets], flush=True)

    model = None
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"gate": "nesso1_davis",
              "benchmark": "DAVIS (TDC), kinase/inhibitor Kd in nM, non-censored only",
              "metric": ("within-target Pearson of predicted pAffinity (= -affinity_pred_value) "
                         "against pKd = 9 - log10(Kd_nM)"),
              "control": "molecular weight alone (RDKit Descriptors.MolWt) vs pKd, same ligands",
              "trunk": args.trunk, "seed": DEFAULT_SEED, "targets": []}

    for target_id, recs in targets:
        tag = target_id.replace("/", "_")
        work = args.scratch / "run" / tag
        pairs = write_yamls(target_id, recs, work / "yaml")
        preds_dir = work / "preds"
        preds_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.out / ("davis_%s.json" % tag)

        t0 = time.perf_counter()
        dataset, manifest, failed = prepare(
            work / "yaml", work, num_workers=0, esm_cache=args.esm_cache)
        feat_s = time.perf_counter() - t0
        by_id = {p["record"]: p for p in pairs}
        row = {"target_id": target_id, "seq_len": len(recs[0]["seq"]),
               "n_pairs": len(pairs), "featurize_s": feat_s, "failed": failed,
               "kd_nM_range": [recs[0]["kd_nM"], recs[-1]["kd_nM"]], "pairs": pairs}
        print("=== %s: %d aa, %d ligands, featurized in %.1fs"
              % (target_id, row["seq_len"], len(pairs), feat_s), flush=True)

        if model is None:
            model = Nesso1.from_pretrained(
                args.weights, use_tenstorrent=True,
                trunk_fp32=args.trunk == "fp32", affinity_fp32=True)
            model.use_kernels = False
            model.predict_args.update(CLI_PREDICT_ARGS)

        for idx, record in enumerate(manifest.records):
            p = by_id.get(record.id)
            if p is None:
                continue
            cached = preds_dir / f"{record.id}.json"
            if cached.exists():                       # resume: never recompute a prediction
                got = json.loads(cached.read_text())
            else:
                torch.manual_seed(DEFAULT_SEED)       # the featurizer's roto-translation draw
                item = dataset[idx]
                if item.get("exception"):
                    got = {"error": "featurizer raised"}
                else:
                    feats = collate(item)
                    t0 = time.perf_counter()
                    try:
                        with torch.no_grad():
                            pred = model.predict(feats)
                        got = {k: float(pred[k].reshape(-1)[0])
                               for k in REPORTED_SCALARS if k in pred}
                    except Exception as exc:          # noqa: BLE001 - a failure is a datapoint
                        got = {"error": "%s: %s" % (type(exc).__name__, exc)}
                    got["n_tokens"] = int(feats["token_pad_mask"].shape[-1])
                    got["seconds"] = time.perf_counter() - t0
                cached.write_text(json.dumps(got, indent=2) + "\n")
            p["pred"] = got.get("affinity_pred_value")
            p["prob"] = got.get("affinity_probability_binary")
            p["n_tokens"] = got.get("n_tokens")
            p["seconds"] = got.get("seconds")
            if "error" in got:
                p["error"] = got["error"]
            print("  %s drug %-9s %5d tok  pKd %.2f  pred %s  %.1fs"
                  % (record.id, p["drug"], p.get("n_tokens") or -1, p["pkd"],
                     ("%.4f" % p["pred"]) if p["pred"] is not None else got.get("error"),
                     p.get("seconds") or 0.0), flush=True)
            row.update(correlate(pairs))
            out_path.write_text(json.dumps(row, indent=2) + "\n")   # survive a mid-run kill

        row["wall_s"] = sum(p.get("seconds") or 0.0 for p in pairs)
        out_path.write_text(json.dumps(row, indent=2) + "\n")
        report["targets"].append({k: v for k, v in row.items() if k != "pairs"})
        print("  -> %s: Pearson %.4f  Spearman %.4f  MW control %.4f  (n=%d)"
              % (target_id, row.get("pearson_pred_vs_pkd", float("nan")),
                 row.get("spearman_pred_vs_pkd", float("nan")),
                 row.get("pearson_mw_vs_pkd", float("nan")), row.get("n_scored", 0)), flush=True)

    done = [t for t in report["targets"] if t.get("n_scored", 0) >= 3]
    if done:
        report["mean_pearson"] = round(
            sum(t["pearson_pred_vs_pkd"] for t in done) / len(done), 4)
        report["mean_spearman"] = round(
            sum(t["spearman_pred_vs_pkd"] for t in done) / len(done), 4)
        report["mean_pearson_mw_control"] = round(
            sum(t["pearson_mw_vs_pkd"] for t in done) / len(done), 4)
    suffix = "" if args.target_index is None else "_t%d" % args.target_index
    (args.out / ("davis%s.json" % suffix)).write_text(json.dumps(report, indent=2) + "\n")
    print("\nwrote %s" % (args.out / ("davis%s.json" % suffix)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
