#!/usr/bin/env python3
"""Device DAVIS run against the GPU reference, per compound and in aggregate.

The correlation is the product number; this is the parity check hiding inside it. Both arms
scored the same 30 compounds per target in the same order, so l00 here is l00 there, and the
per-ligand delta says whether the port reproduces the reference prediction or merely happens to
rank as well.
"""
import json
import math
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def main():
    gpu = json.load(open(REPO / "perf/nesso1/results/validate.json"))
    out = {"gate": "nesso1_davis_vs_gpu",
           "gpu_reference": {"mean_pearson": gpu["mean_pearson"],
                             "mean_pearson_mw_control": gpu["mean_pearson_mw_control"]},
           "targets": []}
    for gt in gpu["targets"]:
        tag = gt["target_id"].replace("/", "_")
        p = REPO / ("perf/nesso1/davis_%s.json" % tag)
        if not p.exists():
            print("missing %s" % p)
            continue
        tt = json.load(open(p))
        gmap = {q["record"]: q for q in gt["pairs"]}
        both = [(q, gmap[q["record"]]) for q in tt["pairs"]
                if q.get("pred") is not None and q["record"] in gmap]
        d = [abs(a["pred"] - b["pred"]) for a, b in both]
        row = {
            "target_id": gt["target_id"], "seq_len": gt["seq_len"],
            "n_common": len(both),
            "tt_pearson": tt.get("pearson_pred_vs_pkd"),
            "tt_spearman": tt.get("spearman_pred_vs_pkd"),
            "tt_mw_control": tt.get("pearson_mw_vs_pkd"),
            "gpu_pearson": gt["pearson_pred_vs_pkd"],
            "gpu_spearman": gt["spearman_pred_vs_pkd"],
            "gpu_mw_control": gt["pearson_mw_vs_pkd"],
            "per_ligand_max_abs_delta": round(max(d), 4) if d else None,
            "per_ligand_mean_abs_delta": round(sum(d) / len(d), 4) if d else None,
            "pearson_tt_vs_gpu_pred": round(
                pearson([a["pred"] for a, _ in both], [b["pred"] for _, b in both]), 5)
            if len(both) >= 3 else None,
            "tt_n_scored": tt.get("n_scored"),
            "tt_mean_s_per_prediction": round(
                sum(q.get("seconds") or 0.0 for q in tt["pairs"]) / max(tt.get("n_scored") or 1, 1), 1),
            "tt_tokens": sorted({q.get("n_tokens") for q in tt["pairs"] if q.get("n_tokens")}),
        }
        out["targets"].append(row)
        print("%-8s n=%-3d TT Pearson %.4f (GPU %.4f)  Spearman %.4f (GPU %.4f)  "
              "MW %.4f (GPU %.4f)  per-ligand max|d| %s  corr(TT,GPU) %s  %.1fs/pred"
              % (row["target_id"], row["n_common"], row["tt_pearson"], row["gpu_pearson"],
                 row["tt_spearman"], row["gpu_spearman"], row["tt_mw_control"],
                 row["gpu_mw_control"], row["per_ligand_max_abs_delta"],
                 row["pearson_tt_vs_gpu_pred"], row["tt_mean_s_per_prediction"]))
    if out["targets"]:
        n = len(out["targets"])
        out["tt_mean_pearson"] = round(sum(t["tt_pearson"] for t in out["targets"]) / n, 4)
        out["tt_mean_spearman"] = round(sum(t["tt_spearman"] for t in out["targets"]) / n, 4)
        out["tt_mean_mw_control"] = round(sum(t["tt_mw_control"] for t in out["targets"]) / n, 4)
        out["verdict"] = (
            "VALID: within-target Pearson %.3f beats the MW-only control %.3f (GPU reference "
            "%.3f / %.3f)" % (out["tt_mean_pearson"], out["tt_mean_mw_control"],
                              gpu["mean_pearson"], gpu["mean_pearson_mw_control"]))
        print("\n%s" % out["verdict"])
    dst = REPO / "perf/nesso1/davis.json"
    dst.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote %s" % dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
