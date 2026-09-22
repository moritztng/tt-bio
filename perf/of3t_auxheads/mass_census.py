#!/usr/bin/env python3
"""PROTOCOL A23: where a sections gradient mass actually is, and the mass-weighted headline.

A23 binds three things on any set statistic: report the fraction of reference mass the set
holds, make the HEADLINE mass-weighted with the median beside it rather than instead of it,
and where mass is concentrated, name the heavy tensors individually.

This does both halves. `--census` ranks every tensor of a section of the 0.4.3 reference by
its share of the sections squared gradient norm, so the heavy tensors are named before any
comparison is taken. `--score` reads an `aux_instrument.py` report and computes the
mass-weighted relative L2 over the concatenated compared set:

    rel_l2_concat = sqrt( sum_i rel_l2_i^2 * ref_sq_i / sum_i ref_sq_i )

which is exact, because ||g_ours - g_ref||^2 over a concatenation is the sum of the
per-tensor squared differences and each of those is (rel_l2_i * ||ref_i||)^2. So the
headline needs no second device run -- it is recoverable from the per-tensor array the
report already carries. (It would NOT be recoverable from a worst-10/best-10 summary, which
is the `result-file-keeping-only-extremes-cannot-be-reanalysed` lesson.)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def census(ref_path: Path, sections: list[str], top: int) -> dict:
    ref = torch.load(ref_path, map_location="cpu", weights_only=False)
    sq = {k: float((v.double() ** 2).sum()) for k, v in ref.items() if v is not None}
    model = sum(sq.values())
    out = {"model_squared_norm": model, "n_tensors": len(sq), "sections": {}}
    for s in sections:
        keys = [k for k in sq if k == s or k.startswith(s + ".")]
        tot = sum(sq[k] for k in keys)
        ranked = sorted(keys, key=lambda k: -sq[k])
        cum, n_for_99 = 0.0, 0
        for k in ranked:
            cum += sq[k]
            n_for_99 += 1
            if tot and cum / tot >= 0.99:
                break
        out["sections"][s] = {
            "n_tensors": len(keys),
            "squared_norm": tot,
            "share_of_model": tot / model if model else None,
            "n_tensors_holding_99pct": n_for_99,
            "top": [{"name": k, "share_of_section": sq[k] / tot if tot else None,
                     "share_of_model": sq[k] / model if model else None,
                     "shape": list(ref[k].shape)} for k in ranked[:top]],
        }
    return out


def score(report_path: Path) -> dict:
    rep = json.loads(report_path.read_text())
    rows = rep["gradient"]["rows"]
    tot = sum(r["ref_sq"] for r in rows)
    num = sum(r["rel_l2"] ** 2 * r["ref_sq"] for r in rows)
    ranked = sorted(rows, key=lambda r: -r["ref_sq"])
    meds = sorted(r["rel_l2"] for r in rows)
    # the mass-weighted share of the set that agrees: mass of tensors inside the bar
    inside = sum(r["ref_sq"] for r in rows if r["rel_l2"] <= 5e-2)
    return {
        "report": str(report_path),
        "n_tensors": len(rows),
        "mass_weighted_rel_l2": math.sqrt(num / tot) if tot else None,
        "median_over_tensors": meds[len(meds) // 2] if meds else None,
        "mass_inside_per_tensor_bar": inside / tot if tot else None,
        "n_inside_per_tensor_bar": sum(1 for r in rows if r["rel_l2"] <= 5e-2),
        "heaviest": [{"name": r["name"], "rel_l2": r["rel_l2"],
                      "share_of_compared_mass": r["ref_sq"] / tot} for r in ranked[:5]],
        "reach": rep["gradient"].get("reach"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-grads", type=Path)
    ap.add_argument("--sections", default="aux_heads,msa_module,input_embedder")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--score", type=Path, action="append", default=[])
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    out = {}
    if a.reference_grads:
        out["census"] = census(a.reference_grads, a.sections.split(","), a.top)
        for s, e in out["census"]["sections"].items():
            print("%-16s %4d tensors  %8.4f %% of the model  99 %% of it in %d tensor(s)"
                  % (s, e["n_tensors"], e["share_of_model"] * 100, e["n_tensors_holding_99pct"]))
            for t in e["top"][:4]:
                print("     %8.4f %% of the section  %8.5f %% of the model  %s %s"
                      % (t["share_of_section"] * 100, t["share_of_model"] * 100,
                         t["name"], t["shape"]))
    if a.score:
        out["scored"] = [score(p) for p in a.score]
        for s in out["scored"]:
            print("\n%s\n  mass-weighted rel_l2 %.4e over %d tensors; median %.4e; "
                  "mass inside the 5.0e-02 bar %.6f %% (%d tensors)"
                  % (s["report"], s["mass_weighted_rel_l2"], s["n_tensors"],
                     s["median_over_tensors"], s["mass_inside_per_tensor_bar"] * 100,
                     s["n_inside_per_tensor_bar"]))
            for h in s["heaviest"][:3]:
                print("    %8.5f %% of compared mass  rel_l2 %.4e  %s"
                      % (h["share_of_compared_mass"] * 100, h["rel_l2"], h["name"]))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
