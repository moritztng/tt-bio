#!/usr/bin/env python3
"""of3t-cotcoh: OUR per-block cotangent error curve in a form `of3t-twoside` can put its own
curve beside, and upstream's own bf16 curve from the SAME entry when it exists.

D241 (pass 380): the trunk clause divides two different experiments. Our arm is handed the
reference's float64 boundary AND its float64 cotangent; the bf16 denominator it is divided by is
a FULL-MODEL bf16 run that drove its own trunk with its own bf16 pair. So a per-block number of
ours is a measurement OF OURS and is not an excess over upstream until upstream is measured from
the same entry. This file states the entry, the mask and the reference on every row so a second
row can produce a comparable curve without reading this one's code.

The `upstream_bf16` columns are populated only when `--upstream-cot` is given. That capture is
produced by the same script that produced the float64 one, `perf/of3t_cotcoh/refcot.py`, with
`--policy bf16auto` and everything else identical, which is what makes the two columns
comparable to each other.
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess

import torch

CONV = {
    "entry": "of3t-modelframe's boundary_model_n384.pt (sha256 583bcd7c91ce6ed46aea59844954fb"
             "2bf7ddc3d553d7f9d4f238998183db99e2) and cot_model_n384.pt (4e66d1ef45da2eec18fdff"
             "9489d68e980df928dc43d10fd4af82d14ec67141ac), the reference's own float64 pair. "
             "EVERY arm in this file is driven from that same pair; that is the whole point.",
    "reference": "upstream OpenFold3 0.4.3's own stack in float64 on that entry, "
                 "perf/of3t_cotcoh/refcot.py --policy f64",
    "quantity": "the cotangent ARRIVING at the LayerNorm output, i.e. what the affine gradient "
                "dW = sum_t g_t xhat_t is a linear function of",
    "mask": "the 56 real tokens of 384. Family A keeps rows 0..55 of the single track; family B "
            "keeps the 56x56 real block of the pair track. Unmasked at this width is pad junk: "
            "of3t-apbleaf measured 1.3827 unmasked against 1.7e-03 masked on the same object.",
    "families": {"A": "attn_pair_bias.layer_norm_a, ours pre_norm_s, 42.004 % of the trunk's "
                      "error mass",
                 "B": "pair_stack.pair_transition.layer_norm, ours transition_z, 32.408 %"},
    "family_B_note": "the pair-track site fires the taped LayerNorm backward 12 times per block "
                     "with 2 of them on the real block. The effective cotangent is their SUM, "
                     "which is exact because dW is linear in g and both fires share the operand "
                     "(x_max_abs_diff_between_fires is 0.0 at all 48 sites).",
    "how_to_reproduce_the_other_side": "run perf/of3t_cotcoh/refcot.py with --policy bf16auto "
                                       "and otherwise identical arguments, then pass its capture "
                                       "to this script as --upstream-cot.",
}


def stats(G, R):
    g = G.to(torch.float64)
    r = R.to(torch.float64)
    ng = float(torch.linalg.vector_norm(g))
    nr = float(torch.linalg.vector_norm(r))
    d = float(torch.linalg.vector_norm(g - r))
    return {"norm": ng, "ref_norm": nr, "err_norm": d,
            "rel_l2_vs_float64": d / nr if nr > 0 else None,
            "cos_vs_float64": (float((g * r).sum()) / (ng * nr)) if ng > 0 and nr > 0 else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--upstream-cot", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    O = torch.load(a.ours, map_location="cpu", weights_only=False)["sites"]
    Rf = torch.load(a.ref, map_location="cpu", weights_only=False)["sites"]
    U = {}
    if a.upstream_cot:
        U = torch.load(a.upstream_cot, map_location="cpu", weights_only=False)["sites"]

    rows, fam_tot = [], {}
    for fam in ("A", "B"):
        eo = er = eu = 0.0
        for b in range(48):
            k = f"{fam}:{b}"
            if k not in O or k not in Rf:
                continue
            ours = stats(O[k]["g"], Rf[k]["g"])
            row = {"family": fam, "block": b, "P": int(O[k]["g"].shape[0]),
                   "C": int(O[k]["g"].shape[1]),
                   "ours": ours}
            eo += ours["err_norm"] ** 2
            er += ours["ref_norm"] ** 2
            if k in U:
                up = stats(U[k]["g"], Rf[k]["g"])
                row["upstream_bf16"] = up
                row["ours_over_upstream_bf16"] = (
                    ours["rel_l2_vs_float64"] / up["rel_l2_vs_float64"]
                    if up["rel_l2_vs_float64"] else None)
                eu += up["err_norm"] ** 2
            rows.append(row)
        fam_tot[fam] = {
            "pooled_ours_rel_l2_vs_float64": (eo / er) ** 0.5 if er > 0 else None,
            "pooled_upstream_bf16_rel_l2_vs_float64": ((eu / er) ** 0.5 if (U and er > 0)
                                                       else None),
            "pooled_ours_over_upstream_bf16": (((eo / eu) ** 0.5) if (U and eu > 0) else None),
            "n_blocks": sum(1 for r in rows if r["family"] == fam)}

    out = {"what": "our per-block cotangent error at the two worst affine leaf families, masked, "
                   "against the reference's own float64, with upstream's own bf16 from the SAME "
                   "entry beside it where measured",
           "host": socket.gethostname(), "device_involved": False,
           "why_no_aiclk": "the device arm that produced the ours capture records its own "
                           "DURING-sampled AICLK; this is a CPU scoring step",
           "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                        text=True).stdout.strip(),
           "D241": "a number in the `ours` column is a measurement OF OURS. It becomes an excess "
                   "only against the `upstream_bf16` column, which is the same recipe from the "
                   "same entry. The 2.9702x trunk multiple in the campaign's clause divides our "
                   "arm on the float64 entry by a FULL-MODEL bf16 run on its own bf16 entry and "
                   "is not like-for-like.",
           "conventions": CONV,
           "upstream_bf16_present": bool(U),
           "inputs": {"ours": a.ours, "float64_reference": a.ref,
                      "upstream_bf16": a.upstream_cot or None},
           "family_totals": fam_tot,
           "curve": rows}
    json.dump(out, open(a.out, "w"), indent=1)
    for fam in ("A", "B"):
        t = fam_tot[fam]
        print(f"{fam}: pooled ours {t['pooled_ours_rel_l2_vs_float64']}")
        if t["pooled_upstream_bf16_rel_l2_vs_float64"]:
            print(f"   upstream bf16 {t['pooled_upstream_bf16_rel_l2_vs_float64']}  "
                  f"ours/upstream {t['pooled_ours_over_upstream_bf16']}")
        for b in (47, 46, 44, 38, 24, 15, 4, 0):
            r = next((x for x in rows if x["family"] == fam and x["block"] == b), None)
            if not r:
                continue
            u = r.get("upstream_bf16", {}).get("rel_l2_vs_float64")
            print("   blk %2d ours %10.6f  upstream %s  ratio %s"
                  % (b, r["ours"]["rel_l2_vs_float64"],
                     ("%10.6f" % u) if u else "        --",
                     ("%8.4f" % r["ours_over_upstream_bf16"])
                     if r.get("ours_over_upstream_bf16") else "      --"))
    print(json.dumps({"out": a.out, "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
