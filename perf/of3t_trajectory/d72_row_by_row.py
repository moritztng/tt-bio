#!/usr/bin/env python3
"""D72's table with the shared subtrahend removed, row by row.

D72 scored each section's distance from float64 against upstream bf16's distance from float64.
Both columns share a reference, and two distances from one reference do not order each other.
This adds the column that does: each section's distance from upstream's bf16 gradient itself.
Nothing is re-run and nothing is copied by hand -- the bf16 and device columns come out of
SCORED_AGAINST_THE_RECIPES_OWN_FLOOR.json and the direct column out of
AGREEMENT_WITH_UPSTREAMS_OWN_STEP.json, and each row names where its numbers came from.
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d72", required=True, type=Path)
    ap.add_argument("--agreement", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    d72 = json.loads(a.d72.read_text())
    ag = json.loads(a.agreement.read_text())

    def sets(label):
        return {s["set"]: s for s in ag["pairs"][label]["sets"]}

    direct = sets("DEVICE_vs_UPSTREAM_BF16")
    permcot = sets("DEVICE_PERMUTED_COTANGENT_vs_UPSTREAM_BF16")
    bf16 = sets("UPSTREAM_BF16_vs_FLOAT64")

    rows = []
    for r in d72["table"]:
        key = f"section {r['section']}"
        d = direct.get(key)
        row = {
            "section": r["section"],
            "pct_of_model_mass_D72": r["pct"],
            "bf16_vs_float64_D72": r["bf16"],
            "device_vs_float64_D72": r["device"],
            "D72_band": r["band"],
            "D72_device_source": r["device_source"],
        }
        if d is None:
            row.update(
                device_vs_bf16_direct=None,
                verdict="outside the device arm's 547-tensor scope, so this row has no direct "
                        "measurement. D72's reading for it stands unchecked.",
                source="not covered by " + a.agreement.name)
            rows.append(row)
            continue
        own_floor = bf16[key]["mass_weighted_rel_l2"]
        h = d["mass_weighted_rel_l2"]
        row.update(
            pct_of_model_mass_here=d["pct_of_model_mass"],
            device_vs_bf16_direct=h,
            median_over_tensors=d["median_rel_l2_over_tensors"],
            norm_ratio=d["mass_weighted_norm_ratio"],
            cos=d["mass_weighted_cos"],
            bf16_vs_float64_here=own_floor,
            multiples_of_this_sections_own_bf16_floor=h / own_floor,
            worst_tensor=d["worst_tensor"],
            worst_rel_l2=d["worst_rel_l2"],
            break_control_permuted_cotangent=permcot[key]["mass_weighted_rel_l2"],
            source=f"{a.agreement.name}, set {key!r}")
        if h <= own_floor:
            row["verdict"] = ("D72's reading SURVIVES the direct test: we are no further from "
                              "their bf16 gradient than their bf16 gradient is from the ideal")
        elif r["band"] == "AT OR BETTER":
            row["verdict"] = ("D72 read this section AT OR BETTER against float64 and the "
                              "direct test does NOT confirm it: the two errors point partly in "
                              "different directions, so our distance from their gradient is "
                              f"{h / own_floor:.2f}x this section's own bf16 floor")
        else:
            row["verdict"] = (f"unchanged by removing the shared subtrahend: "
                              f"{h / own_floor:.1f}x this section's own bf16 floor")
        rows.append(row)

    scope = direct["the device arm's scope (all compared tensors)"]
    out = {
        "what": __doc__.strip().splitlines()[0],
        "scope_of_the_direct_column": {
            "n_tensors": ag["scope"]["n_tensors"],
            "pct_of_model_mass": ag["scope"]["pct_of_model_mass"],
            "what_it_covers": "diffusion_module minus diffusion_conditioning. The four sections "
                              "D72 filled from other rows' arms (conditioning, aux_heads, "
                              "msa_module) and the two with no arm at all are outside it.",
        },
        "headline": (f"{scope['mass_weighted_rel_l2']:.6e} mass-weighted over "
                     f"{scope['n']} tensors, median {scope['median_rel_l2_over_tensors']:.6e}, "
                     f"norm ratio {scope['mass_weighted_norm_ratio']:.6f}, cos "
                     f"{scope['mass_weighted_cos']:.6f}"),
        "table": rows,
        "provenance": {"D72": str(a.d72), "direct": str(a.agreement)},
    }
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    for r in rows:
        print(f"{r['section']:46s} {r['pct_of_model_mass_D72']:7.4f} % "
              f"{str(r['device_vs_bf16_direct'])[:12]:>12s}  {r['verdict'][:60]}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
