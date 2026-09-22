#!/usr/bin/env python3
"""Host-only analysis of the two gradient arms. No device, no new measurement of the model.

Three questions the per-arm JSONs cannot answer on their own:

  1. How much of the gradient VECTOR did the mask actually move? Two distances from a shared
     reference do not order each other (D72), so arm M against arm N is measured directly.
  2. The 89 tensors still over the per-tensor bar with the masks on -- are they at the bf16
     floor, or is something else left? The floor is measured per tensor by round-tripping the
     float64 reference through bf16, which is the best any bf16 gradient could do on storage
     alone. Note what this bounds: a storage floor, not an accumulation floor, so a tensor ABOVE
     it is not thereby a defect -- but a tensor AT it has nothing left to win.
  3. Where does the remaining error sit in mass? A23: a count is not a result.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_PERF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PERF not in sys.path:
    sys.path.append(_PERF)
import refpath                                                            # noqa: E402

import torch

D = Path(__file__).resolve().parent
L = Path("/home/ttuser/of3t_auxgrad_logs")
BAR = 5.0e-2
REF = Path(refpath.BUNDLE) / "grads_f64_043.pt"


def main():
    M = torch.load(L / "grads_M.pt", map_location="cpu", weights_only=False)
    N = torch.load(L / "grads_N.pt", map_location="cpu", weights_only=False)
    R = torch.load(REF, map_location="cpu", weights_only=False)
    rows_M = {r["name"]: r for r in json.loads((D / "grad_M_masked.json").read_text())
              ["gradient"]["rows"]}
    out = {"bars": {"per_tensor": BAR}, "n_dumped": len(M)}

    keys = [k for k in M if k in N and k in R and R[k] is not None]
    dm = sr = dn = mn = 0.0
    per = []
    for k in keys:
        r = R[k].double().flatten()
        m, n = M[k].double().flatten(), N[k].double().flatten()
        # the bf16 STORAGE floor: the reference written into bf16 and read back.
        f = r.to(torch.bfloat16).double()
        rn = float(r.norm()) + 1e-300
        per.append({
            "name": k, "ref_norm": float(r.norm()), "ref_sq": float((r ** 2).sum()),
            "rel_M": float((m - r).norm() / rn), "rel_N": float((n - r).norm() / rn),
            "rel_M_vs_N": float((m - n).norm() / (float(n.norm()) + 1e-300)),
            "bf16_storage_floor": float((f - r).norm() / rn),
        })
        dm += float(((m - r) ** 2).sum()); dn += float(((n - r) ** 2).sum())
        sr += float((r ** 2).sum()); mn += float(((m - n) ** 2).sum())
    out["mass_weighted"] = {
        "rel_M": dm ** 0.5 / sr ** 0.5, "rel_N": dn ** 0.5 / sr ** 0.5,
        "M_vs_N_over_ref": mn ** 0.5 / sr ** 0.5,
        "n_tensors": len(keys),
        "note": "over the dumped tensors, which fuse the p_in/g_in halves the per-row scoring "
                "splits, so this set is 180 tensors against the 176 scored rows",
    }

    # the tensors still over the bar with the masks on, and what they are worth
    over = [p for p in per if p["rel_M"] > BAR]
    inside = [p for p in per if p["rel_M"] <= BAR]
    tot = sum(p["ref_sq"] for p in per) or 1.0
    at_floor = [p for p in over if p["rel_M"] <= 2.0 * p["bf16_storage_floor"]]
    out["still_over_bar"] = {
        "n": len(over), "n_inside": len(inside),
        "share_of_compared_mass": sum(p["ref_sq"] for p in over) / tot,
        "ref_norm_max": max((p["ref_norm"] for p in over), default=None),
        "ref_norm_min": min((p["ref_norm"] for p in over), default=None),
        "largest_inside_ref_norm": max((p["ref_norm"] for p in inside), default=None),
        "n_within_2x_of_bf16_storage_floor": len(at_floor),
        "median_rel_M_over_its_own_bf16_floor": (
            sorted(p["rel_M"] / (p["bf16_storage_floor"] + 1e-300) for p in over)[len(over) // 2]
            if over else None),
        "worst": max(per, key=lambda p: p["rel_M"]),
    }
    per.sort(key=lambda p: -p["rel_M"])
    out["per_tensor"] = per
    (D / "analysis.json").write_text(json.dumps(out, indent=1) + "\n")

    mw = out["mass_weighted"]; so = out["still_over_bar"]
    print(f"mass-weighted over {mw['n_tensors']} dumped tensors: "
          f"M {mw['rel_M']:.6e}  N {mw['rel_N']:.6e}  |M-N|/|ref| {mw['M_vs_N_over_ref']:.6e}")
    print(f"still over the {BAR:.1e} bar with the masks on: {so['n']} of {len(per)}, holding "
          f"{so['share_of_compared_mass']:.4e} of the compared mass")
    print(f"  their reference norms span {so['ref_norm_min']:.4e} to {so['ref_norm_max']:.4e}; "
          f"the largest tensor INSIDE the bar has norm {so['largest_inside_ref_norm']:.4e}")
    print(f"  {so['n_within_2x_of_bf16_storage_floor']} of them are within 2x of their own bf16 "
          f"STORAGE floor; median rel_M / own floor = "
          f"{so['median_rel_M_over_its_own_bf16_floor']:.4g}")
    print(f"  worst {so['worst']['name']}")
    print(f"    rel_M {so['worst']['rel_M']:.4e}  rel_N {so['worst']['rel_N']:.4e}  "
          f"bf16 storage floor {so['worst']['bf16_storage_floor']:.4e}")
    print("\ntop 12 by rel_M (name, rel_M, rel_N, bf16 storage floor, ref_norm):")
    for p in per[:12]:
        print(f"  {p['rel_M']:.4e}  {p['rel_N']:.4e}  {p['bf16_storage_floor']:.4e}  "
              f"{p['ref_norm']:.3e}  {p['name'][len('aux_heads.'):]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
