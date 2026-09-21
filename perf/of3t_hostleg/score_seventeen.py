#!/usr/bin/env python3
"""The seventeen HOST_APPLIED tensors, scored against the float64 reference in the MODEL's
denominator, in `shipped_vs_FLOAT64.json`'s schema.

The arm's own report scores against `sub_boundary.pt`'s `grad_f64`, which is the reference AT THE
DIFFUSION BOUNDARY and normalised by the diffusion section. The campaign's coverage claim is in
the MODEL's denominator (`model_squared_gradient_norm` 10.279642678524981 over 4,170 tensors,
from `grads_f64_043.pt`), so the reading has to be retaken there. The two references are checked
against each other here rather than assumed equal -- a per-tensor norm comparison, printed -- so
that "the boundary capture is a slice of the bundle" is a measurement and not a premise.

The scorer is `perf/of3t_trajectory/agreement.py`, imported, not reimplemented: it already
applies A14 on the float64 gradient for every pair, scores a tensor absent from an arm as ZERO
rather than dropping its mass, and keeps `r` and `cos` beside `rel_l2` so a magnitude error can
be told from a direction error (D35).

    score_seventeen.py --arm shipped=/path/dev.pt --arm refatom=/path/dev.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "perf" / "of3t_trajectory"))
import agreement  # noqa: E402  the scorer, not a copy of it

F64 = "/home/ttuser/of3t_hostleg/grads_f64_043.pt"
F64_SHA = "1d4ea9225f854afe06a8adedcb5f8aa1c53656fbf8cefdaa3a451d0327895cc4"
SECTIONS = REPO / "perf" / "of3t_orchestrator" / "SECTION_MASS_MEASURED.json"
# READABLE_MASS.json's own denominator, which is what the coverage claim divides by.
MODEL_TOTAL_SQ = 10.279642678524981

PREFIXES = ("diffusion_module.atom_attn_enc.ref_atom_feature_embedder.",
            "input_embedder.atom_attn_enc.ref_atom_feature_embedder.")
NAME = "input_embedder.atom_attn_enc.linear_q.0.weight"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[], metavar="LABEL=PATH", required=True)
    ap.add_argument("--f64", default=F64)
    ap.add_argument("--expect-f64", default=F64_SHA)
    ap.add_argument("--boundary-ref", default="/home/ttuser/of3t_hostleg/diffcap043/sub_boundary.pt",
                    help="the arm's OWN reference, compared per tensor against the bundle's")
    ap.add_argument("--out", default=str(Path(__file__).with_name("SEVENTEEN.json")))
    ap.add_argument("--sidecar-dir", default=str(Path(__file__).parent / "sidecar"))
    a = ap.parse_args()

    got = agreement.sha256(a.f64)
    if got != a.expect_f64:
        raise SystemExit(f"STOP: {a.f64} is {got}, the pin says {a.expect_f64}")
    agreement.MODEL_TOTAL_SQ = MODEL_TOTAL_SQ

    full = torch.load(a.f64, map_location="cpu", weights_only=False)
    names = sorted([n for n in full if any(n.startswith(p) for p in PREFIXES)] + [NAME])
    if len(names) != 17:
        raise SystemExit(f"STOP: the two prefixes plus one name resolve to {len(names)}, not 17")
    mass = {n: full[n].to(torch.float64).reshape(-1) if full.get(n) is not None else None
            for n in names}
    del full
    ref = {n: (v.clone() if v is not None else None) for n, v in mass.items()}
    total = sum(float(torch.linalg.vector_norm(v)) ** 2 for v in mass.values() if v is not None)
    sections = json.loads(SECTIONS.read_text()) if SECTIONS.is_file() else {}

    # Is the arm's own reference the same numbers as the bundle's, on the eight it covers?
    bnd_check = {}
    if Path(a.boundary_ref).is_file():
        S = torch.load(a.boundary_ref, map_location="cpu", weights_only=False)
        gb = S.get("grad_f64", {})
        for n in names:
            rel = n[len("diffusion_module."):] if n.startswith("diffusion_module.") else None
            v = gb.get(rel) if rel else None
            if v is None or mass[n] is None:
                continue
            x, y = v.double().reshape(-1), mass[n]
            bnd_check[n] = {
                "boundary_norm": float(torch.linalg.vector_norm(x)),
                "bundle_norm": float(torch.linalg.vector_norm(y)),
                "rel_l2_between_the_two_references":
                    float(torch.linalg.vector_norm(x - y) / (torch.linalg.vector_norm(y) + 1e-300)),
            }
        del S, gb

    out = {
        "instrument": "of3t-hostleg score_seventeen.py -- the 17 HOST_APPLIED tensors against "
                      "the float64 reference, in READABLE_MASS.json's own model denominator",
        "host": "qb1 (tt-quietbox) card 1, Blackhole p150a",
        "inputs": {"float64": {"path": a.f64, "sha256": got, "matches_pin": True}},
        "denominator": {"model_squared_gradient_norm": MODEL_TOTAL_SQ,
                        "seventeen_mass_sq": total,
                        "seventeen_pct_of_model_mass": 100.0 * total / MODEL_TOTAL_SQ},
        "names": names,
        "per_tensor_bar": agreement.PER_TENSOR_BAR,
        "two_references_compared": bnd_check,
        "arms": {},
    }
    Path(a.sidecar_dir).mkdir(parents=True, exist_ok=True)
    for spec in a.arm:
        label, path = spec.split("=", 1)
        arm = agreement.load_subset(path, names)
        rows = agreement.pair_rows(ref, arm, names, mass, sections)
        st = agreement.stat(rows, f"{label}_vs_FLOAT64")
        present = sorted(n for n in names if arm.get(n) is not None)
        out["arms"][label] = {
            "path": path,
            "sha256": agreement.sha256(path),
            "n_present_in_the_arm": len(present),
            "present": present,
            "absent": sorted(set(names) - set(present)),
            "stats": st,
            "over_bar": sorted(
                ({"param": r["param"], "rel_l2": r["rel_l2"],
                  "pct_of_model_mass": r["pct_of_model_mass"], "r": r["r"], "cos": r["cos"]}
                 for r in rows if r["rel_l2"] is not None
                 and r["rel_l2"] > agreement.PER_TENSOR_BAR),
                key=lambda x: -x["rel_l2"]),
            "rows": [{k: r.get(k) for k in
                      ("param", "rel_l2", "r", "cos", "pct_of_model_mass", "ref_norm",
                       "arm_norm", "unmeasurable")} for r in rows],
        }
        (Path(a.sidecar_dir) / f"{label}_vs_FLOAT64.json").write_text(json.dumps(rows))
        print(f"{label}: {len(present)}/17 present, "
              f"n_over_bar={st['n_over_per_tensor_bar']}/{st['n_rel_measurable']}, "
              f"worst={st['worst_rel_l2']} at {st['worst_tensor']}", flush=True)
        del arm

    # did the BREAK control move the eight?
    if "refatom" in out["arms"] and "break" in out["arms"]:
        A = {r["param"]: r["rel_l2"] for r in out["arms"]["refatom"]["rows"]}
        B = {r["param"]: r["rel_l2"] for r in out["arms"]["break"]["rows"]}
        out["break_control"] = {
            n: {"refatom_rel_l2": A.get(n), "break_rel_l2": B.get(n),
                "moved": (A.get(n) is not None and B.get(n) is not None
                          and abs(A[n] - B[n]) > 1e-9)}
            for n in names if A.get(n) is not None}
        out["break_control_n_moved"] = sum(
            1 for v in out["break_control"].values() if v["moved"])

    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print("->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
