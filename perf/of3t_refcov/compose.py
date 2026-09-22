#!/usr/bin/env python3
"""of3t-refcov: the seventeen HOST_APPLIED tensors, composed into the model-scope union.

`of3t-covdefault` read `input_embedder` at `n_compared 0 of 98` and concluded that
`linear_q.0.weight` "cannot enter the reading either". The sentence is true of the PUBLISHED
union and the conclusion does not follow: the arm exists. `of3t-hostleg` built and controlled
both of them, on upstream 0.4.3's own captured cotangent, and neither was ever composed into
`perf/of3t_modelboundary/MODEL_withtrunk_n384.json`.

So this composes them. No shipped code changes, no arm re-runs, no new boundary: the union's
3,643 rows come from the published sidecar, the 17 new rows are scored here against the SAME
pinned float64 reference through `of3t-trajectory`'s scorer, and every gate the composition
has to pass is checked before the answer is written.

    python3 perf/of3t_refcov/compose.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "perf" / "of3t_trajectory"))
import agreement  # noqa: E402

F64 = Path("/home/ttuser/of3t_hostleg/grads_f64_043.pt")
PIN = "1d4ea9225f854afe06a8adedcb5f8aa1c53656fbf8cefdaa3a451d0327895cc4"
IE_ARM = Path("/home/ttuser/of3t_hostleg/ie_grads_f64.pt")
DM_ARM = Path("/home/ttuser/of3t_hostleg/device_grads_hl_refatom.pt")
DM_OFF = Path("/home/ttuser/of3t_hostleg/device_grads_hl_shipped.pt")
SIDE = REPO / "perf/of3t_modelboundary/sidecar_modelboundary"
BOUNDARY = REPO / "perf/of3t_modelboundary/MODEL_withtrunk_n384.json"
SECTIONS = REPO / "perf/of3t_orchestrator/SECTION_MASS_MEASURED.json"
SEVENTEEN = REPO / "perf/of3t_hostleg/SEVENTEEN.json"
IE_PUB = REPO / "perf/of3t_hostleg/IE_ARM_f64.json"
DM_PUB = REPO / "perf/of3t_hostleg/device_gradient_hl_refatom_per_tensor.json"
BAR = 99.2594
OUT = Path(__file__).with_name("COVERAGE_COMPOSED.json")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def load_grads(p: Path) -> dict:
    d = torch.load(p, map_location="cpu", mmap=True)
    for k in ("grads", "state_dict"):
        if isinstance(d, dict) and k in d and isinstance(d[k], dict):
            d = d[k]
            break
    return {k: v for k, v in d.items() if v is not None}


def main() -> int:
    gates: list[dict] = []

    def gate(name, ok, detail):
        gates.append({"gate": name, "pass": bool(ok), "detail": detail})
        print(("  PASS " if ok else "  FAIL ") + name + ": " + str(detail), flush=True)
        return ok

    got = sha256(F64)
    print(f"float64 reference {F64} sha256 {got}", flush=True)
    gate("float64 reference matches the campaign pin", got == PIN, got)

    ref_full = load_grads(F64)
    denom = float(sum(float(v.double().norm() ** 2) for v in ref_full.values()))
    gate("denominator reproduces the published 10.279642678524981",
         abs(denom - 10.279642678524981) / 10.279642678524981 < 1e-14, repr(denom))
    agreement.MODEL_TOTAL_SQ = denom

    mass = {k: float(v.double().norm() ** 2) for k, v in ref_full.items()}
    pct = lambda names: 100.0 * sum(mass[n] for n in names if n in mass) / denom  # noqa: E731

    # --- the published union, from its own sidecar ---------------------------------------
    pub_rows = json.loads((SIDE / "shipped_vs_FLOAT64.json").read_text())
    union = [r["param"] for r in pub_rows]
    bd = json.loads(BOUNDARY.read_text())
    shipped_pct = bd["coverage_total"]["pct_of_model_compared"]
    gate("the sidecar's 3,643 names reproduce the published coverage",
         abs(pct(union) - shipped_pct) < 1e-9,
         f"recomputed {pct(union)!r} vs published {shipped_pct!r}")
    gate("union has no duplicate names", len(set(union)) == len(union), len(union))

    # --- the two arms that were never composed --------------------------------------------
    ie = load_grads(IE_ARM)
    on, off = load_grads(DM_ARM), load_grads(DM_OFF)
    dm_new = sorted(set(on) - set(off))
    new = {n: ie[n].double().reshape(-1) for n in ie}
    new.update({n: on[n].double().reshape(-1) for n in dm_new})

    s17 = set(json.loads(SEVENTEEN.read_text())["names"])
    gate("the entering key set is exactly SEVENTEEN.json",
         set(new) == s17, f"{len(new)} entering, {len(set(new) ^ s17)} symmetric difference")
    gate("nothing entering is already in the union",
         not (set(new) & set(union)), sorted(set(new) & set(union))[:5] or "disjoint")
    gate("the diffusion arm adds exactly 8 keys over its own shipped twin",
         len(dm_new) == 8 and not (set(off) - set(on)), f"{len(dm_new)} added, "
         f"{len(set(off) - set(on))} removed")
    gate("every entering tensor has a gradient in the reference",
         all(n in ref_full for n in new), sorted(n for n in new if n not in ref_full)[:3])

    # --- score the 17 against the SAME reference, through the SAME scorer -------------------
    sections = list(json.loads(SECTIONS.read_text())["sections_pct_of_model"])
    names_new = sorted(new)
    ref_sub = {n: ref_full[n].double().reshape(-1) for n in names_new}
    mass_sub = {n: ref_full[n].double().reshape(-1) for n in names_new}
    new_rows = agreement.pair_rows(ref_sub, new, names_new, mass_sub, sections)

    # P3: the recomputed readings against the MODEL reference vs the arms' own published rows
    pub = {r["param"]: r["rel_l2"] for r in json.loads(IE_PUB.read_text())["per_tensor"]}
    for r in json.loads(DM_PUB.read_text())["per_tensor"]:
        pub["diffusion_module." + r["tensor"]] = r["rel_l2"]
    repro = []
    worst_repro = 0.0
    for r in new_rows:
        p, q = pub.get(r["param"]), r["rel_l2"]
        rel = None if (p is None or q is None or p == 0) else abs(q - p) / p
        worst_repro = max(worst_repro, rel or 0.0)
        repro.append({"param": r["param"], "published": p, "recomputed": q, "rel_diff": rel})
    gate("the arms' published per-tensor readings reproduce against the MODEL reference",
         worst_repro < 1e-6, f"worst relative disagreement {worst_repro:.3e}")

    # --- coverage --------------------------------------------------------------------------
    composed = union + names_new
    cov_new = pct(composed)
    predicted = 99.50523155277438
    gate("coverage matches the pre-registered prediction within 1e-6",
         abs(cov_new - predicted) < 1e-6, f"measured {cov_new!r} vs predicted {predicted!r}")

    by_section: dict = {}
    for n, v in ref_full.items():
        sec = agreement.section_of(n, sections)
        c = by_section.setdefault(sec, {"n_total": 0, "n_compared_before": 0,
                                        "n_compared_after": 0, "mass_sq": 0.0,
                                        "mass_sq_before": 0.0, "mass_sq_after": 0.0})
        c["n_total"] += 1
        c["mass_sq"] += mass[n]
        if n in set(union):
            c["n_compared_before"] += 1
            c["mass_sq_before"] += mass[n]
        if n in set(composed):
            c["n_compared_after"] += 1
            c["mass_sq_after"] += mass[n]
    for c in by_section.values():
        c["pct_of_model"] = 100.0 * c["mass_sq"] / denom
        c["pct_compared_before"] = 100.0 * c["mass_sq_before"] / denom
        c["pct_compared_after"] = 100.0 * c["mass_sq_after"] / denom
        for k in ("mass_sq", "mass_sq_before", "mass_sq_after"):
            c.pop(k)

    # --- accuracy: the headline before and after, both arms --------------------------------
    stats = {}
    for arm in ("shipped", "renorm"):
        rows = json.loads((SIDE / f"{arm}_vs_FLOAT64.json").read_text())
        before = agreement.stat(rows, f"{arm}_vs_FLOAT64_before")
        if arm == "shipped":
            add = new_rows
        else:
            # The renorm arm has NO twin of either new arm. model_scope scores a tensor the
            # arm does not carry as zero, which is the honest reading and not a hole: its
            # rel_l2 is 1.0 and its mass stays in the denominator.
            add = agreement.pair_rows(ref_sub, {}, names_new, mass_sub, sections)
        after = agreement.stat(rows + add, f"{arm}_vs_FLOAT64_after")
        stats[arm] = {"before": before, "after": after}
        gate(f"{arm}: the mass-weighted reading does not get worse",
             after["mass_weighted_rel_l2"] <= before["mass_weighted_rel_l2"] + 1e-12,
             f"{before['mass_weighted_rel_l2']!r} -> {after['mass_weighted_rel_l2']!r}")

    over = [r["param"] for r in new_rows if (r["rel_l2"] or 0.0) > 0.05]
    out = {
        "instrument": "of3t-refcov compose.py -- the 17 HOST_APPLIED tensors composed into "
                      "the model-scope union from two arms of3t-hostleg already built and "
                      "controlled on upstream 0.4.3's own captured cotangent",
        "host": "qb1 (tt-quietbox), CPU only, no device opened",
        "route": "the union, not the training adapter. See BOUNDARY_ADAPTER.json for why the "
                 "adapter cannot contribute to this reading at any precision.",
        "inputs": {
            "float64_reference": {"path": str(F64), "sha256": got, "matches_pin": got == PIN},
            "input_embedder_leg": {"path": str(IE_ARM), "sha256": sha256(IE_ARM),
                                   "n_keys": len(ie),
                                   "arm": "perf/of3t_hostleg/ie_arm.py, seeded with upstream's "
                                          "cotangent at cl/plm/ai (IE_BOUNDARY.json)"},
            "diffusion_refatom": {"path": str(DM_ARM), "sha256": sha256(DM_ARM),
                                  "n_keys": len(on), "n_over_its_shipped_twin": len(dm_new),
                                  "arm": "perf/of3t_diffusion/device_gradient.py "
                                         "--device-refatom, no env flag, no shipped-code change"},
            "published_union": {"path": str(SIDE / "shipped_vs_FLOAT64.json"),
                                "n_names": len(union)},
        },
        "denominator": {"recomputed": denom, "published": 10.279642678524981},
        "gates": gates,
        "all_gates_pass": all(g["pass"] for g in gates),
        "coverage": {
            "before": shipped_pct, "after": cov_new, "delta": cov_new - shipped_pct,
            "bar": BAR, "after_minus_bar": cov_new - BAR, "bar_reached": cov_new >= BAR,
            "n_compared_before": len(union), "n_compared_after": len(composed),
            "n_reference_tensors": len(ref_full),
            "keys_that_entered": names_new,
            "pct_contributed": {
                "diffusion_module_ref_atom_feature_embedder_8": pct(dm_new),
                "input_embedder_leg_9": pct(sorted(ie)),
            },
        },
        "coverage_by_section": by_section,
        "accuracy": {
            "per_tensor_bar": 0.05,
            "entering_over_the_per_tensor_bar": over,
            "n_entering_over_the_bar": len(over),
            "worst_entering": max(new_rows, key=lambda r: r["rel_l2"] or -1.0)["param"],
            "worst_entering_rel_l2": max((r["rel_l2"] or -1.0) for r in new_rows),
            "per_tensor": sorted(new_rows, key=lambda r: -(r["rel_l2"] or -1.0)),
            "reproduction_against_the_model_reference": repro,
            "headline": stats,
        },
    }
    OUT.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print("\ncoverage %.14f -> %.14f (bar %.4f, %s by %.14f)"
          % (shipped_pct, cov_new, BAR, "reached" if cov_new >= BAR else "SHORT",
             abs(cov_new - BAR)), flush=True)
    print("entering: %d tensors, %d over the 5.0e-02 per-tensor bar, worst %s at %.9g"
          % (len(names_new), len(over), out["accuracy"]["worst_entering"],
             out["accuracy"]["worst_entering_rel_l2"]), flush=True)
    for arm, s in stats.items():
        print("  %-8s mass-weighted vs float64 %.15g -> %.15g"
              % (arm, s["before"]["mass_weighted_rel_l2"], s["after"]["mass_weighted_rel_l2"]),
              flush=True)
    print("->", OUT, flush=True)
    return 0 if out["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
