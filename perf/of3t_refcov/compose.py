#!/usr/bin/env python3
"""of3t-refcov: the seventeen HOST_APPLIED tensors, composed into the model-scope union.

`of3t-covdefault` read `input_embedder` at `n_compared 0 of 98` and concluded that
`linear_q.0.weight` "cannot enter the reading either". The sentence is true of the PUBLISHED
union and the conclusion does not follow: the arm exists. `of3t-hostleg` built and controlled
both of them on upstream 0.4.3's own captured cotangent, and neither was ever composed into
`perf/of3t_modelboundary/MODEL_withtrunk_n384.json`. So this composes them.

Two things had to be read rather than assumed, and both changed the answer.

WHICH ARM A DUMP BELONGS TO IS A READING, NOT ITS FILENAME. `tt_bio/autograd.py:87` has read
`env_flag("TT_BIO_SOFTMAX_BW_RENORM", True)` since D56 (2de9355d1, 2026-09-21T12:26:54Z), so the
softmax-backward repair is the DEFAULT and `--softmax-bw-renorm` is a no-op. Both of3t-hostleg
dumps ran that evening at 21:59Z and both stamped `softmax_bw_renorm_live: true`:
`device_grads_hl_shipped.pt` is a RENORM arm whatever its name says. Every arm here is re-run
in both tree states so each enters the arm of the union it belongs to.

THE LEVER DOES NOT ONLY ADD EIGHT TENSORS. `--device-refatom` moves `cl0`/`plm0` from host
float32 to the card, which is an input to the whole DiffusionModule, so all 547 gradients the
scope already had move with it. The scope is therefore REPLACED by its with-lever version, not
augmented with eight rows beside 547 stale ones. Control C1 is what makes that safe to say: the
no-lever arm re-run at HEAD reproduces the published scope exactly.

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
SC = Path("/tmp/of3t/of3t-refcov")
# union arm label -> (with-lever diffusion scope, input-embedder leg, no-lever control)
ARMS = {
    "shipped": (SC / "device_grads_rc_refatom_off.pt", SC / "ie_grads_f64_renorm_off.pt",
                SC / "device_grads_rc_plain_off.pt"),
    "renorm": (SC / "device_grads_rc_refatom_on.pt", SC / "ie_grads_f64_renorm_on.pt",
               Path("/home/ttuser/of3t_hostleg/device_grads_hl_shipped.pt")),
}
SIDE = REPO / "perf/of3t_modelboundary/sidecar_modelboundary"
BOUNDARY = REPO / "perf/of3t_modelboundary/MODEL_withtrunk_n384.json"
SECTIONS = REPO / "perf/of3t_orchestrator/SECTION_MASS_MEASURED.json"
SEVENTEEN = REPO / "perf/of3t_hostleg/SEVENTEEN.json"
BAR = 99.2594
PER_TENSOR_BAR = 0.05
MASS_BAR = 0.02
CTRL_TOL = 1e-6
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
        gates.append({"gate": name, "pass": bool(ok), "detail": str(detail)})
        print(("  PASS " if ok else "  FAIL ") + name + ": " + str(detail), flush=True)
        return ok

    got = sha256(F64)
    gate("float64 reference matches the campaign pin", got == PIN, got)
    ref_full = load_grads(F64)
    denom = float(sum(float(v.double().norm() ** 2) for v in ref_full.values()))
    gate("denominator reproduces the published 10.279642678524981",
         abs(denom - 10.279642678524981) / 10.279642678524981 < 1e-14, repr(denom))
    agreement.MODEL_TOTAL_SQ = denom
    mass = {k: float(v.double().norm() ** 2) for k, v in ref_full.items()}
    pct = lambda ns: 100.0 * sum(mass[n] for n in ns if n in mass) / denom  # noqa: E731
    sections = list(json.loads(SECTIONS.read_text())["sections_pct_of_model"])

    def score(arm_d, names):
        sub = {n: ref_full[n].double().reshape(-1) for n in names}
        return agreement.pair_rows(
            sub, {n: arm_d[n].double().reshape(-1) for n in names if n in arm_d},
            names, dict(sub), sections)

    def drift(rows, published):
        """Worst and mass-weighted disagreement of recomputed rows against published ones."""
        worst, worst_n, num, den = 0.0, None, 0.0, 0.0
        for r in rows:
            p, q = published.get(r["param"]), r["rel_l2"]
            if p in (None, 0) or q is None:
                continue
            d = abs(q - p) / p
            num += r["mass_sq"] * d
            den += r["mass_sq"]
            if d > worst:
                worst, worst_n = d, r["param"]
        return {"worst_rel_diff": worst, "worst_tensor": worst_n,
                "mass_weighted_rel_diff": (num / den) if den else None}

    pub_rows = {a: json.loads((SIDE / f"{a}_vs_FLOAT64.json").read_text()) for a in ARMS}
    union = [r["param"] for r in pub_rows["shipped"]]
    bd = json.loads(BOUNDARY.read_text())
    shipped_pct = bd["coverage_total"]["pct_of_model_compared"]
    gate("the sidecar's 3,643 names reproduce the published coverage",
         abs(pct(union) - shipped_pct) < 1e-9,
         f"recomputed {pct(union)!r} vs published {shipped_pct!r}")
    gate("both arms' sidecars carry the same union",
         [r["param"] for r in pub_rows["renorm"]] == union, len(union))
    s17 = set(json.loads(SEVENTEEN.read_text())["names"])

    per_arm = {}
    for arm, (dm_p, ie_p, ctrl_p) in ARMS.items():
        dm, ie, ctrl = load_grads(dm_p), load_grads(ie_p), load_grads(ctrl_p)
        pub = {r["param"]: r["rel_l2"] for r in pub_rows[arm]}
        scope = sorted(set(ctrl) & set(union))        # the 547 the union already had
        entering = sorted((set(dm) - set(ctrl)) | set(ie))

        # C1 -- the no-lever arm, re-run at HEAD, IS the published scope. Without this the
        # lever's effect below cannot be told from the tree having moved under the campaign.
        c1 = drift(score(ctrl, scope), pub)
        gate(f"{arm} C1: the no-lever arm re-run at HEAD reproduces the published scope",
             c1["worst_rel_diff"] < CTRL_TOL,
             f"{len(scope)} tensors, worst {c1['worst_rel_diff']:.3e} at {c1['worst_tensor']}")

        # C2 -- the lever's reach. NOT a failure: cl0/plm0 are an input to the whole module.
        lever_rows = score(dm, scope)
        c2 = drift(lever_rows, pub)
        gate(f"{arm} C2: the lever is measured on the tensors it was not sold on",
             c2["worst_rel_diff"] > CTRL_TOL,
             f"worst {c2['worst_rel_diff']:.3e} at {c2['worst_tensor']}, mass-weighted "
             f"{c2['mass_weighted_rel_diff']:.3e}")

        gate(f"{arm}: the entering key set is exactly SEVENTEEN.json",
             set(entering) == s17, f"{len(entering)} entering, {len(set(entering) ^ s17)} diff")
        gate(f"{arm}: nothing entering is already in the union",
             not (set(entering) & set(union)), sorted(set(entering) & set(union))[:3] or "none")

        new_rows = score({**dm, **ie}, entering)
        keep = [r for r in pub_rows[arm] if r["param"] not in set(scope)]
        before = agreement.stat(pub_rows[arm], f"{arm}_before")
        after = agreement.stat(keep + lever_rows + new_rows, f"{arm}_after")
        gate(f"{arm}: the composed mass-weighted reading does not get worse",
             after["mass_weighted_rel_l2"] <= before["mass_weighted_rel_l2"] + 1e-12,
             f"{before['mass_weighted_rel_l2']!r} -> {after['mass_weighted_rel_l2']!r}")
        over = [r["param"] for r in new_rows if (r["rel_l2"] or 0.0) > PER_TENSOR_BAR]
        per_arm[arm] = {
            "dumps": {k: {"path": str(p), "sha256": sha256(p), "n_keys": len(d)}
                      for k, p, d in (("with_lever_diffusion_scope", dm_p, dm),
                                      ("input_embedder_leg", ie_p, ie),
                                      ("no_lever_control", ctrl_p, ctrl))},
            "C1_no_lever_reproduces_the_published_scope": {"n_tensors": len(scope), **c1},
            "C2_the_levers_reach_into_the_547_it_was_not_sold_on": {
                "n_tensors": len(scope), **c2,
                "note": "--device-refatom moves cl0/plm0 to the card, and they are an input to "
                        "the whole DiffusionModule. Every gradient in the scope moves. C1 is "
                        "what says this is the lever and not the tree."},
            "headline_vs_float64": {"before": before, "after": after,
                                    "mass_weighted_bar": MASS_BAR},
            "entering": {"n": len(new_rows), "n_over_the_per_tensor_bar": len(over),
                         "over_the_per_tensor_bar": over,
                         "worst": max(new_rows, key=lambda r: r["rel_l2"] or -1.0)["param"],
                         "worst_rel_l2": max((r["rel_l2"] or -1.0) for r in new_rows),
                         "per_tensor": sorted(new_rows, key=lambda r: -(r["rel_l2"] or -1.0))},
        }

    names_new = sorted(s17)
    composed = union + names_new
    cov_new = pct(composed)
    predicted = 99.50523155277438
    gate("coverage matches the pre-registered prediction within 1e-6",
         abs(cov_new - predicted) < 1e-6, f"measured {cov_new!r} vs predicted {predicted!r}")
    gate("the bar is reached", cov_new >= BAR, f"{cov_new!r} vs {BAR}")

    uset, cset = set(union), set(composed)
    by_section: dict = {}
    for n in ref_full:
        c = by_section.setdefault(agreement.section_of(n, sections),
                                  {"n_total": 0, "n_compared_before": 0, "n_compared_after": 0,
                                   "_b": 0.0, "_a": 0.0, "_m": 0.0})
        c["n_total"] += 1
        c["_m"] += mass[n]
        if n in uset:
            c["n_compared_before"] += 1
            c["_b"] += mass[n]
        if n in cset:
            c["n_compared_after"] += 1
            c["_a"] += mass[n]
    for c in by_section.values():
        c["pct_of_model"] = 100.0 * c.pop("_m") / denom
        c["pct_compared_before"] = 100.0 * c.pop("_b") / denom
        c["pct_compared_after"] = 100.0 * c.pop("_a") / denom

    dm_new = sorted(n for n in s17 if n.startswith("diffusion_module."))
    out = {
        "instrument": "of3t-refcov compose.py -- the 17 HOST_APPLIED tensors composed into the "
                      "model-scope union from two arms of3t-hostleg already built on upstream "
                      "0.4.3's own captured cotangent, re-run here in both tree states, with "
                      "the diffusion scope REPLACED by its with-lever version",
        "host": "qb1 (tt-quietbox); scoring CPU-only, the arms ran on card 1",
        "route": "the union, not the training adapter. BOUNDARY_ADAPTER.json measures why the "
                 "adapter cannot contribute to this reading at any precision.",
        "renorm_is_the_default": {
            "where": "tt_bio/autograd.py:87 env_flag('TT_BIO_SOFTMAX_BW_RENORM', True)",
            "since": "2de9355d1 D56, 2026-09-21T12:26:54Z",
            "consequence": "--softmax-bw-renorm is a no-op on this tree and both of3t-hostleg "
                           "dumps are RENORM arms whatever their filenames say. The union's "
                           "`shipped` arm needs TT_BIO_SOFTMAX_BW_RENORM=0.",
        },
        "float64_reference": {"path": str(F64), "sha256": got, "matches_pin": got == PIN,
                              "n_tensors_with_a_gradient": len(ref_full)},
        "denominator": {"recomputed": denom, "published": 10.279642678524981},
        "gates": gates,
        "all_gates_pass": all(g["pass"] for g in gates),
        "coverage": {
            "before": shipped_pct, "after": cov_new, "delta": cov_new - shipped_pct,
            "bar": BAR, "after_minus_bar": cov_new - BAR, "bar_reached": cov_new >= BAR,
            "n_compared_before": len(union), "n_compared_after": len(composed),
            "n_reference_tensors": len(ref_full), "keys_that_entered": names_new,
            "pct_contributed": {
                "diffusion_module_ref_atom_feature_embedder_8": pct(dm_new),
                "input_embedder_leg_9": pct(sorted(s17 - set(dm_new))),
            },
        },
        "coverage_by_section": by_section,
        "accuracy": {"per_tensor_bar": PER_TENSOR_BAR, "mass_weighted_bar": MASS_BAR,
                     "by_arm": per_arm},
    }
    OUT.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print("\ncoverage %.14f -> %.14f (bar %.4f, %s by %.14f), n_compared %d -> %d"
          % (shipped_pct, cov_new, BAR, "REACHED" if cov_new >= BAR else "SHORT",
             abs(cov_new - BAR), len(union), len(composed)), flush=True)
    for arm, v in per_arm.items():
        h, e = v["headline_vs_float64"], v["entering"]
        print("  %-8s mass-weighted %.15g -> %.15g | entering %d over %.2g, worst %s %.9g"
              % (arm, h["before"]["mass_weighted_rel_l2"], h["after"]["mass_weighted_rel_l2"],
                 e["n_over_the_per_tensor_bar"], PER_TENSOR_BAR, e["worst"], e["worst_rel_l2"]),
              flush=True)
    print("->", OUT, flush=True)
    return 0 if out["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
