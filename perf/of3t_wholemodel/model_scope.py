#!/usr/bin/env python3
"""One mass-weighted number for the WHOLE OpenFold3 model, assembled per tensor.

Every scope in this campaign was scored on its own, against its own boundary, in its own
process. Nobody has put them into one vector, and a sum of scope headlines is not the same
statement as a reading over the model's parameters: the scopes carry very different masses, and
two of the arithmetic mistakes this campaign has already made (D17, D51/A23) were exactly a
statistic taken where the gradient is not.

So this does the per-tensor thing. One row per reference tensor, the union of every device arm
that was driven by the model's OWN batch, scored in float64 in the model bundle's denominator.
Scopes that were driven by a DIFFERENT boundary are refused entry to the union and reported as a
composed term with their boundary named -- that separation is the point of the RECONCILE
deliverable and it is enforced here rather than remembered.

The scoring itself is `of3t-trajectory`'s `agreement.py`, imported rather than reimplemented: it
already applies A14 on the float64 gradient for every pair, scores a tensor absent from an arm as
zero instead of dropping its mass, and keeps `r` and `cos` beside `rel_l2` so a magnitude error
can be told from a direction error (D35). A fourth summary-only variant of that arithmetic is
what this campaign keeps being bitten by.

    model_scope.py --arm-part diffusion=/path/dev.pt --arm-part cond=/path/cond.pt \
                   --f64 ... --bf16 ... --f32 ... --sections ... --out ... --sidecar-dir ...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "perf" / "of3t_trajectory"))
import agreement  # noqa: E402  the scorer, not a copy of it


def load_parts(parts, label, unwrap=None):
    """Union the per-scope dumps of one arm. Overlapping keys are a STOP: two scopes claiming
    the same parameter means one of them is measuring something it does not own, and silently
    keeping the last one read would make the result depend on argument order."""
    out, owner = {}, {}
    for spec in parts:
        name, path = spec.split("=", 1)
        d = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(d, dict) and unwrap and unwrap in d:
            d = d[unwrap]
        if not isinstance(d, dict):
            raise SystemExit(f"STOP: {path} is not a name->tensor dict")
        clash = sorted(set(d) & set(out))
        if clash:
            raise SystemExit(
                f"STOP: {name} and {owner[clash[0]]} both carry {clash[0]!r} "
                f"({len(clash)} tensors overlap). Two scopes cannot own the same parameter.")
        for k, v in d.items():
            if v is None:
                continue
            out[k] = v.to(torch.float64).reshape(-1)
            owner[k] = name
        print(f"  {label}: {name} +{len(d)} -> {len(out)}", flush=True)
        del d
    return out, owner


def sha256(p):
    return agreement.sha256(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[], metavar="ARM:SCOPE=PATH",
                    help="repeatable. ARM is the arm label, SCOPE names the scope the dump "
                         "covers, PATH is the gradient dump. Every scope of every arm must be "
                         "driven by the model's own batch; a scope on another boundary belongs "
                         "in --composed-term, not here.")
    ap.add_argument("--composed-term", action="append", default=[], metavar="ARM:SCOPE=share,reading,boundary",
                    help="a scope measured on a DIFFERENT boundary: its share of the model's "
                         "squared gradient norm in percent, its published mass-weighted reading "
                         "against upstream's bf16 step, and a one-line boundary description.")
    ap.add_argument("--f64", required=True, type=Path)
    ap.add_argument("--bf16", required=True, type=Path)
    ap.add_argument("--f32", required=True, type=Path)
    ap.add_argument("--sections", required=True, type=Path)
    ap.add_argument("--expect", action="append", default=[])
    ap.add_argument("--model-total-sq", type=float, default=agreement.MODEL_TOTAL_SQ)
    ap.add_argument("--unwrap-key", default=None, dest="unwrap_key",
                    help="the trunk dumps wrap their gradients under a key alongside the arm's "
                         "own config; naming the key here is how a scope stays readable without "
                         "a second copy of this scorer")
    ap.add_argument("--scope-only", action="store_true", dest="scope_only",
                    help="--f64/--bf16/--f32 hold ONE SCOPE's gradients over that scope's own "
                         "captured boundary, not the model's. The model-drift assert and the "
                         "whole-model coverage table cannot apply and are replaced by the "
                         "scope's own denominator. A scope scored this way may not be "
                         "concatenated into the model union; that is what --composed-term is "
                         "for.")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sidecar-dir", required=True, type=Path)
    args = ap.parse_args()

    agreement.MODEL_TOTAL_SQ = args.model_total_sq

    inputs = {}
    for label, p in (("float64", args.f64), ("upstream_bf16", args.bf16),
                     ("upstream_f32", args.f32)):
        print(f"hashing {label} {p}", flush=True)
        inputs[label] = {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)}
    for e in args.expect:
        label, want = e.split("=", 1)
        got = inputs.get(label, {}).get("sha256")
        if got != want:
            raise SystemExit(f"STOP: {label} is {got} and the pin says {want}.")
        inputs[label]["matches_pin"] = True

    # --- the arms, unioned per arm ------------------------------------------------------------
    by_arm = {}
    for spec in args.arm:
        head, path = spec.split("=", 1)
        arm, scope = head.split(":", 1)
        by_arm.setdefault(arm, []).append(f"{scope}={path}")
    arms, owners = {}, {}
    for arm, parts in by_arm.items():
        print(f"loading arm {arm}", flush=True)
        arms[arm], owners[arm] = load_parts(parts, arm, args.unwrap_key)

    # The union is fixed by the PRIMARY arm so every arm is scored over the same tensor set; an
    # arm missing a tensor the primary has is scored as zero there (agreement.pair_rows), which
    # keeps its mass in the denominator instead of quietly shrinking the scope.
    primary = next(iter(by_arm))
    names = sorted(arms[primary])
    for arm in arms:
        extra = sorted(set(arms[arm]) - set(names))
        if extra:
            raise SystemExit(f"STOP: arm {arm} carries {len(extra)} tensors the primary arm "
                             f"{primary} does not, e.g. {extra[0]!r}. The arms would be scored "
                             f"over different sets and their readings would not be comparable.")
    print(f"union: {len(names)} tensors from {len(by_arm[primary])} scopes", flush=True)

    print(f"loading float64 {args.f64}", flush=True)
    f64_full = torch.load(args.f64, map_location="cpu", weights_only=False)
    if args.unwrap_key and args.unwrap_key in f64_full:
        f64_full = f64_full[args.unwrap_key]
    total_sq = sum(float(torch.linalg.vector_norm(v.to(torch.float64))) ** 2
                   for v in f64_full.values() if v is not None)
    drift = abs(total_sq - args.model_total_sq) / args.model_total_sq
    print(f"float64 squared gradient norm {total_sq!r}, drift {drift:.3e}"
          + ("  -- NOT CHECKED, --scope-only" if args.scope_only else ""), flush=True)
    assert args.scope_only or drift <= agreement.MODEL_TOTAL_SQ_RTOL, (
        f"the float64 file's squared norm {total_sq!r} is not the campaign's denominator")
    if args.scope_only:
        # every share below is then a share of THIS SCOPE, and says so
        agreement.MODEL_TOTAL_SQ = total_sq

    # coverage over the WHOLE reference, by mass (A15/A23), before any headline
    sections = list(json.loads(args.sections.read_text())["sections_pct_of_model"])
    cov, compared = {}, set(names)
    for n, v in f64_full.items():
        if v is None:
            continue
        sq = float(torch.linalg.vector_norm(v.to(torch.float64))) ** 2
        sec = agreement.section_of(n, sections)
        c = cov.setdefault(sec, {"n_total": 0, "n_compared": 0, "mass_sq": 0.0,
                                 "mass_sq_compared": 0.0})
        c["n_total"] += 1
        c["mass_sq"] += sq
        if n in compared:
            c["n_compared"] += 1
            c["mass_sq_compared"] += sq
    for c in cov.values():
        c["pct_of_model"] = 100.0 * c["mass_sq"] / total_sq
        c["pct_of_model_compared"] = 100.0 * c["mass_sq_compared"] / total_sq

    f64 = {n: (f64_full[n].to(torch.float64).reshape(-1) if f64_full.get(n) is not None else None)
           for n in names}
    n_missing_ref = sum(1 for n in names if f64[n] is None)
    del f64_full

    refs = {"FLOAT64": f64}
    for k, p in (("UPSTREAM_BF16", args.bf16), ("UPSTREAM_F32", args.f32)):
        print(f"loading {k}", flush=True)
        if args.unwrap_key:
            dd = torch.load(p, map_location="cpu", weights_only=False)
            dd = dd.get(args.unwrap_key, dd)
            refs[k] = {n: (dd[n].to(torch.float64).reshape(-1) if dd.get(n) is not None else None)
                       for n in names}
            del dd
        else:
            refs[k] = agreement.load_subset(p, names)
    # A16: a model that computes nothing, through the same scorer rather than asserted as 1.0
    arms["ZERO_MODEL"] = {}

    # --- every pair the branches need ---------------------------------------------------------
    pairs = []
    for arm in list(by_arm) + ["ZERO_MODEL"]:
        pairs.append((f"{arm}_vs_UPSTREAM_BF16", refs["UPSTREAM_BF16"], arms[arm]))
        pairs.append((f"{arm}_vs_FLOAT64", refs["FLOAT64"], arms[arm]))
    pairs.append(("UPSTREAM_BF16_vs_FLOAT64", refs["FLOAT64"], refs["UPSTREAM_BF16"]))
    pairs.append(("UPSTREAM_F32_vs_FLOAT64", refs["FLOAT64"], refs["UPSTREAM_F32"]))

    args.sidecar_dir.mkdir(parents=True, exist_ok=True)
    stats, per_section = {}, {}
    for label, ref, arm in pairs:
        print(f"scoring {label}", flush=True)
        rows = agreement.pair_rows(ref, arm, names, f64, sections)
        stats[label] = agreement.stat(rows, label)
        bysec = {}
        for r in rows:
            bysec.setdefault(r["section"], []).append(r)
        per_section[label] = {
            s: dict(agreement.stat(rs, f"{label}::{s}"),
                    ref_sq=sum(r["ref_norm"] ** 2 for r in rs))
            for s, rs in sorted(bysec.items())}
        (args.sidecar_dir / f"{label}.json").write_text(json.dumps(rows))

    # --- the bars, computed from the measurement and not borrowed ----------------------------
    floor = stats["UPSTREAM_BF16_vs_FLOAT64"]["mass_weighted_rel_l2"]
    r_bf16 = stats["UPSTREAM_BF16_vs_FLOAT64"]["mass_weighted_norm_ratio"]
    perfect = floor / r_bf16
    bars = {
        "note": "A26 applies against upstream's bf16 step because that reference carries error; "
                "A26-SCOPE forbids widening the float64 bar the same way.",
        "upstream_own_bf16_floor_vs_float64": floor,
        "norm_ratio_bf16_over_float64": r_bf16,
        "perfect_port_reads_floor_over_r": perfect,
        "A26_reachable_bar_vs_their_bf16": math.sqrt(2.0) * perfect,
        "mass_weighted_bar_vs_float64": 2.0e-02,
        "per_tensor_bar": agreement.PER_TENSOR_BAR,
        "A16_zero_model_measured_vs_their_bf16":
            stats["ZERO_MODEL_vs_UPSTREAM_BF16"]["mass_weighted_rel_l2"],
        "instrument_floor_upstream_f32_vs_float64":
            stats["UPSTREAM_F32_vs_FLOAT64"]["mass_weighted_rel_l2"],
    }

    # --- reconciliation ----------------------------------------------------------------------
    recon = {}
    for arm in by_arm:
        assembled = stats[f"{arm}_vs_UPSTREAM_BF16"]["mass_weighted_rel_l2"]
        secs = per_section[f"{arm}_vs_UPSTREAM_BF16"]
        # (a) the composition AS THE CAMPAIGN STATES IT: each scope's published reading weighted
        # by the scope's share of the model's squared gradient norm, which every table in this
        # campaign takes from the FLOAT64 reference.
        num = sum(s["pct_of_model_mass"] * (s["mass_weighted_rel_l2"] or 0.0) ** 2
                  for s in secs.values())
        den = sum(s["pct_of_model_mass"] for s in secs.values())
        composed = math.sqrt(num / den) if den else None
        # (b) the composition that is an IDENTITY. A reading of the form ||d|| / ||ref|| composes
        # with weights taken from the SAME ref it divides by. Against upstream's bf16 step that
        # is the bf16 norm, not the float64 norm, and the two differ per section by r_s. Both
        # are computed so the size of that mismatch is a measured number rather than an
        # assumption; (b) must reproduce the assembled headline to rounding or this scorer is
        # wrong about its own arithmetic.
        numx = sum(s["ref_sq"] * (s["mass_weighted_rel_l2"] or 0.0) ** 2 for s in secs.values())
        denx = sum(s["ref_sq"] for s in secs.values())
        composed_exact = math.sqrt(numx / denx) if denx else None
        recon[arm] = {
            "assembled_over_union": assembled,
            "composed_from_published_float64_mass_shares": composed,
            "rel_difference_published": (abs(composed - assembled) / assembled
                                         if composed and assembled else None),
            "composed_from_the_arms_own_reference_mass": composed_exact,
            "rel_difference_exact": (abs(composed_exact - assembled) / assembled
                                     if composed_exact and assembled else None),
            "why_they_differ": "a scope's reading divides by ||g_bf16|| on that scope while the "
                               "campaign's mass share is a share of ||g_float64||^2; the two "
                               "weightings differ by r_s = ||g_bf16||/||g_float64|| per section",
            "sections": {s: {"pct_of_model_mass": v["pct_of_model_mass"],
                             "ref_sq": v["ref_sq"],
                             "mass_weighted_rel_l2": v["mass_weighted_rel_l2"],
                             "mass_weighted_norm_ratio": v["mass_weighted_norm_ratio"],
                             "mass_weighted_cos": v["mass_weighted_cos"],
                             "n": v["n"], "n_over_per_tensor_bar": v["n_over_per_tensor_bar"],
                             "worst_tensor": v["worst_tensor"],
                             "worst_rel_l2": v["worst_rel_l2"]} for s, v in secs.items()},
        }
    terms = []
    for spec in args.composed_term:
        head, body = spec.split("=", 1)
        arm, scope = head.split(":", 1)
        share, reading, sfloor, sr, boundary = body.split(",", 4)
        terms.append({"arm": arm, "scope": scope, "pct_of_model_mass": float(share),
                      "mass_weighted_rel_l2": float(reading),
                      "own_bf16_floor_vs_float64": float(sfloor),
                      "norm_ratio_bf16_over_float64": float(sr), "boundary": boundary,
                      "why_not_in_the_union": "measured on a different boundary than "
                                              "batch_step003; concatenating it would mix "
                                              "gradients of two different inputs"})
    for arm in by_arm:
        mine = [t for t in terms if t["arm"] == arm]
        if not mine:
            continue
        a = stats[f"{arm}_vs_UPSTREAM_BF16"]
        fl = stats["UPSTREAM_BF16_vs_FLOAT64"]
        # A reading against upstream's bf16 divides by ||g_bf16||, so it composes with bf16
        # mass. The campaign's shares are float64 shares, and bf16 mass is share * r^2. Getting
        # that wrong is worth 0.55 % at this scope -- measured above, not assumed.
        rows = [(a["pct_of_model_mass"], a["mass_weighted_rel_l2"],
                 fl["mass_weighted_rel_l2"], fl["mass_weighted_norm_ratio"])] + \
               [(t["pct_of_model_mass"], t["mass_weighted_rel_l2"],
                 t["own_bf16_floor_vs_float64"], t["norm_ratio_bf16_over_float64"])
                for t in mine]
        wsum = sum(m * rr * rr for m, _, _, rr in rows)
        reading = math.sqrt(sum(m * rr * rr * x * x for m, x, _, rr in rows) / wsum)
        msum = sum(m for m, _, _, _ in rows)
        # the floor and r divide by the float64 norm, so THEY compose with float64 mass
        cfloor = math.sqrt(sum(m * f * f for m, _, f, _ in rows) / msum)
        cr = math.sqrt(sum(m * rr * rr for m, _, _, rr in rows) / msum)
        recon[arm]["with_other_boundary_terms"] = {
            "pct_of_model_mass": msum, "mass_weighted_rel_l2": reading,
            "composed_floor_vs_float64": cfloor, "composed_norm_ratio": cr,
            "composed_perfect_port_threshold": cfloor / cr,
            "composed_A26_reachable_bar": math.sqrt(2.0) * cfloor / cr,
            "x_reachable_bar": reading / (math.sqrt(2.0) * cfloor / cr),
            "terms": mine,
            "caveat": "NOT one gradient vector: the added terms were taken on another input. "
                      "Reported as a composition and never as the assembled headline.",
        }

    out = {
        "instrument": "of3t-wholemodel model_scope.py -- the union of the same-batch device "
                      "arms, scored per tensor in the model bundle's denominator",
        "host": os.uname().nodename,
        "inputs": inputs,
        "arms": {a: {"scopes": by_arm[a], "n_tensors": len(arms[a])} for a in by_arm},
        "union_n_tensors": len(names),
        "n_union_tensors_absent_from_float64_reference": n_missing_ref,
        "model_squared_gradient_norm_measured": total_sq,
        "scope_only": bool(args.scope_only),
        "bars": bars,
        "stats": stats,
        "per_section": per_section,
        "coverage_by_section": dict(sorted(cov.items(), key=lambda kv: -kv[1]["pct_of_model"])),
        "coverage_total": {
            "pct_of_model_compared": sum(c["pct_of_model_compared"] for c in cov.values()),
            "pct_of_model_uncompared": sum(c["pct_of_model"] - c["pct_of_model_compared"]
                                           for c in cov.values()),
            "n_reference_tensors": sum(c["n_total"] for c in cov.values()),
            "n_compared": sum(c["n_compared"] for c in cov.values()),
        },
        "reconciliation": recon,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items()
                      if k in ("bars", "coverage_total", "union_n_tensors")}, indent=2))
    for label, s in stats.items():
        print(f"{label:42s} rel {s['mass_weighted_rel_l2']!r} median "
              f"{s['median_rel_l2_over_tensors']!r} r {s['mass_weighted_norm_ratio']!r} "
              f"cos {s['mass_weighted_cos']!r} mass {s['pct_of_model_mass']:.4f} %")
        print(f"{'':42s} worst {s['worst_tensor']} {s['worst_rel_l2']!r}")


if __name__ == "__main__":
    main()
