#!/usr/bin/env python3
"""Do we reproduce the gradient OpenFold3's own training step computes?

Every figure this campaign has is a distance from float64, and nobody trains in float64. The
re-scoring in D72 put our deviation beside upstream's bf16 deviation, but both are distances
from the SAME float64 reference and two distances from a shared reference do not order each
other: ours at 7.865e-03 beside bf16's 6.2532e-02 does not establish that we agree with bf16 to
6e-02, because the two errors may point in different directions. This instrument removes the
shared subtrahend. It compares our device gradient DIRECTLY against arm4, upstream's own bf16
autocast step, over the 547 tensors our device arm covers.

A23 governs the reporting. The headline for a set is the mass-weighted `rel_l2` over the
concatenated set, the median over tensors goes beside it and never instead of it, and every set
carries the share of the model's squared gradient norm it holds. Per tensor the sidecar keeps
`rel_l2`, the norm ratio `r` and the cosine, because `rel` alone bounds `r` to [1-rel, 1+rel]
and cannot separate a magnitude error from a direction error. The whole array is written, not
the extremes: this campaign has changed denominator twice and a file keeping only extremes
cannot be re-scored.

Mass weights come from the FLOAT64 gradient for every pair, not from each pair's own reference,
so a section's weight is the same number in every row of every table here and in D72.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

REF_NORM_FLOOR = 1e-12          # A14: below this a relative error carries no information
PER_TENSOR_BAR = 5.0e-02        # PROTOCOL 3d
MODEL_TOTAL_SQ = 10.279642678524985   # the campaign's published denominator
BF16_OWN_FLOOR = 5.852018e-02   # arm4's own distance from float64 on the device arm's scope


def sha256(path):
    """A reference whose content can change under you is not a reference. of3t-refprec rewrites
    its arm files in place with no atomic rename, so the identity of an input here is its digest
    and not its path."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 23), b""):
            h.update(chunk)
    return h.hexdigest()


def load_subset(path, names):
    """Load one gradient file and keep only the scope, so three arms fit in memory at once."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    out = {n: (d[n].to(torch.float64).reshape(-1) if d.get(n) is not None else None)
           for n in names}
    del d
    return out


def section_of(name, sections):
    best = None
    for key in sections:
        if name == key or name.startswith(key + "."):
            if best is None or len(key) > len(best):
                best = key
    return best or name.split(".")[0]


def pair_rows(ref, arm, names, mass, sections):
    rows = []
    for n in names:
        b, a = ref.get(n), arm.get(n)
        m = mass.get(n)
        rb = 0.0 if b is None else float(torch.linalg.vector_norm(b))
        sq = 0.0 if m is None else float(torch.linalg.vector_norm(m)) ** 2
        row = {"param": n, "section": section_of(n, sections), "ref_norm": rb,
               "mass_sq": sq, "pct_of_model_mass": 100.0 * sq / MODEL_TOTAL_SQ}
        if b is None or a is None:
            # An arm with no gradient where the reference has one is not unmeasurable: its
            # value there IS zero. Scoring it as a missing row would drop its mass from the
            # numerator and leave it in the denominator.
            if b is not None and rb >= REF_NORM_FLOOR:
                row.update(arm_norm=0.0, diff_norm=rb, dot=0.0, rel_l2=1.0, r=0.0, cos=0.0,
                           unmeasurable="absent from the arm, scored as zero")
            else:
                row.update(arm_norm=None, diff_norm=None, dot=0.0, rel_l2=None, r=None,
                           cos=None, unmeasurable="absent on one side")
            rows.append(row)
            continue
        ra = float(torch.linalg.vector_norm(a))
        d = float(torch.linalg.vector_norm(a - b))
        row.update(arm_norm=ra, diff_norm=d, dot=float(torch.dot(a, b)))
        if rb < REF_NORM_FLOOR:
            row.update(rel_l2=None, r=None, cos=None,
                       unmeasurable=f"ref_norm {rb:.3e} < {REF_NORM_FLOOR:.0e} (A14)")
        else:
            row.update(rel_l2=d / rb, r=ra / rb,
                       cos=(row["dot"] / (ra * rb)) if ra > 0 else 0.0)
        rows.append(row)
    return rows


def stat(rows, label):
    sq = sum(r["ref_norm"] ** 2 for r in rows)
    mass = sum(r["mass_sq"] for r in rows)
    d2 = sum((r["diff_norm"] or 0.0) ** 2 for r in rows)
    a2 = sum((r["arm_norm"] or 0.0) ** 2 for r in rows)
    dot = sum(r.get("dot") or 0.0 for r in rows)
    rels = sorted(r["rel_l2"] for r in rows if r["rel_l2"] is not None)
    med = (rels[len(rels) // 2] if len(rels) % 2 else
           0.5 * (rels[len(rels) // 2 - 1] + rels[len(rels) // 2])) if rels else None
    worst = max((r for r in rows if r["rel_l2"] is not None),
                key=lambda r: r["rel_l2"], default=None)
    return {
        "set": label,
        "n": len(rows),
        "pct_of_model_mass": 100.0 * mass / MODEL_TOTAL_SQ,
        "mass_weighted_rel_l2": math.sqrt(d2 / sq) if sq else None,
        "mass_weighted_norm_ratio": math.sqrt(a2 / sq) if sq else None,
        "mass_weighted_cos": (dot / math.sqrt(a2 * sq)) if a2 > 0 and sq > 0 else None,
        "median_rel_l2_over_tensors": med,
        "n_rel_measurable": len(rels),
        "n_over_per_tensor_bar": sum(1 for x in rels if x > PER_TENSOR_BAR),
        "worst_rel_l2": worst["rel_l2"] if worst else None,
        "worst_tensor": worst["param"] if worst else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True, type=Path)
    ap.add_argument("--device-permuted", type=Path)
    ap.add_argument("--f64", required=True, type=Path)
    ap.add_argument("--bf16", required=True, type=Path)
    ap.add_argument("--f32", required=True, type=Path)
    ap.add_argument("--upstream-permuted", type=Path)
    ap.add_argument("--diffcap", type=Path,
                    help="the capture whose grad_f64 the device arm was scored against; "
                         "checked against the bundle's own float64 gradient, because that "
                         "identity is what makes a device-vs-arm4 comparison legitimate")
    ap.add_argument("--cap-prefix", default="diffusion_module.", dest="cap_prefix",
                    help="the capture keys its grad_f64 by name RELATIVE to the module it "
                         "hooked, so the prefix to strip is a property of the capture and not "
                         "of this script. Hardcoding it made the check silently vacuous on any "
                         "other scope: every name would miss and `max_abs_diff` would stay 0.0, "
                         "reporting `identical` on zero compared tensors.")
    ap.add_argument("--sections", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sidecar-dir", required=True, type=Path)
    ap.add_argument("--expect", action="append", default=[],
                    help="label=sha256, repeatable. A mismatch stops the run: measuring against "
                         "a file that is not the one the campaign's figures came from would "
                         "make every number downstream unattributable.")
    args = ap.parse_args()

    inputs = {}
    for label, path in (("device", args.device), ("device_permuted", args.device_permuted),
                        ("float64", args.f64), ("upstream_bf16", args.bf16),
                        ("upstream_f32", args.f32),
                        ("upstream_permuted_draws", args.upstream_permuted),
                        ("diffcap", args.diffcap)):
        if path is None:
            continue
        print(f"hashing {label} {path}", flush=True)
        inputs[label] = {"path": str(path), "bytes": path.stat().st_size,
                         "sha256": sha256(path)}
    for e in args.expect:
        label, want = e.split("=", 1)
        got = inputs.get(label, {}).get("sha256")
        if got != want:
            raise SystemExit(
                f"STOP: {label} is {got} and the pin says {want}. The pin was taken against "
                f"something other than the run these numbers came from; every figure "
                f"downstream of it would be unattributable.")
        inputs[label]["matches_pin"] = True

    sections = list(json.loads(args.sections.read_text())["sections_pct_of_model"])

    dev = torch.load(args.device, map_location="cpu", weights_only=False)
    names = sorted(dev)
    dev = {n: dev[n].to(torch.float64).reshape(-1) for n in names}
    print(f"device scope: {len(names)} tensors", flush=True)

    print(f"loading float64 {args.f64}", flush=True)
    f64_full = torch.load(args.f64, map_location="cpu", weights_only=False)
    total_sq = sum(float(torch.linalg.vector_norm(v.to(torch.float64))) ** 2
                   for v in f64_full.values() if v is not None)
    assert abs(total_sq - MODEL_TOTAL_SQ) < 1e-12, (
        f"denominator {total_sq!r} is not the campaign's published {MODEL_TOTAL_SQ!r}; every "
        "share here would be in a different denominator than the one D72 uses")
    missing = [n for n in names if f64_full.get(n) is None]
    f64 = {n: (f64_full[n].to(torch.float64).reshape(-1) if f64_full.get(n) is not None
               else None) for n in names}
    del f64_full
    print(f"float64: model squared norm {total_sq!r}, {len(names) - len(missing)} of "
          f"{len(names)} scope tensors present", flush=True)

    # The device arm was scored against diffcap043's grad_f64, not against the bundle's file.
    # If those two are not the same numbers, our gradient and arm4's are not gradients of the
    # same loss and nothing below means anything.
    cap_check = None
    if args.diffcap:
        S = torch.load(args.diffcap, map_location="cpu", weights_only=False)
        g = S["grad_f64"]
        worst, worst_n, checked = 0.0, None, 0
        for n in names:
            if not n.startswith(args.cap_prefix):
                continue
            k = n[len(args.cap_prefix):]
            if k not in g or f64.get(n) is None:
                continue
            checked += 1
            d = float((g[k].to(torch.float64).reshape(-1) - f64[n]).abs().max())
            if d > worst:
                worst, worst_n = d, n
        # A count of 0 compared tensors would read `identical: true` on nothing at all, which
        # is the vacuous-check shape this campaign has now shipped three times.
        cap_check = {"what": "the capture's grad_f64 against the bundle's float64 gradient, "
                             "over the device scope",
                     "capture": str(args.diffcap), "key_prefix_stripped": args.cap_prefix,
                     "max_abs_diff": worst, "at": worst_n, "n_scope": len(names),
                     "n_compared": checked,
                     "identical": (checked > 0 and worst == 0.0)}
        if checked == 0:
            raise SystemExit(
                f"STOP: no scope tensor name starts with {args.cap_prefix!r}, so the capture "
                f"identity check compared nothing. A check that compares nothing cannot fail.")
        del S, g
        print("diffcap vs bundle float64:", cap_check, flush=True)

    arms = {"UPSTREAM_BF16": args.bf16, "UPSTREAM_F32": args.f32}
    if args.upstream_permuted:
        arms["UPSTREAM_F32_PERMUTED_DRAWS"] = args.upstream_permuted
    loaded = {"DEVICE": dev, "FLOAT64": f64}
    for k, p in arms.items():
        print(f"loading {k} from {p}", flush=True)
        loaded[k] = load_subset(p, names)
    if args.device_permuted:
        d2 = torch.load(args.device_permuted, map_location="cpu", weights_only=False)
        loaded["DEVICE_COTANGENT_PERMUTED"] = {
            n: (d2[n].to(torch.float64).reshape(-1) if d2.get(n) is not None else None)
            for n in names}
        del d2
    loaded["ZERO"] = {n: (torch.zeros_like(f64[n]) if f64.get(n) is not None else None)
                      for n in names}

    pairs = [
        ("DEVICE_vs_UPSTREAM_BF16", "DEVICE", "UPSTREAM_BF16",
         "the question. Our gradient against the gradient upstream's own bf16 training step "
         "computes, no shared subtrahend."),
        ("DEVICE_vs_FLOAT64", "DEVICE", "FLOAT64",
         "continuity with the record: the device arm's published distance from the ideal."),
        ("UPSTREAM_BF16_vs_FLOAT64", "UPSTREAM_BF16", "FLOAT64",
         "the bar. Upstream's own distance from the ideal on this scope."),
        ("UPSTREAM_BF16_vs_UPSTREAM_F32", "UPSTREAM_BF16", "UPSTREAM_F32",
         "the scale. How far upstream's own two precisions are from each other."),
        ("UPSTREAM_F32_vs_FLOAT64", "UPSTREAM_F32", "FLOAT64",
         "the achievable floor: upstream's single precision against the ideal."),
        ("ZERO_vs_UPSTREAM_BF16", "ZERO", "UPSTREAM_BF16",
         "A16, measured not asserted: what a model that computes nothing reads."),
    ]
    if "DEVICE_COTANGENT_PERMUTED" in loaded:
        pairs.append(("DEVICE_PERMUTED_COTANGENT_vs_UPSTREAM_BF16",
                      "DEVICE_COTANGENT_PERMUTED", "UPSTREAM_BF16",
                      "the break control: our own run with every sample paired to the wrong "
                      "cotangent. The headline must move by orders of magnitude."))
    if "UPSTREAM_F32_PERMUTED_DRAWS" in loaded:
        pairs.append(("UPSTREAM_PERMUTED_DRAWS_vs_UPSTREAM_BF16",
                      "UPSTREAM_F32_PERMUTED_DRAWS", "UPSTREAM_BF16",
                      "a second break control, upstream against itself with the sample draws "
                      "permuted."))

    args.sidecar_dir.mkdir(parents=True, exist_ok=True)
    out = {"what": __doc__.strip().splitlines()[0],
           "inputs": inputs,
           "scope": {"n_tensors": len(names),
                     "source": str(args.device),
                     "pct_of_model_mass": 100.0 * sum(
                         float(torch.linalg.vector_norm(v)) ** 2
                         for v in f64.values() if v is not None) / MODEL_TOTAL_SQ,
                     "tensors_absent_from_float64": missing},
           "model_squared_gradient_norm": total_sq,
           "diffcap_is_the_bundles_float64": cap_check,
           "bars": {"upstream_bf16_own_distance_from_float64": BF16_OWN_FLOOR,
                    "per_tensor": PER_TENSOR_BAR},
           "pairs": {}}

    for label, arm_k, ref_k, why in pairs:
        rows = pair_rows(loaded[ref_k], loaded[arm_k], names, f64, sections)
        sets = [stat(rows, "the device arm's scope (all compared tensors)")]
        for sec in sections:
            sub = [r for r in rows if r["section"] == sec]
            if sub:
                sets.append(stat(sub, f"section {sec}"))
        head = sets[0]
        rec = {"arm": arm_k, "reference": ref_k, "why": why, "sets": sets}
        def f(x, spec=".6e"):
            return "n/a" if x is None else format(x, spec)

        rec["headline"] = (
            f"mass-weighted rel_l2 {f(head['mass_weighted_rel_l2'])} over {head['n']} tensors "
            f"holding {f(head['pct_of_model_mass'], '.4f')} % of the model's squared gradient "
            f"norm; median over tensors {f(head['median_rel_l2_over_tensors'])}, norm ratio "
            f"{f(head['mass_weighted_norm_ratio'], '.6f')}, cos "
            f"{f(head['mass_weighted_cos'], '.6f')}")
        out["pairs"][label] = rec
        side = args.sidecar_dir / f"per_tensor_{label}.json"
        side.write_text(json.dumps(
            [{k: r[k] for k in ("param", "section", "pct_of_model_mass", "ref_norm", "arm_norm",
                                "diff_norm", "rel_l2", "r", "cos")}
             for r in sorted(rows, key=lambda r: -r["mass_sq"])], indent=0) + "\n")
        rec["per_tensor_sidecar"] = side.name
        print(label, "->", rec["headline"], flush=True)
        del rows

    # Deliverable 3, decided before the number existed.
    #
    # The threshold is the SCOPE's own floor, measured here, not the 5.852018e-02 constant --
    # that constant is the floor of ONE scope (the 547-tensor diffusion arm) and the floors
    # measured by section range from 3.1012e-02 to 7.0855e-02, a 2.3x spread, so reading any
    # other scope against it would be scoring one set against another set's bar.
    #
    # And it is `floor / r`, not `floor`. The measured quantity rel(device, bf16) divides by
    # ||bf16||, while `floor` = rel(bf16, f64) divides by ||f64||, so a device gradient that
    # equalled float64 EXACTLY would read floor * ||f64|| / ||bf16|| = floor / r, not floor.
    # D76 shipped an unreachable threshold by treating two `rel` figures with different
    # denominators as though they shared one; the correction is one division and it belongs in
    # the instrument rather than in prose beside it.
    #
    # Attainable range of the reading: 0 when the device gradient IS their bf16 gradient,
    # floor / r when it is the float64 ideal, 1.0 when it is zero (A16, measured below), and
    # unbounded above. All three branches sit inside that range.
    h = out["pairs"]["DEVICE_vs_UPSTREAM_BF16"]["sets"][0]["mass_weighted_rel_l2"]
    ours_f64 = out["pairs"]["DEVICE_vs_FLOAT64"]["sets"][0]["mass_weighted_rel_l2"]
    bf16_set = out["pairs"]["UPSTREAM_BF16_vs_FLOAT64"]["sets"][0]
    floor = bf16_set["mass_weighted_rel_l2"]
    r = bf16_set["mass_weighted_norm_ratio"]
    perfect = floor / r
    zero_read = out["pairs"]["ZERO_vs_UPSTREAM_BF16"]["sets"][0]["mass_weighted_rel_l2"]
    if h <= perfect:
        branch = ("at or below floor/r: on this scope we reproduce upstream's actual training "
                  "gradient to within its own distance from the ideal. The published "
                  "float64-scored pass SURVIVES the direct test.")
    elif h < 1.0:
        branch = ("between floor/r and ~1.0: the published pass DOES NOT SURVIVE the direct "
                  "test. The float64-scored reading was hiding a disagreement, exactly as "
                  "layer_norm_s did.")
    else:
        branch = ("at or above ~1.0: on this scope we are no better than a zero-gradient model "
                  "against their step. Re-check the instrument and the tensor subset before "
                  "believing it.")
    out["PRE_REGISTERED_READING"] = {
        "headline_mass_weighted_rel_l2": h,
        "scope_floor_rel_bf16_vs_float64": floor,
        "scope_norm_ratio_r_bf16_over_float64": r,
        "threshold_a_perfect_port_would_read__floor_over_r": perfect,
        "multiples_of_that_threshold": h / perfect,
        "multiples_of_the_scope_floor_itself": h / floor,
        "measured_zero_model_reading_A16": zero_read,
        "our_distance_from_float64_on_this_scope": ours_f64,
        "what_moving_the_reference_from_float64_to_their_bf16_did_to_our_headline":
            (h / ours_f64 if ours_f64 else None),
        "whole_device_arm_floor_constant_for_cross_reference": BF16_OWN_FLOOR,
        "attainable_range": {
            "device_gradient_equals_their_bf16": 0.0,
            "device_gradient_equals_float64": perfect,
            "device_gradient_is_zero": zero_read,
            "worse": "unbounded above",
        },
        "branch": branch,
    }
    # The geometry the threshold cannot show: our error e = d - f against theirs t = b - f.
    # D76 had to infer this cosine from three norms; measuring it directly separates "our error
    # points somewhere else entirely" from "our error is theirs, larger".
    e2 = t2 = et = 0.0
    for n in names:
        d_, b_, f_ = dev.get(n), loaded["UPSTREAM_BF16"].get(n), f64.get(n)
        if d_ is None or b_ is None or f_ is None:
            continue
        e, t = d_ - f_, b_ - f_
        e2 += float(torch.dot(e, e)); t2 += float(torch.dot(t, t))
        et += float(torch.dot(e, t))
    out["ERROR_GEOMETRY"] = {
        "what": "our error against the float64 ideal, versus upstream bf16's error against the "
                "same ideal, over this scope. cos near 0 means the two errors are unrelated "
                "directions and 'no further from float64 than they are' says nothing about "
                "agreement with their step.",
        "our_error_norm": math.sqrt(e2),
        "their_error_norm": math.sqrt(t2),
        "ratio_ours_over_theirs": (math.sqrt(e2 / t2) if t2 else None),
        "cos_between_the_two_errors": (et / math.sqrt(e2 * t2) if e2 > 0 and t2 > 0 else None),
    }
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nthreshold floor/r = {perfect:.6e} (floor {floor:.6e} / r {r:.6f}); "
          f"headline {h:.6e} = {h / perfect:.4f}x it")
    print("error geometry:", json.dumps(out["ERROR_GEOMETRY"], indent=1))
    print("\nBRANCH:", branch)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
