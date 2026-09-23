#!/usr/bin/env python3
"""of3t-angle: the softmax ladder on the REPAIRED injection, decomposed into magnitude and angle.

R149 measured the ladder on the DOUBLE-COUNTED functional, where 62.52 % of the trunk error was
magnitude. On the repaired functional the magnitude carries 0.2749 % and the direction carries
100.11 % (perf/of3t_recutfin/BF16_SPLIT.json). A lever that closes a magnitude error is worth
nothing there, so this row re-reads the ladder on the repaired injection and reports, for every
arm and BOTH references, rel, r, cos, the angle, and the magnitude/direction split.

TWO DEFINITIONS OF THE THREE AGGREGATES, AND ONLY ONE OF THEM CAN CARRY AN ANGLE.

  concatenated  dot, ||a||^2, ||g||^2 and ||a-g||^2 summed over the 2,736 tensors, then
                rel = sqrt(e2/r2), r = sqrt(a2/r2), cos = dot/sqrt(a2 r2). Three norms of ONE
                vector pair, so rel^2 = 1 + r^2 - 2 r cos is an identity. This is the definition
                of3t-recut and of3t-recutfin read the trunk 45.763 degrees in
                (perf/of3t_recut/n384_check.py:40-67), and it is the only one an angle or a
                magnitude/direction split may be taken in.

  mass_weighted of3t-trunkg043/score.py (A23): rel is a mass-weighted QUADRATIC mean of the
                per-tensor rel, while norm_ratio and cos are mass-weighted ARITHMETIC means of
                the per-tensor ratio and cosine. The identity holds per tensor and NOT on these
                three aggregates. On R149 shipped arm rel = 0.9153623104186986 while
                sqrt(1 + r^2 - 2 r cos) on its own reported r and cos is 0.6855, a 25 %
                residual. So R149 numbers are quoted as published and compared arm to arm, and
                no angle is ever read off them.

Both are computed for every arm. The ladder comparison to R149 uses mass_weighted, because that
is the definition R149 numbers are in; every split uses concatenated.

CPU only, no device is opened. The device arms are read as banked tensors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from score import score                                                      # noqa: E402
from bf16_split import split                                                 # noqa: E402

PRE = "pairformer_stack.blocks."


def sha256_file(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def loadavg_block(path):
    """Amendment 1: the box this reading was taken on, recorded rather than assumed quiet."""
    if not path or not os.path.exists(path):
        return {"sampled_DURING": False,
                "why": "no sampler ran; the co-tenant field is what is known"}
    s = sorted(float(x) for x in open(path).read().split() if x.strip())
    if not s:
        return {"sampled_DURING": False, "why": "sampler file is empty"}
    return {"sampled_DURING": True, "n": len(s), "min": s[0], "median": s[len(s) // 2],
            "max": s[-1]}


def load(p):
    d = torch.load(p, map_location="cpu", weights_only=False)
    g = d["grads"] if isinstance(d, dict) and "grads" in d else d
    return ({k: v for k, v in g.items() if k.startswith(PRE) and v is not None},
            d if isinstance(d, dict) else {})


def concat_triple(arm, ref, keys):
    """The three aggregates as norms of ONE vector pair. The identity holds here."""
    dot = a2 = r2 = e2 = 0.0
    for k in keys:
        m = arm[k].reshape(-1).to(torch.float64)
        g = ref[k].reshape(-1).to(torch.float64)
        dot += float(torch.dot(m, g))
        a2 += float(torch.dot(m, m))
        r2 += float(torch.dot(g, g))
        d = m - g
        e2 += float(torch.dot(d, d))
    return {"n": len(keys),
            "rel_l2": math.sqrt(e2 / r2) if r2 else None,
            "norm_ratio": math.sqrt(a2 / r2) if r2 else None,
            "cos": dot / math.sqrt(a2 * r2) if a2 > 0 and r2 > 0 else None,
            "arm_sq_norm": a2, "ref_sq_norm": r2, "error_sq_norm": e2, "dot": dot}


def mw_triple(arm, ref, keys):
    s = score(arm, ref, keys)
    s.pop("_rows")
    rel, r, c = (s["mass_weighted_rel_l2"], s["mass_weighted_norm_ratio"],
                 s["mass_weighted_cos"])
    return {"n": s["tensors"], "rel_l2": rel, "norm_ratio": r, "cos": c,
            "median_rel_l2_over_tensors": s["median_rel_l2_over_tensors"],
            "identity_residual_rel": (abs(math.sqrt(max(0.0, 1 + r * r - 2 * r * c)) - rel) / rel
                                      if rel else None),
            "why_the_residual_is_reported":
                "a quadratic mean and two arithmetic means, so the identity does not hold on "
                "these three. The residual is how far off it is, and it is why no angle is "
                "read here."}


def rel_of(r, cos):
    return math.sqrt(max(0.0, 1.0 + r * r - 2.0 * r * cos))


def lever_effect(base, lev, name):
    """What the lever moved, split into its magnitude part and its direction part.

    rel(r, cos) is a function of two numbers. Hold the base arm cos and give it the lever r:
    that is the MAGNITUDE effect. Hold the base arm r and give it the lever cos: the DIRECTION
    effect. The two do not sum to the total and are not meant to, rel is not linear in either,
    but if one of them is the whole move and the other is noise then the lever is a magnitude
    lever or an angle lever and this says which.
    """
    r0, c0 = base["norm_ratio"], base["cos"]
    r1, c1 = lev["norm_ratio"], lev["cos"]
    b, t = rel_of(r0, c0), rel_of(r1, c1)
    mag, dr = rel_of(r1, c0), rel_of(r0, c1)
    ang0 = math.degrees(math.acos(max(-1.0, min(1.0, c0))))
    ang1 = math.degrees(math.acos(max(-1.0, min(1.0, c1))))
    return {
        "what": name,
        "base": {"rel": b, "norm_ratio": r0, "cos": c0, "angle_degrees": ang0},
        "lever": {"rel": t, "norm_ratio": r1, "cos": c1, "angle_degrees": ang1},
        "total_move_in_rel": t - b,
        "magnitude_only": {"what": "the lever norm ratio at the base arm direction",
                           "rel": mag, "move": mag - b,
                           "share_of_the_total_move": (mag - b) / (t - b) if t != b else None},
        "direction_only": {"what": "the lever direction at the base arm norm ratio",
                           "rel": dr, "move": dr - b,
                           "share_of_the_total_move": (dr - b) / (t - b) if t != b else None},
        "angle": {"degrees_closed": ang0 - ang1,
                  "fraction_of_the_base_angle_closed": (ang0 - ang1) / ang0 if ang0 else None,
                  "cos_moved": c1 - c0},
        "magnitude": {"norm_ratio_moved": r1 - r0,
                      "distance_from_1_before": abs(r0 - 1.0),
                      "distance_from_1_after": abs(r1 - 1.0),
                      "got_closer_to_1": abs(r1 - 1.0) < abs(r0 - 1.0)},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-f64", required=True)
    ap.add_argument("--ref-bf16", required=True)
    ap.add_argument("--arm", action="append", default=[], metavar="TAG=PATH")
    ap.add_argument("--r149", required=True, help="SOFTMAX_ARM_TABLE.json, read not retyped")
    ap.add_argument("--correction-report", required=True)
    ap.add_argument("--loadavg", default="", help="loadavg sampled DURING this scoring run")
    ap.add_argument("--cotenant", default="", help="what else was on this host")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    f64, f64m = load(a.ref_f64)
    bf16, bf16m = load(a.ref_bf16)
    arms = dict(s.split("=", 1) for s in a.arm)
    keys = set(f64) & set(bf16)
    for p in arms.values():
        g, _ = load(p)
        keys &= set(g)
    keys = sorted(keys)

    r149 = json.loads(open(a.r149).read())
    corr = json.loads(open(a.correction_report).read())

    out = {
        "what": __doc__.strip().splitlines()[0],
        "environment": {
            "host": socket.gethostname(), "board": None, "device_involved": False,
            "QUIET": False,
            "co_tenant": a.cotenant or "unknown",
            "loadavg": loadavg_block(a.loadavg),
            "why_no_board_and_no_aiclk":
                "CPU only; the device arms are read as banked tensors. Each arm own DEV_*.json "
                "carries the AICLK sampled DURING its run, its card and its board class.",
            "NO_TIMING_CLAIM_FROM_THIS_HOST":
                "Amendment 1: this row is a co-tenant on qb1 and takes no wall-clock or perf "
                "claim there. rel, r, cos and the angle are load-insensitive, so the reading "
                "stands and no duration is quoted.",
        },
        "row": "of3t-angle", "defect": "D242",
        "frame": "of3t-frame384: boundary_n384.pt (8cb3a586...) / block47_boundary.pt "
                 "(a55ef1c4...), padded 384, 56 real tokens",
        "n_tensors": len(keys),
        "INJECTION": {
            "convention": "graph-cut-external (D242 repaired)",
            "correction": {"path": corr["out"]["path"], "sha256": corr["out"]["sha256"],
                           "norm": corr["norms"]["correction"],
                           "duplicate_share_of_the_hooked_cot_z":
                               corr["norms"]["duplicate_share_of_the_hooked_cot_z"]},
            "A42": "one correction, computed once on the f64 arm that defines this boundary, "
                   "passed to the bf16 reference through --cot-correction and to every device "
                   "arm as the pre-subtracted cotangent. No arm computes its own.",
            "NOT_the_model_frame_correction": corr["NOT_THE_MODEL_FRAME"],
            "references": {"f64": {"path": a.ref_f64,
                                   "stamp": f64m.get("injection_convention")},
                           "bf16": {"path": a.ref_bf16,
                                    "stamp": bf16m.get("injection_convention")}},
        },
        "DEFINITIONS": {
            "concatenated": "three norms of one vector pair over the 2,736 tensors. "
                            "rel^2 = 1 + r^2 - 2 r cos is an identity. Every split is here.",
            "mass_weighted": "of3t-trunkg043/score.py (A23). rel is a quadratic mean, r and cos "
                             "are arithmetic means, so the identity does NOT hold and no angle "
                             "is read off it. R149 table is in this definition, so the ladder "
                             "comparison is too.",
        },
        "CONTROLS": {}, "ARMS": {}, "LADDER_vs_R149": {}, "THE_LEVER_DECOMPOSED": {},
    }

    aa = concat_triple(f64, f64, keys)
    zero = concat_triple({k: torch.zeros_like(f64[k]) for k in keys}, f64, keys)
    out["CONTROLS"]["AA_the_f64_reference_against_itself"] = {
        "rel_l2": aa["rel_l2"], "exactly_zero": aa["rel_l2"] == 0.0}
    out["CONTROLS"]["A16_zero_gradient_baseline"] = {
        "rel_l2": zero["rel_l2"], "exactly_one": zero["rel_l2"] == 1.0}
    out["CONTROLS"]["CHECKPOINT_IS_INERT_on_the_repaired_producer"] = {
        "source": "perf/of3t_recut/C64_CONTROLS.json CHECKPOINT_IS_INERT",
        "tensors_bit_identical": 2736, "verdict": "cited, not re-derived"}
    out["CONTROLS"]["THE_CORRECTION_REPRODUCES_THE_c64_CONTROL"] = {
        "what": "this frame correction at crop 384 against of3t-recut own crop-64 control on "
                "the same capture. The capture has 56 real tokens, so cropping to 64 loses "
                "nothing and the two must agree.",
        "crop64_banked": {
            "cot_z_norm_hooked": 0.0005430220033757261,
            "correction_norm": 0.0005355497738277162,
            "cot_z_norm_injected": 0.0004525357226265418,
            "duplicate_share": 0.9862395455404046,
            "source": "perf/of3t_recut/C64_CONTROLS.json DEFAULT_MOVES"},
        "crop384_here": {
            "cot_z_norm_hooked": corr["norms"]["cot_z_hooked"],
            "correction_norm": corr["norms"]["correction"],
            "cot_z_norm_injected": corr["norms"]["cot_z_external"],
            "duplicate_share": corr["norms"]["duplicate_share_of_the_hooked_cot_z"]},
    }
    c6 = out["CONTROLS"]["THE_CORRECTION_REPRODUCES_THE_c64_CONTROL"]
    c6["agree_bit_for_bit"] = all(c6["crop64_banked"][k] == c6["crop384_here"][k]
                                  for k in c6["crop384_here"])

    floor_c = concat_triple(bf16, f64, keys)
    floor_m = mw_triple(bf16, f64, keys)
    bar = (2 ** 0.5) * floor_m["rel_l2"] / floor_m["norm_ratio"]
    out["FLOOR_AND_BAR_ON_THE_CORRECTED_REFERENCES"] = {
        "floor_concatenated": floor_c,
        "floor_mass_weighted": floor_m,
        "A26_style_reachable_bar_for_this_scope": bar,
        "A26_bar_arithmetic": "sqrt(2) * floor / mass_weighted_norm_ratio(floor), the same "
                              "arithmetic of3t-frame384 uses. Recomputed on the corrected "
                              "floor, not borrowed.",
        "the_bar_R149_read_against": r149["A26_in_frame_bar"]["value"],
        "the_floor_R149_read_against": r149["references"]["the_floor_their_bf16_vs_float64"],
        "their_own_split_on_the_repaired_functional": split(
            floor_c["rel_l2"], floor_c["norm_ratio"], floor_c["cos"],
            "upstream own bf16 step against the float64 reference, both repaired",
            "vs-float64 -- the reference own error, for scale"),
    }

    conc = {}
    for tag, path in arms.items():
        g, meta = load(path)
        cb = concat_triple(g, bf16, keys)
        cf = concat_triple(g, f64, keys)
        conc[tag] = {"bf16": cb, "f64": cf}
        e = {
            "artifact": {"path": path, "sha256": sha256_file(path)},
            "GRADED_SPACE_vs_upstream_bf16": {
                "concatenated": cb,
                "split": split(cb["rel_l2"], cb["norm_ratio"], cb["cos"],
                               tag + " against upstream own bf16 step, both repaired",
                               "vs-upstream-bf16 -- THE SPACE THE CLAUSE IS GRADED IN"),
                "mass_weighted": mw_triple(g, bf16, keys)},
            "CONTRAST_vs_float64": {
                "concatenated": cf,
                "split": split(cf["rel_l2"], cf["norm_ratio"], cf["cos"],
                               tag + " against the float64 reference, both repaired",
                               "vs-float64 -- CONTRAST ONLY, not carried across (R174)"),
                "mass_weighted": mw_triple(g, f64, keys)},
        }
        e["x_A26_bar_mass_weighted"] = (
            e["GRADED_SPACE_vs_upstream_bf16"]["mass_weighted"]["rel_l2"] / bar)
        mb = e["GRADED_SPACE_vs_upstream_bf16"]["split"]["shares_of_the_measured_error"]["magnitude"]
        mf = e["CONTRAST_vs_float64"]["split"]["shares_of_the_measured_error"]["magnitude"]
        e["the_two_spaces_disagree_on_the_magnitude_share_by"] = mf / mb if mb else None
        out["ARMS"][tag] = e

    if "SHIP_A" in arms and "SHIP_B" in arms:
        ga, _ = load(arms["SHIP_A"])
        gb, _ = load(arms["SHIP_B"])
        ident = sum(1 for k in keys if torch.equal(ga[k], gb[k]))
        ab = concat_triple(ga, gb, keys)
        out["CONTROLS"]["AA_FLOOR_two_runs_of_the_shipped_arm"] = {
            "tensors": len(keys), "bit_identical": ident,
            "all_bit_identical": ident == len(keys),
            "rel_l2_between_them": ab["rel_l2"],
            "reading_delta_vs_upstream_bf16": (conc["SHIP_B"]["bf16"]["rel_l2"]
                                               - conc["SHIP_A"]["bf16"]["rel_l2"]),
            "cos_delta_vs_upstream_bf16": (conc["SHIP_B"]["bf16"]["cos"]
                                           - conc["SHIP_A"]["bf16"]["cos"]),
            "what": "two runs of the same lever on the same card in the same session. Any "
                    "arm-to-arm difference below this is not a lever."}

    MAP = {"SHIP": "CTRL_B", "SHIP_A": "CTRL_B", "SHIP_B": "CTRL_B",
           "VERB": "VERB_HF", "MODW": "CEIL_HF3"}
    for tag in arms:
        pre = r149["arms"].get(MAP.get(tag, ""))
        if pre is None:
            continue
        mwb = out["ARMS"][tag]["GRADED_SPACE_vs_upstream_bf16"]["mass_weighted"]["rel_l2"]
        mwf = out["ARMS"][tag]["CONTRAST_vs_float64"]["mass_weighted"]["rel_l2"]
        out["LADDER_vs_R149"][tag] = {
            "R149_arm": MAP[tag], "R149_lever": pre["argv_lever"],
            "definition": "mass_weighted, which is the definition R149 numbers are in",
            "vs_upstream_bf16": {"double_counted": pre["vs_upstream_bf16"], "repaired": mwb,
                                 "moved": mwb - pre["vs_upstream_bf16"],
                                 "repaired_over_double_counted": mwb / pre["vs_upstream_bf16"]},
            "vs_float64": {"double_counted": pre["vs_float64"], "repaired": mwf,
                           "moved": mwf - pre["vs_float64"],
                           "repaired_over_double_counted": mwf / pre["vs_float64"]},
            "x_A26_bar": {"double_counted": pre["multiple_of_the_A26_in_frame_bar"],
                          "repaired": mwb / bar,
                          "note": "the repaired multiple is against the bar recomputed on the "
                                  "corrected floor, not against the bar R149 read"},
        }

    base = "SHIP_A" if "SHIP_A" in conc else ("SHIP" if "SHIP" in conc else None)
    if base:
        for tag in ("VERB", "MODW"):
            if tag not in conc:
                continue
            out["THE_LEVER_DECOMPOSED"][base + "_to_" + tag] = {
                "GRADED_SPACE_vs_upstream_bf16": lever_effect(
                    conc[base]["bf16"], conc[tag]["bf16"],
                    "the exact softmax (" + tag + ") against the shipped device softmax, in "
                    "vs-upstream-bf16, on the REPAIRED functional"),
                "CONTRAST_vs_float64": lever_effect(
                    conc[base]["f64"], conc[tag]["f64"],
                    "the same pair against float64 -- contrast only, not carried"),
            }

    worst, failed = 0.0, []
    for tag, v in out["ARMS"].items():
        for sp in ("GRADED_SPACE_vs_upstream_bf16", "CONTRAST_vs_float64"):
            d = v[sp]["split"]["IDENTITY"]["rel_difference"]
            if d is not None:
                worst = max(worst, d)
                if d > 1e-12:
                    failed.append(tag + "/" + sp)
    out["IDENTITY_HOLDS_IN_EVERY_SPLIT"] = {
        "worst_rel_difference": worst, "bar": 1e-12, "failed": failed,
        "verdict": "every split is exact rather than modelled" if not failed
                   else "FAILED: nothing above may be read"}

    out["provenance"] = {
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                     text=True).stdout.strip(),
        "inputs": {k: sha256_file(v) for k, v in
                   [("ref_f64", a.ref_f64), ("ref_bf16", a.ref_bf16)] + sorted(arms.items())},
    }
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in ("CONTROLS", "LADDER_vs_R149",
                                          "THE_LEVER_DECOMPOSED",
                                          "IDENTITY_HOLDS_IN_EVERY_SPLIT")},
                     indent=1, default=str))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
