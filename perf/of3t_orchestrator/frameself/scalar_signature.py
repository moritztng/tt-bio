#!/usr/bin/env python3
"""D242's signature: the control's overshoot is ONE SCALAR, not a chain effect.

`of3t-twoside` reported its per-block control curve as "flat, 0.6006 to 1.1438 with ratio 1.510
to 2.111". That quotes the extremes. Read as a distribution over all 48 blocks it is a constant
to 4 %, which is a much sharper lead: forty-eight independently-parameterised blocks do not
overshoot by the same factor through 48 different arithmetic paths.

Recomputed from that row's committed artifact rather than transcribed, so this file cannot drift
from it. No device, no card, no model.

  scalar_signature.py --curve perf/of3t_twoside/CTRL_PERBLOCK.json --out SCALAR_SIGNATURE.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import statistics as st
from pathlib import Path

# of3t-twoside's headline control readings, the two scalars this file reasons against.
CTRL_REL_L2 = 0.7945281613194305       # injected f64 trunk vs grads_f64_043's trunk section
CTRL_NORM_RATIO = 1.7584185703064399
CTRL_COS = 0.9840549138041126
BAR = 1e-12                            # pre-registered, commit 2520681ed


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def decompose(ratio: float, cos: float) -> dict:
    """Split g_inj into (a parallel multiple of g_ref) + (the rest), from ratio and cos alone.

    g_inj = g_ref + e.  ratio = |g_inj|/|g_ref|, cos = <g_inj,g_ref>/(|g_inj||g_ref|).
    Then <e,g_ref>/|g_ref|^2 = ratio*cos - 1, and |e|/|g_ref| = sqrt(ratio^2 - 2*ratio*cos + 1).
    """
    proj = ratio * cos - 1.0
    e_rel = math.sqrt(max(ratio ** 2 - 2 * ratio * cos + 1.0, 0.0))
    return {
        "residual_norm_over_reference": e_rel,
        "residual_component_along_reference": proj,
        "cos_residual_to_reference": (proj / e_rel) if e_rel > 0 else None,
        "rel_l2_reconstructed_from_ratio_and_cos": e_rel,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    doc = json.loads(a.curve.read_text())
    curve = doc["per_block"]
    blocks = sorted(curve, key=int)
    ratio = [curve[b]["norm_ratio"] for b in blocks]
    rel = [curve[b]["rel_l2"] for b in blocks]

    mean, sd = st.mean(ratio), st.stdev(ratio)
    out_of_band = {b: curve[b]["norm_ratio"] for b in blocks
                   if not 1.6 <= curve[b]["norm_ratio"] <= 1.9}

    # The row banked a least-squares scalar per block and the residual left after it. Those are
    # the numbers that decide whether this is one defect or two, and its own state doc does not
    # carry them -- it says only "the residual left after that single scalar is small".
    scale = [curve[b]["best_scale_arm_to_ref"] for b in blocks]
    resid = [curve[b]["relative_residual_after_best_scalar"] for b in blocks]

    dec = decompose(CTRL_NORM_RATIO, CTRL_COS)
    # the control's own rel_l2 must fall out of its ratio and cos; if it does not, one of the
    # three published numbers is inconsistent with the other two and none of them can be used.
    dec["control_rel_l2_published"] = CTRL_REL_L2
    dec["agrees_with_published_rel_l2_to"] = abs(dec["rel_l2_reconstructed_from_ratio_and_cos"]
                                                 - CTRL_REL_L2)

    rep = {
        "what": __doc__.strip().splitlines()[0],
        "host": socket.gethostname(),
        "device_involved": False,
        "source": {"path": str(a.curve), "sha256": sha256_file(a.curve), "blocks": len(blocks)},
        "bar_preregistered": BAR,
        "control_rel_l2": CTRL_REL_L2,
        "ratio_over_48_blocks": {
            "mean": mean, "median": st.median(ratio), "stdev": sd,
            "coefficient_of_variation": sd / mean,
            "min": min(ratio), "max": max(ratio),
            "outside_1p6_to_1p9": out_of_band,
            "n_outside": len(out_of_band),
        },
        "rel_l2_over_48_blocks": {
            "mean": st.mean(rel), "stdev": st.stdev(rel),
            "min": min(rel), "max": max(rel),
            "first_block_touched_by_backward": {"block": "47", "rel_l2": curve["47"]["rel_l2"],
                                                "norm_ratio": curve["47"]["norm_ratio"]},
            "last_block_touched_by_backward": {"block": "0", "rel_l2": curve["0"]["rel_l2"],
                                               "norm_ratio": curve["0"]["norm_ratio"]},
        },
        "best_scalar_fit_over_48_blocks": {
            "scale_arm_to_ref": {"mean": st.mean(scale), "median": st.median(scale),
                                 "stdev": st.stdev(scale), "min": min(scale), "max": max(scale),
                                 "reciprocal_of_median": 1.0 / st.median(scale)},
            "relative_residual_after_best_scalar": {
                "mean": st.mean(resid), "median": st.median(resid), "stdev": st.stdev(resid),
                "min": min(resid), "max": max(resid),
                "argmax_block": blocks[resid.index(max(resid))]},
            "orders_of_magnitude_residual_is_above_the_bar":
                math.log10(st.median(resid) / BAR),
            "control_rel_l2_reduced_by": CTRL_REL_L2 / st.median(resid),
        },
        "decomposition": dec,
        "reading": (
            "TWO COMPONENTS, not one. A single scalar near 1.749 explains most of the overshoot: "
            "it is constant to a 4 % coefficient of variation across 48 independently-"
            "parameterised blocks, it is full size at block 47 before any trunk-internal "
            "accumulation can have happened, and fitting it reduces the control from 0.7945 to a "
            "median 0.0895. But 0.0895 is still EIGHT ORDERS above the 1e-12 bar, so removing the "
            "scalar does not make the frame pass. of3t-twoside's own summary calls that residual "
            "'small'; it is small against 0.79 and enormous against the bar, which is the "
            "name-your-reference trap (R133) in a new dress."),
        "what_it_does_not_say": (
            "It does not name the scalar, and it does not say whether the two components share a "
            "cause. It does not distinguish a scaled cotangent from a uniformly scaled backward: "
            "a per-block norm ratio cannot. And the residual is not flat -- it is 0.3003 at block "
            "47, the first the backward touches, against a 0.0895 median, so the structured part "
            "is concentrated at the entry and attenuates with depth. Any fix that removes only "
            "the scale must still be scored against the 1e-12 bar, not against 0.7945."),
    }
    a.out.write_text(json.dumps(rep, indent=1))
    print(json.dumps({"mean": mean, "cv": sd / mean, "n_outside": len(out_of_band),
                      "residual_norm_over_reference": dec["residual_norm_over_reference"],
                      "cos_residual_to_reference": dec["cos_residual_to_reference"],
                      "agrees_with_published_rel_l2_to":
                          dec["agrees_with_published_rel_l2_to"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
