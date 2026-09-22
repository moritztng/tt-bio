#!/usr/bin/env python3
"""Add the per-block cotangent reading to AMPLIFICATION.json.

of3t-bwdaccum's discriminator, applied to the DiT-24 on the reference's OWN arms: the cotangent
entering each block's backward, scored against the in-frame float64. That float64 is the in-frame
one in the strongest available sense -- the f64 arm reproduces the capture's own xl_out AND its
grad_f64 at rel_l2 exactly 0.0, so its cotangents are bit-for-bit the function that produced the
reference, checked rather than declared.
"""
import json
from pathlib import Path

A = Path("/home/ttuser/.coworker/wt/of3t-tapeamp/perf/of3t_tapeamp")
amp = json.load(open(A / "AMPLIFICATION.json"))
b = json.load(open(A / "AMP_bf16_blockcot.json"))
f = json.load(open(A / "AMP_f32_blockcot.json"))
r = json.load(open(A / "AMP_f64_blockcot.json"))

B, F = b["BLOCK_COTANGENT"], f["BLOCK_COTANGENT"]
pb, pf = B["per_block_forward_order"], F["per_block_forward_order"]
ks = sorted(pb, key=int, reverse=True)           # travel order: 23 first, 0 last

rows, prev_b, prev_f = [], None, None
for k in ks:
    rb, rf = pb[k]["rel_l2"], pf[k]["rel_l2"]
    rows.append({"blocks_travelled": 23 - int(k), "block": int(k),
                 "bf16_rel_l2": rb, "bf16_step": (rb / prev_b) if prev_b else None,
                 "f32_rel_l2": rf, "f32_step": (rf / prev_f) if prev_f else None,
                 "bf16_over_f32": rb / rf,
                 "reference_norm": pb[k]["reference_norm"],
                 "bf16_norm_over_reference": pb[k]["arm_norm"] / pb[k]["reference_norm"]})
    prev_b, prev_f = rb, rf

amp["BLOCK_COTANGENT"] = {
 "what": "the cotangent entering each DiT block's backward, upstream 0.4.3's own bf16 and fp32 "
         "recipes against its own in-frame float64, 24 blocks, structures 0/15/31/47 of 48",
 "REFERENCE_IS_IN_FRAME": {
   "f64_arm_forward_median_vs_capture_xl_out": r["FORWARD"]["median"],
   "f64_arm_gradient_median_vs_capture_grad_f64": r["GRADIENT_full_scope"]["median_rel"],
   "reading": "both exactly 0.0, so the arm that produced these reference cotangents is "
              "bit-for-bit the function that produced the capture's reference. Not asserted.",
 },
 "DISCRIMINATOR_ANSWER": "NEITHER of the two shapes the brief named. Not flat at the bf16 floor, "
   "so not a wrong leaf backward: it moves 13.802x over 24 blocks. Not monotone degradation with "
   "depth, so not a uniform per-block injection: 8.191x of that 13.802x is ONE block boundary "
   "(19 -> 18) and the remaining 18 blocks are a plateau that ends slightly LOWER than it began "
   "(2.587e-01 -> 2.510e-01). The shape is a ramp over five blocks, one localised step, then "
   "saturation. That is of3t-bwdaccum's third pre-registered shape, which it found on the "
   "48-block trunk; this is the first sighting on the DiT-24, and it is in the REFERENCE's own "
   "arm rather than ours.",
 "PRECISION_INDEPENDENT_SHAPE": {
   "bf16_total_over_24_blocks": B["degradation_over_24_blocks"],
   "f32_total_over_24_blocks": F["degradation_over_24_blocks"],
   "step_at_19_to_18_bf16": pb["18"]["rel_l2"] / pb["19"]["rel_l2"],
   "step_at_19_to_18_f32": pf["18"]["rel_l2"] / pf["19"]["rel_l2"],
   "bf16_over_f32_min": min(x["bf16_over_f32"] for x in rows),
   "bf16_over_f32_max": max(x["bf16_over_f32"] for x in rows),
   "reading": "The two curves ramp together, step at the SAME block boundary and plateau "
              "together, with the ratio of their absolute errors constant to within 1.6x across "
              "all 24 blocks while the errors themselves are four orders of magnitude apart. A "
              "dtype boundary cannot produce a shape that is the same shape 11,800x to 18,300x "
              "away in absolute error. The shape belongs to the function's reverse pass.",
 },
 "MAGNITUDE_IS_NOT_THE_STEP": {
   "reference_norm_block_23": pb["23"]["reference_norm"],
   "reference_norm_block_19": pb["19"]["reference_norm"],
   "reference_norm_block_18": pb["18"]["reference_norm"],
   "reference_norm_block_0": pb["0"]["reference_norm"],
   "norm_growth_over_24_blocks": pb["0"]["reference_norm"] / pb["23"]["reference_norm"],
   "norm_step_at_19_to_18": pb["18"]["reference_norm"] / pb["19"]["reference_norm"],
   "bf16_norm_over_reference_range": [min(x["bf16_norm_over_reference"] for x in rows),
                                      max(x["bf16_norm_over_reference"] for x in rows)],
   "reading": "At the step the cotangent's own magnitude moves only 1.399x while its RELATIVE "
              "error moves 8.191x, so the step is not a magnitude event. The arm's cotangent is "
              "the right size at every block (0.936 to 0.992 of the reference's norm), so "
              "nothing is blowing up or collapsing. A relative error that jumps at constant "
              "magnitude is a cancellation event, which is the mechanism of3t-bwdaccum measured "
              "directly on the trunk as a site cancellation factor K, with every leaf reading at "
              "or below u*K at u = 2^-8.",
 },
 "TABLE_travel_order": rows,
 "LIMIT": "These are UPSTREAM's per-block curves, not ours. Our own per-block cotangent curve on "
          "the device arm is NOT measured in this pass and is the next arm: it needs the tape's "
          "DiT block outputs hooked the way census_run.py hooks _unshard, plus a block identity "
          "the tape does not carry today. What this reading establishes is that the accumulation "
          "exists in the reference, at 2.5e-01 by the bottom of the DiT-24, which is what makes "
          "our gradient at 0.734x of upstream's own readable as a result.",
}
(A / "AMPLIFICATION.json").write_text(json.dumps(amp, indent=1, sort_keys=False) + "\n")
print(json.dumps({k: amp["BLOCK_COTANGENT"][k] for k in
                  ("DISCRIMINATOR_ANSWER", "PRECISION_INDEPENDENT_SHAPE",
                   "MAGNITUDE_IS_NOT_THE_STEP", "REFERENCE_IS_IN_FRAME")}, indent=1))
