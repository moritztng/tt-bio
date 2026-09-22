#!/usr/bin/env python3
"""of3t-lnreduce step 0: the pre-registration, written BEFORE the first reading.

Run it to emit PREREG.json. Nothing in this file reads a measurement, and the file is
committed before any probe runs, so the grading rule cannot move after the numbers exist.

THE HYPOTHESIS. The LayerNorm affine leaves carry 93.80 % of the frame-matched trunk`s excess
(of3t-modelframe ATTRIBUTION.json). Their gradients are pure reductions over every leading
position,

    dW = sum_positions (dy * xhat)        dB = sum_positions dy

and on the pair track that sum runs over N x N positions while `single_transition` -- 5.00 % of
the error mass at 1.3896x upstream, the one sub-module under the bar -- runs over N. At crop 384
that is 147,456 against 384, a 384x difference in accumulation length. Three rows have excluded
the alternatives: of3t-vjpln (690 float64 substitutions inside the layer-norm and linear
backwards moved R44 the WRONG way, 2.203113 -> 2.206861), of3t-trunkact (NO-GO on the forward
activations) and of3t-apbleaf (a 1.1745x cotangent residue becomes a 2.4044x gradient residue,
99.72 % of the dW damage lying ACROSS the reference cotangent). What is left is the reduction
itself, which is a property of how the sum is computed rather than of how the op differentiates.

THE TEST is the K ladder: at FIXED padded width, vary the number of real (unmasked) accumulated
positions K and read ours against torch-bf16, which is upstream`s own level.

THE FALSIFIER, registered here and not rescued later: if ours/torch-bf16 is FLAT in K, the
accumulation hypothesis is dead and this row says so in its first heading.

THE MAGNITUDE IS REGISTERED TWO-SIDED, because the naive forms of the hypothesis BRACKET the
observation rather than predicting it:

  * a pure sqrt(K) bf16 accumulation, 147,456 against 384, predicts sqrt(147456/384) = 19.6x
    and we observe 3-4.6x. Too big.
  * a bf16 accumulator against an fp32 one predicts 2^-8 / 2^-24 = 2^16 on the accumulator
    term. Far too big.

So neither naive form is what is happening, and a reading that lands between them does NOT
confirm either. The deliverable is a measurement of what the accumulator IS -- dtype, order,
kernel config, counted at runtime -- not a fit to a story.
"""
import json, socket, subprocess, sys, time

REF = {
    "pair_transition.layer_norm": {"over_upstream": 4.581496569954559, "K_pair": 147456},
    "tri_att_start.layer_norm":   {"over_upstream": 3.307964392562707, "K_pair": 147456},
    "attn_pair_bias.layer_norm_a": {"over_upstream": 1.7711188004138814, "K_pair": 147456},
    "single_transition.layer_norm": {"over_upstream": 1.394485036228151, "K_single": 384},
}

PRE = {
 "row": "of3t-lnreduce",
 "what": "Pre-registration of the accumulation hypothesis and its falsifier, committed before "
         "the first reading.",
 "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
 "host": socket.gethostname(),
 "inherited": {
   "clause": 0.27095922968432157,
   "bar": 0.15210099830945006,
   "multiple_of_bar": 1.7814428090278143,
   "trunk_section_now": 0.9969599833682794,
   "trunk_section_allowance": 0.44608901561034203,
   "trunk_must_fall_by": 2.2348902315028583,
   "trunk_over_upstream_bf16": 2.970162431380236,
   "source": "perf/of3t_modelframe/CLAUSE.json, MODEL_FRAMEMATCHED_composed3660_n384.json, "
             "ATTRIBUTION.json at commit 158c94220",
 },
 "per_leaf_reference": REF,

 "hypothesis": "The carrier is the AFFINE REDUCTION, not the differentiation. dW and dB "
               "accumulate over every leading position, the true sums cancel heavily, and both "
               "the cotangent`s error and the accumulator`s own rounding are multiplied by that "
               "cancellation. Therefore ours/torch-bf16 must be GRADED by the number of real, "
               "unmasked accumulated positions K.",

 "test": {
   "name": "the K ladder",
   "design": "At FIXED padded width, vary the real-position count K over at least four rungs by "
             "MASKING leading positions to exact zero, and read dW/dB error against a float64 "
             "reference for ours and for torch-bf16. A zeroed position contributes nothing to "
             "the sum but still occupies an accumulation slot, so this separates `error grows "
             "with real signal` from `error grows with accumulation length`.",
   "masking": "an unmasked statistic at padded 384 is pad junk -- of3t-apbleaf read 1.3827 "
              "unmasked against 1.7e-03 masked on the same tensors. Every statistic here is "
              "computed on the real rows only, and the pad rows are set to exact zero on the "
              "input side.",
   "min_rungs": 4,
 },

 "grading": {
   "CONFIRMED": "ours/torch-bf16 rises monotonically with K across the ladder and the rise is at "
                "least 1.5x from the smallest rung to the largest. Then and only then is the "
                "trunk arm re-run with the affine reduction at the precision the ladder names.",
   "REFUTED":   "ours/torch-bf16 is flat in K -- the largest rung is within 1.25x of the "
                "smallest, or the trend is not monotone in the direction the hypothesis needs. "
                "Then the accumulation hypothesis is DEAD, this row says so in its first "
                "heading, no arm is run, and the row does not go looking for a rescue.",
   "in_between": "a rise present but under 1.5x is reported as PARTIAL with the measured "
                 "exponent, and the arm is run only if the measured exponent predicts the "
                 "2.2349x the trunk must fall.",
 },

 "registered_two_sided": {
   "sqrt_K_bf16_accumulation_predicts": 19.595917942265423,
   "bf16_vs_fp32_accumulator_predicts": 65536.0,
   "observed_range": [1.394485036228151, 4.581496569954559],
   "statement": "Both naive forms are refuted by the observation BEFORE this row starts. A "
                "reading between them confirms neither. What the row owes is the accumulator "
                "itself, read at runtime.",
 },

 "what_would_make_this_row_worthless": [
   "an unmasked statistic at padded width (pad junk, of3t-apbleaf)",
   "a source read of the reduction instead of a runtime count "
   "(eligibility-firing-condition-is-not-a-code-fact)",
   "scoring against another approximation instead of float64 validated by float64 central "
   "finite differences",
   "reporting on-device fp32 and a host float64 reduction as one rung -- TT fp32 is not IEEE "
   "fp32 and a single headline hides a silicon ceiling",
 ],
}

if __name__ == "__main__":
    PRE["git_commit_when_written"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = sys.argv[1] if len(sys.argv) > 1 else "perf/of3t_lnreduce/PREREG.json"
    json.dump(PRE, open(out, "w"), indent=1)
    print("wrote " + out)
