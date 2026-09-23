#!/usr/bin/env python3
"""Are the trunk reduction and the on-device softmax the SAME precision ceiling?

Pass 379 asserted they were, in VERDICT and in R131, on the strength of both being "TT fp32 is
not IEEE fp32". This script tests that assertion by putting both sites in one unit -- mantissa
bits short of IEEE fp32 -- and it does not survive.

The unit is chosen so the cancellation factor cancels out. Each reading is a RATIO of two
computations of the same quantity (ours against a better-precision counterpart), and a factor F
in such a ratio is log2(F) bits of significand. `LADDER_SHAPE.json` validates the unit directly:
a deliberately 2-bit-short accumulator reads 3.928, against the 4.000 the deficit predicts, flat
in K over a 2,300x span.

The softmax numbers are QUOTED from of3t-softmax's cost table as the campaign charter records
them. They are not re-measured here and this script does not open a device.
"""
import json, math, pathlib, platform

TRUE_FP32_SOFTMAX = 1e-7   # "a true fp32 softmax agrees with float64 to ~1e-7"
SOFTMAX = {"shipped": 2.029e-02, "precise_config()": 1.646e-03, "_accurate_softmax": 5.156e-04}
TRUNK = (3.0, 4.6)         # of3t-modelframe's ATTRIBUTION.json, ours against upstream's own bf16

sites = [{"site": "softmax/" + k, "source": "QUOTED from of3t-softmax's cost table",
          "factor_over_better_precision": v / TRUE_FP32_SOFTMAX,
          "bits_short_of_ieee_fp32": math.log2(v / TRUE_FP32_SOFTMAX)} for k, v in SOFTMAX.items()]
sites += [{"site": "trunk LayerNorm affine reduction (%s bound)" % b,
           "source": "of3t-modelframe ATTRIBUTION.json, frame-matched",
           "factor_over_better_precision": f, "bits_short_of_ieee_fp32": math.log2(f)}
          for b, f in (("low", TRUNK[0]), ("high", TRUNK[1]))]

sm = [s["bits_short_of_ieee_fp32"] for s in sites if s["site"].startswith("softmax")]
tr = [s["bits_short_of_ieee_fp32"] for s in sites if s["site"].startswith("trunk")]
gap_bits = min(sm) - max(tr)

out = {
    "instrument": "perf/of3t_orchestrator/lnladder/bit_deficit_sites.py",
    "host": platform.node(),
    "unit": "mantissa bits short of IEEE fp32, = log2 of the ratio between our reading and a "
            "better-precision counterpart of the SAME quantity. The cancellation factor cancels "
            "out of such a ratio, which is why the two sites are comparable at all.",
    "unit_validated_by": "LADDER_SHAPE.json -- a 2-bit-short accumulator reads 3.928 against a "
                         "predicted 4.000, flat in K over a 2,300x span",
    "sites": sites,
    "softmax_bits_short_range": [min(sm), max(sm)],
    "trunk_bits_short_range": [min(tr), max(tr)],
    "gap_in_bits_between_the_two_sites": gap_bits,
    "gap_as_a_factor": 2.0 ** gap_bits,
    "verdict": (
        "NOT one mechanism at one magnitude. The softmax sites sit %.1f to %.1f bits short of "
        "IEEE fp32; the trunk reduction sits %.1f to %.1f. The nearest pair is %.1f bits apart, "
        "a factor of %.0f. Both are on-device precision deficits and both are consistent with "
        "TT fp32 being short of IEEE fp32, but they are not the same ceiling and the trunk will "
        "not be fixed by whatever fixes the softmax."
        % (min(sm), max(sm), min(tr), max(tr), gap_bits, 2.0 ** gap_bits)),
    "consequence": (
        "The trunk's deficit is SMALL -- about two bits. That is the opposite of the softmax "
        "finding, where nothing on-device came within four orders of magnitude of real fp32 and "
        "a host round trip was the only route. A two-bit deficit is the size a higher-precision "
        "on-device accumulate could plausibly close, so of3t-lnreduce should price the on-device "
        "rung BEFORE assuming it needs a host reduction. Testable, and cheaper if it holds."),
    "caveat": (
        "The two readings are different computations -- a softmax output against float64, and a "
        "reduction's dW against upstream's own bf16 step -- so this is an order-of-magnitude "
        "argument in a common unit, not a proof. It is decisive only because the gap is %.1f "
        "bits, far beyond what the shape difference plausibly explains." % gap_bits),
}
p = pathlib.Path(__file__).with_name("BIT_DEFICIT_SITES.json")
p.write_text(json.dumps(out, indent=1) + "\n")
print("site                                        factor      bits short of IEEE fp32")
for s in sites:
    print("  %-40s %9.0fx   %5.1f" % (s["site"], s["factor_over_better_precision"],
                                      s["bits_short_of_ieee_fp32"]))
print("\n" + out["verdict"])
print("\nwrote", p)
