#!/usr/bin/env python3
"""D237 UPDATE 2. Re-project the clause on the lever reading that is a VALID GRADIENT.

At pass 377 I projected the frame-matched clause at 0.8986x the bar -- PASSING -- using a trunk
lever reading of 0.5547455957585244. `of3t-f64route` then established what that configuration
is: CEIL_HF/VERB_HF serves 5285 TAPED calls and, on the 912 untaped blocks of the checkpointed
forward, computes each block's OUTPUT with the device softmax and its JACOBIAN with the float64
one. Forward and backward disagree, so it is not a function anyone can ship and not a gradient of
anything. Its own row reproduced it exactly (ratio 1.0) and then said why it differs.

ROUTE_HF serves 7442 of 7442 tail blocks and makes both the same softmax. It reads
0.7734340172378431 -- worse, and honest. This script re-projects on it.

The projection's shape is unchanged from pass 377 and so is its caveat: the multiple of upstream's
own bf16 is measured IN FRAME and carried onto the model boundary, which has 3.12x less trunk
gradient mass. `of3t-modelframe` measures it. This is arithmetic on a measured lever, not a result.
"""
import json, math, argparse, hashlib, socket, os

T = "pairformer_stack"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument("--graded", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
d = json.load(open(a.graded))
secs = d["per_section"]["renorm_vs_UPSTREAM_BF16"]
fl = d["per_section"]["UPSTREAM_BF16_vs_FLOAT64"]
bar = d["bars"]["A26_reachable_bar_vs_their_bf16"]
W = sum(s["ref_sq"] for s in secs.values())
wt = secs[T]["ref_sq"]
S_other = sum(s["ref_sq"] * s["mass_weighted_rel_l2"] ** 2 for k, s in secs.items() if k != T)
A26T = math.sqrt(2.0) * fl[T]["mass_weighted_rel_l2"] / fl[T]["mass_weighted_norm_ratio"]
admissible = math.sqrt((bar * bar * W - S_other) / wt)


def model(xt):
    return math.sqrt((S_other + wt * xt * xt) / W)


MULT = 2.2340903768268046      # ours / upstream's own bf16, in frame (FRAME_N384.json)
SHIPPED = 1.029395337772341    # trunk in frame, no lever

ARMS = [
 ("no_lever", 1.0, None,
  "the trunk as it ships"),
 ("CEIL_HF_hybrid_RETRACTED_AS_A_TARGET", None, 0.5547455957585244,
  "5285 taped calls; the 912 untaped blocks take the device softmax forward and the float64 "
  "Jacobian. Forward and backward disagree: not shippable, not a gradient. This is the number "
  "my pass-377 projection used and it was wrong to."),
 ("ROUTE_HF_consistent", None, 0.7734340172378431,
  "7442 of 7442 tail blocks, forward AND Jacobian the same softmax. The honest lever."),
]
cases = {}
for name, lev, reading, why in ARMS:
    lever = lev if lev is not None else SHIPPED / reading
    m = MULT / lever
    xt = A26T * m / math.sqrt(2.0)
    r = model(xt)
    cases[name] = {"trunk_in_frame_reading": reading, "lever_factor": lever,
                   "implied_multiple_of_their_bf16": m, "x_t": xt, "model_reading": r,
                   "x_bar": r / bar, "passes": r <= bar, "why": why}

route = cases["ROUTE_HF_consistent"]
out = {
 "defect": "D237 UPDATE 2",
 "what": "the pass-377 projection is CORRECTED DOWNWARD: on the lever reading that is a valid "
         "gradient the frame-matched clause reads %.4fx the bar and FAILS, not 0.8986x passing."
         % route["x_bar"],
 "host": socket.gethostname(), "device_involved": False,
 "inputs": {"graded": {"path": a.graded, "sha256": sha(a.graded)}},
 "lever_source": "of3t-f64route, qb1 card 2, p150a, AICLK 1350 median DURING (374 and 416 "
                 "samples over the two arms). ROUTE_HF 2527 s, VERB_HF 2753 s.",
 "bar": bar, "trunk_allowance": admissible, "trunk_A26_perfect": A26T,
 "cases": cases,
 "what_is_now_necessary_and_what_is_left": {
   "frame_fix_alone": "without it a bit-exactly perfect trunk reads 3.1624x the bar (D237)",
   "lever_alone": "without it the frame-matched clause reads %.4fx (FAILS)"
                  % cases["no_lever"]["x_bar"],
   "both": "%.4fx (FAILS), so both are necessary and neither is sufficient" % route["x_bar"],
   "remaining_factor_on_the_trunk": route["x_t"] / admissible,
   "remaining_reading": "the trunk must reach %.10f and the projection puts it at %.10f"
                        % (admissible, route["x_t"]),
   "owner_of_the_remainder": "of3t-cotterm -- the AttentionPairBias across-component cotangent "
                             "error. This replaces '12.8 % deficit' and '5.25 % left' with a "
                             "quantity that is scoped to the clause the charter applies.",
 },
 "collateral": "of3t-trunkceiling's 12.8 % norm deficit, its located unexplained object, reads "
               "3.6 % on ROUTE_HF (norm ratio 0.964213 against 0.872203). It pays in cosine "
               "(0.856707 -> 0.782945) and the rel L2 is the net.",
 "inference_hard_constraint": "of3t-f64route measured base == off == on BYTE FOR BYTE on "
                              "openfold3, rf3 and protenix-v2, 18 folds, qb1 card 3 p150a, AICLK "
                              "median 1350 on every one and no arm under 1200, with "
                              "TT_BIO_HOST_F64_SOFTMAX_AB=all forced at every site. Digests "
                              "identical. The training-only constraint holds on the landed route.",
 "caveat": "a PROJECTION. The multiple of upstream's own bf16 is measured in frame and carried "
           "onto the model boundary, which has 3.12x less trunk gradient mass. of3t-modelframe "
           "measures it.",
}
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps({"what": out["what"], "cases": {k: {kk: v[kk] for kk in
      ("lever_factor", "x_bar", "passes")} for k, v in cases.items()},
      "left": out["what_is_now_necessary_and_what_is_left"]}, indent=1))
