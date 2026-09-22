#!/usr/bin/env python3
"""of3t-orchestrator pass 389: the sub-module partition is ENTAILED, and one of my own hypotheses dies here.

`of3t-frameself` split the 57 tensors of a block by sub-module and found the two s-branch ones
exact at every depth while all five `pair_stack.*` ones are off by roughly the same factor. It
read that as locating the defect -- *"it is the gradient arriving at the pair branch"* -- and
noted that `base_blocks.py:307` puts `ps_dropout_row_layer` on every pair update and nowhere on
the s branch, *"which matches the partition exactly"*. It then refuted dropout itself, correctly.

**The partition cannot select a candidate, because it is entailed by what the row already
measured.** `TWOBASIS.json` established by measurement that the z-only arm's `ds_in` is exactly
0.0 -- the pair track does not read the single track -- so s-branch parameters reach the loss
ONLY through `cot_s` and pair-branch parameters reach it through both. A defect in `cot_z` and
nothing else therefore produces exactly this partition, at every depth, with no mechanism at the
pair branch at all. `ps_dropout_row_layer` matches the partition for the same reason the words
"pair branch" match it. That is `retrodiction-is-not-prediction`: every structure that
distinguishes the two branches will match, so matching is worth nothing.

The split IS worth something as a consistency check -- 11 + 5 = 16 s-branch tensors per block,
times 48 blocks, is exactly TWOBASIS's 768 -- and the blockprobe beside it is this pass's real
result: the reference gradient was the one quantity never read twice and a second instrument now
puts it at 3.0392623414001263e-15 and 4.1031090433915236e-15 at the two worst blocks.

**The hypothesis I brought to this pass and had to kill.** The row's headline says the captured
`cot_z` is *"roughly 7x too large AND pointing elsewhere"*, taken from a two-scalar fit whose two
regressors have cos 0.9181951994170596. I expected that fit to be ill-conditioned and its `b_z`
to be meaningless. It IS ill-conditioned -- the normalised Gram matrix has condition number
23.4484 -- but the valley is not flat and the fit is real: `b_z` 0.55, which is what "1.8x too
large" would mean, costs residual 0.173577 against the minimum's 0.111621, 55 % worse. **The
published b_z stands.** Recorded so nobody raises it again.

What the arithmetic does do is dissolve the apparent contradiction between "1.8x" and "7x", and
sharpen what a mechanism has to explain. Every number below is computed from banked scalars, and
the reconstruction reproduces the row's own published fit to 2e-14, which is both the check on
my arithmetic and an independent check on its.

Zero card, zero model run.
"""
from __future__ import annotations

import json
import math
import sys

# every one of these is read from perf/of3t_frameself/{TWOBASIS,FIT_SPLIT,SUBMODULE}.json
SS_ALL = 0.5647771875120086     # ||g_s||^2 over all 2,736
ZZ = 0.4071279988965891         # ||g_z||^2
RR_ALL = 0.599115204802637      # ||g_ref||^2
CTRL_ALL = 1.8524857025526955   # ||g_s+g_z||^2
SS_SO = 0.05425332382512837     # s_only class
RR_SO = 0.05425332382512837
CTRL_SO = 0.05425332382512857
RR_MX = 0.5448618809775087      # mixed class
COS_SZ = 0.9181951994170596
A_S_ONLY = 1.0223852405911826   # best scalar fitting a*g_s to g_ref, SONLY arm
A_ONE = 0.5596249552986813      # one shared scalar
A_S_PUB = 0.9109602082446762
B_Z_PUB = 0.14292915919297092
RESID_PUB = 0.11162126551703709
COS_PAIR = {"block47_worst": 0.982841, "block46_worst": 0.948834, "block24_best": 0.998099,
            "block0_best": 0.998232, "typical_at_depth": 0.995}


def main() -> int:
    sz = COS_SZ * math.sqrt(SS_ALL) * math.sqrt(ZZ)
    rs = A_S_ONLY * SS_ALL
    rz = A_ONE * CTRL_ALL - rs
    det = SS_ALL * ZZ - sz * sz
    a = (ZZ * rs - sz * rz) / det
    b = (-sz * rs + SS_ALL * rz) / det

    def resid(aa, bb):
        v = RR_ALL - 2 * (aa * rs + bb * rz) + aa * aa * SS_ALL + 2 * aa * bb * sz + bb * bb * ZZ
        return math.sqrt(max(v, 0.0)) / math.sqrt(RR_ALL)

    valley = []
    for bz in (0.038316, 0.10, B_Z_PUB, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 1.0):
        aa = (rs - bz * sz) / SS_ALL
        valley.append({"b_z": bz, "a_s_reoptimised": aa, "residual_frac_of_ref": resid(aa, bz)})

    tr, dt = 2.0, 1.0 - COS_SZ * COS_SZ
    l1 = (tr + math.sqrt(tr * tr - 4 * dt)) / 2
    l2 = (tr - math.sqrt(tr * tr - 4 * dt)) / 2

    ss_mx = SS_ALL - SS_SO
    ctrl_mx = CTRL_ALL - CTRL_SO
    sz_mx = (ctrl_mx - ss_mx - ZZ) / 2
    b_match = (-2 * sz_mx + math.sqrt(4 * sz_mx ** 2 - 4 * ZZ * (ss_mx - RR_MX))) / (2 * ZZ)

    out = {
        "what": __doc__.strip().splitlines()[0],
        "defect": "D242",
        "row_that_owns_it": "of3t-frameself",
        "device_involved": False,
        "why_no_aiclk": "CPU only, no Tenstorrent device is opened; nothing here is a timing",
        "THE_PARTITION_IS_ENTAILED": {
            "measured_premise": "TWOBASIS.json -- the z-only arm's ds_in is exactly 0.0, so the "
                                "pair track does not read the single track",
            "therefore": "s-branch parameters reach the loss ONLY through cot_s and pair-branch "
                         "parameters reach it through both, so a defect in cot_z and nothing "
                         "else produces exactly this partition at every depth with no mechanism "
                         "at the pair branch",
            "so_ps_dropout_row_layer_matching_it_is_worth_nothing":
                "every structure that distinguishes the two branches matches the partition. The "
                "row refuted dropout on its own merits, which was right; the point here is that "
                "the partition could not have nominated it in the first place.",
            "what_the_split_IS_worth": "a consistency check that passes -- 11 attn_pair_bias + 5 "
                                       "single_transition = 16 exact tensors per block, times 48, "
                                       "is exactly TWOBASIS's 768 s-only class.",
            "and_the_blockprobe_is_the_passs_real_result":
                "the reference gradient was the one quantity never read twice; a second "
                "instrument now puts it at 3.0392623414001263e-15 (block 47) and "
                "4.1031090433915236e-15 (block 46), at the two blocks where the replay is worst.",
        },
        "MY_OWN_HYPOTHESIS_REFUTED": {
            "what_I_expected": "that the two-scalar fit is collinear (cos 0.9181952 between the "
                               "regressors) and its b_z therefore meaningless, so that the "
                               "'7x too large' headline was a fitting artifact",
            "the_fit_IS_ill_conditioned": {"gram_eigenvalues": [l1, l2], "condition_number": l1 / l2},
            "but_the_valley_is_not_flat": valley,
            "verdict": f"REFUTED. b_z = 0.55, which is what a 1.8x scale error would mean, costs "
                       f"residual {resid((rs - 0.55 * sz) / SS_ALL, 0.55):.6f} against the "
                       f"minimum's {RESID_PUB:.6f} -- 55 % worse. The published b_z stands.",
            "reconstruction_check": {
                "a_s_reconstructed": a, "a_s_published": A_S_PUB,
                "b_z_reconstructed": b, "b_z_published": B_Z_PUB,
                "rel_agreement_a": abs(a - A_S_PUB) / abs(A_S_PUB),
                "rel_agreement_b": abs(b - B_Z_PUB) / abs(B_Z_PUB),
                "why_it_matters": "solving the 2x2 normal equations from banked scalars alone "
                                  "reproduces the row's fit to 2e-14, so this file's arithmetic "
                                  "is checked and so is the row's.",
            },
        },
        "THE_1_8x_AND_THE_7x_ARE_THE_SAME_STATEMENT": {
            "mixed_class_norms": {"ref": math.sqrt(RR_MX), "s_arm": math.sqrt(ss_mx),
                                  "z_arm": math.sqrt(ZZ)},
            "pooled_pair_branch_ratio": math.sqrt(ctrl_mx) / math.sqrt(RR_MX),
            "per_submodule_reading": "1.78 to 2.02 at block 47, 1.567 to 2.203 over all four "
                                     "probed blocks, tightening to 1.781-1.835 at block 0",
            "b_that_matches_the_NORM_on_the_mixed_class": b_match,
            "one_over_it": 1.0 / b_match,
            "read": "the correction applies only to the cot_z PART of a pair parameter's "
                    "gradient, and that part is a large fraction of the total, so a ~7x "
                    "overshoot on the part shows up as ~1.8x on the whole. Nothing is in "
                    "conflict.",
        },
        "WHAT_A_MECHANISM_NOW_HAS_TO_EXPLAIN": {
            "magnitude": "cot_z's contribution wants scaling by about 0.14 (least squares) to "
                         f"0.038 (norm match on the pair class), i.e. 7x to {1.0/b_match:.1f}x "
                         "too large",
            "direction": "and about ten degrees, not more. The pair-branch cos is 0.948834 to "
                         "0.998232 and 0.995 at depth, which is a 0.0999 perpendicular "
                         f"component -- the same size as the {RESID_PUB:.6f} residual floor "
                         "after the best two scalars.",
            "so_the_rows_phrase_pointing_elsewhere_overstates_it":
                "the irreducible residual is a ~10 % perpendicular component, not a different "
                "direction. A mechanism must produce a large magnitude error and a small angle "
                "on the z channel only, which is a much tighter target than the phrase suggests.",
            "cos_readings": COS_PAIR,
        },
        "AND_MY_PASS_387_DISCRIMINATOR_IS_STILL_UN_RUN_AND_IS_NOW_CHEAP": {
            "what_blockprobe_did": "torch.autograd.grad(LOSS, params) on the original graph -- "
                                   "loss-driven. That confirms the reference and is not the "
                                   "discriminator.",
            "what_is_still_owed": "torch.autograd.grad(outputs=(s_out, z_out), "
                                  "grad_outputs=(cot_s, cot_z), inputs=...) on the ORIGINAL "
                                  "graph -- cotangent-driven. Same graph, so the only remaining "
                                  "structural difference from the replay is which drive is used.",
            "and_it_now_costs_about_135_s": "blockprobe already prunes the graph to one block "
                                            "(measured 135.2 s against a 340.6 s forward) and "
                                            "capture_model_frame.py:653-656 already holds "
                                            "s_out_t and z_out_t for exactly this. It is one "
                                            "extra grad_outputs= argument.",
            "falsifier_unchanged": "dL/dz_in at 0.000848887340907281 means the replay's fresh "
                                   "forward is the defect; at 0.0014907294032500784 means the "
                                   "injection overcounts on the original graph too.",
        },
        "DOESNOT": "this names no mechanism and moves no published number. It removes a class of "
                   "dead ends, kills one hypothesis of my own, reconciles two readings that "
                   "looked contradictory, and states the target a mechanism has to hit.",
    }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
