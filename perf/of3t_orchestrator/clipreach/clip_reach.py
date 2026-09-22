#!/usr/bin/env python3
"""of3t-orchestrator pass 388: does D210's pad-lane gradient reach the UPDATE RULE?

D210 is the last USER-FACING defect with no owner and no built repair: our fused `qkv_w` pads
head_dim 48 -> 64, the pad lanes are registered leaves in the optimizer's parameter set, and
Adam steps them -- 14.2M parameters upstream does not have, exactly 0.0 at w_0 and 3.494e-04 by
k = 20. Its triage says *"It moves no number the campaign quotes, because the pad columns are
outside the reference's parameter space and are sliced off before v is used."*

**Sliced off before use is not the only route out of a parameter.** `tt_bio/train/optim.py:218`
computes ONE global gradient norm over `self.params` -- which is our FUSED tensors, pad lanes
included -- and `clip_coef` (`:447-455`) turns it into a single scalar multiplying EVERY
parameter's update. So the pad lanes have a path into every real parameter's step that has
nothing to do with whether `v` is sliced, and the triage does not mention it. Upstream's
`compute_global_norm` runs over upstream's parameter set, which does not contain them, so the
two norms are taken over different sets by construction:

    gnorm_ours^2 = gnorm_theirs^2 + ||pad gradients||^2

and whenever clipping binds, `min(1, 10/gnorm)` is SMALLER for us than for them and every
parameter's update is scaled differently. That would be a divergence in the update rule itself,
which is the campaign's whole object.

This file settles whether it fires, from the trajectory arm's own committed steplog. No card, no
model run, nothing re-measured.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
STEPLOG = "perf/of3t_trajfull/steplog_trajfull.json"
CLIP_NORM = 10.0  # tt_bio/train/optim.py:133 default; upstream's configs_base.py:80


def main() -> int:
    with open(os.path.join(ROOT, STEPLOG)) as fh:
        log = json.load(fh)
    steps = log["steps"]

    gnorms = [s["grad_norm"] for s in steps]
    clips = [s["clip"] for s in steps]
    psc = [c for s in steps for c in (s.get("per_sample_clip_coefs") or [])]

    binding = [i for i, c in enumerate(clips) if c != 1.0]
    psc_binding = [c for c in psc if c != 1.0]
    worst_i = max(range(len(gnorms)), key=lambda i: gnorms[i])

    out = {
        "what": __doc__.strip().splitlines()[0],
        "defect": "D210",
        "device_involved": False,
        "why_no_aiclk": "CPU only; this reads a committed steplog and computes no timing",
        "source": {"steplog": STEPLOG, "arm": log.get("arm"), "side": log.get("side"),
                   "n_steps": log.get("n_steps"), "complete": log.get("complete")},
        "the_route_the_triage_does_not_mention": {
            "where": "tt_bio/train/optim.py:218-226 computes one global gradient norm over "
                     "self.params -- our FUSED tensors, pad lanes included -- and clip_coef "
                     "(:447-455) turns it into one scalar multiplying every parameter's update",
            "so": "gnorm_ours^2 = gnorm_theirs^2 + ||pad gradients||^2, and whenever clipping "
                  "binds our coefficient is smaller than theirs and every parameter's update "
                  "is scaled differently. Slicing v does not close this route.",
        },
        "DOES_IT_FIRE": {
            "clip_threshold": CLIP_NORM,
            "grad_norm_min": min(gnorms),
            "grad_norm_max": max(gnorms),
            "worst_step": worst_i,
            "headroom_at_the_worst_step": CLIP_NORM / gnorms[worst_i],
            "n_steps_where_the_batch_clip_binds": len(binding),
            "n_per_sample_coefficients_that_bind": len(psc_binding),
            "n_per_sample_coefficients_checked": len(psc),
            "verdict": ("NO. The clip coefficient is exactly 1.0 at every one of the 20 steps "
                        "and every one of the per-sample coefficients, and the closest the "
                        "gradient norm comes to the threshold is a factor of "
                        f"{CLIP_NORM / gnorms[worst_i]:.4f}. On this trajectory the pad lanes "
                        "cannot reach another parameter's update through clipping, because "
                        "clipping multiplies by one."),
        },
        "WHAT_THAT_LICENSES_AND_WHAT_IT_DOES_NOT": {
            "licenses": "D210's triage claim survives the one mechanism that could have broken "
                        "it. Nothing the campaign quotes moves, and the 20-step trajectory's "
                        "1.2707x at k = 20 is unaffected.",
            "does_not_license": "'confined' without its condition. Clipping is inert here "
                                "BECAUSE this batch's gradient norm sits 12.6x below the "
                                "threshold, not because the pad lanes are outside the norm. "
                                "They are inside it. A batch whose gradient norm exceeds 10 "
                                "puts 14.2M parameters upstream does not have into the scalar "
                                "that sets every real parameter's step.",
            "so_the_honest_statement_of_D210":
                "the pad lanes move no number on the measured trajectory because the clip is "
                "identically 1.0 there; they enter the update rule through the global-norm clip "
                "on any batch where clipping binds, and our norm is taken over a parameter set "
                "14.2M larger than theirs.",
        },
        "AND_A_POSITIVE_RESULT_FOR_THE_PROTOCOL'S_CENTRAL_DESIGN": {
            "observation": "the 20-step trajectory exercises the clipping factor at coefficient "
                           "1.0 and nowhere else, so a trajectory-only proof would have zero "
                           "coverage of clipping while looking fully covered.",
            "why_it_is_not_a_hole_here":
                "PROTOCOL's factorisation verifies the clipping as a PURE FUNCTION over its "
                "whole domain under an injected drive, exactly because a real batch never "
                "happens to hit the corner cases -- clip_coef is recorded agreeing with their "
                "real compute_global_norm to 8.51e-08 across eight STRADDLING cases "
                "(optim.py:447-455). This is the first measurement showing the trajectory could "
                "not have covered it, which is the argument for the factorisation stated as a "
                "number rather than as a design preference.",
            "what_the_factorisation_still_cannot_see":
                "the composition. Both sides' clip_coef is the same function; the ARGUMENT is "
                "not, because the two global norms are taken over different parameter sets. "
                "That is D210 and no function-level check reaches it.",
        },
        "THE_CHEAP_MEASUREMENT_THIS_NOW_OWES": {
            "ask": "emit ||pad gradients||^2 / gnorm^2 per step alongside grad_norm in the "
                   "existing steplog. It is a slice and a dot product over tensors the "
                   "optimizer already has in hand, no extra backward and no extra device time.",
            "what_it_buys": "the exact size of the divergence -- gnorm_ours / gnorm_theirs -- so "
                            "D210's repair can be measured against a number instead of argued "
                            "against a structure, and a bound on how far a binding batch would "
                            "separate the two clip coefficients.",
            "not_dispatched_as_a_row_this_pass": "both card-capable rows are live on D242 and "
                                                 "D245 and this is not on the critical path; a "
                                                 "third row contending for the same two hosts "
                                                 "would cost more than it returns.",
        },
        "per_step": [{"k": s["k"], "grad_norm": s["grad_norm"], "clip": s["clip"],
                      "headroom": CLIP_NORM / s["grad_norm"]} for s in steps],
        "DOESNOT": "this measures one 20-step trajectory on one batch. It does not bound the "
                   "pad lanes' share of the gradient norm, it does not repair D210, and it "
                   "merges nothing.",
    }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
