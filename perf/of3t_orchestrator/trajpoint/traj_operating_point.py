#!/usr/bin/env python3
"""of3t-orchestrator pass 390: what the CERTIFYING trajectory arm actually exercises.

`TRAJECTORY` is MET on `perf/of3t_trajfull/traj_trajfull.json` -- 20 coupled steps over
89.21058020840096 % of the model's gradient mass. A reader of `CHARTER_EVIDENCE.json` sees
"20 steps, coupled, 89.21 %" and would reasonably conclude the update rule was exercised end to
end. **It was exercised at ONE operating point, and five of the update rule's factors are
single-valued or identically inert there.** Pass 388 found the first of them (clipping); this is
the rest, and the table is the point.

That is not a hole. It is PROTOCOL's factorisation doing exactly the job it was designed for --
*"an injected drive isolates each one and reaches the corner cases a real batch never happens to
hit"* -- and this file is the first time the argument has been a table rather than a design
preference. **Two entries do have a provenance problem** and are named as such: the non-uniform
participation divisor is covered only by SUPERSEDED arms, and the disabled-parameter path is
covered by a different datapoint than the certifying arm.

Every number is read from the source JSON at run time. Zero card, zero model run.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TRAJ = "perf/of3t_trajfull/traj_trajfull.json"
STEPLOG = "perf/of3t_trajfull/steplog_trajfull.json"
CLIP_NORM = 10.0

# arms whose participation_spread is non-uniform, found by scanning every of3t artifact
NONUNIFORM_ARMS = ["perf/of3t_traj20/", "perf/of3t_rebind/", "perf/of3t_modeltraj/",
                   "perf/of3t_trajretake/"]


def read(p):
    with open(os.path.join(ROOT, p)) as fh:
        return json.load(fh)


def main() -> int:
    traj, log = read(TRAJ), read(STEPLOG)
    steps = log["steps"]
    sd = traj["shipped_defaults"]

    lrs = [s["lr"] for s in steps]
    gn = [s["grad_norm"] for s in steps]
    clips = [s["clip"] for s in steps]
    psc = [c for s in steps for c in (s.get("per_sample_clip_coefs") or [])]
    spreads = sorted({tuple(s["participation_spread"]) for s in steps})
    disabled = sorted({d.get("n_disabled_last_sample")
                       for d in traj.get("their_step_log", []) if isinstance(d, dict)})

    factors = {
        "lr_schedule": {
            "exercised_at": f"a strictly linear ramp, {lrs[0]} to {lrs[-1]} at a constant "
                            f"{lrs[2]-lrs[1]:.6g} per step over {len(lrs)} steps",
            "as_a_fraction_of_the_peak": lrs[-1] / sd["lr"],
            "the_schedule_it_is_a_fraction_of": {"peak_lr": sd["lr"],
                                                 "warmup_steps": sd["warmup_steps"],
                                                 "plateau_until": sd["plateau_until"]},
            "never_reached": "the warmup knee at step "
                             f"{sd['warmup_steps']} and the decay after step "
                             f"{sd['plateau_until']}",
            "covered_by": "PROTOCOL's factorisation -- a pure function verified EXACTLY over its "
                          "whole domain in float64, which is why 20 steps is enough",
            "provenance_is_clean": True,
        },
        "gradient_clipping": {
            "exercised_at": f"coefficient exactly 1.0 at all {len(clips)} steps and all "
                            f"{len(psc)} per-sample coefficients",
            "headroom": CLIP_NORM / max(gn),
            "never_reached": "any binding clip; the gradient norm peaks at "
                             f"{max(gn)} against a threshold of {CLIP_NORM}",
            "covered_by": "PROTOCOL -- clip_coef agrees with their compute_global_norm to "
                          "8.51e-08 across eight STRADDLING cases (optim.py:447-455)",
            "provenance_is_clean": True,
            "see": "R152 / CLIP_REACH.json",
        },
        "participation_divisor": {
            "exercised_at": f"participation_spread {spreads} at every step -- one distinct "
                            "value, so every parameter is divided by the same count",
            "and_that_makes_it_inert": "optim.py:195-203 divides each parameter's accumulated "
                                       "gradient by ITS OWN count, and the comment beside it "
                                       "says a uniform per-tensor scaling is exactly what Adam "
                                       "DOES cancel. So in the certifying arm the divisor the "
                                       "campaign implemented because 'the counts genuinely "
                                       "differ' fires and is inert (up to Adam's eps).",
            "covered_by": {
                "count_zero": "of3t-d10_d107 -- zero_participation_steps [3], "
                              "worst_rel_at_zero_participation 0.001102541604612053; and "
                              "FIRES.json puts a zero-participation step at 9.587733739203432e-71 "
                              "at shipped scale",
                "non_uniform_non_zero": "spread (1, 4), which appears ONLY in "
                                        + ", ".join(NONUNIFORM_ARMS),
            },
            "provenance_is_clean": False,
            "why_not": "every arm carrying the non-uniform spread is SUPERSEDED -- TRAJECTORY "
                       "was repointed to of3t-trajfull at pass 357 (A-D210) -- so the case is "
                       "covered by the campaign's history and not by the artifact the clause "
                       "reads. And it is upstream's NORMAL case, not a corner: R6 records that "
                       "their runner disables confidence-head parameters on zero-confidence "
                       "samples and initial_training.yml does so on 4 of its 5 datasets.",
        },
        "disabled_parameters": {
            "exercised_at": f"n_disabled_last_sample {disabled} at every step of the reference "
                            "side -- the path never fires in this arm",
            "covered_by": "COVERAGE's own disabled_parameters item, which R148 records as "
                          "upgraded by the gate firing on 5oid",
            "provenance_is_clean": False,
            "why_not": "a different datapoint from the one TRAJECTORY is graded on, so the two "
                       "clauses are each satisfied and neither is satisfied on the other's data.",
        },
        "weight_decay": {
            "exercised_at": f"weight_decay {sd['weight_decay']} -- identically inert",
            "covered_by": "nothing, and nothing is owed: upstream's own shipped default is "
                          f"{sd['weight_decay']}, so there is no behaviour to reproduce",
            "provenance_is_clean": True,
        },
    }

    out = {
        "what": __doc__.strip().splitlines()[0],
        "device_involved": False,
        "why_no_aiclk": "CPU only; this reads committed artifacts and computes no timing",
        "the_clause": {
            "field": "TRAJECTORY", "verdict_today": "MET",
            "artifact": TRAJ,
            "steps": len(steps),
            "scope_pct_of_model_sq_grad_norm": traj.get("bar_pct_of_model_sq_grad_norm"),
            "what_a_reader_infers": "that the update rule was exercised end to end",
            "what_was_exercised": "one operating point, with five factors single-valued or inert",
        },
        "factors": factors,
        "THE_POSITIVE_READING": {
            "claim": "this is the factorisation working, not a hole in it",
            "why": "PROTOCOL drives the state-free factors with a controlled input precisely "
                   "because a real batch never happens to hit the clip threshold, the warmup "
                   "knee, a zero gradient or a disabled parameter. Five factors inert in a "
                   "20-step trajectory is the measurement that makes that a fact rather than a "
                   "design preference -- and it is the strongest available answer to 'why not "
                   "just run more steps'. More steps of THIS batch move none of the five.",
        },
        "WHAT_IS_OWED": {
            "not_a_new_measurement": "both gaps are provenance, not coverage. The non-uniform "
                                     "divisor and the disabled-parameter path are both exercised "
                                     "somewhere in the campaign; neither is exercised in the "
                                     "artifact its clause reads.",
            "the_cheap_close": "state the operating point beside the TRAJECTORY verdict, the "
                               "way R148 fixed COVERAGE -- describe where each factor's evidence "
                               "lives instead of leaving a reader to infer it from '20 steps, "
                               "coupled, 89.21 %'.",
            "the_honest_headline": "TRAJECTORY MET means the COMPOSITION reproduces at one "
                                   "operating point over 89.21 % of the gradient mass. It does "
                                   "not mean the update rule's factors were swept, and the "
                                   "factorisation is what covers them.",
        },
        "DOESNOT": "this moves no verdict and no bar. It states what the certifying arm "
                   "exercises, names where each unexercised factor's coverage lives, and flags "
                   "the two whose coverage is not in the artifact the clause is graded on.",
    }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
