#!/usr/bin/env python3
"""of3t-orchestrator pass 392: the GRADIENTS clause is graded on a frame the campaign withdrew, and DOESNOT is a frame behind that.

Two defects in the two places a reviewer actually reads, both found by checking the fields
against the live artifacts rather than against each other.

**1. The gate's GRADIENTS clause is graded on the disqualified frame and reports a withdrawn
number as its distance to go.** Its artifact is
`perf/of3t_modelframe/MODEL_FRAMEMATCHED_composed3660_n384.json`, the `of3t-modelframe` frame.
That frame **fails its own gating control**: an injected float64 trunk on its captured boundary
and cotangent must return `grads_f64_043.pt`'s `pairformer_stack` section to float64 round-off,
bar pre-registered at 1e-12 in commit 2520681ed, and it reads **0.7945281613194305** (D242).
`of3t-frameself`'s retraction sort lists the clause's **1.7814428090278143x** under CROSS-FRAME
AND THEREFORE WITHDRAWN. The clause still computes it:
0.27095922968432157 / 0.15210099830945006 = 1.7814428090278143.

**No verdict moves** -- the clause is NOT MET either way, and a FAIL is the direction that cannot
manufacture a false GO. But a reviewer reading `CHARTER_EVIDENCE.json` sees a precise failing
number, and the campaign's own ledger says that number is not a statement about our gradient.
**An unflattering error is still an error**, and this one is in the machine-readable file.

**2. `DOESNOT` is one frame behind the clause it describes, and its blocker has concluded.** It
quotes `0.517117 vs 0.152101` where the live artifact reads **0.27095922968432157**, and
`n_over_per_tensor_bar` 3314 where the live artifact reads **3312**. Worse than the numbers: it
says the 5.83 % *"awaits `of3t-modelframe`"*. **That row concluded.** It published a clause from
a frame whose gating control had not been run, the control then failed by twelve orders of
magnitude, and what the 5.83 % awaits today is D242's repair under `of3t-frameself`. "Awaits a
pending row" reads as routine work in flight; "sits on a frame that failed its own control" does
not, and the second is what is true.

The whole field is also framed on **D237**, which D241 and D242 have superseded -- D241 found the
clause divides two different experiments, D242 found the frame fails its own control.

Fixed the way R148 fixed COVERAGE and R154 fixed TRAJECTORY: by stating the artifact's status in
the gate spec's `why`, so it regenerates into `CHARTER_EVIDENCE.json` every compose and reaches
the reviewer. No check moved, no bar moved, no number invented, and the clause is NOT re-scored
-- `of3t-frameself` has the standing instruction not to, and re-scoring on a frame that still
fails its own control would be the same defect with a newer figure.

Zero card. Reads committed artifacts.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ART = "perf/of3t_modelframe/MODEL_FRAMEMATCHED_composed3660_n384.json"

STALE_IN_DOESNOT = {
    "stats.renorm_vs_UPSTREAM_BF16.mass_weighted_rel_l2": 0.517117,
    "stats.renorm_vs_FLOAT64.n_over_per_tensor_bar": 3314,
}
D242_CONTROL = 0.7945281613194305
D242_BAR = 1e-12


def main() -> int:
    with open(os.path.join(ROOT, ART)) as fh:
        a = json.load(fh)

    def g(path):
        o = a
        for k in path.split("."):
            o = o[k]
        return o

    live = {k: g(k) for k in STALE_IN_DOESNOT}
    ratio = (g("stats.renorm_vs_UPSTREAM_BF16.mass_weighted_rel_l2")
             / g("bars.A26_reachable_bar_vs_their_bf16"))

    out = {
        "what": __doc__.strip().splitlines()[0],
        "device_involved": False,
        "why_no_aiclk": "CPU only; this reads committed artifacts and computes no timing",
        "DEFECT_1_the_clause_is_graded_on_a_withdrawn_frame": {
            "clause": "GRADIENTS",
            "artifact": ART,
            "the_frame": "of3t-modelframe",
            "that_frame_fails_its_own_gating_control": {
                "what_the_control_requires": "an injected float64 trunk on the frame's own "
                                             "captured boundary and cotangent must return "
                                             "grads_f64_043.pt's pairformer_stack section to "
                                             "float64 round-off",
                "bar_preregistered_in": "commit 2520681ed",
                "bar": D242_BAR,
                "reads": D242_CONTROL,
                "orders_over": 11,
                "defect": "D242",
            },
            "the_clause_still_computes": {
                "numerator": g("stats.renorm_vs_UPSTREAM_BF16.mass_weighted_rel_l2"),
                "bar": g("bars.A26_reachable_bar_vs_their_bf16"),
                "ratio": ratio,
            },
            "and_that_ratio_is_listed_as_WITHDRAWN":
                "of3t-frameself's retraction sort puts the clause's 1.7814428090278143x under "
                "CROSS-FRAME AND THEREFORE WITHDRAWN, together with the trunk's 2.9702x, block "
                "47's 4.1613x and of3t-cotcoh's 0.1100",
            "no_verdict_moves": "the clause is NOT MET either way, and a FAIL cannot manufacture "
                                "a false GO -- which is why this survived. An unflattering error "
                                "is still an error, and this one is in the machine-readable file "
                                "a reviewer reads.",
            "what_is_NOT_done_here": "the clause is not re-scored and no number is invented. "
                                     "of3t-frameself has the standing instruction not to "
                                     "re-score, and re-scoring on a frame that still fails its "
                                     "own control would be the same defect with a newer figure.",
        },
        "DEFECT_2_DOESNOT_is_a_frame_behind_and_its_blocker_concluded": {
            "stale_numbers": [
                {"key": k, "DOESNOT_says": v, "artifact_reads": live[k]}
                for k, v in STALE_IN_DOESNOT.items()
            ],
            "the_worse_half": {
                "DOESNOT_says": "the 5.83 % 'awaits of3t-modelframe'",
                "but": "that row CONCLUDED. It published a clause from a frame whose gating "
                       "control had not been run; the control then failed by twelve orders of "
                       "magnitude.",
                "what_it_awaits_today": "D242's repair, owned by of3t-frameself",
                "why_the_wording_matters": "'awaits a pending row' reads as routine work in "
                                           "flight; 'sits on a frame that failed its own "
                                           "control' does not, and the second is what is true.",
            },
            "and_the_framing_is_two_defects_old": "the field is written around D237; D241 found "
                                                  "the clause divides two different experiments "
                                                  "and D242 found the frame fails its own "
                                                  "control.",
        },
        "THE_PATTERN_THIS_MAKES_THREE_OF": {
            "R148": "COVERAGE published MET beside prose naming three paths that do not fire",
            "R154": "TRAJECTORY published MET without the operating point five factors are "
                    "single-valued or inert at",
            "R157": "GRADIENTS published a precise FAIL on a frame the campaign had withdrawn",
            "the_common_shape": "a clause's CHECKS stay live because a script recomputes them, "
                                "while the PROSE beside them is written once and then rots. All "
                                "three were fixed the same way -- describe where the evidence "
                                "lives, or what the artifact's status is, instead of restating a "
                                "value the check already computes.",
            "and_the_direction_is_not_always_flattering": "R148 understated us, R157 overstates "
                                                          "the distance to go. Rot has no "
                                                          "preferred sign, so an audit cannot "
                                                          "skip the clauses that look bad.",
        },
        "DOESNOT": "this moves no verdict, no bar and no measurement. It records that the "
                   "GRADIENTS clause's artifact sits on a disqualified frame and that DOESNOT "
                   "quotes superseded values and a concluded blocker, and it fixes both where a "
                   "reviewer reads them.",
    }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
