#!/usr/bin/env python3
"""of3t-orchestrator pass 394: the clause's allowance is 1.4172x upstream's own bf16, and it was already pre-registered.

Pass 392 established that the GRADIENTS clause is graded on a frame the campaign withdrew, so
its 1.7814428090278143x is not a distance to go. That left the campaign with no distance to go
at all. This pass looked for one **in what is already banked** rather than deriving a new bar,
and `perf/of3t_modelframe/CLAUSE.json` has it: a five-level ladder, pre-registered, each level
a hypothesised trunk reading substituted into the pooled clause and re-evaluated.

    level                                    clause value            x the bar   clears
    a bit-exact float64 trunk                0.1026990533692057      0.675203    yes
    upstream's own bf16 floor here           0.12967067993359122     0.852530    yes
    the A26 section level                    0.1475459165605533      0.970052    yes
    projection from the in-frame multiple    0.19635234908555793     1.290934    no
    the frame mismatch alone                 0.47443757904241024     3.119227    no

**So "if D242 is repaired, does GRADIENTS pass?" already has a conditional answer, and it is
yes for any trunk at or better than its A26 section level.** The clause is satisfiable and has
been since the ladder was written; what is missing is one input, not a bar.

**The number worth stating, and it is computed inside ONE frame.** The trunk's own allowance is
`0.44608901561034203` and upstream's own bf16 floor for the trunk here is
`0.3147698293887927`, both trunk readings against float64 in the published frame's pooling. So
**our trunk may be up to 1.4172x upstream's own bf16 and the clause still clears.** That is a
like-for-like quotient in one frame and it is the cleanest statement of the target this campaign
has.

**What I then did NOT do, and the reason is the point.** The D242-immune estimate of our trunk's
actual multiple is `of3t-twoside`'s two-sided **1.7997765325758555**, measured on the MODEL
frame. Comparing it to the 1.4172 allowance is comparing an in-frame multiple against a
threshold-multiple from a DIFFERENT frame, which is exactly the form D214 and D218 bar and
exactly how this campaign published two wrong targets in two passes (R129, R130). It is offered
below as a PREDICTION carrying its frames and its falsifier, per A37, and **no bar, allowance or
done-check is derived from it.**

**The actionable consequence is one line.** The moment D242 is repaired, the single number to
compute is the trunk's rel_l2 against float64 on the repaired frame; then read the ladder. No
new bar is needed and none should be written — a bar re-derived after the frame moves is how
A37 got written.

Zero card. Reads committed artifacts.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CLAUSE = "perf/of3t_modelframe/CLAUSE.json"
TWOSIDED = 1.7997765325758555          # of3t-twoside, model frame, two-sided, D242-immune
TWOSIDED_SEPARATOR = 2.0               # pre-registered in commit 2520681ed before the arm ran


def main() -> int:
    with open(os.path.join(ROOT, CLAUSE)) as fh:
        c = json.load(fh)
    pub, pre = c["published"], c["preregistered"]
    allowance = pre["trunk_allowance_for_the_clause_to_clear"]
    floor = pub["trunk"]["upstreams_own_floor_here_vs_float64"]

    out = {
        "what": __doc__.strip().splitlines()[0],
        "device_involved": False,
        "why_no_aiclk": "CPU only; reads committed artifacts and computes no timing",
        "source": CLAUSE,
        "THE_LADDER_IS_ALREADY_PREREGISTERED": {
            "levels": {k: {"clause_value": v["clause_value"], "x_bar": v["x_bar"],
                           "clears": v["clears"]}
                       for k, v in pre["levels"].items()},
            "so": "the conditional answer to 'if D242 is repaired, does GRADIENTS pass?' exists "
                  "and is yes for any trunk at or better than its A26 section level. The clause "
                  "is satisfiable; what is missing is one input, not a bar.",
        },
        "THE_ALLOWANCE_AS_A_MULTIPLE_OF_UPSTREAMS_OWN_BF16": {
            "trunk_allowance_for_the_clause_to_clear": allowance,
            "upstreams_own_bf16_floor_for_the_trunk_here_vs_float64": floor,
            "allowance_as_a_multiple_of_that_floor": allowance / floor,
            "both_are": "trunk readings against float64 in the published frame's pooling, so "
                        "this is a like-for-like quotient inside ONE frame",
            "read": "our trunk may be up to 1.4172x upstream's own bf16 and the clause still "
                    "clears. That is the cleanest statement of the target this campaign has.",
        },
        "A_PREDICTION_AND_NOT_A_TARGET": {
            "a37": "a projection may be published as a PREDICTION, never as a TARGET; it carries "
                   "its own falsifier and its frame; and no bar, allowance or done-check may be "
                   "derived from one",
            "the_prediction": "if our trunk's multiple over upstream's own bf16 transfers across "
                              "the frame repair, the clause misses",
            "our_multiple_D242_immune": TWOSIDED,
            "its_frame": "the of3t-modelframe frame, two-sided, both legs on one injected "
                         "cotangent, separator 2.0 pre-registered in commit 2520681ed before "
                         "the arm ran",
            "the_allowance_multiple": allowance / floor,
            "its_frame_": "the published frame's pooling",
            "why_the_comparison_is_NOT_made_as_a_verdict":
                "it sets an in-frame multiple against a threshold-multiple from a different "
                "frame, which is the form D214 and D218 bar and how R129 and R130 were "
                "published wrong in two consecutive passes.",
            "shortfall_IF_it_transferred": TWOSIDED / (allowance / floor),
            "falsifier": "measure the trunk's rel_l2 against float64 on the REPAIRED frame and "
                         "read the ladder. That settles it with no new bar and no projection.",
        },
        "FOR_SCALE_the_frame_effect_on_this_very_multiple": {
            "published_cross_frame_multiple_of_upstreams_own_bf16":
                pub["trunk"]["multiple_of_upstreams_own_bf16"],
            "the_in_frame_two_sided_multiple": TWOSIDED,
            "ratio": pub["trunk"]["multiple_of_upstreams_own_bf16"] / TWOSIDED,
            "read": "the frame is worth 3.81x on the one quantity the campaign most wants to "
                    "know, which is why no cross-frame comparison here is worth making and why "
                    "D242 is the critical path rather than a bookkeeping item.",
        },
        "THE_ONE_LINE_THAT_FOLLOWS_THE_REPAIR":
            "compute the trunk's rel_l2 against float64 on the repaired frame, then read the "
            "ladder above. Do not write a new bar: a bar re-derived after the frame moves is "
            "how A37 came to be written.",
        "DOESNOT": "this measures nothing new and moves no verdict. It assembles the campaign's "
                   "own pre-registered ladder into the target statement it never made, and "
                   "records one comparison deliberately NOT made and why.",
    }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
