#!/usr/bin/env python3
"""of3t-orchestrator pass 390: D242's two remaining candidates are ONE mechanism, and my own R146 is blind to it.

`of3t-frameself` reduced D242 to a single block reproducible in **13.2 s** and named two
remaining candidates, *"both differences between running the block inside the real graph and
running it standalone, neither of which changes the forward"*:

  1. the checkpoint recomputation -- `checkpoint_blocks`, `blocks_per_ckpt` 1,
     `use_reentrant` False, so the block is re-executed at backward time
  2. the cast policy context, live during the real forward and gone by the time the
     recomputation runs

**They are not alternatives. Composed, they are a single mechanism with exactly D242's
signature.** Under `use_reentrant=False`, the block's OUTPUTS come from the first forward, which
runs inside `bm.cast_policy(...)`; the graph that the backward differentiates is REBUILT by the
recomputation at backward time. `torch.utils.checkpoint` saves and restores torch's own RNG and
autocast state across that recomputation -- it knows nothing about a user-written context
manager. So a policy that is live for the first forward and absent for the recomputation gives:

    outputs        bit-exact, because they came from the in-policy forward   <- observed
    backward       built from a DIFFERENT graph than the one that produced them
    forward check  passes, and structurally cannot see this                  <- observed

That is the premise pass 387 named -- a bit-exact forward does not imply an identical backward
graph -- with a named mechanism attached, and it is the first candidate that predicts every
observation rather than retrodicting the partition.

**AND MY OWN R146 CANNOT SEE IT.** `perf/of3t_orchestrator/frameself/ckpt_break.py` contains no
`policy`, no `autocast` and no `cast_policy` -- zero occurrences. It exercised `checkpoint_blocks`
with no context manager to lose across the recomputation, so its 228-of-228 bit-identical result
is true of candidate 1 IN ISOLATION and says nothing about candidate 1 COMPOSED WITH candidate 2.
R146 stands as written and does not cover this; I am recording that rather than letting a
successor read it as an exclusion.

**THE DIRECTION MATTERS AND NOBODY HAS STATED IT.** If the recomputation runs outside the policy,
then it is the REFERENCE's backward that was built through the odd graph and the standalone
replay -- which the row runs inside `policy2` -- that is differentiated consistently with its own
forward. The replay is **1.679x LARGER**. So this mechanism makes `grads_f64_043.pt` the
attenuated side, not ours. **Nobody should "repair" the replay to match the reference until the
2x2 below has said which side is which**, and if the reference is the attenuated one then every
ratio this campaign has graded against it inherits that, which would be the most consequential
finding of the campaign.

Zero card, zero model run. The 2x2 belongs to `of3t-frameself` and its `ONEBLOCK_<tag>.pt`.
"""
from __future__ import annotations

import json
import sys

OBS = {
    "standalone_in_policy_no_checkpoint": {
        "measured": True,
        "param_ratio": 1.6790846854374906,
        "param_rel_l2": 0.7849281738435927,
        "cos": 0.9538569348402676,
        "din_z_mine": 0.0005083278038663214,
        "din_z_real": 0.00030085507552087443,
        "bit_identical_params": 16,
        "note": "the row's current arm -- forward bit-exact, din_s bit-identical",
    },
}

ARMS = [
    {"arm": "A", "policy": "in", "checkpoint": "off",
     "status": "ALREADY MEASURED", "reads": 1.6790846854374906},
    {"arm": "B", "policy": "in", "checkpoint": "on",
     "status": "owed", "isolates": "candidate 1 alone, WITH the policy present for both the "
                                   "forward and the recomputation"},
    {"arm": "C", "policy": "out", "checkpoint": "off",
     "status": "owed", "isolates": "candidate 2 alone, no recomputation involved"},
    {"arm": "D", "policy": "out", "checkpoint": "on",
     "status": "owed", "isolates": "nothing on its own; it is the control that says the two "
                                   "single-factor arms were not each half of one effect"},
]

READINGS = {
    "B alone reaches 1.0": "candidate 1 is the cause after all and R146 missed it for a reason "
                           "other than the policy -- check R146's block against the real one",
    "C alone reaches 1.0": "candidate 2 is the cause and it needs no recomputation, which means "
                           "the policy changes the FORWARD GRAPH's saved tensors directly",
    "neither alone, A and D bracket it":
        "THE INTERACTION, which is the predicted answer: the outputs come from the in-policy "
        "forward and the backward from an out-of-policy recomputation. Neither factor alone "
        "reproduces it and a one-at-a-time sweep would have cleared both.",
    "none of them moves 1.679": "the mechanism is elsewhere and this whole line is dead; the "
                                "cost of finding that out is four runs of 13.2 s",
}


def main() -> int:
    out = {
        "what": __doc__.strip().splitlines()[0],
        "defect": "D242",
        "row_that_owns_the_arms": "of3t-frameself",
        "device_involved": False,
        "why_no_aiclk": "CPU only, no Tenstorrent device is opened; nothing here is a timing",
        "THE_TWO_CANDIDATES_ARE_ONE_MECHANISM": {
            "composed": "under use_reentrant=False the OUTPUTS come from the first forward, "
                        "which is inside bm.cast_policy(...), while the graph the backward "
                        "differentiates is REBUILT by the recomputation at backward time. "
                        "torch.utils.checkpoint saves and restores torch's own RNG and autocast "
                        "state; it knows nothing about a user-written context manager.",
            "and_it_predicts_every_observation": [
                "forward bit-exact, because the outputs came from the in-policy pass",
                "the backward differs, because it is built from a different graph",
                "a forward check cannot see it, which is why every forward control passed",
            ],
            "why_this_one_is_different": "it PREDICTS rather than retrodicts. R153 closed the "
                                         "class of candidates that merely match the s/pair "
                                         "partition; this one is derived from the observation "
                                         "that the forward is bit-exact and the backward is not.",
            "the_row_s_own_supporting_counts": "822 autocast contexts entered with 0 actually "
                                               "enabled, and 240 Tensor.float calls that changed "
                                               "dtype, 2 of them FROM float64 -- a .float() on a "
                                               "float64 tensor in a float64 reference run is "
                                               "worth locating whatever the 2x2 says",
        },
        "MY_OWN_R146_IS_BLIND_TO_IT": {
            "what_R146_said": "checkpoint_blocks is EXACT -- 228 of 228 parameter gradients "
                              "bit-identical to a bare loop on upstream's real PairFormerBlock "
                              "at use_reentrant None, True and False",
            "what_it_did_not_have": "a cast policy. perf/of3t_orchestrator/frameself/"
                                    "ckpt_break.py contains zero occurrences of policy, autocast "
                                    "or cast_policy, so there was no context manager to lose "
                                    "across the recomputation.",
            "so": "R146 is true of candidate 1 IN ISOLATION and says nothing about candidate 1 "
                  "composed with candidate 2. Recorded here so a successor does not read it as "
                  "an exclusion it never was.",
        },
        "THE_DIRECTION_NOBODY_HAS_STATED": {
            "if_the_recomputation_runs_outside_the_policy":
                "then the REFERENCE's backward was built through the odd graph and the "
                "standalone replay -- which runs inside policy2 -- is the one differentiated "
                "consistently with its own forward.",
            "and_the_replay_is_the_LARGER_side": 1.6790846854374906,
            "so": "this mechanism makes grads_f64_043.pt the ATTENUATED side, not ours.",
            "the_instruction_that_follows": "do not 'repair' the replay to match the reference "
                                            "until the 2x2 has said which side is which. If the "
                                            "reference is attenuated, every ratio graded against "
                                            "it inherits that.",
            "what_does_NOT_follow": "the reference is still internally consistent -- blockprobe "
                                    "confirms it at 3.0392623414001263e-15 with a second "
                                    "instrument in the same process. Internal consistency is "
                                    "not correctness, and the 2x2 is what separates them.",
        },
        "THE_2x2": {
            "cost": "four runs at about 13.2 s each on the row's own ONEBLOCK_<tag>.pt -- under "
                    "a minute, against the 340 s forward plus 2310 s backward it replaces",
            "arms": ARMS,
            "preregistered_readings": READINGS,
            "why_a_2x2_and_not_two_tests": "a one-at-a-time sweep clears both factors when the "
                                           "effect is their interaction, and the interaction is "
                                           "the predicted answer here. Arm D is the control that "
                                           "makes the other three interpretable.",
        },
        "observations_this_rests_on": OBS,
        "DOESNOT": "this runs nothing and moves no number. It says the row's two candidates "
                   "compose into one mechanism that predicts every observation, records that my "
                   "own R146 cannot exclude it, states which side the mechanism would make "
                   "wrong, and specifies the four arms that decide it.",
    }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
