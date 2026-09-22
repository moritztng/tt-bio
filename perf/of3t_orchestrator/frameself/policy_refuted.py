#!/usr/bin/env python3
"""of3t-orchestrator pass 391: I RETRACT the cast-policy mechanism I proposed last pass. Both halves are inert inside the block.

Pass 390 (R155) proposed that D242 is the checkpoint recomputation composed with the cast policy:
the block's OUTPUTS come from the first forward, which runs inside `bm.cast_policy(...)`, while
the graph the backward differentiates is rebuilt by the recomputation at backward time, outside
it. I put it in `of3t-frameself`'s brief as Amendment 6 and called it "the first D242 candidate
that predicts rather than retrodicts".

**The scoping half is right and the acting half is wrong.** The policy really does exit before
the backward -- `bundle_min.py:650-654` is `forward_loss(..., cast_ctx=policy)` and then a bare
`loss.backward()`, and `forward_loss:457` wraps only `model(private)` and `loss_fn(...)`, so
`torch.Tensor.float` and `torch.amp.autocast` are restored by the time the recomputation runs.
But a context that is absent can only matter if it was DOING something, and inside the
checkpointed pairformer block it was not. Both of its halves are inert there:

  `.float()`   `cast_policy`'s special case is `if mode == "removed" and dtype is float64:
               return self_t` -- a no-op inside, a downcast outside. It has **ZERO call sites in
               the entire `openfold3/core/model/` subtree**, checked by grep over every `.py`.
               The 240 calls the capture counted are elsewhere, and their sources are
               {"float64": 2, "int32": 84, "int64": 154} -- 238 are integer featurisation casts
               that the special case does not touch and that behave identically inside and
               outside, and the 2 float64 ones are outside the per-block checkpoint.
  autocast     five sites under `core/model/primitives/` (linear.py:123,136,
               attention.py:115,151, normalization.py:65) and every one names
               `device_type="cuda"` on a CPU box. Not an argument -- MEASURED: the capture's own
               report is `n_autocast_contexts_entered: 822`,
               `n_autocast_contexts_torch_actually_enabled: 0`.

So the reference run is `mode: "removed"`, `device: "cpu"`, and inside `PairFormerBlock` the
policy has nothing to patch. **The mechanism cannot fire and Amendment 6's arms C and D would
have cost the row a pass to learn it.**

**What that leaves, and it is a harder place than last pass.** The row's candidate 1, the
checkpoint recomputation on its own, is covered: R146 ran upstream's REAL `PairFormerBlock`
through upstream's own `checkpoint_blocks` and got 228 of 228 parameter gradients bit-identical
to a bare loop at `use_reentrant` None, True and False, with independent random cotangents on
both outputs. Pass 390 said R146 was blind because it had no policy to lose; with the policy now
inert, that objection dies too and R146 covers what is left of candidate 1. **Both remaining
candidates are dead and the hypothesis space is empty again.**

**Which is the point: stop nominating candidates.** The row now has a 13.2 s one-block
reproducer and a portable `ONEBLOCK_<tag>.pt`. At that cost the honest move is not another
mechanism, it is a DIRECT DIFF of the two graphs -- specified below, candidate-free, and it
returns the site rather than a verdict on a guess.

Zero card, zero model run. Code reads and the capture's own banked counters.
"""
from __future__ import annotations

import json
import sys

BANKED = {
    "cast_policy_report_from_CAPTURE_selftest_n384": {
        "mode": "removed", "device": "cpu",
        "n_autocast_contexts_entered": 822,
        "n_autocast_contexts_torch_actually_enabled": 0,
        "n_tensor_float_calls": 240,
        "n_tensor_float_calls_that_changed_dtype": 240,
        "tensor_float_source_dtypes": {"float64": 2, "int32": 84, "int64": 154},
    },
    "float_call_sites_in_openfold3_core_model": 0,
    "autocast_sites_in_core_model_primitives": 5,
    "all_five_name_device_type": "cuda",
    "oneblock_ratio": 1.6790846854374906,
    "oneblock_seconds": 13.2,
}


def main() -> int:
    out = {
        "what": __doc__.strip().splitlines()[0],
        "defect": "D242",
        "retracts": "R155's mechanism and Amendment 6's arms C and D",
        "does_not_retract": "R155's direction warning, which is about what a repair would mean "
                            "and stands on its own; and R153, which closed the retrodiction "
                            "class. Amendment 6's arm B is still worth its 13.2 s as a "
                            "confirmation of R146 on the real frame.",
        "device_involved": False,
        "why_no_aiclk": "CPU only, no Tenstorrent device is opened; nothing here is a timing",
        "THE_SCOPING_HALF_IS_RIGHT": {
            "bundle_min.py:650-654": "forward_loss(..., cast_ctx=policy) and then a bare "
                                     "loss.backward()",
            "forward_loss:457": "with ac, ctx: wraps model(private) and loss_fn(...) only",
            "so": "torch.Tensor.float and torch.amp.autocast ARE restored before the "
                  "checkpoint recomputation runs. That part of the reasoning survives.",
        },
        "THE_ACTING_HALF_IS_WRONG": {
            "float_patch": {
                "what_it_does": "cast_policy.__enter__ patches torch.Tensor.float; in mode "
                                "'removed' a float64 source returns unchanged, so it is a no-op "
                                "inside the policy and a downcast outside it",
                "call_sites_inside_the_block": 0,
                "checked_by": "grep -rn '\\.float()' openfold3/core/model/ --include=*.py over "
                              "the whole subtree",
                "and_the_240_counted_calls": "are elsewhere, sources {float64: 2, int32: 84, "
                                             "int64: 154}. 238 are integer featurisation casts "
                                             "the special case does not touch; the 2 float64 "
                                             "ones sit outside the per-block checkpoint.",
            },
            "autocast_patch": {
                "sites_inside_the_block": 5,
                "where": "linear.py:123,136, attention.py:115,151, normalization.py:65",
                "every_one_names": "device_type='cuda', on a CPU box",
                "and_this_is_measured_not_argued": "the capture's own report reads "
                                                   "n_autocast_contexts_entered 822 and "
                                                   "n_autocast_contexts_torch_actually_enabled 0",
            },
            "verdict": "inside PairFormerBlock the policy has nothing to patch, so its absence "
                       "during the recomputation cannot change the recomputed graph.",
        },
        "AND_R146_IS_NO_LONGER_BLIND": {
            "pass_390_said": "R146 could not exclude the checkpoint candidate because "
                             "ckpt_break.py had no policy to lose across the recomputation",
            "with_the_policy_inert_that_objection_dies": True,
            "so_R146_covers_what_is_left": "228 of 228 parameter gradients bit-identical to a "
                                           "bare loop on upstream's REAL PairFormerBlock at "
                                           "use_reentrant None, True and False, with "
                                           "independent random cotangents on both outputs",
            "net": "both remaining candidates are dead and the hypothesis space is empty again.",
        },
        "THE_REPLACEMENT_IS_AN_INSTRUMENT_NOT_A_CANDIDATE": {
            "why": "at 13.2 s per run with a portable ONEBLOCK artifact, guessing is more "
                   "expensive than looking. Three passes of candidates have each died; the "
                   "observation that survives all of them is that the forward is bit-exact and "
                   "the backward is not, which means the two GRAPHS differ. Diff the graphs.",
            "the_arm": "register a forward hook on every submodule of the block. Record every "
                       "output during the real first forward, record again during the backward's "
                       "recomputation (use_reentrant=False re-executes the same modules, so the "
                       "same hooks fire), and diff the two recordings tensor by tensor.",
            "pre_registered_two_sided": {
                "some_intermediate_differs": "that submodule is the site, returned directly, in "
                                             "one run and with no candidate involved",
                "all_bit_identical": "the recomputation reproduces every value, so the defect is "
                                     "not in what the recomputation COMPUTES but in which "
                                     "tensors the graph SAVES -- and the next instrument is the "
                                     "saved-tensor set, not another mechanism",
            },
            "why_it_cannot_be_retrodiction": "it reads the difference rather than testing "
                                             "whether a proposal matches a partition already "
                                             "entailed by cot_z (R153).",
        },
        "banked": BANKED,
        "DOESNOT": "this refutes one mechanism and names the instrument that replaces it. It "
                   "moves no published number, it does not repair D242, and it merges nothing.",
    }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
