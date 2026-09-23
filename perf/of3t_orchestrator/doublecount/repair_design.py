#!/usr/bin/env python3
"""of3t-orchestrator pass 396: how D242's repair lands -- four decisions, one of which inverts a standing rule of mine.

R160 located D242 at `perf/of3t_trunkg043/ref_grad.py:201` and specified the fix. Fifteen
namespaces drive that file, so how the fix lands is an arbitration question rather than an
edit, and it is mine. Four decisions, taken here so the row does not have to guess and so no
part of them is improvised inside a running pass.

**1. THE CORRECT BEHAVIOUR IS THE DEFAULT, AND THAT INVERTS MY OWN OPT-IN RULE ON PURPOSE.**
The standing rule is that a change to a shared instrument must be opt-in by default so every
existing arm reproduces what it produced before, with `of3t-trajfull`'s `--namemap structural`
as the model. **That rule protects a CORRECT baseline from a SPECULATIVE change. Here the
baseline is measured-wrong and the change is measured-right, so applying the rule unchanged
would leave every future arm inheriting a known double count unless its author remembered a
flag.** Opt-in is the right default for a LEVER and the wrong default for a BUG FIX. So: the
graph-cut-correct injection becomes the default, the old behaviour stays reachable as
`--legacy-total-cotangent` for reproducing a banked number, and **every artifact stamps which
convention produced it** so no reading is ever ambiguous about which functional it is on.

**2. THE RE-SCORES LAND IN ONE NAMESPACE, NOT FIFTEEN.** The corrected readings must not be
written into `perf/of3t_f64route/`, `perf/of3t_trunkceiling/` and the rest: those rows have
concluded and a row may not edit another's namespace
(`sibling-perf-campaigns-need-namespaced-output-paths`). They read their banked inputs from
wherever they live and write every output into one new directory, which also gives the campaign
a single place to read the corrected picture instead of a diff across fifteen.

**3. NO ARM NEEDS A FULL RE-RUN, AND THIS IS THE ONE THAT SAVES REAL DEVICE TIME.** The
parameter gradient is LINEAR in the injected cotangent -- not assumed, measured: `TWOBASIS.json`
puts `|| g_sonly + g_zonly - g_ctrl || / || g_ctrl ||` at **6.435383259300361e-15**. So with
`delta = autograd.grad(outputs=s_out, grad_outputs=cot_s, inputs=z_out)` computed ONCE,

    g_corrected  =  g(cot_s, cot_z)  -  g(0, delta)

and `g(cot_s, cot_z)` is already banked for every arm. **Each configuration needs one
additional arm driven by `(0, delta)`, not a re-run from scratch** -- which matters most for
the DEVICE arms, where a naive re-score would be card time the campaign does not have to spend.
The subtraction carries its own control for free: re-running one arm end to end with the
corrected cotangent must reproduce the subtraction, and the sum identity above is the
pre-existing evidence that the linearity holds in this exact setup.

**4. `of3t-frameself` MAY EDIT `ref_grad.py`, AND THIS FILE IS THE AUTHORISATION.** It is
`of3t-trunkg043`'s namespace and that row has concluded; the change is the repair of the defect
`of3t-frameself` root-caused; and splitting the fix from the diagnosis across two rows would
cost a full ramp-up to save nothing. The re-scores still go to one new namespace per decision 2.

Zero card. This is a design record; it runs nothing.
"""
from __future__ import annotations

import json
import sys

NAMESPACES = ["of3t_apbback", "of3t_apbleaf", "of3t_bwdaccum", "of3t_condtrans", "of3t_frame384",
              "of3t_gradients", "of3t_modelframe", "of3t_orchestrator", "of3t_trajwiden",
              "of3t_trunkact", "of3t_trunkback", "of3t_trunkdepth", "of3t_trunkg043",
              "of3t_trunkgrad", "of3t_widthattr"]
SUM_IDENTITY = 6.435383259300361e-15


def main() -> int:
    out = {
        "what": __doc__.strip().splitlines()[0],
        "defect": "D242",
        "device_involved": False,
        "why_no_aiclk": "CPU only; a design record that runs nothing",
        "scale": {"namespaces_driving_ref_grad_py": len(NAMESPACES), "which": NAMESPACES},
        "D1_THE_FIX_IS_THE_DEFAULT": {
            "decision": "the graph-cut-correct injection becomes the DEFAULT; the old behaviour "
                        "stays reachable as --legacy-total-cotangent; every artifact stamps "
                        "which convention produced it",
            "why_this_inverts_my_own_rule": "the standing rule makes a shared-instrument change "
                                            "opt-in so existing arms reproduce. That protects a "
                                            "CORRECT baseline from a SPECULATIVE change. Here "
                                            "the baseline is measured-wrong and the change is "
                                            "measured-right.",
            "the_distinction_worth_keeping": "opt-in is the right default for a LEVER and the "
                                             "wrong default for a BUG FIX. A lever that defaults "
                                             "on is an unreviewed change; a bug fix that "
                                             "defaults off is a known defect left armed.",
            "what_the_stamp_buys": "no reading is ever ambiguous about which functional it is "
                                   "on, which is the whole difficulty R160 had to untangle "
                                   "retrospectively.",
        },
        "D2_ONE_NAMESPACE_FOR_THE_RESCORES": {
            "decision": "corrected readings are written into ONE new directory, never back into "
                        "the concluded rows' namespaces",
            "why": "those rows have concluded and a row may not edit another's namespace "
                   "(sibling-perf-campaigns-need-namespaced-output-paths)",
            "and_the_second_benefit": "one place to read the corrected campaign instead of a "
                                      "diff across fifteen namespaces",
        },
        "D3_LINEARITY_MEANS_NO_ARM_IS_RE_RUN_FROM_SCRATCH": {
            "the_identity": "g_corrected = g(cot_s, cot_z) - g(0, delta), with delta = "
                            "autograd.grad(outputs=s_out, grad_outputs=cot_s, inputs=z_out)",
            "why_it_holds": "the parameter gradient is linear in the injected cotangent",
            "and_it_is_measured_not_assumed": {
                "source": "perf/of3t_frameself/TWOBASIS.json",
                "sum_identity_rel_l2": SUM_IDENTITY,
                "read": "|| g_sonly + g_zonly - g_ctrl || / || g_ctrl ||, in this exact setup",
            },
            "what_it_saves": "g(cot_s, cot_z) is already banked for every arm, so each "
                             "configuration needs ONE additional arm driven by (0, delta) "
                             "rather than a re-run. That is the difference between a cheap "
                             "re-score and card time on every device arm.",
            "its_own_control_is_free": "re-run ONE arm end to end with the corrected cotangent "
                                       "and it must reproduce the subtraction. If it does not, "
                                       "the linearity assumption is the thing that broke and "
                                       "the shortcut is withdrawn, not the repair.",
        },
        "D4_AUTHORISATION": {
            "decision": "of3t-frameself may edit perf/of3t_trunkg043/ref_grad.py",
            "why": "trunkg043 has concluded; the change repairs the defect frameself "
                   "root-caused; splitting fix from diagnosis would cost a full ramp-up to save "
                   "nothing",
            "limit": "the edit is to that one file. Re-scores still go to one new namespace per "
                     "decision 2, and no other concluded row's namespace is touched.",
        },
        "WHAT_THIS_DOES_NOT_DECIDE": "whether the corrected readings clear the clause. That is "
                                     "pass 394's pre-registered ladder and one measurement, and "
                                     "no bar may be re-derived to meet it.",
        "DOESNOT": "this runs nothing and measures nothing. It settles four questions the "
                   "repair would otherwise have improvised, one of which deliberately inverts a "
                   "standing rule of mine and says why.",
    }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
