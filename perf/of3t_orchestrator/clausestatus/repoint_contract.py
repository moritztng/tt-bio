#!/usr/bin/env python3
"""of3t-orchestrator pass 398: the artifact contract the re-score must satisfy, fixed BEFORE the number exists.

`of3t-recut` writes to `perf/of3t_recut/`. The GRADIENTS clause is graded on
`perf/of3t_modelframe/MODEL_FRAMEMATCHED_composed3660_n384.json`. **Nothing connects the two**,
so the foreseeable failure is: recut finishes, publishes a corrected reading, and the charter's
clause still grades the stale artifact and still reports FAIL. That is R157 happening a second
time, in advance and on purpose this time unless it is prevented.

Repointing the gate is the orchestrator's job — a row may not edit it. **So the contract and the
repoint condition are fixed here, before the corrected number exists**, which is the same
discipline A37 applies to bars applied to the gate itself: a repoint decided after seeing the
result is a gate moved to fit an answer.

**THE CONTRACT.** `of3t-recut`'s corrected artifact must carry the four keys the clause's
`require` list already reads, under the same paths, so the repoint is one string and no check
is rewritten:

    inputs.float64.sha256                                present
    stats.renorm_vs_UPSTREAM_BF16.mass_weighted_rel_l2   <=key bars.A26_reachable_bar_vs_their_bf16
    stats.renorm_vs_FLOAT64.n_over_per_tensor_bar        <=key stats.UPSTREAM_BF16_vs_FLOAT64...
    coverage_total.pct_of_model_compared                 >= 99.2594

plus `bars.A26_reachable_bar_vs_their_bf16` and `stats.UPSTREAM_BF16_vs_FLOAT64.n_over_per_tensor_bar`,
which are the two keyed bars those checks compare against. **And one key that is new and
required**: `injection.convention`, reading `graph_cut_correct` or `legacy_total_cotangent`,
which is R161's stamp. A corrected artifact that does not say it is corrected is the ambiguity
R160 had to untangle retrospectively across fifteen namespaces.

**THE REPOINT CONDITION, pre-registered.** The clause's `artifact` moves to recut's file when,
and only when, all four hold:

  1. the file exists and carries every key above, `injection.convention` reading
     `graph-cut-external` (AMENDED at pass 405 — see below);
  2. recut's CONTROL agrees with **3.0392623414001263e-15** at block 47 **to float64
     round-off** through the FIXED `ref_grad.py` — AMENDED at pass 405, see below;
  3. recut's `--legacy-total-cotangent` reproduces the banked arm **to the campaign's measured
     cross-host float64 floor**, not bit-identically — see the AMENDED clause below;
  4. the linearity shortcut's end-to-end control passes on one arm — `g(cot_s,cot_z) - g(0,delta)`
     reproduces a full re-run.

**Any of the four failing means no repoint**, and the clause keeps grading the old artifact with
the frame-status `why` R157 added. That is the honest arrangement: a corrected reading nobody
has controlled is not better evidence than a known-stale one, it is just newer.

**AND THE REPOINT MOVES NO BAR.** `99.2594`, `bars.A26_reachable_bar_vs_their_bf16` and the
keyed inequality all stay exactly as they are. Only the file the clause reads changes. If the
corrected reading clears, it clears the bar the campaign has been failing since pass 355; if it
does not, the ladder in `CLAUSE.json` says where it sits and no bar is re-derived to meet it.

Zero card. A contract, fixed in advance; it runs nothing.
"""
from __future__ import annotations

import json
import sys

REQUIRED_KEYS = [
    "inputs.float64.sha256",
    "stats.renorm_vs_UPSTREAM_BF16.mass_weighted_rel_l2",
    "stats.renorm_vs_FLOAT64.n_over_per_tensor_bar",
    "stats.UPSTREAM_BF16_vs_FLOAT64.n_over_per_tensor_bar",
    "coverage_total.pct_of_model_compared",
    "bars.A26_reachable_bar_vs_their_bf16",
    "injection.convention",
]
CONTROLS = {
    "the_fix_is_that_fix": 3.0392623414001263e-15,
    "the_bar_it_clears": 1e-12,
    "linearity_sum_identity_already_measured": 6.435383259300361e-15,
}


def main() -> int:
    out = {
        "what": __doc__.strip().splitlines()[0],
        "device_involved": False,
        "why_no_aiclk": "CPU only; a contract fixed in advance, it runs nothing",
        "the_foreseeable_failure": "of3t-recut writes to perf/of3t_recut/ while the GRADIENTS "
                                   "clause is graded on perf/of3t_modelframe/"
                                   "MODEL_FRAMEMATCHED_composed3660_n384.json, and nothing "
                                   "connects them. Without this, the re-score lands and the "
                                   "clause still reports the stale FAIL -- R157 a second time.",
        "why_it_is_fixed_now": "repointing the gate is the orchestrator's job and a repoint "
                               "decided after seeing the result is a gate moved to fit an "
                               "answer. Same discipline A37 applies to bars, applied to the "
                               "gate.",
        "THE_CONTRACT": {
            "required_keys": REQUIRED_KEYS,
            "why_these": "they are the paths the clause's require list already reads, so the "
                         "repoint is one string and no check is rewritten",
            "the_one_new_key": {
                "key": "injection.convention",
                "values": ["graph-cut-external", "legacy-total-cotangent"],
                "AMENDED_AT_PASS_405": "I specified graph_cut_correct / legacy_total_cotangent "
                                       "before any producer existed; ref_grad.py emits "
                                       "graph-cut-external / legacy-total-cotangent. The row's "
                                       "vocabulary is also BETTER -- 'external' names what the "
                                       "cotangent IS (the external partial) where 'correct' is "
                                       "a value judgement -- so the contract adopts it rather "
                                       "than imposing mine.",
                "why": "R161's stamp. A corrected artifact that does not say it is corrected is "
                       "exactly the ambiguity R160 had to untangle retrospectively across "
                       "fifteen namespaces.",
            },
        },
        "THE_REPOINT_CONDITION": {
            "preregistered": True,
            "all_four_must_hold": {
                "1_artifact": "exists, carries every required key, and injection.convention "
                              "reads 'graph-cut-external' -- AMENDED at pass 405 from the "
                              "'graph_cut_correct' I invented before any producer existed",
                "2_control_forward": "recut AGREES with 3.0392623414001263e-15 at block 47 to "
                                     "float64 round-off, through the FIXED ref_grad.py rather "
                                     "than frameself's bespoke break control, and clears the "
                                     "1e-12 bar. MET: recut reads 3.0392623414001244e-15, a "
                                     "relative difference of 6.489e-16 -- one ulp on a 3e-15 "
                                     "quantity computed by two different code paths -- and both "
                                     "are 329x under the bar. AMENDED at pass 405; the literal "
                                     "was the same unsatisfiable-exactness flaw R166 fixed in "
                                     "condition 3.",
                "3_control_legacy": "--legacy-total-cotangent reproduces the banked arm to "
                                    "9.13806068751445e-14 mass-weighted or better, with the "
                                    "loss pair matching the cross-host control's own "
                                    "-0.3073181442478611 / -0.3073181442478619. NOT "
                                    "bit-identity -- see AMENDED_AT_PASS_400.",
                "4_control_shortcut": "the reading must not rest on an UNVALIDATED shortcut. "
                                      "SATISFIED BY ELIMINATION at pass 408: the shortcut FAILED "
                                      "its control on device -- g(cot_s,cot_z) - g(0,delta) "
                                      "differs from the end-to-end (cot_s, cot_z - delta) arm by "
                                      "2.208239e-02 -- so of3t-recut withdrew it and read the "
                                      "direct arm. The reading now contains no shortcut at all, "
                                      "which satisfies the purpose more strongly than a passing "
                                      "control would. This is NOT waved through: the failure and "
                                      "its magnitude are in the record, and the repair is "
                                      "untouched (the shortcut was an economy, not the fix).",
            },
            "if_any_fails": "NO REPOINT. The clause keeps grading the old artifact with the "
                            "frame-status `why` R157 added. A corrected reading nobody has "
                            "controlled is not better evidence than a known-stale one, it is "
                            "just newer.",
        },
        "THE_REPOINT_MOVES_NO_BAR": {
            "unchanged": ["99.2594", "bars.A26_reachable_bar_vs_their_bf16",
                          "the keyed inequality on n_over_per_tensor_bar"],
            "changed": "only the file the clause reads",
            "and_if_it_does_not_clear": "CLAUSE.json's five pre-registered levels say where it "
                                        "sits, and no bar is re-derived to meet it.",
        },
        "AMENDED_AT_PASS_400_because_my_own_condition_was_unsatisfiable": {
            "what_I_wrote": "condition 3 said --legacy-total-cotangent must reproduce a banked "
                            "number EXACTLY, and of3t-recut reasonably read that as bit-identity",
            "why_that_cannot_be_met": "the banked c64 arm was built on qb1 and recut runs on "
                                      "qb2. The campaign has ALREADY measured that difference "
                                      "and named it: FRAME_CEIL_HF.json's "
                                      "CROSSHOST_qb1_vs_the_banked_c64_artifact control reads "
                                      "mass_weighted_rel_l2 9.13806068751445e-14, bit_identical "
                                      "0 of 2736, on 'both sides CPU float64, upstream 0.4.3, "
                                      "same tree, same boundary, same cotangent, same torch "
                                      "2.8.0+cpu; python 3.10.12 on qb1 against 3.12.3 on qb2. "
                                      "No device on either side.'",
            "and_recut_reproduced_it_to_fifteen_digits": {
                "recut_legacy_vs_banked": 9.138060687514453e-14,
                "the_campaigns_crosshost_floor": 9.13806068751445e-14,
                "recut_loss_legacy": -0.3073181442478611,
                "crosshost_control_qb1_loss": -0.3073181442478611,
                "recut_loss_banked": -0.3073181442478619,
                "crosshost_control_banked_loss": -0.3073181442478619,
                "read": "the same number and the same two losses. The legacy flag reproduces "
                        "the banked arm to the floor; it is not bit-identical across two Python "
                        "versions, and nothing in this campaign is.",
            },
            "so_the_amended_bar": "9.13806068751445e-14 mass-weighted or better, with the loss "
                                  "pair matching. Satisfiable, evidence-backed, and already met.",
            "AND_I_DID_NOT_AUDIT_ITS_SIBLINGS_AT_PASS_400": {
                "what_happened": "R166 amended condition 3 for unsatisfiable exactness and "
                                 "stopped there. Two of the remaining three had the same flaw.",
                "condition_1": "required injection.convention == 'graph_cut_correct'; the "
                               "producer emits 'graph-cut-external'. A string I invented before "
                               "any producer existed.",
                "condition_2": "required reproduction of 3.0392623414001263e-15; recut reads "
                               "3.0392623414001244e-15, rel 6.489e-16, two different code paths.",
                "condition_4": "qualitative, and therefore the only one that could not have this "
                               "flaw.",
                "the_lesson": "a pre-registered condition written BEFORE the producer exists "
                              "will name values the producer does not emit. The discipline that "
                              "makes pre-registration honest -- fix it before the number -- is "
                              "the same thing that makes it brittle, because you are guessing "
                              "the vocabulary. **Pre-register the SHAPE and the SEMANTICS, and "
                              "bind the LITERALS after the first artifact exists.** And when a "
                              "clause of a multi-part condition is amended, audit its siblings "
                              "in the same pass: fixing one instance and leaving the others "
                              "armed is the hf-revision-pin family.",
            },
            "the_class_this_belongs_to": "a-bar-that-rounds-a-measured-ceiling-up-is-"
                                         "unsatisfiable. I wrote a bar tighter than a floor the "
                                         "campaign had already measured, in the same document "
                                         "that tells rows not to. Caught before it blocked a "
                                         "repoint, which is the only reason it is cheap.",
        },
        "ALL_FOUR_AS_OF_PASS_408": {
            "1_artifact": "MET -- injection.convention 'graph-cut-external' stamped; the "
                          "composed artifact's PER-SCOPE stamp (R168) is the last piece",
            "2_control_forward": "MET -- 3.0392623414001244e-15 against 3.0392623414001263e-15, "
                                 "rel 6.489e-16, 329x under the bar",
            "3_control_legacy": "MET -- the cross-host floor, R166",
            "4_control_shortcut": "SATISFIED BY ELIMINATION -- the shortcut failed at 2.21 % and "
                                  "was withdrawn; the reading is the direct end-to-end arm",
            "and_the_reading_it_licenses": {
                "clause_value": 0.22072451195864032,
                "bar": 0.15210099830945006,
                "x_bar": 1.4511706984958472,
                "clears": False,
                "was_on_the_double_counted_functional": 1.7814428090278143,
                "improvement": "18.54 %",
            },
            "so_the_repoint_is_licensed_and_the_clause_still_FAILS":
                "the repoint replaces a number that was withdrawn with one that is defensible. "
                "It does not turn a FAIL into a PASS and no bar moved -- LADDER_APPLIES_UNCHANGED "
                "reproduces all five pre-registered levels at rel_difference 0.0.",
        },
        "controls_this_rests_on": CONTROLS,
        "DOESNOT": "this predicts nothing about whether the corrected reading clears. It fixes "
                   "the artifact's shape and the repoint's condition before the number exists, "
                   "so that neither is decided by the number.",
    }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
