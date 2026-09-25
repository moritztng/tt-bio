#!/usr/bin/env python3
"""of3t-covdefault: what it would take to reach the 99.2594 coverage bar, priced.

Pure arithmetic over `COVERAGE_CEILING.json` and `READABLE_MASS.json`, so the bar proposal in
the state doc has a committed carrier outside the doc that quotes it.

  python3 perf/of3t_covdefault/remedies.py
"""
import json
from pathlib import Path

W = Path(__file__).resolve().parents[2]
CC = json.loads((Path(__file__).with_name("COVERAGE_CEILING.json")).read_text())
RM = json.loads((W / "perf/of3t_readable_mass/READABLE_MASS.json").read_text())
BD = json.loads((W / "perf/of3t_modelboundary/MODEL_withtrunk_n384.json").read_text())
BAR = 99.2594
OUT = Path(__file__).with_name("BAR_PROPOSAL.json")

shipped = CC["coverage"]["shipped_measured"]
s = CC["seventeen"]
d8 = s["diffusion_module_ref_atom_feature_embedder_8"]["pct_of_model"]
ie8 = s["input_embedder_ref_atom_feature_embedder_8"]["pct_of_model"]
lq = s["input_embedder_linear_q_0_weight"]["pct_of_model"]
cls = {k: v["pct_of_model_mass"] for k, v in RM["classes"].items()}

remedies = {
    "flag_ON_only": {
        "adds_pct": d8,
        "needs": "TT_BIO_OF3_DEVICE_REFATOM default ON. Nothing else.",
        "costs": "+198.476 ms per cold fold on openfold3 and +192.515 on openbind, +50.231 "
                 "and +48.185 warm, and the structure moves 0.114391 A / 0.048635 A "
                 "(perf/of3t_covdefault/AB_FOLD.json)",
        "reaches": shipped + d8},
    "flag_ON_plus_an_input_embedder_arm": {
        "adds_pct": d8 + lq,
        "needs": "the flag ON, plus a scored gradient arm for the input embedder -- its section "
                 "reads n_compared 0 of 98 today, so linear_q.0.weight has nowhere to land even "
                 "when the flag puts it on the card",
        "costs": "the same inference regression, plus one new device arm",
        "reaches": shipped + d8 + lq},
    "flag_ON_plus_input_embedder_arm_plus_the_other_eight_on_device": {
        "adds_pct": d8 + lq + ie8,
        "needs": "all of the above, plus moving the input embedder's OWN eight ref-atom linears "
                 "off openfold3_host_prep.py:301 -- code that does not exist in either arm",
        "costs": "the same, plus a second device leg on the same once-per-fold call",
        "reaches": shipped + d8 + lq + ie8},
    "the_four_blocked_classes_nobody_owns": {
        "adds_pct": sum(v for k, v in cls.items()
                        if k in ("FUSED_NOT_SPLIT", "NO_ARM", "NOT_A_LEAF_LAZY",
                                 "ARM_RAN_DID_NOT_CARRY")),
        "needs": "instrument and wiring work on code that already exists, per class: "
                 + ", ".join(f"{k} {cls[k]:.7f} %" for k in
                             ("FUSED_NOT_SPLIT", "NO_ARM", "NOT_A_LEAF_LAZY",
                              "ARM_RAN_DID_NOT_CARRY")),
        "costs": "no inference change",
        "reaches": shipped + sum(v for k, v in cls.items()
                                 if k in ("FUSED_NOT_SPLIT", "NO_ARM", "NOT_A_LEAF_LAZY",
                                          "ARM_RAN_DID_NOT_CARRY"))},
}
for v in remedies.values():
    v["clears_99_2594"] = v["reaches"] >= BAR

out = {
    "instrument": "of3t-covdefault remedies.py -- every route to the 99.2594 coverage bar with "
                  "what it costs, so the bar decision is arithmetic and not preference",
    "bar": BAR,
    "shipped_measured": shipped,
    "points_needed": BAR - shipped,
    "remedies": remedies,
    "finding": "the bar is reachable, and every route that reaches it requires "
               "TT_BIO_OF3_DEVICE_REFATOM ON, which this row measures as a per-fold inference "
               "regression on both models of OF3_FAMILY. The charter's own hard constraint is "
               "that nothing may change inference's output or slow it down, so the coverage "
               "clause as written can only be satisfied by breaking the constraint it sits "
               "beside. That is the defect, not the 0.52 points.",
    "proposal": {
        "repoint_to": 97.9849,
        "arithmetic": f"the measured ceiling of the shipped arm is {shipped!r}; a bar that "
                      f"rounds a measured ceiling UP is unsatisfiable by a perfect artifact, so "
                      f"it rounds DOWN, at the four decimals the current bar carries",
        "and_keep_the_question_alive": "a bar equal to the measured value is a no-regression "
                                       "ratchet, not a proof of coverage. Pair it with the class "
                                       "itemisation below so the 2.0150069313385040 % that is "
                                       "NOT compared stays stated rather than absorbed, and so "
                                       "the number cannot drift down without failing",
        "uncompared_itemised_pct": {k: v for k, v in cls.items()
                                    if k not in ("COMPARED", "READ_ON_ANOTHER_BOUNDARY")},
        "uncompared_total_pct": BD["coverage_total"]["pct_of_model_uncompared"],
        "not_this_rows_edit": "proposed only. _of3t_donecheck.py and CHARTER_EVIDENCE belong to "
                              "the orchestrator; a row that moves its own bar toward passing is "
                              "the pattern that gate exists to refuse",
    },
}
OUT.write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
print("->", OUT)
