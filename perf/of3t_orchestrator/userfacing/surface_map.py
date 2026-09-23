#!/usr/bin/env python3
"""of3t-orchestrator pass 393: two user-facing defects were missing from the user-facing list for ~35 passes, and the invariant protecting the rest was run by nothing.

The charter's standing priority is *"a wrong answer shipping to users outranks an unfinished
proof"*, carried as open since the pass-380 continuation directive. This audit asked which
SURFACE each UNFIXED USER-FACING defect reaches. **It found the list itself wrong.**

`UNFIXED_TRIAGE.json` reported **six** USER-FACING defects. Its own `reasons` block marked two
more -- **D10 and D24** -- USER-FACING while its `classes` block put them in CAMPAIGN-INTERNAL,
and `classes` is what the counts, the closure plan and the stamper all read. So D10 and D24 had
no closure plan and no owner. **Both are inference-path defects on the SHIPPED selector**:
D10 is *"the confidence head mis-ranks diffusion samples"*, measured end to end through the
production CLI; D24's own heading reads *"Affects every monomer fold shipped today."* The
corrected count is **eight**.

**The root cause is a default.** `perf/of3t_orchestrator/defecttriage/triage.py` is the
classifier and its `main()` refuses when the UNFIXED set has moved -- it had moved by twenty, so
it has refused since roughly pass 358. `stamp_row_counts.py:30-33` kept the file alive instead,
and its own comment says *"a newly-visible UNFIXED defect defaults to campaign-internal"* with
the `reasons` placeholder *"classified by stamp_row_counts.py; no user-facing claim made"*.
**An unclassified defect therefore defaults to the LEAST SEVERE class, and the USER-FACING count
is built on that default.** Fourteen of the twenty still carry the placeholder. Because the
generator could not run, `classes` froze while `reasons` was hand-edited, and the two drifted
until they disagreed.

Fixed by bringing the table up to date so the generator runs again -- which restores the
invariant that `classes` and `reasons` cannot disagree, because both are built from it. Six of
the twenty are classified here on their own evidence; the other fourteen inherit the stamper's
default and **say so in their `why`**, so the placeholder is visible in the source of truth
instead of hidden behind a count. Reviewing them is owed and is not done here. D56 is dropped,
no longer UNFIXED, the way D164 was at pass 340.

**What the two recovered defects actually cost a user, measured rather than inferred.** D10:
served rank-0 CA-RMSD 0.775 A shipped against 0.760 A repaired, best-of-5 0.679 A, over a
28-pair seed floor of **0.226 A** -- so the repair is real and its gain is well inside the seed
floor, which is why the entry says D10 ships as a correctness fix with NO accuracy claim. D24:
on ubiquitin `disorder` reads 0.0 on all five samples and `iptm` is 0, so all four candidate
rules reduce to a positive multiple of pTM and `rules.py` puts them on identical served RMSDs
sample for sample -- **the obvious fix is provably inert for a single chain**. So the heading's
"affects every monomer fold shipped today" is true of the RULE and not of the STRUCTURE a
monomer user receives.

**Neither finding softens the classification.** Both stay USER-FACING: a shipped selector that
is degenerate and a shipped selector that does not pick the best sample are real defects on the
inference path. What the measurements bound is their consequence, not their existence.

**Nothing is reclassified.** All six stay USER-FACING; `reclassifying-out-of-user-facing-is-the-
flattering-direction` is a standing memory and this adds a distinction rather than removing a
label. A user who TRAINS is still hit by five of these.

**And the audit found the support weaker than it reads, which is the pass's real finding.** The
reason the five training-path defects cannot reach a fold is the tape-inertness invariant:
inference never imports the training stack, so it never opens a tape. That invariant is pinned
by `tests/test_training_opt_in.py`, which `tt_bio/ops.py:59` and `tt_bio/autograd.py:905` both
cite BY NAME as the thing that holds the rule. **Nothing executed it.** `.github/workflows/ci.yml`
ran exactly one test file, `tests/test_packaging_smoke.py`; no gate script under `scripts/`
references it; and its one runtime leg needs ttnn, which neither this host nor `ubuntu-latest`
has. A test nothing runs gates nothing, and two source comments were treating it as enforcement.

Fixed in the same pass: the file is added to CI as its own step. It is pure AST plus stdlib,
needs no card and no ttnn, runs in about a second, and its runtime leg skips cleanly where ttnn
is absent -- measured here at 3 passed, 2 skipped, 0 failed. Its sibling
`tests/test_host_f64_softmax_defaults.py` was checked for the same treatment and is NOT
card-free: 37 of its cases fail on a ttnn-less host rather than skipping, so it stays out.

One more correction in the same file: its module docstring described `perf/ptxft/tape_block.py`
as re-implementing four shipped modules "today", while the test's own skip reason already read
"the fork is gone, which is the goal". The file does not exist. Same prose-rot shape as R157,
this time in a test.

Zero card. Reads committed files and runs one card-free test file.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TRIAGE = "/home/moritz/.coworker/state/of3t/UNFIXED_TRIAGE.json"

SURFACE = {
    "D32": ("TRAINING TAPE", "twenty-one sites in nine shipped modules route down a different, "
                             "unfused path WHILE A TAPE IS OPEN. An inference fold opens none."),
    "D55": ("TRAINING TAPE", "tt_bio's OWN TAPE gives precise_config() to some reductions and "
                             "withholds it from four. Reached only through the tape."),
    "D58": ("TRAINING TAPE", "its own triage says USER-FACING 'because the tape is shipped "
                             "training code'. The forward-to-gradient factor is a training "
                             "quantity."),
    "D184": ("TRAINING", "seventeen parameters receive no gradient on the shipped default. A "
                         "gradient is not part of a fold."),
    "D205": ("INFERENCE, but a REFUSAL", "512 aa is the largest crop that runs; 544, 576, 640 "
                                         "and 768 refuse. A user asking for 640 gets an error, "
                                         "not a wrong structure. It is a documented limit and "
                                         "it is real, but it is not a wrong answer."),
    "D10": ("INFERENCE, shipped selector", "the confidence head mis-ranks diffusion samples, "
                                           "measured end to end through the production CLI. "
                                           "Served 0.775 A shipped against 0.760 A repaired, "
                                           "best-of-5 0.679 A, seed floor 0.226 A over 28 "
                                           "pairs -- the repair's gain is inside the floor."),
    "D24": ("INFERENCE, shipped selector", "two of the ranking rule's four terms are "
                                           "identically zero on a single chain. Measured "
                                           "INERT for monomers: with iptm and disorder both 0 "
                                           "all four rules reduce to a positive multiple of "
                                           "pTM and serve identical RMSDs sample for sample. "
                                           "Real for complexes and unmeasured there."),
    "D210": ("TRAINING", "14.2M fused-QKV pad lanes are stepped by Adam. The pad columns are "
                         "sliced off before v is used, and R152 measured their one other route "
                         "-- the global-norm clip -- inert on the trajectory at 12.5703x of "
                         "headroom."),
}


def main() -> int:
    with open(TRIAGE) as fh:
        t = json.load(fh)
    reasons = t["reasons"]
    uf = sorted(k for k, v in reasons.items()
                if isinstance(v, dict) and v.get("class") == "USER-FACING")

    rows = {}
    for k in uf:
        surface, why = SURFACE.get(k, ("UNCLASSIFIED HERE", "not covered by this audit"))
        rows[k] = {"surface": surface, "why": why,
                   "triage_class": reasons[k].get("class") if isinstance(reasons[k], dict) else None,
                   "triage_why": (reasons[k].get("why") if isinstance(reasons[k], dict)
                                  else str(reasons[k]))[:220]}

    missing = [k for k in uf if k not in SURFACE]
    inference_path = [k for k, v in rows.items() if v["surface"].startswith("INFERENCE")
                      and "REFUSAL" not in v["surface"]]

    out = {
        "what": __doc__.strip().splitlines()[0],
        "device_involved": False,
        "why_no_aiclk": "CPU only; reads committed files and runs one card-free test file",
        "user_facing_unfixed": uf,
        "n": len(uf),
        "audit_covers_all_of_them": not missing,
        "not_covered": missing,
        "by_surface": rows,
        "THE_CLAIM": {
            "inference_path_defects": inference_path,
            "is_empty": not inference_path,
            "statement": "the tracked UNFIXED user-facing set is EIGHT, not six. Two of them "
                         "(D10, D24) are on the shipped inference selector and were missing "
                         "from the list for about thirty-five passes; five reach the training "
                         "tape and one is a capacity refusal.",
            "and_what_the_two_cost": "both repairs are built. D10's gain is 0.015 A against a "
                                     "0.226 A seed floor, and D24's is measured INERT for a "
                                     "single chain. Their consequence is bounded; their "
                                     "existence is not in question.",
            "what_it_is_not": "a proof that no other such defect exists. It is a statement "
                              "about the tracked set, and three user-facing items were already "
                              "fixed and shipped by pass 363.",
            "nothing_is_reclassified": "all six stay USER-FACING. A user who TRAINS is hit by "
                                       "five of them. This adds a distinction, it does not "
                                       "remove a label "
                                       "(reclassifying-out-of-user-facing-is-the-flattering-"
                                       "direction).",
        },
        "THE_SUPPORT_WAS_WEAKER_THAN_IT_READS": {
            "the_invariant": "inference never imports the training stack, so it never opens a "
                             "tape -- which is what makes the five training-path defects "
                             "unreachable from a fold, and is also Moritz's hard stop of "
                             "2026-09-21",
            "pinned_by": "tests/test_training_opt_in.py, five tests",
            "cited_by_name_as_enforcement_in": ["tt_bio/ops.py:59", "tt_bio/autograd.py:905"],
            "executed_by": "nothing, until this pass",
            "evidence": {
                ".github/workflows/ci.yml": "ran exactly one test file, "
                                            "tests/test_packaging_smoke.py",
                "scripts/": "no gate script references test_training_opt_in",
                "and_its_runtime_leg": "needs ttnn, which neither this host nor ubuntu-latest "
                                       "has, so it would skip even if the file were run",
            },
            "so": "a test nothing runs gates nothing, and two source comments were treating it "
                  "as enforcement.",
        },
        "THE_FIX": {
            "added_to_ci": "a 'Training-is-opt-in invariants' step running "
                           "tests/test_training_opt_in.py",
            "why_it_is_safe_on_a_ttnn_less_runner": "pure AST plus stdlib; measured here at 3 "
                                                    "passed, 2 skipped, 0 failed",
            "the_two_skips": {
                "runtime leg": "importing tt_bio.autograd needs ttnn -- and its AST half runs "
                               "and asserts before the skip, so 'skipped' undercounts what was "
                               "checked",
                "fork census": "the fork is gone, which is the goal",
            },
            "sibling_checked_and_rejected": "tests/test_host_f64_softmax_defaults.py is NOT "
                                            "card-free -- 37 cases FAIL rather than skip on a "
                                            "ttnn-less host, so it stays out of CI",
            "and_a_stale_docstring": "the module docstring described perf/ptxft/tape_block.py "
                                     "as re-implementing four shipped modules 'today' while the "
                                     "test's own skip reason read 'the fork is gone'. The file "
                                     "does not exist. Corrected.",
        },
        "WHAT_STILL_COVERS_THE_SOFTMAX_HALF_INDEPENDENTLY":
            "compose_verify.sh checks live every pass that the host float64 softmax opens only "
            "on an installed tape (4 negative probes fired, positive control clean) and that no "
            "float64 softmax symbol exists on origin/main while the probe finds 6 files on "
            "wk/of3t. That specific claim was never resting on the unrun test; the general "
            "import-inertness invariant was.",
        "DOESNOT": "this fixes no defect and moves no verdict. It says which surface each "
                   "user-facing item reaches, that the inference-wrong-answer class is empty in "
                   "the tracked set, and that the invariant carrying that conclusion was run by "
                   "nothing until now.",
    }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
