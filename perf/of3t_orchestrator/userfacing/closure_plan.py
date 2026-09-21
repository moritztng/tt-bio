#!/usr/bin/env python3
"""What it would actually take to clear GO condition 5, one USER-FACING defect at a time.

Condition 5 is "no unfixed or user-facing defect in GAP". The campaign has argued about the
CONDITION for twenty passes (D122: it is a keyword test on prose) and has never written down the
answer to the obvious question -- what would closing it cost? With the ledger finally complete
(D133/D135 took the invisible set from 29 to 0), that question is answerable for the first time,
because the USER-FACING set is now the whole of it rather than whatever happened to carry a word.

Every entry names a CLOSURE CONDITION, what it NEEDS, and the evidence already held. The table is
asserted against the live triage: if the USER-FACING set moves and this file does not, it refuses
to report rather than printing a stale plan.

CPU only. Reads two files. No device.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

TRIAGE = Path("/home/moritz/.coworker/state/of3t/UNFIXED_TRIAGE.json")
OUT = Path(__file__).resolve().parent / "CLOSURE_PLAN.json"

#: needs -> what has to happen before the defect can change status.
DECISION = "DECISION"      # Moritz's call; no measurement will move it
CARD = "CARD"              # a device row, already dispatched or dispatchable
RELEASE = "RELEASE"        # a merge/ship decision on an existing, measured repair
MERGE = "MERGE"            # DECIDED by Moritz and built; what is left is landing it on main,
                           # which is his gate and not a measurement this campaign can take

PLAN = {
    # D155 was here for one pass and is gone because it was WITHDRAWN, not closed: the
    # non-determinism is pc card 0, a faulty card root-caused 2026-08-17, not a protenix
    # property. Filing it USER-FACING was my error -- a row reporting a digest instability
    # from that card has not measured determinism, and the exclusion is standing.
    # D1 was here until pass 280 and is gone because it CLOSED, not because the plan shrank:
    # Moritz decided it on ask 9629 ("fix it everywhere"), of3t-d1-pairbias concluded GO against
    # his one reopen condition (4 targets, 6 seeds, 48 folds, sign test p = 0.541, pooled median
    # negative, sole regressor 1UBQ which is the earlier objection's own target), and the repair
    # is in the composition with the shipped-default assertion moved to match. The plan refused
    # to publish while it still listed D1 -- "the plan and the live USER-FACING set disagree" --
    # which is the check doing its job rather than an inconvenience.
    "D10": {
        "needs": MERGE,
        "one_line": "the confidence head mis-ranks diffusion samples, and that is what makes D1's repair serve worse",
        "closes_when": ("it lands on main. DECIDED 2026-09-21 on ask 9629: unify, on consistency "
                        "grounds, because no accuracy argument survives either way -- of3t-rankunify "
                        "withdrew it. Built, verified in the tree at pass 278, not merged"),
        "evidence_held": ("+0.046 A and +0.020 A on the two changed models, 9 of 12 changed folds "
                          "the WRONG way, every difference inside its seed floor, p = 0.146"),
        "would_a_row_help": False,
        "asked": "pin 9629, together with D24 -- ANSWERED 2026-09-21 (state/ask-9629-decision.md)",
    },
    "D24": {
        "needs": MERGE,
        "one_line": "three shipped models computed three different ranking rules; one rule now exists and nothing is merged",
        "closes_when": "the same as D10: the decision is made and the merge is not",
        "evidence_held": "the family computes ONE rule on the branch; the accuracy claim is explicitly WITHDRAWN",
        "would_a_row_help": False,
        "asked": "pin 9629, together with D10 -- ANSWERED 2026-09-21",
    },
    "D56": {
        "needs": MERGE,
        "one_line": "a diffusion-scope concentration that the softmax-backward repair collapses 333x",
        "closes_when": ("TT_BIO_SOFTMAX_BW_RENORM stops being default-off. The mechanism is refuted "
                        "and the magnitude collapsed on the repaired arm; what keeps it UNFIXED is "
                        "that the SHIPPED configuration is still the unrepaired one"),
        "evidence_held": ("matched same-branch A/B: leaf error mass 878.85 -> 2.636 (333x), block 8 "
                          "norm ratio 87.643 -> 1.732 with cos -0.169 -> +0.694, 523 tensors both arms"),
        "what_it_costs": ("read from taped_ttnn.py at pass 259, because the ask should not have gone "
                          "out without it. The lever adds EXACTLY TWO OPS inside the softmax "
                          "backward rule -- one `ttnn.sum(y, dim, keepdim=True, "
                          "compute_kernel_config=precise_config())` and one `ttnn.divide` -- on the "
                          "shapes the existing `inner` reduction already uses. It lives in `bw`, so "
                          "it runs ONLY under the tape: an inference fold cannot execute it and the "
                          "inference cost is zero STRUCTURALLY, not by measurement. The training "
                          "cost is unmeasured, and there is no baseline to measure it against -- "
                          "that is D32, `no training throughput can be projected`. So the decision "
                          "does not wait on a number: nothing it could cost is currently knowable, "
                          "and nothing it could cost reaches a user's fold."),
        "would_a_row_help": False,
        "asked": ("pin 9629 -- ANSWERED 2026-09-21: SHIP IT ON. It is default-ON in the "
                  "composition since pass 274, verified backward-only by AST, and not merged"),
    },
    "D30": {
        "needs": CARD,
        "one_line": "the diffusion module's backward costs 19.6x the forward it is taken at",
        "closes_when": ("the cause of the backward-over-forward amplification is located. D58 "
                        "already moved this from a diffusion-module property to a property of the "
                        "tape, and of3t-ditcot is measuring the same object one level down"),
        "evidence_held": ("forward median 8.34e-03, gradient median 1.6588e-01, ratio 19.6x over 48 "
                          "structures against the rebuilt 0.4.3 reference"),
        "would_a_row_help": True,
        "row": "of3t-ditcot",
        "shares_object_with": ["D58", "D129"],
    },
    "D58": {
        "needs": CARD,
        "one_line": "the ~20x amplification belongs to the tape, not to any module: 19.6x and 19.8x in two independent modules",
        "closes_when": ("the same locating measurement as D30. Two independent sightings make it a "
                        "property to explain rather than a coincidence to chase"),
        "evidence_held": ("diffusion 0.85 % forward / 16.6 % gradient; msa_module 0.82 % / 16.2 % -- "
                          "different ops, different track, same factor to two significant figures"),
        "would_a_row_help": True,
        "row": "of3t-ditcot",
        "shares_object_with": ["D30", "D129"],
    },
    "D129": {
        "needs": CARD,
        "one_line": "a LayerNorm affine leaf at 4.388x its own bf16 floor, all of it the arriving cotangent",
        "closes_when": ("of3t-ditcot names the op carrying the flat 2.28x cotangent excess and it is "
                        "repaired or shown to be a floor. Separately, the 0.4.3 ratio needs OUR arm "
                        "at the 0.4.3 capture, which no artifact holds and which needs a lease"),
        "evidence_held": ("isolation 1.5217e-03 (456x under the reading), input exact to 1.98e-08, "
                          "substitution reproduces the reference gradient at 8.877e-09; the 0.4.3 "
                          "bar is 2.2530588761e-01, 0.73 % from the 0.5.0 one"),
        "would_a_row_help": True,
        "row": "of3t-ditcot",
        "shares_object_with": ["D30", "D58"],
    },
    "D55": {
        "needs": CARD,
        "one_line": "four of the tape's ten reductions still lack the precise kernel config, two of them inside the LayerNorm cancellation the rule's own comment is about",
        "closes_when": ("all four are pulled together with a LoFi break control. Six of the ten are "
                        "already measured inert with reach proven, so this is one arm, not a study"),
        "evidence_held": ("the pass-237 AST census, keyed by symbol; dn_mean x4 inert at 2736/2736 "
                          "with the break control moving 2733; softmax:inner x2 inert"),
        "would_a_row_help": True,
        "row": "of3t-ditcot",
    },
    "D32": {
        "needs": CARD,
        "one_line": "21 sites in 9 shipped modules route differently while a tape is open, so no training throughput can be projected from inference",
        "closes_when": ("a full training step is timed on the shipped default and published as "
                        "s/step. This is a MEASUREMENT, not a repair: the routing is deliberate and "
                        "the defect is that nobody can price training without it"),
        "evidence_held": ("the 21 sites enumerated by module and line; the trunk-only taped step "
                          "measured at 230.11 s against 9.721 s for inference, 24x, at crop 384. "
                          "D164 (pass 304): the OTHER trunk figure on the record, 870.75 s, is a "
                          "memory-ladder run's wall clock at 3.53 ms per verb call and is not a "
                          "timing at all -- the same backward clean reads 222.48 s, so this row "
                          "must re-establish its own baseline before it extends one"),
        "would_a_row_help": True,
        "row": "of3t-stepfloor",
    },
}


def main() -> int:
    if not TRIAGE.is_file():
        print(f"REFUSING: {TRIAGE} is absent, so there is no live set to plan against")
        return 1
    live = set(json.loads(TRIAGE.read_text())["classes"]["USER-FACING"])
    planned = set(PLAN)
    if live != planned:
        print("REFUSING: the plan and the live USER-FACING set disagree, so this would be a "
              "stale plan reported as a current one.")
        print(f"  live but unplanned: {sorted(live - planned, key=lambda d: int(d[1:]))}")
        print(f"  planned but not live: {sorted(planned - live, key=lambda d: int(d[1:]))}")
        return 1

    order = sorted(PLAN, key=lambda d: int(d[1:]))
    by_need: dict[str, list[str]] = {}
    for n in order:
        by_need.setdefault(PLAN[n]["needs"], []).append(n)

    print(f"GO condition 5: {len(order)} USER-FACING defects, and they are not "
          f"{len(order)} problems.\n")
    for need in (MERGE, DECISION, RELEASE, CARD):
        ns = by_need.get(need, [])
        if not ns:
            continue
        print(f"--- {need}: {len(ns)}  ({', '.join(ns)})")
        for n in ns:
            e = PLAN[n]
            print(f"  {n}: {e['one_line']}")
            print(f"      closes when: {e['closes_when']}")
            print(f"      held:        {e['evidence_held']}")
            if e.get("row"):
                print(f"      row:         {e['row']}")
            if e.get("shares_object_with"):
                print(f"      same object: {', '.join(e['shares_object_with'])}")
            if e.get("what_it_costs"):
                print(f"      costs:       {e['what_it_costs']}")
            if e.get("asked"):
                print(f"      asked:       {e['asked']}")
        print()

    dec = by_need.get(DECISION, []) + by_need.get(RELEASE, [])
    mrg = by_need.get(MERGE, [])
    card = by_need.get(CARD, [])
    shared = [n for n in card if PLAN[n].get("shares_object_with")]
    if dec:
        print(f"SUMMARY. {len(dec)} of {len(order)} need a DECISION or a RELEASE and no "
              f"measurement will move them: {', '.join(dec)}.")
    else:
        print(f"SUMMARY. NONE of the {len(order)} is waiting on a decision -- that half closed on "
              f"2026-09-21 when Moritz answered pin 9629.")
    if mrg:
        print(f"{len(mrg)} are DECIDED and BUILT and wait only to be merged: {', '.join(mrg)}. "
              f"That is not a measurement this campaign can take: the merge gate is Moritz's, and "
              f"it is now on the critical path to condition 5 rather than beside it.")
    print(f"{len(card)} need a card: {', '.join(card)}. Of those, {len(shared)} "
          f"({', '.join(shared)}) are the SAME OBJECT -- the tape's backward -- and "
          f"`of3t-ditcot` is dispatched against it and held until a card frees.")
    # Pass 304. D32 sat here with `row: None` for 180 passes and this summary never said so:
    # it counted the card-bound defects and named the row for three of them, which reads as
    # "all of them are dispatched". An unowned defect is the one thing a closure plan must not
    # let pass silently, so derive it rather than narrate it.
    _orphan = [n for n in card if not PLAN[n].get("row")]
    if _orphan:
        _verb = "needs a card and has" if len(_orphan) == 1 else "need a card and have"
        print(f"  UNOWNED, and this is the line that was missing: {', '.join(_orphan)} "
              f"{_verb} NO ROW. Dispatch one or say why not.")
    else:
        _owned = ", ".join(f"{n} -> {PLAN[n]['row']}" for n in card)
        print(f"  Every card-bound defect has a row: {_owned}.")
    asked = [n for n in order if PLAN[n].get("asked")]
    print(f"All {len(asked)} of the decision/release items were asked as one bundle (pin 9629) and "
          f"MORITZ ANSWERED on 2026-09-21, by delegating: \"for all of those. think hard. use your "
          f"own judgement. and do the right thing.\" The calls are recorded with their reasoning in "
          f"state/ask-9629-decision.md. So these are no longer waiting on him -- they are waiting "
          f"on a MERGE, which is a different gate and still his.")
    print("So condition 5 is now one merge, one card-bound measurement each for the rest, and no "
          "open question. Stated as a plan, not a promise: naming a closure condition is not "
          "meeting it, and a decided defect is not a merged one.")

    OUT.write_text(json.dumps({
        "what": ("What it would take to clear GO condition 5, per USER-FACING defect. Asserted "
                 "against state/of3t/UNFIXED_TRIAGE.json, which the gate reads."),
        "user_facing": order,
        "by_need": by_need,
        "plan": PLAN,
    }, indent=1, sort_keys=True))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
