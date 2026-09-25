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
SOURCE = "SOURCE"          # a source repair with NO measurement in it: no card, no decision,
                           # no ship gate. Added pass 414 for D246 rather than filing it CARD,
                           # because calling a zero-device fix "needs a card" is how a cheap item
                           # inherits an expensive item's excuse for still being open.

PLAN = {
    # D246 and D247 were here for one pass and are gone because they CLOSED, not because the plan
    # shrank to look better: `of3t-verbinstall` fixed both at `8ab8c791f` the same pass they were
    # filed, and both repairs were verified in source rather than taken from the commit message.
    # D246's fix is better than the one proposed to it -- the flag now sits on both halves, so the
    # docstring's advertised equivalence is RESTORED rather than withdrawn. Release-gated, unmerged.
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
    # D174 was here from pass 324 and is gone because it was REFUTED at pass 325, not because
    # the plan shrank: at a fixed width the pad output moves 7.7862e+02 and the real block stays
    # bit-identical through 48 blocks, so the transition mask our port omits cannot be what costs
    # the trunk. Its residue is D175, which is the measurement rather than the mechanism.
    # D175 was here from pass 325 and is gone because `of3t-padshape` REFUTED it as filed at pass
    # 327, not because the plan shrank. It is not a width law: the sweep is non-monotone, peaking
    # at 192/256 (which agree to seven digits) and coming back down at 384, so the two banked
    # endpoints were the ends of a curve with a maximum between them. Its premise was also D56-OFF
    # and the shipped repair removes 26.6x. The forward does not follow it (1.0345x against
    # 3.487x), so it is backward-only. Worth 21.1 % of the trunk's error; closing it entirely
    # leaves the trunk at 5.4139x upstream's own bf16, carried by the attention-pair-bias and the
    # single transition. That residual is the trunk's real object and is not yet a plan item
    # because it has no owner -- when it gets one it belongs here.
    # D10, D24 and D56 were here and are GONE because they LANDED, verified at pass 363 against
    # git rather than against a row's prose. tt_bio/ranking.py on origin/main is the unified
    # sample-ranking rule, called from openfold3_fold.py:295, rf3/confidence.py:122 and
    # worker.py:1068, with the superseded per-model names surviving only in its own docstring --
    # that closes D10 and D24. tt_bio/autograd.py:86 reads
    # SOFTMAX_BW_RENORM = env_flag("TT_BIO_SOFTMAX_BW_RENORM", True), default ON, which closes
    # D56. Removing an item from this plan is the flattering direction, so each one is recorded
    # with the file and line a reader can check.
    "D184": {
        "needs": MERGE,
        "one_line": "seventeen parameters get no gradient at all on the shipped default; the fix is built, measured and default-off",
        "closes_when": ("REWRITTEN pass 363: not the flag. `of3t-covdefault` NO-GO'd flipping it "
                        "(+198.476 ms cold / +50.231 ms warm on openfold3) and `of3t-refcov` then "
                        "reached those gradients WITHOUT it -- `git diff -- tt_bio/` empty, the arm "
                        "constructing RefAtomFeatureEmbedder directly -- taking coverage to "
                        "99.50523155277438 %. The symbol does not exist on origin/main at all. What "
                        "is left is the narrower true statement that on the shipped inference "
                        "default those seventeen parameters receive no gradient, which matters to "
                        "TRAINING and not to a user's fold; it closes when a shipped training route "
                        "reaches them. Superseded text: `of3t-hostleg` wired "
                        "both legs and measured them: the `all` arm reads mass-weighted rel_l2 "
                        "0.05013681 against float64 over the seventeen, 6 of 17 over the 5.0e-02 "
                        "per-tensor bar, worst 1.175005e-01, and its inference default is "
                        "byte-identical to the base tree on both models that execute the changed "
                        "path. Nothing further is measurable until it lands"),
        "evidence_held": ("the shipped arm reads rel_l2 exactly 1.00000000 on all seventeen -- the "
                          "A16 zero-model signature, so the default computes no gradient for them "
                          "at all. The arm ladder is shipped 1.00000000, refatom 0.71215816, all "
                          "0.05013681, break 0.82521107. CORRECTED at pass 358 (D212): this field "
                          "said landing the flag takes GRADIENTS' coverage to 99.50523155277438 %, "
                          "crediting all seventeen. `of3t-covdefault` measured the key-set diff and "
                          "the flag reaches EIGHT -- 97.98499306866148 % + 0.7530394291090192 = "
                          "98.73803249777050 %, which is 0.52 points SHORT of the 99.2594 % bar. "
                          "The other nine cannot enter the reading at all: `input_embedder` reads "
                          "n_compared 0 of 98 in the boundary artifact. And the flag is now a "
                          "measured inference regression (+198.476 ms cold, +50.231 ms warm on "
                          "openfold3), so the coverage route does not run through it -- it runs "
                          "through the TRAINING ADAPTER, which `tt_bio/train/` confines to training "
                          "by construction. Row `of3t-refcov`"),
        "would_a_row_help": False,
        "asked": ("not yet asked. It belongs with D126 in one merge question rather than as a "
                  "separate ask: both are built, measured, release-gated and waiting only on "
                  "Moritz"),
    },
    # Added pass 393. D10 and D24 were classified USER-FACING in triage.py's own table and
    # were absent from the live USER-FACING list for ~35 passes, because the generator had
    # refused since roughly pass 358 and stamp_row_counts.py kept the file alive by appending
    # new defects to CAMPAIGN-INTERNAL. This plan's own guard is what caught it, on the first
    # regeneration -- "live but unplanned: ['D10', 'D24']".
    "D10": {
        "needs": MERGE,
        "one_line": "the confidence head mis-ranks diffusion samples on the shipped selector, so the served structure is not the best of the five",
        "closes_when": ("the repaired rule merges. It is BUILT and MEASURED end to end through "
                        "the production CLI -- what is left is a merge, which is Moritz's gate "
                        "and not this campaign's"),
        "evidence_held": ("`of3t-confhead`, 1UBQ, production CLI, 200 sampling steps, arms "
                          "interleaved, p300c at AICLK 1350 sampled during, nine ship-arm seeds "
                          "and eight fix-arm seeds over a 28-pair seed floor, read from "
                          "perf/of3t_confhead/analyze.json at 5588d889a rather than from the "
                          "row's prose. Served rank-0 CA-RMSD: shipped 0.775 A, D10 alone "
                          "0.760 A, best-of-5 0.679 A in both. **The 0.015 A the fix gains is "
                          "well inside the 0.226 A seed floor**, which is why the entry says "
                          "D10 ships as a CORRECTNESS fix with NO ACCURACY CLAIM. D1 alone "
                          "serves 1.201 A and D1+D10 0.924 A, so D1 does not ship"),
        "would_a_row_help": False,
        "asked": ("belongs in the same merge question as D184 and D126: built, measured, "
                  "release-gated, waiting only on Moritz"),
    },
    "D24": {
        "needs": MERGE,
        "one_line": "on a single chain OpenFold3's ranking rule has two of its four terms identically zero, and it is the only model of five that leaves it unhandled",
        "closes_when": ("the ipTM->pTM fallback the other four models carry is given to "
                        "OpenFold3 and merged. **It is measured to change nothing a "
                        "single-chain user is served** and it is still the right consistency "
                        "change for complexes, so what closes it is a merge plus one "
                        "measurement on a COMPLEX, where the terms are not degenerate"),
        "evidence_held": ("machine-checked in perf/of3t_confhead/rank_rule.py on the shipped "
                          "default, independent of D1. On ubiquitin the rule reduces one step "
                          "further than filed: `disorder` reads 0.0 on all five samples -- a "
                          "compact 76-residue fold never pushes a 25-residue smoothed RASA "
                          "window past the 0.581 threshold -- so every rank_score is 0.2*pTM to "
                          "the last digit. With iptm = 0 and disorder = 0 the shipped, family, "
                          "no_disorder and ptm rules all reduce to a POSITIVE MULTIPLE of pTM, "
                          "and a positive multiple cannot change an ordering: rules.py puts all "
                          "four on identical served RMSDs, sample for sample. So the heading's "
                          "'affects every monomer fold shipped today' is true of the RULE and "
                          "not of the STRUCTURE a monomer user receives. It stays USER-FACING: "
                          "the shipped selector is degenerate and that is a real defect; what "
                          "the measurement bounds is its consequence, not its existence. "
                          "`disorder = 0` is a property of a compact 76-residue monomer, not of "
                          "monomers, so a larger or genuinely disordered single chain is "
                          "unmeasured"),
        "would_a_row_help": True,
        "asked": ("not yet asked. The complex-side measurement has no owner"),
    },
    "D210": {
        "needs": RELEASE,
        "one_line": "the diffusion transformer trains 14.2M parameters upstream does not have -- fused-QKV pad lanes that Adam steps anyway",
        "closes_when": ("the pad lanes are masked out of the optimizer's parameter set, or a "
                        "measurement establishes they are harmless. Masking is a model change on "
                        "the SHARED diffusion path, so it owes an inference A/B against an A/A "
                        "floor on every model that executes it -- which is what makes this a "
                        "release item rather than a one-line fix"),
        "evidence_held": ("`of3t-trajfull` found it outside its own scored set: our fused `qkv_w` "
                          "pads head_dim 48 -> 64 and the pad columns are registered leaves. They "
                          "are exactly 0.0 at `w_0` and reach 3.494e-04 by k = 20. The mechanism is "
                          "Adam's scale invariance -- a numerically tiny device-backward gradient "
                          "in a lane that should have none still takes a full lr-sized step. It "
                          "moves no number the campaign quotes, because the pad columns are outside "
                          "the reference's parameter space and are sliced off before v is used, "
                          "which is exactly why it sat unnoticed"),
        "would_a_row_help": True,
        "asked": ("not yet asked, and not yet owned. It is the only USER-FACING item whose repair "
                  "has not been built"),
    },
    # Pass 415: D58 closed on both legs (`of3t-msaamp`, GO) and the forward gap it found in its
    # place is D250.
    "D250": {
        "needs": CARD,
        "one_line": "msa_module's forward is 3.54x less accurate than upstream 0.4.3 bf16 at the same boundary (8.176e-03 against 2.311e-03)",
        "closes_when": ("the gap is located to an op and either repaired, default-off and "
                        "tape-gated, or shown to be the reference's own at op level. An "
                        "inference-side repair is Moritz's call under the 2026-09-21 "
                        "no-inference-regression constraint"),
        "evidence_held": ("`of3t-msaamp` (GO): device A/A bit-identical, qb1 p150a reproduces "
                          "qb2 p300c bit for bit, and the backward propagates the gap rather than "
                          "amplifying it (factor 2.38x against upstream's 5.29x). Gradient-side "
                          "family excess: pair_transition 2.79x, tri_att_end 2.59x, "
                          "tri_att_start 2.17x; the triangle multiplications beat upstream"),
        "would_a_row_help": True,
        "row": "of3t-msafwd",   # dispatched pass 415
    },
    "D205": {
        "needs": CARD,
        "one_line": "512 is the largest crop that RUNS; 544, 576, 640 and 768 all refuse",
        "closes_when": ("the CONTIGUITY wall is addressed or documented as the shipped limit. 640 "
                        "and 768 die with the card full, but 544 and 576 die on contiguity with "
                        "6.30 GB and 6.67 GB still free -- 576 refused a 2,717,908,992 B buffer "
                        "inside ttnn::concat -> tilize_with_val_padding, short by 77,930,560 B "
                        "per bank at 88.45 %% occupancy. A capacity extrapolation cannot see that "
                        "wall; the row's own 2.08 fit said 576 would clear with 14 %% of margin. "
                        "Closing it is an allocator or a chunking question, not more memory"),
        "evidence_held": ("544/576/640/768 all measured to refuse, the 544 fixture built for the "
                          "purpose; the 576 refusal reproduces byte for byte across card 0 and "
                          "card 1 (29,970,916,352 B high-water, 5,622 allocations, identical "
                          "per-bank largest-free-block); every refused rung's high-water is a "
                          "LOWER bound, so 768's 1.558x overshoot is a floor; and odd 32-tile "
                          "counts (480, 544) narrow the fp32-softmax L1 plan to 0 B where every "
                          "even count measured keeps it"),
        "row": "of3t-cropwall",   # dispatched pass 414; it overturned the mechanism (D248)
        "owner": "of3t-crop768, CONCLUDED 2026-09-21 -- absorbed into the ledger at pass 349 "
                 "(D204: nothing checked that it ever was)",
    },
    # D55's BACKWARD half closed at pass 311 and the entry stays, rewritten to its forward half.
    # Not removed: `of3t-ditcot`'s commit subject reads "D55 closed" and it is not, it is half.
    # The four it measured are backward reductions inside bw rules; the forward half is one
    # missing compute_kernel_config on the softmax FORWARD, which changes what the card computes.
    "D55": {
        "needs": CARD,
        "one_line": ("the softmax FORWARD is missing one compute_kernel_config, worth 11.2-12.3x "
                     "accuracy for 1.35-1.50x cost on fp32 and inverting to 1.96x for 2.32x on bf16. "
                     "The BACKWARD half of this defect is CLOSED (pass 311, all four inert)"),
        "closes_when": ("of3t-fwdkcfg takes the forward arm with a fold A/B against an A/A floor. It "
                        "is a FORWARD change, so it moves fold output and the 2026-09-21 inference "
                        "constraint binds: an accuracy gain that costs inference time is a regression "
                        "here. The row is HELD behind of3t-ditcot, which owns the same file"),
        "evidence_held": ("BACKWARD, closed: pulled on the RENORM arm, A/A and PULL identical on every "
                          "scope (diffusion 547/547, trunk 2736/2736, T1 3/3, T3 4/4) with the LoFi "
                          "break control MOVING on all four (546/547, 2732/2736, 2/3, 3/4), and reach "
                          "measured rather than assumed. FORWARD, open: the four-rung fp32 ladder at a "
                          "pinned 1350 MHz plus the bf16 rung that inverts the trade"),
        "would_a_row_help": True,
        "row": "of3t-fwdkcfg",
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
    for need in (MERGE, DECISION, RELEASE, SOURCE, CARD):
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
    # Pass 414. The owner checks below ran on the CARD bucket alone, so adding a need
    # would have bought D246 an exemption from the one check this file exists to make.
    # A defect that needs no card still needs an owner -- more so, since nothing else
    # would ever notice it stalling. Same lesson as the D32 `row: None` miss, one need over.
    owned_needs = card + by_need.get(SOURCE, [])
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
    # Pass 311: this line used to end "and `of3t-ditcot` is dispatched against it and held until
    # a card frees." It was held when that was typed and has been running on qb2 card 0 since
    # 19:20 CEST. A closure plan has no business asserting SCHEDULING state: which row owns a
    # defect changes rarely and is derived two lines down, while whether that row is on a card
    # right now changes by the minute and is stale the moment it is written. Same lesson as the
    # D32 orphan comment below -- derive it or drop it, never narrate it.
    print(f"{len(card)} need a card: {', '.join(card)}. Of those, {len(shared)} "
          f"({', '.join(shared)}) are the SAME OBJECT -- the tape's backward.")
    # Pass 304. D32 sat here with `row: None` for 180 passes and this summary never said so:
    # it counted the card-bound defects and named the row for three of them, which reads as
    # "all of them are dispatched". An unowned defect is the one thing a closure plan must not
    # let pass silently, so derive it rather than narrate it.
    # Pass 312. The check above derived the row-is-None case and stopped there, so a defect whose
    # named row has CONCLUDED still read as owned: `of3t-ditcot` concluded STOP at 20:0x CEST and
    # D30, D58 and D129 went on printing "Every card-bound defect has a row." A finished owner is
    # not an owner, and it is the harder case to notice precisely because the field is populated.
    # Same shape as the D32 `row: None` miss this block was written for, one step over.
    _CONC = Path("/home/moritz/.coworker/state/concluded")
    _done = lambda r: bool(r) and _CONC.is_dir() and (_CONC / r).exists()
    _orphan = [n for n in owned_needs if not PLAN[n].get("row")]
    _stale = [n for n in owned_needs if _done(PLAN[n].get("row"))]
    if _orphan:
        _verb = "needs a card and has" if len(_orphan) == 1 else "need a card and have"
        print(f"  UNOWNED, and this is the line that was missing: {', '.join(_orphan)} "
              f"{_verb} NO ROW. Dispatch one or say why not.")
    if _stale:
        _byrow = {}
        for n in _stale:
            _byrow.setdefault(PLAN[n]["row"], []).append(n)
        for _r, _ns in _byrow.items():
            print(f"  OWNER FINISHED: {', '.join(_ns)} name `{_r}`, which has CONCLUDED. "
                  f"A concluded row is not an owner -- dispatch a successor or say why not.")
    if not _orphan and not _stale:
        _owned = ", ".join(f"{n} -> {PLAN[n]['row']}" for n in owned_needs)
        print(f"  Every defect that needs an owner has a LIVE row: {_owned}.")
    asked = [n for n in order if PLAN[n].get("asked")]
    print(f"All {len(asked)} of the decision/release items were asked as one bundle (pin 9629) and "
          f"MORITZ ANSWERED on 2026-09-21, by delegating: \"for all of those. think hard. use your "
          f"own judgement. and do the right thing.\" The calls are recorded with their reasoning in "
          f"state/ask-9629-decision.md. So these are no longer waiting on him -- they are waiting "
          f"on a MERGE, which is a different gate and still his.")
    # Pass 414: this sentence used to read "one merge, one card-bound measurement each for the
    # rest, and no open question". It was true when typed and D246 falsified it the moment it was
    # filed -- a SOURCE item is neither a merge nor a measurement. Derived now, for the same
    # reason the scheduling claim above was: a hand-written census is wrong on its first new row.
    _src = by_need.get(SOURCE, [])
    _bits = [f"{len(by_need.get(MERGE, []))} merge(s)",
             f"a card-bound measurement each for {len(card)}"]
    if _src:
        _bits.append(f"{len(_src)} source repair(s) needing no measurement at all ({', '.join(_src)})")
    print(f"So condition 5 is now {', '.join(_bits)}, and no open question. Stated as a plan, not "
          f"a promise: naming a closure condition is not meeting it, and a decided defect is not "
          f"a merged one.")

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
