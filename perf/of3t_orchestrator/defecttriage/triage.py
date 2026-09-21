#!/usr/bin/env python3
"""Classify every UNFIXED OF3T defect into the three classes GO condition 5 conflates.

CPU only. No card. Reads `state/of3t/DEFECTS.md` and writes `UNFIXED_TRIAGE.json`.

WHY THIS EXISTS
---------------
The campaign's gate (`workstreams/_of3t_donecheck.py`, `_charter_gate`) refuses GO with:

    if re.search(r"inference users get today|unfixed|still open|remains? open", gap, re.I):
        fail.append("VERDICT: GO while GAP still names unfixed or user-facing defects ...")

That is a KEYWORD test on the orchestrator's GAP prose, and `audit_evidence.py` separately
requires GAP to NAME every UNFIXED defect by number. Two consequences, both measured by this
script rather than argued:

  1. GAP contains D2 ("AF2 receives no gradient at all. UNFIXED, out of this campaign's scope")
     and D3 ("RFdiffusion3 cannot train. UNFIXED, out of scope"). Neither can ever be closed by
     this campaign, because each is recorded as belonging to another one. Read literally, GO is
     therefore unreachable no matter how much engineering is done -- not because the port is
     short, but because of two bookkeeping rows.
  2. The check reads PROSE, so it is satisfiable by naming all forty-four defects while avoiding
     four English words. A gate that a rewrite passes and an honest sentence fails is not testing
     what its own message says it tests ("the charter is not met while a defect ships to users").

This script does not change the gate. It supplies the number the gate's own message asks for --
how many defects SHIP TO USERS -- so that the question can be decided on evidence, and so that a
later answer to the open ask can be implemented against a machine-checked set instead of a
judgement remembered from one pass.

THE CLASSES
-----------
  SCOPE-EXCLUDED     the defect's own entry records it as outside this campaign's scope. It
                     cannot be closed here and no amount of work on OF3 will close it.
  USER-FACING        it changes what someone using the shipped tt-bio gets today: an inference
                     output, a crash, or the result of a training run on the shipped default.
  CAMPAIGN-INTERNAL  it is in this campaign's own measurement, instruments, references, captured
                     artifacts or bookkeeping. No shipped behaviour depends on it. Fixing it
                     changes what we KNOW, not what anyone RUNS.

The classification is a judgement and is recorded as one: every entry carries its reason, and the
boundary cases are listed in BOUNDARY below with the argument on both sides, because a triage that
hides its close calls is worth less than no triage.

ANTI-STALENESS
--------------
The table is ASSERTED against the live UNFIXED set on every run: a defect that is closed, or one
that is newly filed, makes this script FAIL rather than quietly report last week's answer. That is
the failure mode of every hand-maintained list this campaign has filed (D74, D83).
"""
import json
import re
import sys
from pathlib import Path

D = Path("/home/moritz/.coworker")
DEFECTS = D / "state" / "of3t" / "DEFECTS.md"
# Canonical location is the campaign state dir, NOT this branch's perf namespace: D112
# is a concluded row's worktree being pruned and taking the campaign's 0.4.3 reference with
# it, and the gate reads this file.
OUT = D / "state" / "of3t" / "UNFIXED_TRIAGE.json"

# The SAME parse audit_evidence.py uses: a defect's status is the last status word on its LATEST
# heading, and a heading with no status word conservatively keeps the previous one.
import sys as _sys_vocab
_sys_vocab.path.insert(0, __file__.rsplit("/", 2)[0])
from status_vocab import statuses_by_defect   # the ONE definition; see that file (pass 241)

SCOPE, USER, CAMP = "SCOPE-EXCLUDED", "USER-FACING", "CAMPAIGN-INTERNAL"

TABLE = {
    # --- outside this campaign by the defect's own words -------------------------------------
    "D2":  (SCOPE, "AF2 reads 0 of 184 parameters with a gradient; the entry says 'out of this "
                   "campaign's scope' and asks for its own row."),
    "D3":  (SCOPE, "RFdiffusion3's three rfd3_bias entry points return a tensor rather than an "
                   "optional; the entry says 'out of scope, recorded so it is not lost'."),
    "D123": (SCOPE, "UPSTREAM: their yaml generator leaves the custom-kernel flags on for EVAL "
                    "while disabling them for TRAIN, so their training test cannot start without "
                    "Triton. Not in our tree and not ours to fix."),
    "D124": (SCOPE, "UPSTREAM: 7kud_A.npz is in their own subset manifest and 404s on their S3, "
                    "and the sdist's train_pdb_subset.yaml is stale against its own generator."),

    # --- ships to users: inference -----------------------------------------------------------
    "D10": (USER, "The confidence head mis-ranks diffusion samples on the SHIPPED selector, "
                  "measured end to end through the production CLI on 1UBQ."),
    "D24": (USER, "On a single chain OpenFold3's ranking rule has two of its four terms "
                  "identically zero -- the shipped default, machine-checked in rank_rule.py."),

    # --- ships to users: training on the shipped default -------------------------------------
    "D30": (USER, "The diffusion module -- 89.2 % of the gradient mass -- agrees to 0.85 % on the "
                  "forward and is 9.3 % out on the gradient; 11.03x backward amplification after "
                  "the repair, re-measured from ONE harness at pass 222."),
    "D32": (USER, "Twenty-one sites in nine shipped modules route down a different, unfused path "
                  "while a tape is open, so a training step is a materially different execution."),
    "D55": (USER, "tt_bio's own tape gives precise_config() to the reductions feeding weight "
                  "gradients and withholds it from four sitting inside near-cancellations."),
    "D56": (USER, "A constant ~2,172x device arithmetic floor against torch fp32 on the worst "
                  "gradient component, which the entry itself calls a port gap."),
    "D58": (USER, "The ~20x backward-over-forward amplification is a property of the TAPE, "
                  "measured on two independent modules -- and the tape is shipped training code."),

    # --- this campaign's own measurement, instruments, references and bookkeeping -------------
    "D22": (CAMP, "Reference-bundle revision skew: our port is 0.4.3 and the bundle was built "
                  "with 0.5.0. A property of the reference, not of the port."),
    "D23": (CAMP, "The reference bundle runs the preview2 checkpoint on a version upstream "
                  "declares unsupported -- again the reference, not what we ship."),
    "D26": (CAMP, "Instrument A's reach denominator excludes `absent` tensors, so its reported "
                  "share is a share of what the instrument happened to see."),
    "D27": (CAMP, "Median and norm-share rank the same two arms in opposite directions -- a "
                  "choice-of-statistic defect in the scoring, not in the port."),
    "D28": (CAMP, "A18 fires: gradients were taken at forwards that disagree above the bar, which "
                  "voids those COMPARISONS. The forward disagreement itself is D19/D87."),
    "D35": (CAMP, "A single rel_l2 cannot say whether a gradient is too small, too big or "
                  "uncorrelated -- an identifiability defect in the metric."),
    "D37": (CAMP, "The measured error direction is the opposite of the one the campaign "
                  "pre-registered; a correction to our own registered claim."),
    "D42": (CAMP, "Both reference manifests name the wrong upstream revision through a hardcoded "
                  "getattr fallback in perf/of3t_reference/bundle_min.py:643 -- campaign code."),
    "D46": (CAMP, "'51.14 % of the gradient norm compared' is 89.211 % x 57.32 % and not a "
                  "sample -- a coverage-arithmetic defect."),
    "D48": (CAMP, "The 41-point coverage gap is diffusion_conditioning: which tensors the "
                  "campaign compared, not how the port computes them."),
    "D51": (CAMP, "Arms were allocated and results scored by section name and by median while "
                  "84.6 % of the mass sits in 1-D LayerNorm vectors -- a granularity defect."),
    "D53": (CAMP, "Error is concentrated on the mass-carrying tensors and the run kept no "
                  "per-tensor array, so the mass-weighted headline is underivable until a re-run."),
    "D59": (CAMP, "A shape-inferred transpose contaminated 48 entries of a campaign artifact -- "
                  "an instrument defect in the comparison, not in the model."),
    "D62": (CAMP, "Two determinations of block 8's cancellation ratio differ by 7,532x, one of "
                  "them interpolated off the curve it was explaining."),
    "D63": (CAMP, "RE-CLASSIFIED pass 222, see BOUNDARY. Its own heading says 'UNFIXED as a "
                  "REVIEWING CONVENTION': the measurement is complete and nothing in the shipped "
                  "models is wrong; what is wrong is estimating blast radius from a construction "
                  "count."),
    "D64": (CAMP, "Everything was measured against float64 and never against what OpenFold3's own "
                  "training precision achieves -- a choice of reference."),
    "D71": (CAMP, "The negative control the reference-precision result rests on is not a "
                  "permutation: 44 of its 45 entries replay. A control-validity defect."),
    "D73": (CAMP, "Nothing had ever measured our gradient against upstream's OWN gradient; "
                  "largely answered by of3t-refprec/of3t-wholemodel but not yet retired here."),
    "D78": (CAMP, "The shared denominator is an order-dependent sum carried on the record in two "
                  "spellings, and an instrument asserts one of them exactly."),
    "D82": (CAMP, "Zero of D72's three at-or-better scopes survive the direct test -- a "
                  "re-scoring of the campaign's own earlier re-scoring."),
    "D86": (CAMP, "A shape test that cannot fire on a square weight scored 87 tensors against "
                  "their own transposes. Instrument defect, plus a defect-numbering collision."),
    "D89": (CAMP, "One sub-figure in the arithmetic carrying the campaign's own retraction could "
                  "not be reproduced by the orchestrator."),
    "D91": (CAMP, "A withdrawal of a withdrawal, as pre-registered: the 'one module' refutation "
                  "is withdrawn and the trunk's gradient returns to unread."),
    "D92": (CAMP, "The revision-inertness audit covered the DiT path only; three scopes the "
                  "campaign relies on sat outside it."),
    "D93": (CAMP, "A second 0.4.3-vs-0.5.0 difference in the trunk's single track -- a property "
                  "of which upstream revision the comparison is against."),
    "D94": (CAMP, "D92's three unread diffs, read: msa_module inert, aux_heads' not. A finding "
                  "about the reference's coverage."),
    "D112": (CAMP, "A concluded row's worktree was pruned and took the captured 0.4.3 reference "
                   "tree with it, breaking 26 campaign scripts. Infrastructure."),
    "D118": (CAMP, "A triage record: seven open defects and the largest headline share one cause. "
                   "It is a set of follow-ups, and D120 already refuted two of its attributions."),
    "D119": (CAMP, "The observational floor built for the crop ladder does not test what it "
                   "claimed, and project.py still carries the unit error. Campaign tooling."),
    "D69": (CAMP, "Upstream's own single precision reproduces its float64 gradient to 8.107441e-05, "
                  "247x inside the bar, so the share our device failed is a port gap and not a bar "
                  "problem. A statement about this campaign's bar, never disposed of; it changes "
                  "nothing a user of the shipped tree gets."),
    "D1": (USER, "The trunk pair bias ships at 1/sqrt(24) = 0.204 of its intended value in every "
                 "OF3 fold served. The repair is written and HELD because applying it measured "
                 "0.149 A WORSE at rank 0, so what ships is a deviation with a repair in hand. "
                 "Whether to take the regression is Moritz's call, like D10 and D24."),
    "D136": (CAMP, "GO condition 3's headline was attributed to the shipped default and measured "
                   "on the repin arm, which the row recorded as default-off and unmerged. A "
                   "correction to this campaign's own record; of3t-trajwide is measuring the "
                   "post-fix shipped default now."),
    "D137": (CAMP, "The host float64 softmax is gated on a global env flag rather than on the tape "
                   "and its entry point accepts a raw inference tensor. Nothing ships -- no float64 "
                   "softmax symbol exists on main, asserted every compose -- so it changes nothing a "
                   "user gets today; it blocks land-standing for that path."),
    "D140": (CAMP, "Two of of3t-trajwide's arms -- norebind and zero, both CONTROLS -- died with "
                   "no error and no done marker while the row was between passes. A fleet/row "
                   "observability defect in this campaign's own execution, not something a user "
                   "of the shipped tree can reach."),
    "D141": (CAMP, "The shared diffusion capture records missing_keys and not unexpected_keys, so "
                   "a load that drops 24 trained tensors reads clean in every artifact derived "
                   "from it. An instrument-provenance defect in this campaign's own reference "
                   "chain; it changes nothing a user of the shipped tree gets."),
    "D151": (CAMP, "TT_BIO_SOFTMAX_BW_RENORM is read twice at module level and the two reads "
                   "reach different softmax backends. Both default off today, so nothing a user "
                   "runs differs; it becomes user-relevant only once the flag ships on, and the "
                   "guard fails the compose before that can happen silently."),
    "D148": (CAMP, "DIRECTIVE-STATUS's closing summary, stamped pass 199, was read at pass 269 "
                   "still saying D8/D9 were open and of3t-nanfloor owed a check that D111 UPDATE 3 "
                   "discharged inside pass 199 itself. A defect in this campaign's own record of "
                   "what it owes Moritz; no shipped behaviour depends on it."),
    "D49": (CAMP, "`fp32_softmax=False` improves gradient parity on five of seven trunk blocks "
                  "and the shipped default is the other way; by mass it is 2.5 %. A training-"
                  "gradient decision, not an inference output a user sees."),
    "D110": (CAMP, "The precise_config() softmax lever installs via setdefault and every diffusion "
                   "call site already passes a config, so it fires 1,440 times and cannot take "
                   "effect. A lever in this campaign's own instruments."),
    "D121": (CAMP, "A lever can be UNREACHED while the numbers MOVE, so the win gets credited to "
                   "the wrong lever; it asks for two counters where the instruments have one. A "
                   "measurement discipline for this campaign."),
    "D120": (CAMP, "0.4.3 and 0.5.0 are different FUNCTIONS at the diffusion boundary, not two "
                   "roundings of one, so a cross-version difference there measures a model change "
                   "and a precision change at once. A reading discipline for this campaign's own "
                   "figures; it changes nothing a user of the shipped tree gets."),
    "D122": (CAMP, "GO condition 5 is a keyword test on GAP prose and, read literally, is "
                   "unreachable while D2 and D3 stand. A defect in this campaign's own gate."),
    "D129": (USER, "conditioned_transition.layer_norm.layer_norm_s.weight reads 4.388x its own "
                   "bf16 floor and 3.10x A26's bar on 28 of 30 instances -- the first leaf to "
                   "survive the floor check that closed D8. Measured at the 0.5.0 boundary."),
    "D125": (CAMP, "Four more defects are declared closed inside another entry's body; three of "
                   "the four do not survive reading. A bookkeeping discipline, not a port defect."),
}

# Close calls, recorded with the argument on both sides. A triage that hides these is worth less
# than none, and the next reader is entitled to disagree with a named judgement rather than to
# discover an unnamed one.
BOUNDARY = {
    "D8":  "Could be read as CAMPAIGN-INTERNAL: pass 90 re-attributed most of it to D23, a "
           "reference defect. Kept USER-FACING because what SURVIVES the re-attribution is a "
           "gradient our port computes, and no measurement has yet shown that residue is zero.",
    "D37": "Could be read as USER-FACING: 'block 47's gradient is INFLATED by 1.18x' is a fact "
           "about our port. Kept CAMPAIGN-INTERNAL because the DEFECT as filed is that the "
           "campaign registered the wrong direction; the magnitude itself is D8's.",
    "D63": "MOVED from USER-FACING to CAMPAIGN-INTERNAL at pass 222, and the move is in the "
           "direction that flatters me, so the argument is given in full and can be reversed in "
           "one line. The USER-FACING test published here is 'changes what someone using the "
           "shipped tt-bio gets today -- an inference output, a crash, or a training run's result "
           "on the shipped default'. D63 changes none of those: its table is complete and "
           "CORRECT, Boltz-2 and RF3 come back byte-identical because they take the fused-SDPA "
           "branch, and the negative controls move (1.3207 A, 0.2005 A) so the instrument works. "
           "It is a true measured fact ABOUT the shipped tree, not a defect IN it -- its own "
           "heading says 'UNFIXED as a reviewing convention'. It stays UNFIXED either way, so the "
           "gate's keyword clause is unaffected; only the USER-FACING count moves, 12 -> 11. I "
           "looked for a defect to move the other way at the same time and did not find one; that "
           "absence is recorded rather than balanced by a manufactured move.",
    "D73": "Arguably already closed by of3t-refprec and of3t-wholemodel, which measured exactly "
           "what it says was never measured. Left UNFIXED and CAMPAIGN-INTERNAL here because "
           "closing a defect is a status edit in DEFECTS.md, not a side effect of a triage.",
    "D126": "RE-RE-CLASSIFIED and then CLOSED, pass 225, and the history is the point. Filed "
            "USER-FACING at 222; withdrawn by me at 223 on finding recipes.py:186 calls "
            "params.rebind(); RESTORED by of3t-rebind at 225, which showed rebind() is not the "
            "seam -- it maintains the MODEL SLOT while _PARAMS is keyed on id(raw), and "
            "modeltraj's own step log reads `rebound 26` beside `tape_resolves 0` on the same "
            "step, in the artifact I had already read. Three callers replace t.value and none "
            "re-keyed: optim.py:253 every step, checkpoint.py:70 every resume, autograd.py:180 "
            "on L1 eviction. Now FIXED at autograd.py:167-168 on the composition and LIVE on "
            "main, so it leaves this table as a closed defect rather than as a re-classified "
            "one. The lesson is not about ranking rules or tapes: I stopped checking once the "
            "library produced a satisfying answer.",

    "D28": "Could be read as USER-FACING: the forwards really do disagree. Kept CAMPAIGN-INTERNAL "
           "because the defect it FILES is that the gradient comparisons taken there are void; "
           "the forward disagreement is D19 (closed for the trunk) and D87 (refuted).",
}


def live_unfixed(text: str) -> list:
    # Was a local copy of the parse, uppercased, which read D3's "UNFIXED, out of scope, recorded
    # so it is not lost" as RECORDED the moment that word joined the vocabulary (pass 241).
    last = statuses_by_defect(text)
    return sorted((n for n, s in last.items() if s == "UNFIXED"), key=lambda d: int(d[1:]))


def main() -> int:
    text = DEFECTS.read_text(errors="replace")
    unfixed = live_unfixed(text)

    missing = [d for d in unfixed if d not in TABLE]
    extra = [d for d in TABLE if d not in unfixed]
    if missing or extra:
        print("TRIAGE IS STALE -- refusing to report a classification of a set that has moved.")
        if missing:
            print(f"  UNFIXED but unclassified: {', '.join(missing)}")
        if extra:
            print(f"  classified but no longer UNFIXED: {', '.join(sorted(extra, key=lambda d: int(d[1:])))}")
        return 1

    by = {SCOPE: [], USER: [], CAMP: []}
    for d in unfixed:
        by[TABLE[d][0]].append(d)

    result = {
        "generated_from": str(DEFECTS),
        "unfixed_total": len(unfixed),
        "counts": {k: len(v) for k, v in by.items()},
        "classes": {k: v for k, v in by.items()},
        "reasons": {d: {"class": TABLE[d][0], "why": TABLE[d][1]} for d in unfixed},
        "boundary_cases": BOUNDARY,
        "class_definitions": {
            SCOPE: "the defect's own entry records it as outside this campaign's scope; it "
                   "cannot be closed here",
            USER: "changes what someone using the shipped tt-bio gets today -- an inference "
                  "output, a crash, or a training run's result on the shipped default",
            CAMP: "in this campaign's own measurement, instruments, references, artifacts or "
                  "bookkeeping; fixing it changes what we know, not what anyone runs",
        },
        "gate_note": "workstreams/_of3t_donecheck.py::_charter_gate refuses GO when the word "
                     "'unfixed' appears in GAP, while audit_evidence.py requires GAP to name "
                     "every UNFIXED defect. Read literally, GO is unreachable while D2 and D3 "
                     "stand, and both are recorded as belonging to other campaigns. This file "
                     "supplies the number the gate's own message asks for and changes nothing.",
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")

    print(f"UNFIXED: {len(unfixed)}")
    for k in (SCOPE, USER, CAMP):
        print(f"  {k:18s} {len(by[k]):3d}   {', '.join(by[k])}")
    print(f"\nwritten {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
