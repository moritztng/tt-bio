#!/usr/bin/env python3
"""p126 -- L5b at the fold: the digest gate, then the A/B, then the decline rung.

Drives `scripts/rfd3_port/fold_ab.py`, the harness extracted from p68/p73/p75/p85/p91, so this
file holds only what is specific to L5b: which flag, which fixture, and what the cost model
predicted. `RFD3_SOFTMAX_PV_FUSED` is default OFF; the harness flips it in-process so both arms
are one lease and one card-warmth trajectory.

Three invocations, in the order `state/rfd3-fusion-programme.md` §4 pre-committed:

    # 1. the digest gate -- 3 timesteps, off/on/off/on. Costs two minutes and it is the check that
    #    caught the wide pair-Transition variant AFTER that variant had passed torch.equal.
    p126_l5b_fold_ab.py perf/p126/digest_R3.json 3 off,on,off,on R3

    # 2. the A/B -- 200 timesteps, five folds, A/A control first (the two leading off reps)
    p126_l5b_fold_ab.py perf/p126/ab_R3.json 200 off,off,on,off,on R3

    # 3. the decline rung -- R2 is `blk=4` against `in0_block_w=2`, so the lever MUST decline.
    #    A pass that only runs an addressable rung never exercises the decline path. Here the
    #    verdict is `served == 0 in the on arm` and an unchanged digest, not a delta.
    p126_l5b_fold_ab.py perf/p126/decline_R2.json 3 off,on,off,on R2

On pc card 0 every number is PROVISIONAL-ON-PC-CARD0 and is never pooled with a qb1/qb2/H200
denominator (`pc-card0-512aa-fold-nondeterminism`).
"""
import os
import pathlib
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tt_bio import softmax_generic                                       # noqa: E402
from fold_ab import fold_ab                                              # noqa: E402

# Cost-model ESTIMATE, not a measurement. §10.4: 22.1 ms/step of deletable traffic at the R3
# padded axis of 4576, and the lineage's central exchange rate is 75 %, so 16.6 ms/step over 200
# steps. The block-sparse 10x overestimate (§"Traps") is why this stays labelled an estimate until
# the A/B answers.
PREDICTED = {"R2": 0.0, "R3": -3.32, "R4": -3.45}

FIXTURES = pathlib.Path("perf/dsfix/fixtures")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "perf/p126/l5b_fold_ab.json"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    arms = (sys.argv[3] if len(sys.argv) > 3 else "off,off,on,off,on").split(",")
    rung = sys.argv[4] if len(sys.argv) > 4 else "R3"

    rec = fold_ab(flag="RFD3_SOFTMAX_PV_FUSED",
                  set_enabled=softmax_generic.set_pv_enabled,
                  counter=softmax_generic.PVSTATS,
                  fixture=FIXTURES / ("rfd3_%s.json" % rung),
                  out=out, steps=steps, arms=arms, tag="p126_%s" % rung,
                  predicted_delta_s=PREDICTED.get(rung),
                  extra={"rung": rung, "provisional_on": "pc-card0",
                         "expect_decline": rung == "R2"})

    # Why the lever declined, whenever it did, so a decline is evidence rather than a silence.
    declines = dict(softmax_generic.PVDECLINES)
    declines.pop("lever off", None)          # every off arm and the warmup contributes this one
    if declines:
        print("\ndecline reasons (excluding the off arms): %s" % declines)
    rec["decline_reasons"] = declines
    pathlib.Path(out).write_text(__import__("json").dumps(rec, indent=2) + "\n")

    if rec["rung"] == "R2":
        # The decline rung's verdict is provenance plus an unchanged digest, not a delta.
        served = [r["served_calls"] for r in rec["rows"] if r["arm"] == "on"]
        ok = bool(served) and max(served) == 0 and rec["bit_exact"] and bool(declines)
        print("decline rung: on arm served %s fused calls, digests %s  ->  %s"
              % (served, "unchanged" if rec["bit_exact"] else "MOVED",
                 "DECLINED CORRECTLY" if ok else "DECLINE PATH BROKEN"))
        return 0 if ok else 2

    if not rec["bit_exact"]:
        print("\nDIGEST MOVED -- failed lever, not a tolerance question")
        return 2
    if not rec["arms_distinct"]:
        print("\nARMS NOT DISTINCT -- the on arm did not serve the fused path, so this is an A/A")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
