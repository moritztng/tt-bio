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

    # 3. the decline rung -- a design size whose padded atom axis falls in the half of the axis
    #    L5b cannot be exact on. A pass that only runs an addressable rung never exercises the
    #    decline path. The verdict is `zero fused calls at the atom key width` plus that width
    #    appearing in the decline census, plus an unchanged digest -- not a delta.
    p126_l5b_fold_ab.py perf/p126/decline_D1.json 3 off,on,off,on D1 --expect-decline

    Which rung declines is a MEASURED property of the fixture, not a property of its name: the
    exactness ladder in the state doc is keyed by synthetic key width, and a fixture named after
    one of those rungs need not fold to it (root-caused on the `R2` fixture, §12.1 -- it served
    the fused path, because its atom axis is addressable, and the hardcoded
    `expect_decline = rung == "R2"` then reported a broken decline path that was not broken).

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
PREDICTED = {"R2": -0.05, "R3": -3.32, "R4": -3.45}

FIXTURES = pathlib.Path("perf/dsfix/fixtures")


def main():
    argv = [a for a in sys.argv[1:] if a != "--expect-decline"]
    expect_decline = "--expect-decline" in sys.argv
    out = argv[0] if len(argv) > 0 else "perf/p126/l5b_fold_ab.json"
    steps = int(argv[1]) if len(argv) > 1 else 200
    arms = (argv[2] if len(argv) > 2 else "off,off,on,off,on").split(",")
    rung = argv[3] if len(argv) > 3 else "R3"

    rec = fold_ab(flag="RFD3_SOFTMAX_PV_FUSED",
                  set_enabled=softmax_generic.set_pv_enabled,
                  counter=softmax_generic.PVSTATS,
                  fixture=FIXTURES / ("rfd3_%s.json" % rung),
                  out=out, steps=steps, arms=arms, tag="p126_%s" % rung,
                  predicted_delta_s=PREDICTED.get(rung),
                  extra={"rung": rung, "provisional_on": "pc-card0",
                         "expect_decline": expect_decline})

    # BOTH halves of the census, keyed by padded key width, because a decline count alone cannot
    # say which site of the fold the lever actually served.
    declines = dict(softmax_generic.PVDECLINES)
    served_widths = dict(softmax_generic.PVSERVED)
    if declines:
        print("\ndecline reasons, keyed by padded key width: %s" % declines)
    print("served, keyed by padded key width: %s" % (served_widths or "{} (nothing served)"))
    rec["decline_reasons"] = declines
    rec["served_widths"] = served_widths

    # The atom-attention site is the widest key the fold presents; the token sites are far narrower.
    widths = [int(k) for k in served_widths]
    widths += [int(k.rsplit(" ", 1)[-1]) for k in declines if k.rsplit(" ", 1)[-1].isdigit()]
    atom_key = max(widths) if widths else None
    rec["atom_key"] = atom_key
    served_at_atom = served_widths.get(atom_key, 0)
    declined_at_atom = sum(c for k, c in declines.items()
                           if k.rsplit(" ", 1)[-1] == str(atom_key))
    print("atom site: padded key %s  ->  served %d, declined %d"
          % (atom_key, served_at_atom, declined_at_atom))
    rec["served_at_atom"] = served_at_atom
    rec["declined_at_atom"] = declined_at_atom
    pathlib.Path(out).write_text(__import__("json").dumps(rec, indent=2) + "\n")

    if expect_decline:
        # The decline rung's verdict is provenance plus an unchanged digest, not a delta: the
        # lever must refuse the atom site outright and the fold must land on the same structure.
        ok = served_at_atom == 0 and declined_at_atom > 0 and rec["bit_exact"]
        print("decline rung: atom site served %d / declined %d, digests %s  ->  %s"
              % (served_at_atom, declined_at_atom,
                 "unchanged" if rec["bit_exact"] else "MOVED",
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
