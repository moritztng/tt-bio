#!/usr/bin/env python3
"""p133 -- both funded levers on in the same fold, which is the programme's standing prize.

Region 1 (`RFD3_SOFTMAX_PV_FUSED`, fused softmax+PV at the atom sites) was measured with region 2
off, and region 2 (`RFD3_FC1_SPLIT_SILU`, the split `fc1` on the pair Transition) with region 1
off. `state/rfd3-fusion-programme.md` §14.2's "the two levers do not interact" is an argument
about digests, not about time: they share no op, but they share the card, the dispatch queue and
the DRAM bus, so whether -1.255 and -6.363 add is a measurement (§15.6 X3).

Drives `scripts/rfd3_port/fold_ab.py` unchanged, so this file holds only the composition:

    p133_both_levers_fold_ab.py perf/p132/both_R3.json 200 off,off,on,off,on R3
    p133_both_levers_fold_ab.py perf/p132/both_decline_R3.json 200 off,off,on,off,on R3 \
        --l1-bytes=90000000        # the decline rung, where region 2 serves hidden=256 only

`--only=pv|fc1` runs one lever through the same script, for a same-process comparison against
the composed arm rather than against an older run's median.

The harness reads provenance from one counter, so the composed arm hands it one that returns the
SUM of the two levers' own counters. That is deliberately not a third counter in the harness: a
counter the harness maintains could report calls neither lever made.

On pc card 0 every number is PROVISIONAL-ON-PC-CARD0 and is never pooled with a qb1/qb2/H200
denominator (`pc-card0-512aa-fold-nondeterminism`).
"""
import json
import os
import pathlib
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tt_bio import softmax_generic                                        # noqa: E402
from tt_bio.rfd3 import model as rfd3_model                               # noqa: E402
from fold_ab import fold_ab                                               # noqa: E402

# ESTIMATE, not a measurement: the two levers' own fold measurements at this fixture and card,
# added. -1.255 to -1.322 (region 1, perf/p126/ab_R3*.json) and -6.363 (region 2,
# perf/p129/ab_R3.json, and -6.243 on its first run). Additivity is the hypothesis under test, so
# this number is what the run is allowed to refute.
# Keyed on (rung, l1_bytes) because the composed prize is not one number: at the shipped budget
# region 2 serves both hidden widths and X3 measured -7.634; at 90 MB hidden=512 declines on the
# height guard and region 2 pays only its hidden=256 half, so the composed prediction is region 1's
# -1.255..-1.322 plus X4's -3.411. `None` means "whatever `_PAIR_TRANSITION_L1_BYTES` ships as".
PREDICTED = {("R3", None): -7.65, ("R3", 90_000_000): -4.70}

FIXTURES = pathlib.Path("perf/dsfix/fixtures")

LEVERS = {"pv": (softmax_generic.set_pv_enabled, softmax_generic.PVSTATS),
          "fc1": (rfd3_model.set_fc1_split_enabled, rfd3_model.FC1STATS)}


class SumCounter:
    """`counter[0]` over several levers' own counters. Read-only, and it owns no state."""

    def __init__(self, counters):
        self._c = counters

    def __getitem__(self, i):
        return sum(c[i] for c in self._c)


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    only = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--only=")), None)
    names = [only] if only else list(LEVERS)
    assert all(n in LEVERS for n in names), "unknown lever: %s" % names

    # `--l1-bytes=N` is p129's knob, verbatim: it shrinks `_PAIR_TRANSITION_L1_BYTES` for this run
    # only and so moves region 2's height-parity predicate, NOT either lever. Without it this script
    # silently ran the shipped 138 MB budget whatever the caller asked for, which would have
    # relabelled X3's rung as the decline rung.
    l1_bytes = None
    for a in sys.argv[1:]:
        if a.startswith("--l1-bytes="):
            l1_bytes = int(a.split("=", 1)[1])
            rfd3_model._PAIR_TRANSITION_L1_BYTES = l1_bytes
            print("[p133] _PAIR_TRANSITION_L1_BYTES = %d (shipped %d) -- moves the parity "
                  "predicate, NOT the levers" % (l1_bytes, 138_000_000))
    out = argv[0] if len(argv) > 0 else "perf/p132/both_R3.json"
    steps = int(argv[1]) if len(argv) > 1 else 200
    arms = (argv[2] if len(argv) > 2 else "off,off,on,off,on").split(",")
    rung = argv[3] if len(argv) > 3 else "R3"

    setters = [LEVERS[n][0] for n in names]

    def set_enabled(on):
        return [s(on) for s in setters][0]

    print("[p133] levers on in the `on` arm: %s" % ", ".join(names))
    rec = fold_ab(flag="+".join(names),
                  set_enabled=set_enabled,
                  counter=SumCounter([LEVERS[n][1] for n in names]),
                  fixture=FIXTURES / ("rfd3_%s.json" % rung),
                  out=out, steps=steps, arms=arms, tag="p133_%s" % rung,
                  predicted_delta_s=PREDICTED.get((rung, l1_bytes)) if not only else None,
                  extra={"rung": rung, "levers": names, "provisional_on": "pc-card0",
                         "l1_bytes": l1_bytes or rfd3_model._PAIR_TRANSITION_L1_BYTES})

    # Per-lever provenance, because a composed arm that served only one lever's calls is that
    # lever's A/B under a composed label.
    per = {n: LEVERS[n][1][0] for n in names}
    print("\nper-lever served calls over the whole run: %s" % per)
    print("region 1 census, served by padded key width: %s" % dict(softmax_generic.PVSERVED))
    print("region 2 census, served: %s" % dict(rfd3_model.FC1SERVED))
    print("region 2 census, declined: %s" % dict(rfd3_model.FC1DECLINES))
    rec["served_per_lever"] = per
    rec["pv_served_widths"] = dict(softmax_generic.PVSERVED)
    rec["fc1_served"] = dict(rfd3_model.FC1SERVED)
    rec["fc1_declines"] = dict(rfd3_model.FC1DECLINES)
    pathlib.Path(out).write_text(json.dumps(rec, indent=2) + "\n")

    if not rec["bit_exact"]:
        print("\nDIGEST MOVED -- failed lever, not a tolerance question")
        return 2
    if not rec["arms_distinct"]:
        print("\nARMS NOT DISTINCT -- no lever served, so this is an A/A")
        return 3
    if any(v == 0 for v in per.values()):
        print("\nA LEVER SERVED NOTHING -- the composed delta belongs to the other one")
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
