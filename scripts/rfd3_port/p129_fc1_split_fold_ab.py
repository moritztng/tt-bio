#!/usr/bin/env python3
"""p129 -- the split `fc1` at the fold: the digest gate, then the A/B, then the no-site rung.

Region 2's lever, specified in `state/rfd3-fusion-programme.md` §13.11. `RFD3_FC1_SPLIT_SILU`
takes the silu out of `ttnn.linear(activation="silu")` -- bit-identical at all eight live keys,
p128 -- so `fc1` can take a pinned program config like its two siblings and its output can join
`b` and `m` in L1.

Drives `scripts/rfd3_port/fold_ab.py`, so this file holds only what is specific to the lever.
Three invocations, in the order §13.11 pre-committed:

    # 1. the digest gate -- 3 timesteps, off/on/off/on
    p129_fc1_split_fold_ab.py perf/p129/digest_R3.json 3 off,on,off,on R3

    # 2. the no-site rung -- 418 tokens, below the 512-token residency gate, so the chunked
    #    branch never runs, the lever has no site and the digest must not move
    p129_fc1_split_fold_ab.py perf/p129/nosite_R2.json 3 off,on,off,on R2 --expect-no-site

    # 3. the A/B -- 200 timesteps, five folds, A/A control first
    p129_fc1_split_fold_ab.py perf/p129/ab_R3.json 200 off,off,on,off,on R3

The lever declines three ways and all three are in the census, because an `on` arm that declined
every call is an A/A wearing an A/B's label: no bit-exact config (or a default under
`_TUNE_MIN_MS`), the chunk-height guard keeping `fc1`'s output in DRAM, or no chunked site at all.

`chunk_h` is reported per arm and per site because h is the one thing that could change shape
between arms, and §15.2 folded what that costs: at a moved height the CIF digest moves, at any
height, in both arms, with the lever's served count at zero. So the guard admits the split only
at an EQUAL height and this script's job is to show the height did not move rather than to
tolerate it moving.

On pc card 0 every number is PROVISIONAL-ON-PC-CARD0 and is never pooled with a qb1/qb2/H200
denominator (`pc-card0-512aa-fold-nondeterminism`).
"""
import json
import os
import pathlib
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tt_bio.rfd3 import model as rfd3_model                              # noqa: E402
from fold_ab import fold_ab                                             # noqa: E402

# Cost-model ESTIMATE, not a measurement, and derived per fixture from the call count the fold
# actually makes rather than from a per-step figure. At R3 the census counts 12768 served body
# chunks and p128 measured 0.390 ms saved per chunk at hidden=512 and 0.475 at hidden=256, so the
# isolated prize is 11.04 s/design; §13.3's realisation band of 0.53 of an isolated screen puts the
# fold at -5.9 s. The block-sparse 10x overestimate is why this stays labelled an estimate.
#
# The first run of this script predicted -3.1 s and measured -6.4, which was NOT the model beating
# its own cost model: -3.1 came from scaling §13.9's per-step figure, and that figure assumes four
# fc1 calls per step at each hidden width where the fold makes eight (199 steps x 8 + 4 from the
# one-time initializer = the 1596 calls per hidden the census reports). Same trap as
# `perf-page-matched-batch-protocol-recurrence`: the estimate and the measurement counted
# different amounts of traffic.
PREDICTED = {"R2": 0.0, "R3": -5.9, "R4": -8.9}

FIXTURES = pathlib.Path("perf/dsfix/fixtures")


def chunk_plan(tokens, hidden):
    """What `Transition.__call__` will decide at this size, from the shipped function."""
    w_pad = -(-tokens // 32) * 32
    h2 = rfd3_model._pair_transition_chunk_h(w_pad, hidden, tokens)
    h3 = rfd3_model._pair_transition_chunk_h(w_pad, hidden, tokens, residents=3)
    split = h3 == h2
    return {"tokens": tokens, "hidden": hidden, "w_pad": w_pad,
            "chunked": w_pad >= rfd3_model._PAIR_TRANSITION_MIN_W,
            "chunk_h_off": h2, "chunk_h_on": h2, "chunk_h_third_resident": h3,
            "n_chunks_off": -(-tokens // h2), "n_chunks_on": -(-tokens // h2),
            "fc1_output_in_l1": split,
            # Under the height guard the served height never moves, so this records what the
            # third resident WOULD have cost -- the decline reason, not a served shape.
            "chunk_count_kept": -(-tokens // h3) == -(-tokens // h2) and not split}


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    expect_no_site = "--expect-no-site" in sys.argv
    # `--l1-bytes=N` shrinks `_PAIR_TRANSITION_L1_BYTES` for this run only. It is the SECOND knob
    # on the height-parity predicate: the guard compares the chunk height at two and three
    # residents, and that comparison moves with the L1 budget exactly as it moves with the token
    # count. So a host that cannot fold a >693-token fixture can still fold both size regimes the
    # guard distinguishes (§10.1: pc tops out near a 5900-wide atom axis, R4 needs 6080).
    # At R3 (514 tokens, w_pad 544) the two rungs the shipped 138 MB budget cannot reach are:
    #   100_000_000 -- the third resident wants h 64 -> 59 at hidden=512 at 9 chunks either way.
    #                  The chunk-COUNT guard served this and moved the digest; the height guard
    #                  declines it. This is the negative control for X1, and the same class as
    #                  the census fixture's own 64 -> 63 at 685 tokens.
    #   90_000_000  -- hidden=512 declines on either guard (h 64 -> 53, 9 -> 10 chunks),
    #                  hidden=256 still serves. Same class as every size from 694 tokens up.
    # In both, the OFF arm's h stays 64 at both widths, so the off path is byte-identical to the
    # shipped one and its digest must still be the shipped digest.
    l1_bytes = None
    for a in sys.argv[1:]:
        if a.startswith("--l1-bytes="):
            l1_bytes = int(a.split("=", 1)[1])
            rfd3_model._PAIR_TRANSITION_L1_BYTES = l1_bytes
            print("[p129] _PAIR_TRANSITION_L1_BYTES = %d (shipped %d) -- moves the parity "
                  "predicate, NOT the lever" % (l1_bytes, 138_000_000))
    out = argv[0] if len(argv) > 0 else "perf/p129/fc1_split_fold_ab.json"
    steps = int(argv[1]) if len(argv) > 1 else 200
    arms = (argv[2] if len(argv) > 2 else "off,off,on,off,on").split(",")
    rung = argv[3] if len(argv) > 3 else "R3"

    spec = json.loads((FIXTURES / ("rfd3_%s.json" % rung)).read_text())
    tokens = sum(int(p.split("-")[-1]) if "-" in p else int(p)
                 for p in list(spec.values())[0]["contig"].split(","))
    plans = {h: chunk_plan(tokens, h) for h in (512, 256)}
    print("[p129] %s: %d tokens, predicted site plan (arithmetic, not a measurement):" % (rung, tokens))
    for h, p in plans.items():
        print("   hidden=%-4d chunked=%-5s h %d, third resident wants %d, chunks %d, "
              "fc1 in L1 %s%s"
              % (h, p["chunked"], p["chunk_h_off"], p["chunk_h_third_resident"],
                 p["n_chunks_off"], p["fc1_output_in_l1"] and p["chunked"],
                 "  (equal count, MOVED height -- declines)" if p["chunk_count_kept"] else ""))

    rec = fold_ab(flag="RFD3_FC1_SPLIT_SILU",
                  set_enabled=rfd3_model.set_fc1_split_enabled,
                  counter=rfd3_model.FC1STATS,
                  fixture=FIXTURES / ("rfd3_%s.json" % rung),
                  out=out, steps=steps, arms=arms, tag="p129_%s" % rung,
                  predicted_delta_s=PREDICTED.get(rung),
                  extra={"rung": rung, "tokens": tokens, "site_plan": plans,
                         "provisional_on": "pc-card0", "expect_no_site": expect_no_site,
                         "l1_bytes": l1_bytes or rfd3_model._PAIR_TRANSITION_L1_BYTES})

    served, declines = dict(rfd3_model.FC1SERVED), dict(rfd3_model.FC1DECLINES)
    print("\nfc1 census -- served (pinned config AND L1 output): %s" % (served or "{} (none)"))
    print("fc1 census -- declined, with the reason: %s" % (declines or "{} (none)"))
    print("tune_matmul active at this size: %s (floor %d atoms)"
          % (rfd3_model._TUNE_MATMUL, rfd3_model._TUNE_MATMUL_MIN_ATOMS))
    rec["fc1_served"] = served
    rec["fc1_declines"] = declines
    rec["tune_matmul"] = bool(rfd3_model._TUNE_MATMUL)
    pathlib.Path(out).write_text(json.dumps(rec, indent=2) + "\n")

    if expect_no_site:
        # The verdict here is provenance plus an unchanged digest, not a delta: below the
        # 512-token gate the chunked branch never runs, so the split must never be reached.
        ok = rec["bit_exact"] and not served and not declines and rfd3_model.FC1STATS[0] == 0
        print("no-site rung: %d split calls, digests %s  ->  %s"
              % (rfd3_model.FC1STATS[0], "unchanged" if rec["bit_exact"] else "MOVED",
                 "NO SITE, AS PREDICTED" if ok else "UNEXPECTED -- read the census"))
        return 0 if ok else 2

    if not rec["bit_exact"]:
        print("\nDIGEST MOVED -- failed lever, not a tolerance question")
        return 2
    if not rec["arms_distinct"]:
        print("\nARMS NOT DISTINCT -- the on arm never took the split, so this is an A/A")
        return 3
    if not served:
        print("\nSPLIT RAN BUT NOTHING SERVED -- every call declined, so any delta is the split's "
              "extra DRAM round trip and not the lever")
        return 4

    # The census has to agree with the arithmetic PER HIDDEN WIDTH, not in total. A rung where one
    # width serves and the other declines reads as a healthy A/B on the totals alone, which is the
    # `gate-fixture-existence-vs-content-inversion` shape: check the content.
    ok = True
    for h, pl in plans.items():
        want = pl["chunked"] and pl["fc1_output_in_l1"]
        got = any(("hidden=%d" % h) in k for k in served)
        rise = any(("hidden=%d chunk-height-would-move" % h) in k for k in declines)
        print("hidden=%-4d predicted served=%-5s  observed served=%-5s  parity-decline logged=%s"
              % (h, want, got, rise))
        if want != got or (not want and pl["chunked"] and not rise):
            ok = False
    print("per-width census agrees with the arithmetic: %s" % ok)
    return 0 if ok else 5


if __name__ == "__main__":
    sys.exit(main())
