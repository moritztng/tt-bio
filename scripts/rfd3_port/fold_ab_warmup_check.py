#!/usr/bin/env python3
"""Host-only check of `fold_ab`s warmup protocol. No device, no fold, no card time.

The harness discarded ONE warmup fold and ran it in the OFF configuration, so the off arm entered
its first measured rep warm and the on arm paid the fused paths first contact inside a measured
rep. Both runs of the RFD3 composed R4 fold showed it, by 0.667 s and by 2.55 s, the second larger
than the margin the decision missed by. `fold_ab` is shared by 13 drivers, so the fix is checked
here rather than trusted: `_fold` is stubbed, so this runs in seconds and asserts the loop, not a
number.

    PYTHONPATH=$PWD python3 scripts/rfd3_port/fold_ab_warmup_check.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fold_ab as F                                                       # noqa: E402

FIXTURE = pathlib.Path("/tmp/fold_ab_warmup_check_fixture.json")


def run(arms, warm_digest="D0"):
    """`arms` through the harness with a stubbed fold. Returns (record, [(out_dir, on)])."""
    FIXTURE.write_text(json.dumps({"case": {"input": "unused, the fold is stubbed"}}))
    seq, state, n, counter = [], {"on": False}, {"i": 0}, [0]

    def stub(specs, out_dir, steps, seed, weights):
        n["i"] += 1
        if state["on"]:
            counter[0] += 10569
        seq.append((out_dir, state["on"]))
        return 90.0 + n["i"], warm_digest if "_warm_" in out_dir else "D0", 1

    F._fold = stub
    rec = F.fold_ab(flag="check", set_enabled=lambda on: state.__setitem__("on", bool(on)),
                    counter=counter, fixture=FIXTURE, out="/tmp/fold_ab_warmup_check.json",
                    steps=200, arms=arms, tag="check")
    return rec, seq


def main():
    rec, seq = run(["off", "off", "on", "off", "on", "off", "on"])
    assert [w["arm"] for w in rec["warmups"]] == ["off", "on"], rec["warmups"]
    assert rec["warmups"][0]["served_calls"] == 0, "the off warmup must serve nothing"
    assert rec["warmups"][1]["served_calls"] == 10569, "the on warmup must serve, or it warms nothing"
    assert [s[0] for s in seq][:2] == ["/tmp/check_warm_off", "/tmp/check_warm_on"]
    assert sum(1 for s in seq if s[1]) == 4, "one on warmup plus three measured on reps"
    assert rec["discarded_warmup_s"] == rec["warmups"][0]["sampler_s"], "key keeps its meaning"
    assert len(rec["rows"]) == 7 and rec["arms_distinct"] and rec["aa_spread_s"] is not None
    assert rec["bit_exact"] and rec["warmup_digests_match"]

    # An all-`off` arm list runs exactly one warmup, so every artifact written before the fix is
    # unchanged in protocol.
    off_only, seq2 = run(["off", "off", "off"])
    assert [w["arm"] for w in off_only["warmups"]] == ["off"] and len(seq2) == 4

    # Negative control: break the digest of the warmups ONLY. `bit_exact` reads measured reps and
    # must stay true; the new field is the one that has to go false, or it reads nothing.
    diverged, _ = run(["off", "off", "on", "off", "on", "off", "on"], warm_digest="D1")
    assert diverged["bit_exact"] is True, "bit_exact must keep its old meaning"
    assert diverged["warmup_digests_match"] is False, "the warmup gate did not read the digest"

    print("\nfold_ab warmup protocol OK: one discarded warmup per distinct arm, the on warmup "
          "serves, an all-off list is unchanged, and the warmup digest gate has a negative control")


if __name__ == "__main__":
    sys.exit(main())
