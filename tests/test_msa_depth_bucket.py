"""The MSA row bucket is a pure subtraction from the padding, never an addition.

No device, no weights. The row axis is bucketed for the same recompilation reason the token axis
is, and it tiles the same way, so it inherits the same two obligations: every bucket is a multiple
of 32, and the map from a real depth to its bucket is a pure function of that depth.

It carries a third, which is specific to this axis and is the one that failed. The ladder is
OPTIONAL -- an env flag over a shipped single 1024 bucket -- so it is only ever judged on the
accuracy its shorter reduction costs. That judgement is only valid if turning it on cannot make a
fold slower. A ladder that doubled past 1024 broke it: 2500 rows landed on 4096 against an
incumbent 3072, and 7904 of the first 12000 row counts padded MORE with the flag on than off, by
up to 7168 rows -- 11.42 s per fold on Wormhole at the measured 0.0995 ms per padded row per
MSALayer over 16 calls (perf/b2z2_msa/msa_depth_cost_512_whglx_c7.json).
"""
import sys

from tt_bio import token_axis as TA

CEILING = 20001          # past the 8192 deep-MSA default and its next 1024 step


def _fail(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    return ok


def _buckets(on):
    import os
    old = os.environ.get("TT_BIO_MSA_DEPTH_LADDER")
    if on:
        os.environ["TT_BIO_MSA_DEPTH_LADDER"] = "1"
    else:
        os.environ.pop("TT_BIO_MSA_DEPTH_LADDER", None)
    try:
        return [TA.msa_depth_bucket(n) for n in range(1, CEILING)]
    finally:
        if old is None:
            os.environ.pop("TT_BIO_MSA_DEPTH_LADDER", None)
        else:
            os.environ["TT_BIO_MSA_DEPTH_LADDER"] = old


def check_off_is_the_incumbent():
    off = _buckets(False)
    bad = [n for n, b in enumerate(off, 1) if b != TA.bucketed_width(n, TA.MSA_AXIS_MULTIPLE[2])]
    return _fail(not bad, "flag off reproduces the single %d bucket exactly"
                 % TA.MSA_AXIS_MULTIPLE[2] + ("" if not bad else "; first bad row %d" % bad[0]))


def check_ladder_never_pads_more_than_the_incumbent():
    off, on = _buckets(False), _buckets(True)
    bad = [n for n, (a, b) in enumerate(zip(off, on), 1) if b > a]
    return _fail(not bad, "the ladder never pads above the incumbent"
                 + ("" if not bad else "; %d row counts do, worst +%d rows at %d"
                    % (len(bad), max(on[n - 1] - off[n - 1] for n in bad), bad[0])))


def check_every_bucket_holds_its_rows():
    on = _buckets(True)
    bad = [n for n, b in enumerate(on, 1) if b < n]
    return _fail(not bad, "no bucket is smaller than the alignment it has to hold"
                 + ("" if not bad else "; first bad row %d" % bad[0]))


def check_every_bucket_is_a_row_tile_multiple():
    on = _buckets(True)
    bad = [n for n, b in enumerate(on, 1) if b % 32]
    return _fail(not bad, "every bucket is a multiple of the 32 row tile"
                 + ("" if not bad else "; first bad row %d -> %d" % (bad[0], on[bad[0] - 1])))


def check_the_bucket_is_a_pure_function_of_the_depth():
    a, b = _buckets(True), _buckets(True)
    return _fail(a == b, "two reads of the same depth give the same bucket, so two folds of one "
                         "alignment run one program")


def check_the_ladder_is_monotonic():
    on = _buckets(True)
    bad = [n for n in range(1, len(on)) if on[n] < on[n - 1]]
    return _fail(not bad, "a deeper alignment never gets a shallower bucket"
                 + ("" if not bad else "; first inversion at row %d" % (bad[0] + 1)))


def check_the_shallow_end_actually_moves():
    """The point of the flag. 35 rows is the perf fixture; 1 row is every --single_sequence fold."""
    on = _buckets(True)
    return _fail(on[0] == 64 and on[34] == 64,
                 "a 1-row and a 35-row alignment both land on 64, not 1024 "
                 "(got %d and %d)" % (on[0], on[34]))


CHECKS = (
    check_off_is_the_incumbent,
    check_ladder_never_pads_more_than_the_incumbent,
    check_every_bucket_holds_its_rows,
    check_every_bucket_is_a_row_tile_multiple,
    check_the_bucket_is_a_pure_function_of_the_depth,
    check_the_ladder_is_monotonic,
    check_the_shallow_end_actually_moves,
)


def run_checks() -> bool:
    ok = True
    for c in CHECKS:
        ok = c() and ok
    off, on = _buckets(False), _buckets(True)
    saved = sum(a - b for a, b in zip(off, on))
    print("\nrows 1-%d: the ladder removes %d padded rows in total and adds none"
          % (CEILING - 1, saved))
    print("ALL OK" if ok else "FAILURES")
    return ok


def test_msa_depth_bucket():
    assert run_checks(), "a check above printed FAIL"


if __name__ == "__main__":
    sys.exit(0 if run_checks() else 1)
