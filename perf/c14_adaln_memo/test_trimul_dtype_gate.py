#!/usr/bin/env python3
"""`trimul_tail.eligible`'s dtype clauses as a truth table, on the CPU with no device.

Three of the four hand-transcribed fast paths route their dtype gate through
`mm_generic.fast_dtypes_ok`, which requires every operand to be the SAME dtype. `trimul_tail`
cannot use it: it takes an activation pair and a weight pair and deliberately allows the two pairs
to differ. So it needs the narrower rule -- uniformity is required only once bfp8 is in the set --
and this pins that rule against both the OLD behaviour and the alternative of reusing the helper.

The point of the table is the two columns on the right. `today` must be identical to `main` for
every row that the fold can actually issue, and `helper` must differ, because reusing the helper
here would decline mixed bf16/fp32 pairs that production serves.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "tt_bio" / "trimul_tail.py"

BF16, BFP8, FP32 = "bf16", "bfp8", "fp32"
FAST = {BF16, BFP8}


def main_rule(xa, xb, wa, wb):
    """`origin/main` today: membership only, then the pair tests."""
    if not (FAST & {xa, xb, wa, wb}):
        return "dtype"
    if xa != xb or wa != wb:
        return "dtype_pair"
    return None


def new_rule(xa, xb, wa, wb):
    """This change: membership, then uniformity ONLY when bfp8 is present, then the pair tests."""
    seen = {xa, xb, wa, wb}
    if not (FAST & seen):
        return "dtype"
    if BFP8 in seen and len(seen) != 1:
        return "dtype_mixed_bfp8"
    if xa != xb or wa != wb:
        return "dtype_pair"
    return None


def helper_rule(xa, xb, wa, wb):
    """The rejected alternative: reuse `fast_dtypes_ok` over all four operands."""
    seen = {xa, xb, wa, wb}
    return None if (len(seen) == 1 and seen <= FAST) else "dtype"


# (row, reachable-on-the-default-path-today)
CASES = [
    ((BF16, BF16, BF16, BF16), True),    # what the 512 aa fold actually issues
    ((BF16, BF16, FP32, FP32), True),    # mixed pairs, no bfp8: deliberate and pre-existing
    ((FP32, FP32, BF16, BF16), True),
    ((FP32, FP32, FP32, FP32), False),   # no fast dtype at all
    ((BFP8, BFP8, BFP8, BFP8), False),   # uniform bfp8: measured 0.0283 vs 0.0267, fine
    ((BFP8, BFP8, BF16, BF16), False),   # mixed with bfp8: the 12.55 rel_rms failure mode
    ((BFP8, BFP8, FP32, FP32), False),
    ((BF16, BF16, BFP8, BFP8), False),
    ((BF16, BFP8, BF16, BF16), False),   # non-uniform inside the activation pair
]


def main():
    fail = []

    src = SRC.read_text()
    for needle in ("seen = {xa.dtype, xb.dtype, wa.dtype, wb.dtype}",
                   "if ttnn.bfloat8_b in seen and len(seen) != 1:",
                   'return "dtype_mixed_bfp8"'):
        if needle not in src:
            fail.append(f"source no longer contains {needle!r}: transcription is stale")

    print(f"{'xa,xb,wa,wb':<26} {'main':<16} {'this change':<18} {'helper':<8} reachable")
    for row, reachable in CASES:
        m, n, h = main_rule(*row), new_rule(*row), helper_rule(*row)
        print(f"{','.join(row):<26} {str(m):<16} {str(n):<18} {str(h):<8} {reachable}")
        # 1. Nothing the fold issues today may change verdict.
        if reachable and m != n:
            fail.append(f"{row}: reachable today and the verdict moved, {m} -> {n}")
        # 2. Every set containing bfp8 and more than one dtype must now decline.
        if BFP8 in set(row) and len(set(row)) != 1 and n != "dtype_mixed_bfp8":
            fail.append(f"{row}: mixed set containing bfp8 returned {n}, want dtype_mixed_bfp8")
        # 3. Uniform bfp8 must still be served -- it is measured fine, and narrowing it would
        #    delete the bfp8 sub-campaign's own target.
        if set(row) == {BFP8} and n is not None:
            fail.append(f"{row}: uniform bfp8 declined with {n}, want served")

    # 4. The helper is genuinely the wrong rule here, or this file has no reason to exist.
    served_by_main = [r for r, _ in CASES if main_rule(*r) is None]
    if not any(helper_rule(*r) is not None for r in served_by_main):
        fail.append("the helper declines nothing main serves -- then trimul_tail should just use it")

    for x in fail:
        print(f"FAIL: {x}")
    if fail:
        return 1
    n_mixed = sum(1 for r, _ in CASES if BFP8 in set(r) and len(set(r)) != 1)
    print(f"\nOK: {n_mixed} mixed-bfp8 rows now decline, uniform bfp8 still served, "
          f"and no row reachable today changed verdict")
    return 0


if __name__ == "__main__":
    sys.exit(main())
