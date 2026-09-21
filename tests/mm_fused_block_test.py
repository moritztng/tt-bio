#!/usr/bin/env python3
"""The derived fused entry must reproduce the six literals it replaced, exactly.

`_MM_BLOCK` carried six hand-written fused keys -- (4, 16)/(4, 17) for boltz2 and openfold3,
(8, 32)/(8, 33) for protenix-v2, (2, 8)/(2, 9) for protenix-v2's template stack -- each landed with
the same argument: the fused weight is two registered widths concatenated on the output axis, and
the entry folds K the way the separate matmuls fold it. `_mm_fused_block` states that argument once.
This pins the two against each other, so the rule cannot drift from the table it replaced.

    python3 tests/mm_fused_block_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The values as they shipped at 47810889f, before the rule replaced them.
SHIPPED_FUSED = {
    (4, 16): (4, 4, 1, 4, 1),
    (4, 17): (4, 4, 1, 4, 1),
    (8, 32): (4, 8, 1, 4, 1),
    (8, 33): (4, 8, 1, 4, 1),
    (2, 8): (4, 2, 1, 4, 1),
    (2, 9): (4, 2, 1, 4, 1),
}

# Widths a model presents that are NOT a fusion of registered operands. These must keep getting no
# config: the point of the rule is to serve concatenations, never to configure an unmeasured op.
NOT_DERIVABLE = [
    (8, 1),    # protenix-v2's one-tile pair-bias projection at c_z=256
    (4, 1), (12, 1), (2, 1),
    (4, 13), (8, 25), (12, 37),   # base + 1, which is not a fusion of two
    (16, 48),  # a kt with no registered width at all
]


def main() -> int:
    from tt_bio.tenstorrent import _MM_BLOCK, _mm_fused_block

    bad = []
    for key, want in SHIPPED_FUSED.items():
        assert key not in _MM_BLOCK, f"{key} is back in the table; the rule derives it"
        got = _mm_fused_block(*key)
        if got != want:
            bad.append(f"{key}: derived {got}, shipped {want}")
    for key in NOT_DERIVABLE:
        if key in _MM_BLOCK:
            continue
        got = _mm_fused_block(*key)
        if got is not None:
            bad.append(f"{key}: derived {got}, must stay unconfigured")

    # opendde's fused twins, the keys this change exists for. Derived from (12, 36) + (12, 12),
    # tie broken to the wider component, so the fused call runs the block the qkv matmul was
    # swept with.
    for key in ((12, 48), (12, 49)):
        got = _mm_fused_block(*key)
        if got != _MM_BLOCK[(12, 36)]:
            bad.append(f"{key}: derived {got}, want {_MM_BLOCK[(12, 36)]}")

    # `swiglu_fused` and `trimul_tail` keep their own allow-lists of keys their descriptors were
    # swept at and then INDEX the table for the value. swiglu's list is {(4, 16)}, which is a fused
    # width: deleting the literal without giving them the resolver turns the allow-list into a
    # KeyError on boltz2's and openfold3's pair Transition.
    from tt_bio import swiglu_fused as SW
    from tt_bio import trimul_tail as TT1

    class _W:
        def __init__(self, kt, nt):
            self.shape = (kt * 32, nt * 32)

    if SW._block(_W(4, 16)) != (4, 4, 1, 4, 1):
        bad.append(f"swiglu_fused._block((4, 16)) = {SW._block(_W(4, 16))}, want (4, 4, 1, 4, 1)")
    if SW._block(_W(4, 12)) is not None:
        bad.append("swiglu_fused._block served a key outside its allow-list")
    for key in sorted(TT1.F1_BLOCK_KEYS):
        if TT1._block_for(*key) is None:
            bad.append(f"trimul_tail._block_for{key} resolved to None")

    for line in bad:
        print(f"FAIL  {line}")
    print(f"{'FAIL' if bad else 'PASS'}  {len(SHIPPED_FUSED)} shipped fused keys reproduced, "
          f"{len(NOT_DERIVABLE)} non-fusions left unconfigured, opendde (12, 48)/(12, 49) derived")
    return 1 if bad else 0


def test_mm_fused_block():
    """`main` is a script entry point, so pytest collected this file and ran nothing.

    A file under tests/ that reports "no tests ran" is a vacuous pass: the suite goes green whether
    the rule holds or not. This is the collected name; it asserts on the same check.
    """
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
