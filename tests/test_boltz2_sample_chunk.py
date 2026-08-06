"""Host-only tests for boltz2.resolve_sample_chunk_width.

No Tenstorrent device needed: the resolver is pure arithmetic. The device-side
acceptance check (same seed, different chunking, same per-sample digests via
TT_BIO_SAMPLE_DIGEST) is a separate on-card run.

Run: python3 tests/test_boltz2_sample_chunk.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tt_bio.boltz2 import resolve_sample_chunk_width


def _old_ragged_widths(m, mps):
    """The pre-fix split: ceil chunk count, then torch.chunk's equalise-or-short-tail."""
    n_chunks = max(1, (m + mps - 1) // mps)
    return [c.numel() for c in torch.arange(m).chunk(n_chunks)]


def test_cap_divides():
    assert resolve_sample_chunk_width(50, 5) == 5
    assert resolve_sample_chunk_width(50, 10) == 10
    assert resolve_sample_chunk_width(50, 25) == 25
    print("[PASS] cap divides multiplicity: exact width")


def test_cap_not_dividing():
    # 50 at cap 3: 17 chunks, widest needs only ceil(50/17) = 3.
    assert resolve_sample_chunk_width(50, 3) == 3
    # 50 at cap 7: 8 chunks, ceil(50/8) = 7.
    assert resolve_sample_chunk_width(50, 7) == 7
    print("[PASS] cap not dividing: rebalanced width")


def test_rebalance_is_free():
    # 16 at cap 10: 2 chunks either way, but 2x8 pads nothing where 10+6 would.
    assert resolve_sample_chunk_width(16, 10) == 8
    print("[PASS] rebalance: 16 at cap 10 -> width 8")


def test_cap_edges():
    assert resolve_sample_chunk_width(50, 1) == 1
    assert resolve_sample_chunk_width(1, 5) == 1
    assert resolve_sample_chunk_width(1, None) == 1
    assert resolve_sample_chunk_width(5, 50) == 5   # cap above M is one chunk
    print("[PASS] edge caps: mps=1, M=1, cap > M")


def test_no_cap_is_one_chunk():
    for m in (1, 2, 5, 50, 200):
        assert resolve_sample_chunk_width(m, None) == m
    print("[PASS] no cap: whole multiplicity is one chunk")


def test_matches_old_split_wherever_old_split_was_even():
    """Where the pre-fix ragged split produced equal widths (the only configs that
    ran without crashing), the resolved width must be identical -- those configs
    stay bit-identical."""
    for m in range(1, 65):
        for mps in range(1, 65):
            old = _old_ragged_widths(m, mps)
            if len(set(old)) == 1:
                assert resolve_sample_chunk_width(m, mps) == old[0], (m, mps)
    print("[PASS] identical width on every previously-even split (4096 combos)")


def test_covers_multiplicity_with_bounded_padding():
    for m in range(1, 130):
        for mps in (1, 2, 3, 5, 7, 10, 16, 25, 64):
            w = resolve_sample_chunk_width(m, mps)
            assert 1 <= w <= max(1, min(m, mps)), (m, mps, w)
            n_chunks = -(-m // w)
            # Padding costs at most n_chunks-1 extra sample-steps out of m.
            assert n_chunks * w - m <= n_chunks - 1, (m, mps, w)
    print("[PASS] coverage and padding bound (1161 combos)")


if __name__ == "__main__":
    test_cap_divides()
    test_cap_not_dividing()
    test_rebalance_is_free()
    test_cap_edges()
    test_no_cap_is_one_chunk()
    test_matches_old_split_wherever_old_split_was_even()
    test_covers_multiplicity_with_bounded_padding()
    print("\nALL BOLTZ2 SAMPLE-CHUNK WIDTH TESTS PASS")
