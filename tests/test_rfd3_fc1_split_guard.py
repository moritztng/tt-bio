"""`_FC1_SPLIT_SILU` may only take the split where the chunk height does not move.

Region 2's lever admits `fc1`'s output to L1, which costs a third L1 resident and so shrinks
the chunk height. The height is not a footprint detail: at 514 tokens h=64 is the only height of
64, 63, 59 and 53 that reproduces the whole-tensor path, and a moved height moved the fold's CIF
digest with the lever's own served count at zero (state doc §15.2/§15.3). The first guard
compared chunk COUNTS, which 49 sizes keep while moving the height.

This file locks the arithmetic the guard reads. The guard itself -- `h3 == h` in
`Transition.__call__` -- is checked at the fold, where the negative control is that the moved
height rung's digest goes back to the shipped one (§15.6 X1).

Host-only, and it doubles as a parity gate on `_PAIR_TRANSITION_L1_BYTES` and
`_PAIR_TRANSITION_H_CHUNK`: both select the model's arithmetic now, so a retune of either would
land here rather than in a user's structure.
"""
from __future__ import annotations

from tt_bio.rfd3.model import _pair_transition_chunk_h

TOKENS = range(512, 1201)               # every chunked size the census fixture range covers


def classify(hidden):
    served, moved_height_equal_count, declined = [], [], []
    for tokens in TOKENS:
        w_pad = -(-tokens // 32) * 32
        h = _pair_transition_chunk_h(w_pad, hidden, tokens)
        h3 = _pair_transition_chunk_h(w_pad, hidden, tokens, residents=3)
        if h3 == h:
            served.append(tokens)
        else:
            declined.append(tokens)
            if -(-tokens // h3) == -(-tokens // h):
                moved_height_equal_count.append(tokens)
    return served, moved_height_equal_count, declined


def test_hidden_512_addressable_set():
    served, moved, declined = classify(512)
    assert (len(served), len(declined)) == (161, 528)
    assert max(served) == 672                       # 685, the census fixture, is NOT addressable
    assert 514 in served                            # R3, where every fold so far was measured


def test_the_equal_count_sizes_are_the_ones_the_first_guard_got_wrong():
    """49 sizes in two windows keep the chunk count and move the height -- 673-693 at 64 -> 63
    and 705-732 at 64 -> 61. The count guard served all of them."""
    _, moved, declined = classify(512)
    assert len(moved) == 49
    assert (min(moved), max(moved)) == (673, 732)
    assert set(moved) <= set(declined)


def test_hidden_256_keeps_every_size():
    served, moved, declined = classify(256)
    assert len(served) == len(list(TOKENS)) == 689
    assert not declined and not moved
