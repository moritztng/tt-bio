"""`sdpa_generic.balanced_split`: the busiest core's unit count, never worse than the factory's split."""
import itertools

from tt_bio.sdpa_generic import _div_up, balanced_split


def factory(B, H, Q, cores):
    b = min(B, cores)
    h = min(cores // b, H)
    q = min(cores // (b * h), Q)
    return b, h, q


def work(B, H, Q, s):
    return _div_up(B, s[0]) * _div_up(H, s[1]) * _div_up(Q, s[2])


def test_dit_shapes():
    assert balanced_split(5, 16, 3, 72) == (5, 4, 3)     # Wormhole, c730 padded to 768, q chunk 256
    assert work(5, 16, 3, factory(5, 16, 3, 72)) == 6
    assert work(5, 16, 3, (5, 4, 3)) == 4
    assert work(5, 16, 3, balanced_split(5, 16, 3, 130)) == 2   # Blackhole 13x10


def test_never_worse_and_fits():
    for B, H, Q, cores in itertools.product((1, 2, 5, 8), (1, 4, 16, 24), (1, 2, 3, 6, 12), (64, 72, 110, 130)):
        s = balanced_split(B, H, Q, cores)
        assert s[0] * s[1] * s[2] <= cores and s[0] <= B and s[1] <= H and s[2] <= Q
        assert work(B, H, Q, s) <= work(B, H, Q, factory(B, H, Q, cores))
