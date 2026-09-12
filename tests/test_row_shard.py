"""The i-axis shard plan: tile-aligned slabs, and the load-balance ceiling they imply.

Pure arithmetic, no device. The device half of this lives in
`perf/b2z2_trunkshard/block_split_bitexact.py`, which proves the split itself is bit-exact.
"""

import pytest

from tt_bio.row_shard import (PAIR_CHAIN, gathers_per_block, row_shard_bounds, row_shard_ceiling)
from tt_bio.token_axis import TOKEN_BUCKET


@pytest.mark.parametrize("n_rows,n_shards,expected", [
    (512, 2, ((0, 256), (256, 512))),
    (512, 4, ((0, 128), (128, 256), (256, 384), (384, 512))),
    (320, 2, ((0, 160), (160, 320))),
    # 9 tiles two ways cannot be even, and the leading slab takes the spare tile
    (288, 2, ((0, 160), (160, 288))),
    (288, 3, ((0, 96), (96, 192), (192, 288))),
    (160, 2, ((0, 96), (96, 160))),
    (512, 1, ((0, 512),)),
])
def test_bounds(n_rows, n_shards, expected):
    assert row_shard_bounds(n_rows, n_shards) == expected


@pytest.mark.parametrize("n_rows", [64, 128, 160, 288, 320, 512, 1024])
@pytest.mark.parametrize("n_shards", [1, 2, 4])
def test_bounds_are_a_tile_aligned_partition(n_rows, n_shards):
    """Contiguous, gapless, covering, and every boundary a whole number of tiles."""
    if n_rows // TOKEN_BUCKET < n_shards:
        pytest.skip("axis too short to shard that many ways")
    b = row_shard_bounds(n_rows, n_shards)
    assert b[0][0] == 0 and b[-1][1] == n_rows
    assert all(b[i][1] == b[i + 1][0] for i in range(len(b) - 1))
    assert all(r0 % TOKEN_BUCKET == 0 and r1 % TOKEN_BUCKET == 0 for r0, r1 in b)
    # at most one tile of imbalance, which is the best a 32-aligned split can do
    extents = [r1 - r0 for r0, r1 in b]
    assert max(extents) - min(extents) <= TOKEN_BUCKET


def test_refuses_an_unbucketed_axis():
    """The token axis is bucketed to a multiple of 32 before it gets here; 298 never should."""
    with pytest.raises(ValueError, match="not a multiple"):
        row_shard_bounds(298, 2)


def test_refuses_an_axis_too_short_to_shard():
    """One tile is one tile. Rounding that into an empty slab would be a silent wrong answer."""
    with pytest.raises(ValueError, match="not shardable"):
        row_shard_bounds(32, 2)
    with pytest.raises(ValueError, match="not shardable"):
        row_shard_bounds(64, 4)


@pytest.mark.parametrize("n_rows,n_shards,ceiling", [
    (512, 2, 2.0), (512, 4, 4.0), (320, 2, 2.0), (288, 2, 1.8), (160, 2, 5 / 3), (512, 1, 1.0),
])
def test_ceiling(n_rows, n_shards, ceiling):
    assert row_shard_ceiling(n_rows, n_shards) == pytest.approx(ceiling)


def test_ceiling_never_exceeds_the_shard_count():
    for n_rows in (128, 160, 288, 320, 512):
        for n_shards in (1, 2, 4):
            if n_rows // TOKEN_BUCKET < n_shards:
                continue
            assert 1.0 <= row_shard_ceiling(n_rows, n_shards) <= n_shards


def test_gather_count_matches_the_chain():
    """Four with the row_slab API as it stands; three once the triangle bias can be injected."""
    assert gathers_per_block() == sum(1 for _, need in PAIR_CHAIN if need == "FULL") == 4
    assert gathers_per_block(injectable_bias=True) == 3
