"""A lone MSA row chunk aliases the tensor it came from, so the source must not be freed.

`MSALayer.__call__` takes a row-chunked branch past `SEQ_LEN_MORE_CHUNKING` and slices the MSA
depth into `MSA_CHUNK_SIZE` blocks. When the depth fits in one block the slice spans the whole
tensor, and a full-span `ttnn.slice` hands back the input's own buffer rather than a copy. The
branch freed the source before joining the blocks, which at one block freed the block too, and
`ttnn.reallocate` two lines later threw "Buffer is not allocated".

BoltzGen design is the shape that reaches it: one MSA row from `dummy_msa`, and a target past
1536 tokens. It killed 1831 and 1911 tokens on a qb1 p150a. Boltz-2 at the same token counts
carries a deeper MSA, gets partial slices, which are real copies, and takes the concat branch
where the free is right.

Needs a card: whether a slice copies is the device allocator's answer, not something a host
model of it can be trusted to reproduce.
"""
from __future__ import annotations

import pytest
import torch
import ttnn

from tt_bio import tenstorrent as T

pytestmark = pytest.mark.device

SEQ, CHAN = 512, 64


@pytest.fixture(scope="module")
def dev():
    d = ttnn.open_device(device_id=0)
    yield d
    ttnn.close_device(d)


def _rows(d, n):
    return ttnn.from_torch(torch.randn(1, n, SEQ, CHAN), dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=d)


@pytest.mark.parametrize("depth", [1, 2, 8])
def test_a_full_span_slice_shares_the_source_buffer(dev, depth):
    """The device fact the guard rests on. One chunk means the slice covers everything."""
    assert depth <= T.MSA_CHUNK_SIZE, "this case must be a one-chunk depth"
    m = _rows(dev, depth)
    mc = m[:, 0:depth, :]
    assert mc.buffer_address() == m.buffer_address()
    ttnn.deallocate(m)
    assert not mc.is_allocated(), "freeing the source must be what kills the chunk"


def test_the_shipped_order_survives_one_chunk(dev):
    """The fixed sequence: at one chunk the source is the result, so nothing is freed."""
    m = _rows(dev, 1)
    parts = [m[:, 0:1, :]]
    assert len(parts) == 1
    m = parts[0]
    m = ttnn.reallocate(m)          # threw "Buffer is not allocated" before the fix
    assert m.is_allocated()
    ttnn.deallocate(m)


def test_the_old_order_still_breaks(dev):
    """Negative control: without the guard the same sequence dies, so the test above bites."""
    m = _rows(dev, 1)
    parts = [m[:, 0:1, :]]
    ttnn.deallocate(m)              # the free the fix moved into the concat branch
    with pytest.raises(RuntimeError, match="Buffer is not allocated"):
        ttnn.reallocate(parts[0])


def test_many_chunks_still_copy(dev):
    """The concat branch is unchanged: a partial slice is a real copy and the free is right."""
    depth = T.MSA_CHUNK_SIZE + 1
    m = _rows(dev, depth)
    parts = [m[:, s:min(s + T.MSA_CHUNK_SIZE, depth), :]
             for s in range(0, depth, T.MSA_CHUNK_SIZE)]
    assert len(parts) > 1
    assert all(p.buffer_address() != m.buffer_address() for p in parts)
    ttnn.deallocate(m)
    assert all(p.is_allocated() for p in parts)
    joined = ttnn.concat(parts, dim=1)
    for p in parts:
        ttnn.deallocate(p)
    assert tuple(joined.shape)[1] == depth
    ttnn.deallocate(ttnn.reallocate(joined))
