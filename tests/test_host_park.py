"""`host_park` / `host_unpark` hand back the same tile bytes, and only past the offload size.

The Protenix/OpenDDE trunk parks its cycle-invariant z_init and template projections on the host
past 1 GiB and uploads them once per recycling cycle, so the round trip has to be lossless for
every dtype the trunk may hold, including bfloat8_b whose shared exponents a to_torch round trip
would re-quantise.
"""
import os

import pytest
import torch
import ttnn

from tt_bio import tenstorrent as T

pytestmark = pytest.mark.device


@pytest.fixture
def dev():
    # Per test: conftest closes the chip after every test, so a module-scoped handle goes stale.
    return T.get_device()


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32])
def test_a_parked_tensor_comes_back_bit_for_bit(dev, dtype):
    torch.manual_seed(0)
    x = torch.randn(1, 96, 96, 64)
    ref = ttnn.to_torch(ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dtype))
    ts = [ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dtype) for _ in range(2)]
    os.environ["TT_BIO_MSA_HOST_OFFLOAD_MIN_BYTES"] = "0"
    try:
        hs = T.host_park(ts)
    finally:
        del os.environ["TT_BIO_MSA_HOST_OFFLOAD_MIN_BYTES"]
    assert not any(t.is_allocated() for t in ts)
    assert all(h.storage_type() != ttnn.StorageType.DEVICE for h in hs)
    for _ in range(2):                                  # every cycle reads the same parked copy
        d = T.host_unpark(hs[1])
        assert d.storage_type() == ttnn.StorageType.DEVICE
        assert torch.equal(ttnn.to_torch(d), ref)
        ttnn.deallocate(d)


def test_below_the_offload_size_nothing_moves(dev):
    t = ttnn.from_torch(torch.randn(1, 64, 64, 32), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16)
    assert T.host_park(t) is t and T.host_unpark(t) is t and t.is_allocated()
    assert T.host_park([]) == []
