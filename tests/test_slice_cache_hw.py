"""Hardware test for the ttnn 0.68.0 slice program-cache collision that collapsed l1536 under transition_shard.

A 4-D DRAM slice of a 3-row block and a 3-D slice of a 64-row block that end on the same index share a
program-cache entry, so whichever runs second returns the wrong rows. In the fold the triangle attention's
3-D chunks ([1536,1536,256][128:192]) poisoned every 64th 3-row block of the pair Transition. Locks:

  1. the collision itself, so a ttnn upgrade that fixes it shows up here as an xpass;
  2. the L1 slice transition_shard's row blocks use stays exact behind the same poison;
  3. the tile-aligned DRAM blocks the default paths cut stay exact behind it too.

Run: TT_VISIBLE_DEVICES=<card> python3 -m pytest tests/test_slice_cache_hw.py
"""
import pytest
import torch
import ttnn

from tt_bio import tenstorrent as T
from tt_bio.main import ensure_p300_mesh_descriptor

pytestmark = pytest.mark.device

S, C, E = 1536, 32, 192


@pytest.fixture(scope="module")
def dev():
    ensure_p300_mesh_descriptor()
    d = ttnn.open_device(device_id=0)
    yield d
    T._close_device_locked(d)


@pytest.fixture(scope="module")
def pair(dev):
    torch.manual_seed(0)
    a = torch.randn(1, S, S, C, dtype=torch.bfloat16)
    return a, ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _poison(dev):
    dev.clear_program_cache()
    z = ttnn.from_torch(torch.zeros(S, S, C, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(ttnn.slice(z, [E - 64, 0, 0], [E, S, C]))
    ttnn.deallocate(z)


def _err(t, a, lo, hi):
    return float((ttnn.to_torch(t).float() - a[:, lo:hi].float()).abs().max())


@pytest.mark.xfail(strict=True, reason="ttnn 0.68.0 slice program-cache collision; xpass = fixed upstream")
def test_dram_three_row_block_is_poisoned(dev, pair):
    a, x = pair
    _poison(dev)
    assert _err(x[:, E - 3:E], a, E - 3, E) == 0


def test_l1_three_row_block_is_exact(dev, pair):
    a, x = pair
    _poison(dev)
    t = ttnn.slice(x, [0, E - 3, 0, 0], [1, E, S, C], memory_config=ttnn.L1_MEMORY_CONFIG)
    assert _err(t, a, E - 3, E) == 0


@pytest.mark.parametrize("h", [32, 128])
def test_tile_aligned_dram_blocks_are_exact(dev, pair, h):
    a, x = pair
    _poison(dev)
    lo = (E // h - 1) * h if E % h == 0 else E - h
    assert _err(x[:, lo:lo + h], a, lo, lo + h) == 0
