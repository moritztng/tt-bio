"""`_tape` finds a view by asking whether an output and its parent share storage.

It used to ask whether their `buffer_address()` values were equal, and that is a different
question, wrong two ways that each have a shipped producer:

- `ttnn.reallocate` is a move. It frees its input and puts the output at the address the input
  had. The trimul tail runs it on every chunk when it chunks, so under a tape the parent of the
  taped move is already freed. ttnn refuses `buffer_address()` on it, logs `TT_FATAL: Tensor is
  not allocated` at critical and throws, and the old check swallowed the throw. The shipped code
  then deallocates the freed input, and `Tensor.free` evicted it: `ttnn.to_memory_config` of a
  released L1 handle copies whatever now sits at that address into a new DRAM buffer, and the
  tape held that copy.
- L1 and DRAM addresses are separate spaces over the same integers, so an L1 tensor and a DRAM
  one can hold the same number. The old check called that pair a view and pinned both.

Each test below fails on the address check and passes on the storage check. The last one is the
control: a real view must still be found, or the others would pass by detecting nothing.

Needs a card: which address the allocator hands back, and what ttnn does with a freed handle,
are the device's answers.
"""
from __future__ import annotations

import pytest
import torch
import ttnn

from tt_bio import autograd as ag
from tt_bio import taped_ttnn as tt
from tt_bio import tenstorrent as T

pytestmark = pytest.mark.device

TILE = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)


@pytest.fixture
def dev():
    # Per test: conftest closes the device after every test, so a module-scoped handle is stale
    # from the second test on.
    return T.get_device()


def _linked(a, b) -> bool:
    """Whether the tape treats `a` and `b` as one buffer: both pinned in place, or, where the
    tape keeps storage groups, one group.

    A storage group holds its members weakly, so its entries are `weakref.ref` objects and
    comparing one to a tensor is always False. `ag._members` dereferences them."""
    group = getattr(a, "shares", None)
    if group is not None:
        return any(t is b for t in ag._members(group))
    return not a.evictable and not b.evictable


def _leaf(dev, shape, mc):
    return ag.Tensor(ttnn.from_torch(torch.randn(*shape), device=dev, memory_config=mc, **TILE),
                     requires_grad=True)


@pytest.mark.parametrize("mc", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
                         ids=["dram", "l1"])
def test_a_moved_parent_is_not_asked_for_its_address(dev, mc, capfd):
    x = _leaf(dev, (1, 1, 64, 64), mc)
    with tt.tape():
        a = tt.taped_ttnn().typecast(x, ttnn.float32, memory_config=mc)
        capfd.readouterr()
        b = tt.taped_ttnn().reallocate(a)
        log = "".join(capfd.readouterr())
    assert "Tensor is not allocated" not in log, log[-400:]
    assert not a.value.is_allocated(), (
        "reallocate frees its input, and the tape then gave the freed handle a live DRAM copy "
        "of whatever sits at its old L1 address")
    assert not _linked(a, b)


def test_freeing_a_moved_l1_parent_allocates_nothing(dev):
    """The trimul tail's own sequence: reallocate a chunk, then deallocate the old handle."""
    mc = ttnn.L1_MEMORY_CONFIG
    x = _leaf(dev, (1, 1, 64, 64), mc)
    with tt.tape():
        a = tt.taped_ttnn().typecast(x, ttnn.float32, memory_config=mc)
        b = tt.taped_ttnn().reallocate(a)
        tt.taped_ttnn().deallocate(a)
    assert not a.value.is_allocated(), (
        "free() moved a released L1 handle to DRAM, which copies the bytes that now occupy its "
        f"address into a buffer the tape holds ({a.value.memory_config().buffer_type})")
    ag.backward([b], [None])
    # The move's backward is the identity, so the gradient of sum(b) is ones.
    torch.testing.assert_close(ttnn.to_torch(x.grad).float(), torch.ones(1, 1, 64, 64),
                               rtol=0, atol=0)


def _dram_filler_to(dev, addr):
    """Hold DRAM up to per-bank address `addr`, so the next DRAM buffer starts there.

    Interleaved DRAM hands out addresses bottom-up and reserves the same range in every bank, so
    a one-row row-major tensor takes exactly one page of its row's width in each. Probe where
    the next buffer would land, then hold a row whose width is the distance to `addr`."""
    probe = ttnn.from_torch(torch.zeros(1, 1, 32, 32), device=dev, **TILE)
    nxt = probe.buffer_address()
    ttnn.deallocate(probe)
    gap = addr - nxt
    # DRAM buffers align to 32 B on Wormhole and 64 B on Blackhole.
    assert gap > 0 and gap % (32 if T.is_wormhole() else 64) == 0, (addr, nxt)
    return ttnn.from_torch(torch.zeros(1, gap // 2), dtype=ttnn.bfloat16,
                           layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)


def test_an_l1_tensor_and_a_dram_tensor_at_the_same_address_are_not_a_view(dev):
    x = _leaf(dev, (1, 1, 32, 32), ttnn.L1_MEMORY_CONFIG)
    addr = x.value.buffer_address()
    filler = _dram_filler_to(dev, addr)
    try:
        with tt.tape():
            y = tt.taped_ttnn().to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        assert y.value.buffer_address() == addr, (
            f"precondition: the DRAM copy was meant to land at {addr}, it is at "
            f"{y.value.buffer_address()} -- the collision was not constructed, fix the filler")
        assert y.value.memory_config().buffer_type == ttnn.BufferType.DRAM
        assert not _linked(x, y), "an L1 buffer and a DRAM buffer were taped as one storage"
    finally:
        ttnn.deallocate(filler)


def test_a_real_view_is_still_found(dev):
    """The control. `ttnn.reshape` of (1,4,32,64) to (1,128,64) returns the input's buffer."""
    x = _leaf(dev, (1, 4, 32, 64), ttnn.DRAM_MEMORY_CONFIG)
    with tt.tape():
        y = tt.taped_ttnn().reshape(x, (1, 128, 64))
    assert y.value.buffer_address() == x.value.buffer_address(), "precondition: a view"
    assert _linked(x, y)


def test_a_dead_dram_view_group_in_a_kept_tape_goes_back_to_the_card(dev):
    """A view group no backward reads is released like a lone tensor would be.

    Both members are mid-graph intermediates, so both require a gradient; neither is read. The
    group used to be kept on that alone, which is how BindCraft 2's masked outer product mean
    held one pair tensor per resident Evoformer block once it reshaped its matmul output."""
    x = _leaf(dev, (1, 4, 32, 64), ttnn.DRAM_MEMORY_CONFIG)
    with tt.tape():
        a = tt.taped_ttnn().typecast(x, ttnn.float32)
        v = tt.taped_ttnn().reshape(a, (1, 128, 64))
        w = tt.taped_ttnn().permute(v, (0, 2, 1))
    assert _linked(a, v), "precondition: the reshape is a view"
    assert a.requires_grad and v.requires_grad and not (a.pinned or v.pinned)
    ag.release_dead([w], [x])
    assert not a.value.is_allocated(), "a view group nothing reads stayed on the card"
    ag.backward([w], [None])
    torch.testing.assert_close(ttnn.to_torch(x.grad).float(), torch.ones(1, 4, 32, 64),
                               rtol=0, atol=0)


def test_a_view_group_a_backward_reads_is_kept(dev):
    """The control: the matmul's weight gradient reads its left operand, a view."""
    x = _leaf(dev, (1, 4, 32, 64), ttnn.DRAM_MEMORY_CONFIG)
    wt = _leaf(dev, (64, 32), ttnn.DRAM_MEMORY_CONFIG)
    with tt.tape():
        a = tt.taped_ttnn().typecast(x, ttnn.float32)
        v = tt.taped_ttnn().reshape(a, (1, 128, 64))
        y = tt.taped_ttnn().matmul(v, wt)
    assert _linked(a, v), "precondition: the reshape is a view"
    ag.release_dead([y], [x, wt])
    assert a.value.is_allocated() and v.value.is_allocated()
