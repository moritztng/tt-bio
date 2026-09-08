"""The pair FFN shrinks its row block before it gives up a lever.

ESMFold2's row-blocked pair FFN carries three levers that only move a destination: C-in
slices each block into L1 lazily, F keeps fc2's output in L1 so the per-block residual add
reads it on chip, and G writes each block straight into a pre-allocated output with
`fill_cache`. When a block-sized L1 resident does not fit, `_row_blocked` used to drop G,
then F, then C-in and re-run the loop, which is how the Wormhole Galaxy census came to read
F and G at 0 served / 538 declined at 896 and 1024 aa while every rung up to 768 read 538/0.

The block is `rows` rows of [1, rows, L, C], so at a fixed rows=32 it grows with L until it
stops fitting. Shrinking it is the rung that matches the cause; G and F are destinations, not
sizes. These tests pin the resulting order: halve while the block can still halve, drop a
lever only when it cannot.

Host-only. The device is a stub whose `_ffn` throws above a pinned block height, so the test
is the control flow and not the allocator.
"""
from __future__ import annotations

import pytest
import ttnn

from tt_bio import esmc


class FakeTensor:
    """Enough of a ttnn.Tensor for `_row_blocked`: a shape, a padded shape, a device."""

    dtype = "bfloat8_b"
    layout = "TILE"

    def __init__(self, shape):
        self.shape = tuple(shape)
        self.padded_shape = tuple(shape)

    def device(self):
        return "fake-device"


@pytest.fixture
def stub(monkeypatch):
    """A `SwiGLUFFN` whose every ttnn call is a stub, plus a clean lever state.

    `_ffn` raises for any block taller than `fits`, the way an L1 exhaustion presents to
    `_row_blocked`: an exception out of program creation, with nothing to distinguish it
    from any other. Returns a namespace the tests set `fits` on.
    """
    calls = {"ffn": 0, "raised": 0, "fits": 8}

    def fake_ffn(self, part, split=False, l1_gated=False, out_mc=None):
        calls["ffn"] += 1
        if part.shape[1] > calls["fits"]:
            calls["raised"] += 1
            raise RuntimeError(
                f"TT_THROW @ program.cpp:1052 out of L1 for {part.shape[1]} rows")
        return FakeTensor(part.shape)

    monkeypatch.setattr(esmc.SwiGLUFFN, "_ffn", fake_ffn)
    monkeypatch.setattr(ttnn, "chunk", lambda x, n, dim: [
        FakeTensor((1, -(-x.shape[1] // n), x.shape[2], x.shape[3])) for _ in range(n)])
    monkeypatch.setattr(ttnn, "slice", lambda x, start, end, memory_config=None:
                        FakeTensor((1, end[1] - start[1], x.shape[2], x.shape[3])))
    monkeypatch.setattr(ttnn, "allocate_tensor_on_device",
                        lambda shape, dtype, layout, device, mc: FakeTensor(tuple(shape)))
    monkeypatch.setattr(ttnn, "Shape", lambda s: tuple(s))
    monkeypatch.setattr(ttnn.experimental, "view", lambda dst, shape: FakeTensor(tuple(shape)))
    monkeypatch.setattr(ttnn, "fill_cache", lambda view, out, i: None)
    monkeypatch.setattr(ttnn, "add", lambda a, b, memory_config=None: FakeTensor(a.shape))
    monkeypatch.setattr(ttnn, "deallocate", lambda t: None)
    monkeypatch.setattr(ttnn, "concat", lambda outs, dim: FakeTensor(
        (1, sum(o.shape[1] for o in outs), outs[0].shape[2], outs[0].shape[3])))

    monkeypatch.setattr(esmc, "_PAIR_FFN_L1_SLICE", True)
    monkeypatch.setattr(esmc, "_PAIR_FFN_FUSED_RESIDUAL", True)
    monkeypatch.setattr(esmc, "_PAIR_FFN_FILL_ASSEMBLY", True)
    # raising=False so the fixture still builds against an engine without the shrink:
    # a negative control has to reach the assertion that reads the lever counts, not
    # die setting up (negative-control-must-break-what-check-reads).
    monkeypatch.setattr(esmc, "_ROW_BLOCK_SHRUNK", {}, raising=False)
    monkeypatch.setattr(esmc, "_L1_SLICE_REFUSED", set())
    monkeypatch.setattr(esmc, "_FUSED_RESID_REFUSED", set())
    monkeypatch.setattr(esmc, "_FILL_ASSEMBLY_REFUSED", set())
    monkeypatch.setattr(esmc, "L1_SLICE_STATS", [0, 0])
    monkeypatch.setattr(esmc, "FUSED_RESID_STATS", [0, 0])
    monkeypatch.setattr(esmc, "FILL_ASSEMBLY_STATS", [0, 0])
    return calls


def _ffn_instance():
    return esmc.SwiGLUFFN.__new__(esmc.SwiGLUFFN)


def test_a_block_that_does_not_fit_is_halved_and_both_levers_stay_lit(stub):
    """The 1024 aa case: 32 rows throws, 16 fits, F and G must come back served."""
    stub["fits"] = 16
    x = FakeTensor((1, 1024, 1024, 128))
    esmc.SwiGLUFFN._row_blocked(_ffn_instance(), x, 32, residual=True)

    assert esmc.FUSED_RESID_STATS == [1, 0], "F declined despite a block that could shrink"
    assert esmc.FILL_ASSEMBLY_STATS == [1, 0], "G declined despite a block that could shrink"
    assert stub["raised"] == 1, "one exception per fold, not one per block"
    assert esmc._ROW_BLOCK_SHRUNK[tuple(x.padded_shape)] == 16
    assert esmc.L1_SLICE_STATS == [1, 0]
    assert not esmc._FILL_ASSEMBLY_REFUSED and not esmc._FUSED_RESID_REFUSED


def test_the_shrink_halves_repeatedly_before_it_gives_up_anything(stub):
    """32 -> 16 -> 8 with G and F still lit: the ladder is the block height, not the levers."""
    stub["fits"] = 8
    x = FakeTensor((1, 1024, 1024, 128))
    esmc.SwiGLUFFN._row_blocked(_ffn_instance(), x, 32, residual=True)

    assert esmc.FUSED_RESID_STATS == [1, 0]
    assert esmc.FILL_ASSEMBLY_STATS == [1, 0]
    assert stub["raised"] == 2
    assert esmc._ROW_BLOCK_SHRUNK[tuple(x.padded_shape)] == 8


def test_below_the_floor_the_old_ladder_still_runs(stub):
    """Nothing fits, so the block cannot shrink past PAIR_FFN_ROW_BLOCK_MIN and G, then F,
    then C-in are dropped as before. The shrink adds a rung, it does not remove one."""
    stub["fits"] = 0
    x = FakeTensor((1, 1024, 1024, 128))
    with pytest.raises(RuntimeError):
        esmc.SwiGLUFFN._row_blocked(_ffn_instance(), x, 32, residual=True)

    key = tuple(x.padded_shape)
    assert esmc._ROW_BLOCK_SHRUNK[key] == esmc.PAIR_FFN_ROW_BLOCK_MIN
    assert stub["raised"] > 3, "the levers were dropped without the block shrinking first"
    assert key in esmc._FILL_ASSEMBLY_REFUSED
    assert key in esmc._FUSED_RESID_REFUSED
    assert key in esmc._L1_SLICE_REFUSED


def test_the_working_height_is_cached_so_the_next_call_pays_no_exception(stub):
    """The same contract the three *_REFUSED sets keep: a fold pays one exception, not one
    per block. 538 pair-FFN calls per fold is 538 exceptions without this."""
    stub["fits"] = 16
    x = FakeTensor((1, 1024, 1024, 128))
    inst = _ffn_instance()
    esmc.SwiGLUFFN._row_blocked(inst, x, 32, residual=True)
    raised_after_first = stub["raised"]
    esmc.SwiGLUFFN._row_blocked(inst, FakeTensor(x.shape), 32, residual=True)

    assert stub["raised"] == raised_after_first, "second call re-paid the exception"
    assert esmc.FUSED_RESID_STATS == [2, 0]
    assert esmc.FILL_ASSEMBLY_STATS == [2, 0]


def test_a_block_that_fits_takes_the_same_path_it_always_did(stub):
    """Inert where nothing throws: no exception, no shrink, no cache entry. This is why the
    change cannot reach a model or a card whose census shows the levers already served."""
    stub["fits"] = 32
    x = FakeTensor((1, 512, 512, 128))
    esmc.SwiGLUFFN._row_blocked(_ffn_instance(), x, 32, residual=True)

    assert stub["raised"] == 0
    assert esmc._ROW_BLOCK_SHRUNK == {}
    assert esmc.FUSED_RESID_STATS == [1, 0]
    assert esmc.FILL_ASSEMBLY_STATS == [1, 0]


def test_g_is_dropped_when_the_shrunk_block_no_longer_tiles_the_length(stub):
    """G needs the blocks to tile L exactly (`fill_cache` writes whole blocks). 896 rows is
    28 blocks of 32 and 56 of 16, so it survives a halving; a length that stops tiling must
    lose G rather than write a short last block."""
    stub["fits"] = 16
    x = FakeTensor((1, 544, 544, 128))   # 17 x 32, but 34 x 16 as well
    esmc.SwiGLUFFN._row_blocked(_ffn_instance(), x, 32, residual=True)
    assert esmc.FILL_ASSEMBLY_STATS == [1, 0]

    esmc._ROW_BLOCK_SHRUNK.clear()
    esmc.FILL_ASSEMBLY_STATS[:] = [0, 0]
    esmc.FUSED_RESID_STATS[:] = [0, 0]
    esmc.L1_SLICE_STATS[:] = [0, 0]
    y = FakeTensor((1, 40, 40, 128))     # 2 x 32 with a short tail; 3 x 16 with one too
    esmc.SwiGLUFFN._row_blocked(_ffn_instance(), y, 32, residual=True)
    assert esmc.FILL_ASSEMBLY_STATS == [0, 1], "G assembled a length it does not tile"
    assert esmc.FUSED_RESID_STATS == [1, 0], "F does not need the exact tiling G does"
