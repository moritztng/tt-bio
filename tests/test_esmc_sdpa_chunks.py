"""The LM SDPA chunk choice, and lm_sdpa's fall back to k64 when the device refuses the few-k program."""
from __future__ import annotations

import types

import pytest
import torch

from tt_bio import esmc


@pytest.mark.parametrize("k_few, q_len, want", [
    (False, 544, (544, 64)),
    (True, 544, (544, 192)),   # 3 x 192 = 576, the same padded length as 9 x 64
    (True, 512, (512, 128)),   # 4 x 128 = 512 beats 3 x 192 = 576
    (True, 64, (64, 64)),
])
def test_lm_sdpa_chunks(k_few, q_len, want, monkeypatch):
    monkeypatch.setattr(esmc, "_ESMC_SDPA_Q_WHOLE", True)
    assert esmc._lm_sdpa_chunks(q_len, q_len, k_few) == want


class _Q:
    shape = (7, 15, 544, 64)
    padded_shape = shape

    def memory_config(self):
        return types.SimpleNamespace(buffer_type=esmc.ttnn.BufferType.L1)


def test_lm_sdpa_falls_back_once(monkeypatch):
    monkeypatch.setattr(esmc, "_ESMC_SDPA_Q_WHOLE", True)
    monkeypatch.setattr(esmc, "_ESMC_SDPA_K_FEW", True)
    monkeypatch.setattr(esmc, "_SDPA_K_FEW_REFUSED", set())
    monkeypatch.setattr(esmc, "_sdpa_program_config", lambda q, k: (q, k))
    calls = []

    def sdpa(q, k, v, attn_mask=None, *, scale, program_config):
        calls.append(program_config)
        if program_config[1] > 64:
            raise RuntimeError("TT_THROW: Statically allocated circular buffers in program 77 clash with L1 buffers")
        return "o"

    monkeypatch.setattr(esmc, "fused_sdpa", sdpa)
    q = _Q()
    assert esmc.lm_sdpa(q, q, q, None, scale=0.125) == "o"
    assert esmc.lm_sdpa(q, q, q, None, scale=0.125) == "o"
    assert calls == [(544, 192), (544, 64), (544, 64)]


@pytest.mark.parametrize("wormhole", [True, False])
def test_k_few_default_is_wormhole_only(wormhole, monkeypatch):
    from tt_bio import tenstorrent
    monkeypatch.setattr(esmc, "_ESMC_SDPA_K_FEW", None)
    monkeypatch.setattr(tenstorrent, "is_wormhole", lambda: wormhole)
    assert esmc._sdpa_k_few() is wormhole


def _pad_mask(lens, L):
    m = torch.zeros(len(lens), L, L)
    for b, n in enumerate(lens):
        m[b, :, n:] = float("-inf")
    return m


def test_key_lens_reads_pure_key_padding_only(monkeypatch):
    monkeypatch.setattr(esmc, "_ESMC_SHSDPA", True)
    assert esmc._key_lens(None) is None
    assert esmc._key_lens(_pad_mask([544, 300, 37], 544)) == [544, 300, 37]
    m = _pad_mask([544, 300], 544)
    m[1, 5, 10] = float("-inf")                  # one query row differs: not key padding
    assert esmc._key_lens(m) is None
    m = _pad_mask([544, 300], 544)
    m[0, :, 100] = float("-inf")                 # a hole, not a suffix
    assert esmc._key_lens(m) is None
    monkeypatch.setattr(esmc, "_ESMC_SHSDPA", False)
    assert esmc._key_lens(_pad_mask([544, 300], 544)) is None


@pytest.mark.parametrize("wormhole", [True, False])
def test_shsdpa_default_is_wormhole_only(wormhole, monkeypatch):
    from tt_bio import tenstorrent
    monkeypatch.setattr(esmc, "_ESMC_SHSDPA", None)
    monkeypatch.setattr(tenstorrent, "is_wormhole", lambda: wormhole)
    assert (esmc._key_lens(_pad_mask([544, 300], 544)) is not None) is wormhole
    with tenstorrent.levers("lm_sdpa_stock"):     # ESMFold2's Wormhole sets keep the stock op
        assert esmc._key_lens(_pad_mask([544, 300], 544)) is None


class _T(_Q):
    def __init__(self, dram=True, d=64, sharded=False):
        self.shape = self.padded_shape = (7, 15, 544, d)
        self.dtype = esmc.ttnn.bfloat16
        self._mc = esmc.ttnn.DRAM_MEMORY_CONFIG if dram else esmc.ttnn.L1_MEMORY_CONFIG
        self._sharded = sharded

    def memory_config(self):
        return self._mc

    def is_sharded(self):
        return self._sharded


@pytest.mark.parametrize("dram, sharded, d, route", [(True, False, 64, "shsdpa"), (False, False, 64, "shsdpa"),
                                                     (False, True, 64, "stock"), (True, False, 24, "stock")])
def test_lm_sdpa_keypad_route(dram, sharded, d, route, monkeypatch):
    """A KeyPad goes to shsdpa for bf16 interleaved (DRAM or L1) q/k/v on whole tiles; anything else runs the stock op
    over the row repeated to the full [B, 1, S, S] mask."""
    from tt_bio import shsdpa
    monkeypatch.setattr(shsdpa, "sdpa", lambda q, k, v, scale, pad, ckc=None: ("shsdpa", pad))
    monkeypatch.setattr(esmc.ttnn, "repeat", lambda t, reps: ("rep", t, tuple(reps)))
    monkeypatch.setattr(esmc, "_sdpa_k_few", lambda: False)
    monkeypatch.setattr(esmc, "_sdpa_program_config", lambda q, k: (q, k))
    monkeypatch.setattr(esmc, "fused_sdpa", lambda q, k, v, attn_mask=None, *, scale, program_config: ("stock", attn_mask))
    pad = esmc.KeyPad("table", "row")
    q = _T(dram, d, sharded)
    got = esmc.lm_sdpa(q, q, q, pad, scale=0.125)
    assert got == (("shsdpa", pad) if route == "shsdpa" else ("stock", ("rep", "row", (1, 1, 17, 1))))


def test_shsdpa_steps_down_the_q_chunk_on_an_l1_refusal(monkeypatch):
    """A q chunk whose circular buffers clash with L1 operands is re-planned one size smaller, and that shape keeps
    the cap; anything that is not an L1 refusal still raises."""
    from tt_bio import shsdpa, tenstorrent
    monkeypatch.setattr(shsdpa, "_Q_TILES_CAP", {})
    monkeypatch.setattr(shsdpa.ttnn, "allocate_tensor_on_device", lambda *a, **kw: "out")
    monkeypatch.setattr(tenstorrent, "COMPUTE_GRID_MAIN", (8, 9))
    tried = []

    def sg(dev, q, k, v, row, out, q_chunk, k_chunk, grid, ckc, scale, **kw):
        tried.append(q_chunk)
        if q_chunk > 288:
            raise RuntimeError("Statically allocated circular buffers in program 33 clash with L1 buffers")
        return q_chunk

    monkeypatch.setattr(shsdpa.SG, "sdpa", sg)
    q = _T(False)
    q.shape = q.padded_shape = (7, 20, 544, 64)
    q.device = lambda: "dev"
    assert shsdpa.sdpa(q, q, q, 0.125, ("table", "row")) == 288
    assert shsdpa.sdpa(q, q, q, 0.125, ("table", "row")) == 288
    assert tried == [544, 288, 288]

    def boom(*a, **kw):
        raise RuntimeError("something else")

    monkeypatch.setattr(shsdpa.SG, "sdpa", boom)
    with pytest.raises(RuntimeError, match="something else"):
        shsdpa.sdpa(q, q, q, 0.125, ("table", "row"))
