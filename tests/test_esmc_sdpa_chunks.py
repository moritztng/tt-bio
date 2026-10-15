"""The LM SDPA chunk choice, and lm_sdpa's fall back to k64 when the device refuses the few-k program."""
from __future__ import annotations

import types

import pytest

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
