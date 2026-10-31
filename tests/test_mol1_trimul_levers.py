"""Mol-1's trimul levers reach Mol-1 alone: protenix.mode_levers adds MOL1_BH only at c_z 448, Blackhole normal.

Run: TT_VISIBLE_DEVICES= python3 -m pytest tests/test_mol1_trimul_levers.py
"""
import pytest
import torch

from tt_bio import protenix as P
from tt_bio import tenstorrent as T


def _levers(c_z, wh, fast, monkeypatch):
    monkeypatch.setattr(T, "is_wormhole", lambda: wh)
    monkeypatch.setattr(T, "_FAST_MODE", fast)
    monkeypatch.delenv("TT_BIO_LEVERS", raising=False)
    return P.mode_levers({"layernorm_z_cycle.weight": torch.zeros(c_z)})


@pytest.mark.parametrize("wh", [False, True])
@pytest.mark.parametrize("fast", [False, True])
def test_mol1_set_only_at_448_bh_normal(wh, fast, monkeypatch):
    base = _levers(128, wh, fast, monkeypatch)
    got = _levers(448, wh, fast, monkeypatch)
    assert got == (base | P.MOL1_BH if not wh and not fast else base)
    for c_z in (128, 256, 384):           # Protenix-v1, Protenix-v2, OpenDDE: no Mol-1 lever
        assert not _levers(c_z, wh, fast, monkeypatch) & {"trimul_t1_c320", "trimul_t1_c448"}
    assert P.MOL1_BH <= set(T.LEVERS)
