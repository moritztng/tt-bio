"""Mol-1's levers reach Mol-1 alone: protenix.mode_levers adds them only at c_z 448 in normal mode, MOL1_BH on Blackhole
and MOL1_WH on Wormhole, and leaves every other checkpoint's set as it was.

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
def test_mol1_set_only_at_448_normal(wh, fast, monkeypatch):
    base = _levers(128, wh, fast, monkeypatch)
    got = _levers(448, wh, fast, monkeypatch)
    assert got == (base if fast else base | P.WIDE_NORMAL | (P.MOL1_WH if wh else P.MOL1_BH))
    for c_z in (128, 256, 384):           # Protenix-v1, Protenix-v2, OpenDDE: no Mol-1 lever
        assert not _levers(c_z, wh, fast, monkeypatch) & {"trimul_t1_c320", "trimul_t1_c448"}
    assert P.MOL1_BH | P.MOL1_WH <= set(T.LEVERS)


def _size(c_z, wh, fast, monkeypatch, env=None):
    monkeypatch.setattr(T, "is_wormhole", lambda: wh)
    monkeypatch.setattr(T, "_FAST_MODE", fast)
    monkeypatch.delenv("TT_BIO_LEVERS", raising=False)
    if env:
        monkeypatch.setenv("TT_BIO_LEVERS", env)
    return P.size_levers({"layernorm_z_cycle.weight": torch.zeros(c_z)})


def test_size_set_only_mol1_bh_normal(monkeypatch):
    assert _size(448, False, False, monkeypatch) == P.MOL1_BH_512 <= set(T.LEVERS)
    assert not _size(448, True, False, monkeypatch)          # Wormhole
    assert not _size(448, False, True, monkeypatch)          # --fast names its whole set
    assert not _size(448, False, False, monkeypatch, env="normal")   # a harness set is exact
    for c_z in (128, 256, 384):
        assert not _size(c_z, False, False, monkeypatch)


@pytest.mark.parametrize("nt,big", [(245, False), (480, False), (481, True), (509, True), (1000, True)])
def test_fold_takes_size_set_from_512_padded_tokens(nt, big):
    class M:
        _levers, _size_levers = frozenset({"opm_jd"}), frozenset({"transition_b8"})

        @P._under_size_levers
        def fold(self, feats):
            return T._LEVERS

    feats = {"atom_to_token_idx": torch.tensor([0, nt - 1])}
    assert M().fold(feats) == (M._levers | M._size_levers if big else M._levers)
    assert M().fold([feats, feats]) == M().fold(feats)       # fold_many keys on the shared token count
