"""The swept qkv/gate config at a ragged token count is AF2's alone.

Dropping the row guard kept AF2-IG bit-exact and saved 0.12 s per complex, but OpenFold3 and
OpenBind-0 fold some targets at a ragged token count too and their structures moved with it
(spd-integrate d60). So the relaxation is opt-in per attention, and AF2's two opt in.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tt_bio import af2  # noqa: E402
from tt_bio import tenstorrent as TT  # noqa: E402


def _t(*shape):
    return SimpleNamespace(shape=shape)


@pytest.fixture
def config_on(monkeypatch):
    monkeypatch.setattr(TT, "_MM_CFG", True)
    monkeypatch.setattr(TT, "_MM_RAGGED_M", True)
    monkeypatch.setattr(TT.ttnn, "MinimalMatmulConfig", lambda **kw: kw, raising=False)


def test_ragged_rows_take_the_config_only_when_the_attention_asks(config_on):
    x, w = _t(195, 195, 128), _t(128, 384)       # c_z 128 qkv: kt 4, nt 12, M_block 4
    assert TT._qkv_mm_config(x, w) is None
    assert TT._qkv_mm_config(x, w, ragged_m=True)["M_block_size"] == 4


def test_aligned_rows_are_unchanged_either_way(config_on):
    x, w = _t(224, 224, 128), _t(128, 384)
    assert TT._qkv_mm_config(x, w) == TT._qkv_mm_config(x, w, ragged_m=True) is not None


def test_the_flag_is_a_kill_switch(config_on, monkeypatch):
    monkeypatch.setattr(TT, "_MM_RAGGED_M", False)
    assert TT._qkv_mm_config(_t(195, 195, 128), _t(128, 384), ragged_m=True) is None


def test_attention_defaults_off_and_af2_opts_in():
    assert inspect.signature(TT.TriangleAttention.__init__).parameters["ragged_m"].default is False
    src = inspect.getsource(af2.AF2PairBlock.__init__)
    assert src.count("ragged_m=True)") == 2
