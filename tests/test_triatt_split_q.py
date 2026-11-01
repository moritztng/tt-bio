"""Where `triatt_sdpa.split_q_chunk` puts a padded q chunk ahead of the dividing ladder, and where not.

The pick is bit-exact by construction (same k chunk, padded rows dropped), so what can regress is
only WHICH shapes take it. Two properties are locked: the Mol-1 Wormhole cell it was measured on
(1024 padded, 10 heads, 72 cores: q160 runs 107.3 -> 70.5 ms, perf/mol1_triattn/split_ab.py) still
takes it, and the 4-head models (Protenix-v2, OpenDDE, Boltz-2, OpenFold3) take it on Wormhole only at
the one measured shape, 800 tokens on 72 cores (1.20x, torch.equal).

Opens no device.
"""

import pytest

ttnn = pytest.importorskip("ttnn")

from tt_bio import tenstorrent as T                                          # noqa: E402
from tt_bio import triatt_sdpa as TS                                         # noqa: E402

GALAXY_WH = 72


def _pick(seq, heads, cores=GALAXY_WH, dtype=None):
    dtype = dtype or ttnn.bfloat16
    k = T._sdpa_chunks_shipped(seq, seq)[1]
    return TS.split_q_chunk(seq, heads, 32, cores, k, dtype, tuple(T._tri_att_q_chunks(seq, seq)))


def test_mol1_trunk_cell_takes_the_padded_chunk():
    assert _pick(1024, 10) == 160


@pytest.mark.parametrize("seq, heads", [(512, 10), (512, 14)])
def test_512_stays_on_the_ladder(seq, heads):
    # 512 already lights 70 of 72 cores.
    assert _pick(seq, heads) == 0


def test_mol1_14_head_cell_takes_the_padded_chunk():
    # predicted 12.5 %, measured 1.14x at the 12 % margin.
    assert _pick(1024, 14) == 224


@pytest.mark.parametrize("cores", [64, 72])
@pytest.mark.parametrize("dtype", ["bfloat16", "bfloat8_b"])
def test_four_heads_fire_only_at_the_measured_shape(cores, dtype):
    fired = {seq for seq in range(256, 1025, 32) if _pick(seq, 4, cores, getattr(ttnn, dtype))}
    assert fired == ({800} if (cores, dtype) == (72, "bfloat16") else set())


def test_above_the_cap_returns_zero():
    assert _pick(TS._Q_SPLIT_MAX_S + 32, 10) == 0


def test_off_switch(monkeypatch):
    monkeypatch.setattr(TS, "SPLIT_Q", False)
    TS.split_q_chunk.cache_clear()
    try:
        assert _pick(1024, 10) == 0
    finally:
        TS.split_q_chunk.cache_clear()


@pytest.mark.parametrize("on, declines, want", [
    (False, (), [256]),             # off: the split q at the ladder's k only
    (True, (), [1024]),             # triatt_k1s: the whole row first
    (True, (1024,), [1024, 256]),   # one k over L1: sdpa declines it, the ladder's k serves
])
def test_k1s_offers_the_split_q_one_k_chunk_first(monkeypatch, on, declines, want):
    from types import SimpleNamespace as NS
    t = NS(shape=(1, 10, 1024, 32), dtype=ttnn.bfloat16)
    calls = []

    def sdpa(q, k, v, bias, scale, qc, kc, **kw):
        calls.append(kc)
        assert qc == 160 and kw["padded_mask"]
        return None if kc in declines else "o"

    monkeypatch.setattr(T, "_tri_att_fused_large_s", lambda *a: None)
    monkeypatch.setattr(T, "_tri_att_k_chunks", lambda *a: (256,))
    monkeypatch.setattr(T, "is_wormhole", lambda: True)
    monkeypatch.setattr(TS, "split_q_chunk", lambda *a: 160)
    monkeypatch.setattr(TS, "sdpa", sdpa)
    with T.levers({"triatt_k1s"} if on else ()):
        assert T._tri_att_sdpa_at(t, t, t, t, 1.0) == "o"
    assert calls == want and T.SDPA_CHUNK_PICKS[(1024, 1024)] == [160, want[-1], "fused"]
