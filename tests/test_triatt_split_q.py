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


@pytest.mark.parametrize("heads, want", [(10, 224), (14, 160), (8, 0), (12, 0)])
def test_blackhole_picks_at_the_wide_k(heads, want):
    # p150a/p300 grid 11x10, the served k1024: measured 1.14x (10 heads) and 1.27x (14), 8 and 12 unchanged.
    TS.split_q_chunk.cache_clear()
    got = TS.split_q_chunk(1024, heads, 32, 110, 1024, ttnn.bfloat16, tuple(T._tri_att_q_chunks(1024, 1024)),
                           TS.SPLIT_Q_ITER_ROWS["blackhole"], TS.SPLIT_Q_MARGIN["blackhole"])
    assert got == want


def test_above_the_cap_returns_zero():
    assert _pick(TS._Q_SPLIT_MAX_S + 32, 10) == 0


def test_off_switch(monkeypatch):
    monkeypatch.setattr(TS, "SPLIT_Q", False)
    TS.split_q_chunk.cache_clear()
    try:
        assert _pick(1024, 10) == 0
    finally:
        TS.split_q_chunk.cache_clear()
