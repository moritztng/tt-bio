"""Where `triatt_sdpa.split_q_chunk` puts a padded q chunk ahead of the dividing ladder, and where not.

The pick is bit-exact by construction (same k chunk, padded rows dropped), so what can regress is
only WHICH shapes take it. Two properties are locked: the Mol-1 Wormhole cell it was measured on
(1024 padded, 10 heads, 72 cores: q160 runs 107.3 -> 70.5 ms, perf/mol1_triattn/split_ab.py) still
takes it, and the 4-head models (Protenix-v2, OpenDDE, Boltz-2, OpenFold3) never do on any grid.

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


@pytest.mark.parametrize("seq, heads", [(512, 10), (512, 14), (1024, 14)])
def test_512_and_the_14_head_cell_stay_on_the_ladder(seq, heads):
    # 512 already lights 70 of 72 cores; 1024 x 14 is predicted 12.5 %, under the 20 % margin.
    assert _pick(seq, heads) == 0


@pytest.mark.parametrize("cores", [64, 72, 110, 130])
@pytest.mark.parametrize("dtype", ["bfloat16", "bfloat8_b"])
def test_four_heads_never_fire(cores, dtype):
    for seq in range(256, 1025, 32):
        assert _pick(seq, 4, cores, getattr(ttnn, dtype)) == 0, seq


def test_above_the_cap_returns_zero():
    assert _pick(TS._Q_SPLIT_MAX_S + 32, 10) == 0


def test_off_switch(monkeypatch):
    monkeypatch.setattr(TS, "SPLIT_Q", False)
    TS.split_q_chunk.cache_clear()
    try:
        assert _pick(1024, 10) == 0
    finally:
        TS.split_q_chunk.cache_clear()
