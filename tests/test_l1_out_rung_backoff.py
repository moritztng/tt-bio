"""A refused L1 output loses one drain rung; it does not retire L1 for the whole shape class.

Two gates in `tt_bio/tenstorrent.py` ask the device for something the static budget said would
fit, because that budget reads the IDLE L1 bank and cannot see what the live block already holds:
the pair-projection L1 output (`_L1_OUT_RUNG`) and the batched-matmul program config
(`_BMM_CFG_RUNG`). Both used to retire the whole shape class on the first refusal, which is the
bug `rf3-1024aa-exponent-gate` fixed in the fp32-softmax tail and left standing here.

MEASURED on the RF3 trunk at 768 aa (qb2 card 1, perf/rf3/latch_census.sh): the triangle attention
out-projection class `(1,768,768,64) x (64,64)` served 17 calls, hit
"Statically allocated circular buffers in program 173 clash with L1 buffers" on the 18th, and the
retirement then sent the remaining 30 of 48 calls back through a DRAM round trip. `_BMM_CFG_RUNG`
did not fire at either 768 or 1024 aa (0 refusals in 166278 and 442226 served calls), so its ladder
is insurance rather than a measured win.

Every rung leaves `in0_block_w` alone -- `out_block_h`, `out_block_w` and `per_core_M` are drain and
occupancy parameters -- so the contraction accumulates in the same order on every rung. That is what
makes the walk bit-exact, verified on device in perf/rf3/latch_rung_bits.py.

Host-only: no device, no fold. The refusal itself needs hardware and lives in that harness.
"""
from __future__ import annotations

import pytest

from tt_bio import tenstorrent as T

# The real RF3 768 aa triangle attention out-projection class, read off the census. `per_core_M`
# follows the core grid, so it is derived the way the factory derives it and not pinned to one host.
RF3_M_TILES = 18432         # x = (1, 768, 768, 64): 768 batch rows of ceil(768/32) tiles
RF3_N_TILES = 2             # w = (64, 64)
RF3_CORES = T.COMPUTE_GRID_MAIN[0] * T.COMPUTE_GRID_MAIN[1]
RF3_PER_CORE_M = -(-(-(-RF3_M_TILES // RF3_CORES)) // 5) * 5
BANK = 1461760              # Blackhole L1 bytes per bank, what _l1_bank_bytes() reports on qb2


@pytest.fixture(autouse=True)
def clean_state():
    saved = dict(T._L1_OUT_RUNG), set(T._L1_OUT_REFUSED), dict(T._BMM_CFG_RUNG)
    for c in (T._L1_OUT_RUNG, T._BMM_CFG_RUNG):
        c.clear()
    T._L1_OUT_REFUSED.clear()
    yield
    T._L1_OUT_RUNG.clear()
    T._L1_OUT_RUNG.update(saved[0])
    T._L1_OUT_REFUSED.clear()
    T._L1_OUT_REFUSED.update(saved[1])
    T._BMM_CFG_RUNG.clear()
    T._BMM_CFG_RUNG.update(saved[2])


# ---------------------------------------------------------------- the drain-block ladder

def test_rung_zero_is_the_callers_ask():
    """A class that never refuses must see exactly the shipped config, on both call shapes."""
    assert T._pair_proj_l1_rungs(RF3_PER_CORE_M, RF3_N_TILES)[0] == (5, 2)
    # the pair FFN fc1 site: out_block_w = 16 of n_tiles = 32
    assert T._pair_proj_l1_rungs(RF3_PER_CORE_M, 16)[0] == (5, 16)


def test_the_rf3_class_has_three_rungs_below_the_shipped_one():
    """The regression this file exists for: a refusal narrows, it does not retire."""
    assert T._pair_proj_l1_rungs(RF3_PER_CORE_M, RF3_N_TILES) == [(5, 2), (5, 1), (1, 2), (1, 1)]


def test_every_rung_divides_and_shrinks():
    for per_core_M, ask in ((145, 16), (170, 2), (130, 8), (23, 4)):
        rungs = T._pair_proj_l1_rungs(per_core_M, ask)
        assert rungs, (per_core_M, ask)
        for h, w in rungs:
            assert per_core_M % h == 0 and ask % w == 0, (per_core_M, ask, h, w)
        cost = [h * w for h, w in rungs]
        assert cost == sorted(cost, reverse=True), rungs
        assert len(set(rungs)) == len(rungs)


def test_a_prime_per_core_m_still_has_a_ladder():
    """out_block_h = 5 does not divide 23, so the ladder starts at 1 rather than being empty."""
    assert T._pair_proj_l1_rungs(23, 2) == [(1, 2), (1, 1)]


def test_a_refusal_narrows_the_class_by_one_rung():
    key = ((1, 768, 768, 64), (64, 64), "DataType.BFLOAT16", None, None)
    assert T._l1_out_rung(key) == 0
    T._l1_out_narrow(key)
    assert T._l1_out_rung(key) == 1
    T._l1_out_narrow(key)
    assert T._l1_out_rung(key) == 2
    assert key not in T._L1_OUT_REFUSED, "narrowing is not retirement"


def test_the_config_factory_walks_the_ladder_and_never_moves_in0_block_w(monkeypatch):
    monkeypatch.setattr(T, "_l1_bank_bytes", lambda: BANK)
    T._pair_proj_program_config.cache_clear()
    seen = []
    for rung in range(6):
        cfg = T._pair_proj_program_config(RF3_M_TILES, 2, 2, 2, 2, True, None, rung)
        if cfg is None:
            break
        seen.append((cfg.out_block_h, cfg.out_block_w))
        assert cfg.in0_block_w == 2, "in0_block_w is the one parameter parity depends on"
        assert cfg.per_core_M == RF3_PER_CORE_M and cfg.per_core_N == RF3_N_TILES
    assert seen == [(5, 2), (5, 1), (1, 2), (1, 1)], seen
    T._pair_proj_program_config.cache_clear()


def test_the_ladder_runs_out_rather_than_returning_a_wider_config(monkeypatch):
    monkeypatch.setattr(T, "_l1_bank_bytes", lambda: BANK)
    T._pair_proj_program_config.cache_clear()
    assert T._pair_proj_program_config(RF3_M_TILES, 2, 2, 2, 2, True, None, 4) is None
    assert T._pair_proj_program_config(RF3_M_TILES, 2, 2, 2, 2, True, None, -1) is None
    T._pair_proj_program_config.cache_clear()


def test_a_narrower_rung_frees_circular_buffer_bytes(monkeypatch):
    """The point of the walk: rung 1 has to actually be smaller, or the retry is pointless."""
    monkeypatch.setattr(T, "_l1_bank_bytes", lambda: BANK)
    T._pair_proj_program_config.cache_clear()

    def cb(cfg):
        tile = 2048
        return (2 * cfg.in0_block_w * (cfg.out_block_h + cfg.out_block_w) * tile
                + cfg.out_block_h * cfg.out_block_w * (tile + 4096))

    costs = [cb(T._pair_proj_program_config(RF3_M_TILES, 2, 2, 2, 2, True, None, r)) for r in range(4)]
    assert costs == sorted(costs, reverse=True) and len(set(costs)) == 4, costs
    T._pair_proj_program_config.cache_clear()


# --------------------------------------------------------------- the per_core_M ladder

# A class with several legal `per_core_M`, which is what there is to narrow: 256 tile rows over a
# 130-core grid, so 8 is the tuned choice and 4 and 2 are the rungs below it.
BMM = dict(batch=32, m_tiles=8, k_tiles=2, n_tiles=1, elem_bytes=2, grid=(13, 10), l1=BANK)


def test_batched_matmul_rung_zero_is_the_tuned_choice():
    T._batched_matmul_search.cache_clear()
    base = T._batched_matmul_search(**BMM)
    assert base is not None
    assert T._batched_matmul_search(**BMM, rung=0).per_core_M == base.per_core_M


def test_batched_matmul_walks_per_core_m_down_and_never_moves_in0_block_w():
    T._batched_matmul_search.cache_clear()
    base = T._batched_matmul_search(**BMM)
    seen, rung = [], 0
    while True:
        cfg = T._batched_matmul_search(**BMM, rung=rung)
        if cfg is None:
            break
        seen.append(cfg.per_core_M)
        assert cfg.in0_block_w == base.in0_block_w, "the one parameter parity depends on"
        assert cfg.per_core_N == BMM["n_tiles"]
        rung += 1
    assert len(seen) > 1, "a class with one legal per_core_M has nothing to narrow to"
    assert seen == sorted(seen, reverse=True) and len(set(seen)) == len(seen), seen
    assert seen[0] == base.per_core_M


def test_every_batched_matmul_rung_keeps_the_correctness_filter():
    """`per_core_M != m_tiles` with more blocks than cores returns WRONG results, not slow ones."""
    T._batched_matmul_search.cache_clear()
    cores = BMM["grid"][0] * BMM["grid"][1]
    rung = 0
    while True:
        cfg = T._batched_matmul_search(**BMM, rung=rung)
        if cfg is None:
            break
        p = cfg.per_core_M
        assert BMM["m_tiles"] % p == 0
        assert p == BMM["m_tiles"] or BMM["batch"] * BMM["m_tiles"] // p <= cores, p
        rung += 1


def test_batched_matmul_ladder_runs_out_rather_than_wrapping():
    T._batched_matmul_search.cache_clear()
    rung = 0
    while T._batched_matmul_search(**BMM, rung=rung) is not None:
        rung += 1
        assert rung < 64
    assert T._batched_matmul_search(**BMM, rung=rung + 5) is None
    assert T._batched_matmul_search(**BMM, rung=-1) is None


# ------------------------------------------- who may walk the ladder, and who must retire

class _FakeTensor:
    def __init__(self, shape, dtype="bf16"):
        self.shape = self.padded_shape = tuple(shape)
        self.dtype = dtype


def _drive_pair_proj(monkeypatch, l1_block_w, calls=3):
    """Call `_pair_proj_linear` with the device refusing every L1 destination.

    Returns the rung the class ended on, whether it was retired, and how many times it
    asked for an L1 destination at all. `_pair_proj_config` and `ttnn.linear` are
    the only device-facing pieces, so stubbing those two leaves the dispatch under test.
    """
    monkeypatch.setattr(T, "_pair_proj_config", lambda *a, **k: "cfg")
    monkeypatch.setattr(T, "_pair_proj_minimal_matmul", lambda *a, **k: None)
    attempts = []

    def fake_linear(x, w, **kw):
        l1 = kw.get("memory_config") is T.ttnn.L1_MEMORY_CONFIG
        attempts.append(l1)
        if l1:
            raise RuntimeError("Statically allocated circular buffers clash with L1 buffers")
        return "dram_out"

    monkeypatch.setattr(T.ttnn, "linear", fake_linear)
    x, w = _FakeTensor((1, 32, 768, 256)), _FakeTensor((256, 1024))
    for _ in range(calls):
        T._pair_proj_linear(x, w, None, "bf16", l1_out=True, l1_bw=1, l1_block_w=l1_block_w)
    key = (x.padded_shape, w.shape, "bf16", 1, l1_block_w)
    return T._L1_OUT_RUNG.get(key, 0), key in T._L1_OUT_REFUSED, attempts.count(True)


def test_a_gate_chosen_block_walks_the_ladder(monkeypatch):
    """No named block, so the gate picked the drain schedule and owns narrowing it."""
    rung, retired, l1_attempts = _drive_pair_proj(monkeypatch, None)
    assert (rung, retired) == (3, False)
    assert l1_attempts == 3          # every call still tried an L1 destination, one rung lower


def test_a_caller_named_block_retires_instead_of_narrowing(monkeypatch):
    """The pair FFN's fc1 names its own swept block, and its consumer -- not this gate -- is what
    runs out of L1. MEASURED: narrowing it instead of retiring kills an ESMFold2 768 aa fold."""
    rung, retired, l1_attempts = _drive_pair_proj(monkeypatch, 16)
    assert (rung, retired) == (0, True)
    assert l1_attempts == 1          # one refusal, then the class is done asking
