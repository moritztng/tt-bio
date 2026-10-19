"""`sdpa_generic.cb_budget` prices a Wormhole core by its own L1, not Blackhole's.

On a Wormhole Galaxy chip (.107, 10-18) OpenFold3 --fast at 512 aa built the fused-HiFi triangle attention at
q512/k512 and the device refused it twice per process: CB region ends 1584416 B (kv double-buffered) and 1518880 B
(single), against an L1 of 1499136 B. Both are the CB base 103712 B plus `cb_bytes`, so the plan is priced exactly
and the budget was what was wrong. q256/k512, the config that then served, must still fit.

Run: python3 -m pytest tests/test_sdpa_cb_budget_wormhole.py   (needs ttnn importable, no device)
"""
import ttnn

from tt_bio import sdpa_generic as SG
from tt_bio import triatt_sdpa as TS

WH_BASE = 1499136 - SG.WH_CB_BUDGET
HIFI = (ttnn.MathFidelity.HiFi4, False, True, False)


def _bytes(q_chunk, kvbf, seq=512, heads=4, grid=(8, 9)):
    cores = grid[0] * grid[1]
    qpf = TS.q_parallel_factor(seq, heads, q_chunk, cores)
    t = SG._ShapeOnly([seq, heads, seq, 32], ttnn.bfloat16)
    bias = SG._ShapeOnly([1, heads, seq, seq], ttnn.bfloat16)
    p = SG.plan(t, t, t, bias, t, q_chunk, seq, grid, HIFI, 1.0, (cores // (heads * qpf), heads, qpf), kvbf)
    return SG.cb_bytes(p, mask_cb_tiles=p["k_num_chunks"] * p["Sq_chunk_t"] * p["Sk_chunk_t"])


def test_the_two_wormhole_refusals_are_base_plus_cb_bytes():
    assert WH_BASE + _bytes(512, 2) == 1584416
    assert WH_BASE + _bytes(512, 1) == 1518880


def test_wormhole_budget_declines_them_and_keeps_the_config_that_served():
    assert _bytes(512, 2) > SG.cb_budget(wormhole=True)
    assert _bytes(512, 1) > SG.cb_budget(wormhole=True)
    assert _bytes(256, 2) <= SG.cb_budget(wormhole=True)


def test_blackhole_budget_is_unchanged():
    assert SG.cb_budget(wormhole=False) == 1572864 - 109056


def test_triatt_k1_one_chunk_config_fits_wormhole_as_built():
    # triatt_k1's (q512, k512) is one k chunk, so `build` allocates the trimmed table. The pre-build
    # price must use it too: the untrimmed one refused every call on Wormhole (cen12, 10-19).
    cores = 72
    qpf = TS.q_parallel_factor(512, 4, 512, cores)
    t = SG._ShapeOnly([512, 4, 512, 32], ttnn.bfloat16)
    bias = SG._ShapeOnly([1, 4, 512, 512], ttnn.bfloat16)
    p = SG.plan(t, t, t, bias, t, 512, 512, (8, 9), HIFI, 1.0, (cores // (4 * qpf), 4, qpf), 2)
    pers = p["k_num_chunks"] * p["Sq_chunk_t"] * p["Sk_chunk_t"]
    assert p["k_num_chunks"] == 1
    assert SG.cb_bytes(p, mask_cb_tiles=pers, one_k_trim=True) <= SG.cb_budget(wormhole=True)
