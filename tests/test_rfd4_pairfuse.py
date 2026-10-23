"""RFD4's Hadamard pair write folded into the output triangle (`pair_outer_fold`) on the card.

Under `pair_bf16z` the unfused path writes z' = z + pj * pi with `outer_residual` (rounded once to even) and the
output triangle reads it twice: T1 normalises it, the tail adds its update to it. The lever has T1 and the tail form
z' themselves with outer_residual's own sequence, so z' is never written. The pair update must come out bit for bit
as the unfused one, and both forms must have served (a declined form falls back to the same bits, so equality alone
would not show the fold ran). The update is also graded against float64 (bar 0.05).

Run: TT_VISIBLE_DEVICES=<card> TT_BIO_LEASE_CARDS=<card> python3 -m pytest tests/test_rfd4_pairfuse.py
"""
import pytest
import torch
import ttnn

from tt_bio import tenstorrent as T
from tt_bio import trimul_tail as TT1
from tt_bio.rfd4 import trunk as TR

import rfd4_reference as REF

BAR = 5e-2
BASE = TR.NORMAL_LEVERS | {"pair_bf16z"}


def _dev(t, dev):
    return ttnn.from_torch(t.to(torch.float32).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _update(levers, w, x, z, dev):
    """One pair update under `levers`, from host x [1, n, 768] and the bf16 pair z [1, n, n, 384]."""
    with T.lever_scope(frozenset(levers)):
        mod = TR.PairUpdate(T.WeightScope({k: v.to(torch.float32) for k, v in w.items()}).child("pu"),
                            TR.compute_kernel_config(), residual_dtype=ttnn.bfloat16)
        mod.run(_dev(x, dev), _dev(z, dev), None, None, None)   # warmup: a first trimul may retry without T1
        before = list(TT1.OUTER_STATS)
        zt = _dev(z, dev)
        out, _ = mod.run(_dev(x, dev), zt, None, None, None)
        served = [a - b for a, b in zip(TT1.OUTER_STATS, before)]
    return ttnn.to_torch(out).double().reshape(z.shape), served


@pytest.mark.device
@pytest.mark.parametrize("n", [256, 192])
def test_pair_outer_fold_bit_identical(n):
    dev = T.get_device()
    w = REF.pair_update_weights(768, 384, 256, seed=5, prefix="pu")
    x, z, _, _ = REF.block_inputs(1, n, 768, 384, 256, seed=1)
    zb = REF.bf16(z.double())
    base, served0 = _update(BASE, w, x, zb, dev)
    fold, served1 = _update(BASE | {"pair_outer_fold"}, w, x, zb, dev)
    print(f"\npair_outer_fold n={n}: forms served T1/tail {served1}, equal {torch.equal(base, fold)}, "
          f"max |diff| {(base - fold).abs().max().item():.3e}")
    assert served0 == [0, 0] and served1 == [1, 1]
    assert torch.equal(base, fold)


@pytest.mark.device
def test_pair_outer_fold_tri_out_vs_float64():
    """The output triangle's update on the folded path against float64 on z' (the bf16 sum outer_residual writes)."""
    dev = T.get_device()
    n = 256
    w = REF.pair_update_weights(768, 384, 256, seed=5, prefix="pu")
    x, z, _, _ = REF.block_inputs(1, n, 768, 384, 256, seed=1)
    zb = REF.bf16(z.double())
    with T.lever_scope(frozenset(BASE | {"pair_outer_fold"})):
        mod = TR.PairUpdate(T.WeightScope({k: v.to(torch.float32) for k, v in w.items()}).child("pu"),
                            TR.compute_kernel_config(), residual_dtype=ttnn.bfloat16)
        pi, pj = mod._factors(_dev(x, dev), None)
        pj4 = ttnn.reshape(pj, (1, n, 1, 384))
        mod.tri_out(_dev(zb, dev), None)   # warmup
        zt = _dev(zb, dev)
        got = mod.tri_out(zt, None, stream32=zt, outer=(pi, pj4))
        assert got is zt and mod.tri_out.took_residual
        pih = ttnn.to_torch(pi).double().reshape(1, n, 384)
        pjh = ttnn.to_torch(pj).double().reshape(1, n, 384)
    zp = REF.bf16(zb + pjh[:, :, None, :] * pih[:, None, :, :])
    ref = REF.tri_mult({k: v.double() for k, v in w.items()}, "pu.tri_mult_out", zp,
                       torch.ones(1, n, n, dtype=torch.float64), True)
    upd = ttnn.to_torch(zt).double().reshape(zp.shape) - zp
    err = ((upd - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    print(f"\npair_outer_fold tri_out: update rel_rms {err:.3e} vs float64")
    assert err < BAR
