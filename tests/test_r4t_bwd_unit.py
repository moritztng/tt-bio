"""Pure-CPU unit tests for r4t-bwd (B1 penultimate reuse, B2/B3 config planner, B4 lnbw L1 budget)."""
from __future__ import annotations

import types

import pytest
import torch


def test_b1_penultimate_reuse_exact_against_reference():
    """Upstream LocalLatentsTransformer has zero dropout and zero cross-batch coupling:
    slicing penultimate (s_{d-1}[win], z_{d-1}[win]) from a K-candidate forward into cycle d-1
    is bit-exact in float64 to running the winner alone through cycles 0..d-1, and negating
    s_{d-1} (broken arm) diverges loudly."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import rfd4_reference as REF

    K, n, dim, dp, dc, h, nlayers = 4, 16, 64, 32, 32, 4, 3
    w = {k: v.double() for k, v in REF.trunk_weights(nlayers, dim, dp, dc, h, seed=11).items()}
    s0, z0, c0, mask = REF.block_inputs(K, n, dim, dp, dc, lengths=[16, 12, 14, 10], seed=3)
    gc = torch.Generator().manual_seed(19)
    c1 = torch.randn(K, n, dc, generator=gc, dtype=torch.float64)
    z0 = z0 * (mask[:, :, None] * mask[:, None, :])[..., None]
    s0 = s0 * mask[..., None]

    # K-candidate cycles 0..0 (penultimate), then cycle 1
    s_pen_k, z_pen_k = REF.trunk(w, s0, z0, [c0], mask, nlayers, n_heads=h)
    for win in range(K):
        sw_full, zw_full = REF.trunk(w, s0[win:win + 1], z0[win:win + 1], [c0[win:win + 1], c1[win:win + 1]],
                                     mask[win:win + 1], nlayers, n_heads=h)
        sw_reuse, zw_reuse = REF.trunk(w, s0[win:win + 1], z0[win:win + 1], [c1[win:win + 1]],
                                       mask[win:win + 1], nlayers, n_heads=h,
                                       state=(s_pen_k[win:win + 1], z_pen_k[win:win + 1]))
        assert torch.allclose(sw_reuse, sw_full, atol=1e-13, rtol=1e-13)
        assert torch.allclose(zw_reuse, zw_full, atol=1e-13, rtol=1e-13)

    # Broken arm: negated penultimate state must diverge strongly
    sw_bad, _ = REF.trunk(w, s0[0:1], z0[0:1], [c1[0:1]], mask[0:1], nlayers, n_heads=h,
                          state=(-s_pen_k[0:1], z_pen_k[0:1]))
    sw_ref, _ = REF.trunk(w, s0[0:1], z0[0:1], [c0[0:1], c1[0:1]], mask[0:1], nlayers, n_heads=h)
    rel = float((sw_bad - sw_ref).norm() / sw_ref.norm())
    assert rel > 0.05, f"broken arm did not diverge enough: rel={rel}"


def test_mmvjp_config_fills_grid_on_rfd4_dw_shapes():
    """Unified grid-filling MinimalMatmulConfig planner must light up >= 72 cores on BH p300c/Galaxy
    for every RFD4 PairUpdate dW shape instead of falling back to 2-8 cores."""
    import ttnn
    from tt_bio import autograd as ag

    class FakeDev:
        def __init__(self, x, y):
            self._g = ttnn.CoreCoord(x, y)

        def compute_with_storage_grid_size(self):
            return self._g

    dev_bh = FakeDev(11, 10)  # 110 cores (p300c)
    for M, N in ((256, 384), (384, 384), (384, 1024), (384, 768), (768, 384)):
        cfg = ag._mmvjp_config(M, 65536, N, {}, dev_bh, fill_small=True)
        assert cfg is not None
        Mt, Nt = M // 32, N // 32
        blocks = (Mt // cfg.M_block_size) * (Nt // cfg.N_block_size)
        assert blocks >= 72, f"{M}x{N}: only {blocks} blocks ({cfg})"
        assert cfg.subblock_h * cfg.subblock_w <= 4
        assert cfg.M_block_size % cfg.subblock_h == 0
        assert cfg.N_block_size % cfg.subblock_w == 0


def test_trunk_call_prev_and_return_penultimate_ownership(monkeypatch):
    """Trunk.__call__(..., prev=..., return_penultimate=True) never deallocates caller-owned prev
    or returned penultimate tensors, and matches 2-cycle output when split into pre + final."""
    import ttnn
    from tt_bio.rfd4 import trunk as TR

    freed = []

    class MockT:
        def __init__(self, val, dtype=ttnn.float32):
            self.val = val
            self.dtype = dtype

    monkeypatch.setattr(ttnn, "clone", lambda t: MockT(t.val, t.dtype))
    monkeypatch.setattr(ttnn, "deallocate", lambda t: freed.append(t))

    t = object.__new__(TR.Trunk)
    t.dtype = ttnn.bfloat16
    t.residual_dtype = ttnn.float32
    t.pair_residual_dtype = ttnn.float32
    t._levers = frozenset()
    t.s_norm = t.s_proj = t.z_norm = t.z_proj = None

    def fake_recycle(init, prev, _nw, _pw, res_dtype, low=None):
        out = MockT(init.val + 10 * prev.val, res_dtype)
        return (out, MockT(out.val, ttnn.bfloat16)) if low is not None else out

    def fake_run_pass(s_in, z_in, c, _masks, zb=None):
        return MockT(s_in.val + c, ttnn.float32), MockT(z_in.val + 2 * c, ttnn.float32)

    t._recycle = fake_recycle
    t.run_pass = fake_run_pass
    masks = (None, None, None, None, None, None)
    s0, z0 = MockT(1), MockT(2)

    s_full, z_full, pen = t(s0, z0, [3, 5], masks, return_penultimate=True)
    assert pen[0] is not None and pen[1] is not None
    assert pen[0] not in freed and pen[1] not in freed

    freed.clear()
    s_prev, z_prev = pen
    s_split, z_split = t(s0, z0, [5], masks, prev=(s_prev, z_prev))
    assert s_split.val == s_full.val and z_split.val == z_full.val
    assert s_prev not in freed and z_prev not in freed


def test_lnbw_rms_l1_budget_accepts_k768_and_declines_ln_k768():
    """CENTER=0, DO_BETA=0 (RMSNorm) frees enough L1 to fit K=768 with gamma weights,
    while CENTER=1, DO_BETA=1 (LayerNorm) at K=768 stays declined for L1."""
    import ttnn
    from tt_bio import lnbw

    class FakeMC:
        memory_layout = ttnn.TensorMemoryLayout.INTERLEAVED

    class FakeT:
        def __init__(self, shape, dtype=ttnn.bfloat16):
            self.shape = shape
            self.dtype = dtype
            self.layout = ttnn.TILE_LAYOUT

        def memory_config(self):
            return FakeMC()

    old_rms, old_w = lnbw.RMS, lnbw.WEIGHTS
    try:
        lnbw.RMS, lnbw.WEIGHTS = True, True
        x768, g768, w768 = FakeT([1, 256, 768]), FakeT([1, 256, 768]), FakeT([1, 768])
        x384, g384 = FakeT([1, 256, 256, 384]), FakeT([1, 256, 256, 384])
        assert lnbw.eligible(x768, g768, w768, weights=True, center=False, do_beta=False) is True
        assert lnbw.eligible(x768, g768, None, weights=False, center=False) is True
        assert lnbw.eligible(x384, g384, None, weights=False, center=False) is True
        assert lnbw.eligible(x768, g768, w768, weights=True, center=True, do_beta=True) is False
    finally:
        lnbw.RMS, lnbw.WEIGHTS = old_rms, old_w
