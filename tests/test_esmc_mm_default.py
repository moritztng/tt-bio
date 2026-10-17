"""ESMC's minimal_matmul projections default by arch: on for Wormhole, off for Blackhole; the switch overrides both."""
from __future__ import annotations

import types

import pytest

from tt_bio import esmc, tenstorrent


@pytest.mark.parametrize("wormhole", [True, False])
@pytest.mark.parametrize("switch", [None, True, False])
def test_esmc_mm_default_by_arch(wormhole, switch, monkeypatch):
    monkeypatch.setattr(tenstorrent, "is_wormhole", lambda: wormhole)
    monkeypatch.setattr(esmc, "_ESMC_MM", switch)
    assert esmc._esmc_mm() is (wormhole if switch is None else switch)


class _Ck:
    def __init__(self, math_fidelity=None, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False):
        self.math_fidelity, self.math_approx_mode = math_fidelity, math_approx_mode
        self.fp32_dest_acc_en, self.packer_l1_acc = fp32_dest_acc_en, packer_l1_acc


class _T:
    def __init__(self, rows, cols, dtype):
        self.shape, self.dtype, self._v = (rows, cols), dtype, rows * cols

    def volume(self):
        return self._v

    def device(self):
        return types.SimpleNamespace(compute_with_storage_grid_size=lambda: types.SimpleNamespace(x=8, y=8))


@pytest.mark.parametrize("act", [None, "silu"])
def test_mm_carries_fc1_silu(act, monkeypatch):
    """The SwiGLU's fc1a (activation="silu") takes minimal_matmul with the SiLU fused, not the linear."""
    calls = []
    monkeypatch.setattr(esmc, "_ESMC_MM", True)
    monkeypatch.setattr(esmc.ttnn.experimental, "minimal_matmul", lambda x, w, **kw: calls.append(kw) or "mm",
                        raising=False)
    lin = esmc._MMLin()
    lin.compute_kernel_config = None
    bf16 = esmc.ttnn.bfloat16
    kw = {"activation": act} if act else {}
    assert lin._lin(_T(esmc.ESMC_MM_MIN_ROWS, 960, bf16), _T(960, 2560, bf16), **kw) == "mm"
    assert ("fused_activation" in calls[0]) is bool(act)


def test_mm_takes_fast_block_fp8_weights(monkeypatch):
    """--fast's block-fp8 projection weights take minimal_matmul by default, like the bf16 ones."""
    monkeypatch.setattr(esmc, "_ESMC_MM", True)
    monkeypatch.setattr(esmc.ttnn.experimental, "minimal_matmul", lambda x, w, **kw: "mm", raising=False)
    lin = esmc._MMLin()
    lin.compute_kernel_config = _Ck()
    t = esmc.ttnn
    assert lin._lin(_T(esmc.ESMC_MM_MIN_ROWS, 960, t.bfloat16), _T(960, 960, t.bfloat8_b)) == "mm"


@pytest.mark.parametrize("wdtype", ["bfloat16", "bfloat8_b"])
def test_mm_resid_takes_bf16_weights_only(wdtype, monkeypatch):
    """The residual-fused projection serves bf16 weights; --fast's block-fp8 ones go to matmul + add."""
    monkeypatch.setattr(esmc, "_ESMC_MM", True)
    monkeypatch.setattr(esmc.ttnn.experimental, "dit_minimal_matmul_addcmul_fused", lambda *a, **kw: "fused",
                        raising=False)
    monkeypatch.setattr(esmc.ttnn, "from_torch", lambda *a, **kw: _T(1, 960, esmc.ttnn.bfloat16))
    lin = esmc._MMLin()
    lin.compute_kernel_config = None
    t = esmc.ttnn
    out = lin._lin_resid(_T(esmc.ESMC_MM_MIN_ROWS, 960, t.bfloat16), _T(960, 960, getattr(t, wdtype)), "x")
    assert out == ("fused" if wdtype == "bfloat16" else None)


@pytest.mark.parametrize("switch", [None, True, False])
@pytest.mark.parametrize("wdtype", ["bfloat16", "bfloat8_b"])
def test_mm_bf16_acc_follows_fast_weights(wdtype, switch, monkeypatch):
    """bf16 dest accumulation is on by default only for --fast's block-fp8 weights; the switch overrides both."""
    monkeypatch.setattr(esmc, "_ESMC_MM_BF16_ACC", switch)
    lin = esmc._MMLin()
    lin.compute_kernel_config = _Ck()
    w = _T(960, 960, getattr(esmc.ttnn, wdtype))
    on = wdtype == "bfloat8_b" if switch is None else switch
    assert lin._mm_ck(w).fp32_dest_acc_en is not on


@pytest.mark.parametrize("wormhole", [True, False])
@pytest.mark.parametrize("switch", [None, True, False])
@pytest.mark.parametrize("wdtype", ["bfloat16", "bfloat8_b"])
def test_qk_l1_large_default(wormhole, switch, wdtype, monkeypatch):
    """The 6B's q/k L1 path defaults on for Wormhole block-fp8 weights only; the switch overrides it."""
    monkeypatch.setattr(tenstorrent, "is_wormhole", lambda: wormhole)
    monkeypatch.setattr(esmc, "_ESMC_QK_L1_LARGE", switch)
    want = (wormhole and wdtype == "bfloat8_b") if switch is None else switch
    assert esmc._qk_l1_large(getattr(esmc.ttnn, wdtype)) is want


@pytest.mark.parametrize("wormhole", [True, False])
@pytest.mark.parametrize("switch", [None, True, False])
def test_qkrh_default(wormhole, switch, monkeypatch):
    """qk_rope_heads defaults on for Wormhole unless lm_qk_chain is active; the switch overrides both."""
    monkeypatch.setattr(tenstorrent, "is_wormhole", lambda: wormhole)
    monkeypatch.setattr(esmc, "_ESMC_QKRH", switch)
    assert esmc._qkrh() is (wormhole if switch is None else switch)
    with tenstorrent.levers("lm_qk_chain"):
        assert esmc._qkrh() is (False if switch is None else switch)
