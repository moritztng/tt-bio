"""ESMC's minimal_matmul projections default by arch: on for Wormhole, off for Blackhole; the switch overrides both."""
from __future__ import annotations

import types

import pytest
import ttnn

from tt_bio import esmc, tenstorrent


@pytest.mark.parametrize("wormhole", [True, False])
@pytest.mark.parametrize("switch", [None, True, False])
def test_esmc_mm_default_by_arch(wormhole, switch, monkeypatch):
    monkeypatch.setattr(tenstorrent, "is_wormhole", lambda: wormhole)
    monkeypatch.setattr(esmc, "_ESMC_MM", switch)
    assert esmc._esmc_mm() is (wormhole if switch is None else switch)



@pytest.mark.parametrize("wormhole", [True, False])
@pytest.mark.parametrize("switch", [None, True, False])
@pytest.mark.parametrize("width", [480, 1152, 1280])
def test_saprot_mm_default_by_width(wormhole, switch, width, monkeypatch):
    from tt_bio import saprot
    monkeypatch.setattr(tenstorrent, "is_wormhole", lambda: wormhole)
    monkeypatch.setattr(saprot, "_SAPROT_MM", switch)
    want = (wormhole and width <= saprot._MM_MAX_WIDTH) if switch is None else switch
    assert saprot._saprot_mm(width) is want

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



def test_mm_exact_acc_keeps_fp32_for_block_fp8(monkeypatch):
    """Block-fp8 weights under a normal-mode forward (the Wormhole ESMC-6B embed) keep fp32 accumulation."""
    monkeypatch.setattr(esmc, "_ESMC_MM_BF16_ACC", None)
    lin = esmc._MMLin()
    lin.compute_kernel_config = _Ck()
    lin.mm_exact_acc = True
    assert lin._mm_ck(_T(960, 960, esmc.ttnn.bfloat8_b)).fp32_dest_acc_en

@pytest.mark.parametrize("saprot", [False, True])
@pytest.mark.parametrize("large", [False, True])
@pytest.mark.parametrize("wdtype", ["bfloat16", "bfloat8_b"])
def test_mm_bfp8_hifi2_lowers_fast_weights_only(wdtype, large, saprot, monkeypatch):
    """The HiFi2 switch lowers HiFi3 to HiFi2 for block-fp8 weights only; the 6B (lm_large) keeps its fidelity, and
    by default (None) only ESMC takes it, not SaProt."""
    from tt_bio import saprot as sp
    monkeypatch.setattr(esmc, "_ESMC_MM_BF16_ACC", None)
    lin = sp._SaprotLin() if saprot else esmc._MMLin()
    lin.compute_kernel_config = _Ck(math_fidelity=ttnn.MathFidelity.HiFi3)
    lin.lm_large = large
    w = _T(960, 960, getattr(esmc.ttnn, wdtype))
    monkeypatch.setattr(esmc, "_ESMC_MM_BFP8_HIFI2", False)
    assert lin._mm_ck(w).math_fidelity == ttnn.MathFidelity.HiFi3
    monkeypatch.setattr(esmc, "_ESMC_MM_BFP8_HIFI2", True)
    lowered = wdtype == "bfloat8_b" and not large
    assert lin._mm_ck(w).math_fidelity == (ttnn.MathFidelity.HiFi2 if lowered else ttnn.MathFidelity.HiFi3)
    monkeypatch.setattr(esmc, "_ESMC_MM_BFP8_HIFI2", None)
    lowered = lowered and not saprot
    assert lin._mm_ck(w).math_fidelity == (ttnn.MathFidelity.HiFi2 if lowered else ttnn.MathFidelity.HiFi3)


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


@pytest.mark.parametrize("wormhole", [True, False])
@pytest.mark.parametrize("switch", [None, True, False])
def test_saprot_qkrh_default(wormhole, switch, monkeypatch):
    """SaProt's packed RoPE + head split program defaults on with block-fp8 weights (--fast) on both arches; the switch
    overrides."""
    from tt_bio import saprot
    monkeypatch.setattr(tenstorrent, "is_wormhole", lambda: wormhole)
    monkeypatch.setattr(saprot, "_SAPROT_QKRH", switch)
    for dtype in (ttnn.bfloat8_b, ttnn.bfloat16):
        w = type("W", (), {"dtype": dtype})()
        want = (dtype == ttnn.bfloat8_b) if switch is None else switch
        assert saprot._saprot_qkrh(w) is want


@pytest.mark.parametrize("wormhole", [True, False])
@pytest.mark.parametrize("switch", [None, True, False])
def test_lm_gate_legacy_default(wormhole, switch, monkeypatch):
    """The LM SwiGLU gate takes the legacy multiply on Wormhole unless lm_gate_ng is active; the switch overrides both."""
    monkeypatch.setattr(tenstorrent, "is_wormhole", lambda: wormhole)
    monkeypatch.setattr(esmc, "_ESMC_LM_GATE_LEGACY", switch)
    assert esmc._lm_gate_legacy() is (wormhole if switch is None else switch)
    with tenstorrent.levers("lm_gate_ng"):
        assert esmc._lm_gate_legacy() is (False if switch is None else switch)


@pytest.mark.parametrize("wormhole", [True, False])
@pytest.mark.parametrize("switch", [None, True, False])
def test_mm_gate_default(wormhole, switch, monkeypatch):
    """The LM FFN's fc1 and gate run as one gated minimal_matmul on Wormhole unless lm_gate_ng is active."""
    monkeypatch.setattr(tenstorrent, "is_wormhole", lambda: wormhole)
    monkeypatch.setattr(esmc, "_ESMC_MM_GATE", switch)
    assert esmc._mm_gate_on() is (wormhole if switch is None else switch)
    with tenstorrent.levers("lm_gate_ng"):
        assert esmc._mm_gate_on() is (False if switch is None else switch)


@pytest.mark.parametrize("wdtype,bf16_acc", [("bfloat16", False), ("bfloat8_b", True)])
@pytest.mark.parametrize("on", [True, False])
def test_mm_config_blocks_8x8_under_bf16_acc(wdtype, bf16_acc, on, monkeypatch):
    """--fast's bf16-accumulated block-fp8 projections take 8 x 8 blocks; every other minimal_matmul keeps 16 x 4."""
    monkeypatch.setattr(esmc, "_ESMC_MM_CFG", True)
    monkeypatch.setattr(esmc, "_ESMC_MM_CFG_BF16_ACC_8X8", on)
    monkeypatch.setattr(esmc.ttnn, "MinimalMatmulConfig", lambda **kw: kw, raising=False)
    x = types.SimpleNamespace(device=lambda: types.SimpleNamespace(
        compute_with_storage_grid_size=lambda: types.SimpleNamespace(x=8, y=9)))
    cfg = esmc._mm_config(x, _T(1152, 3456, getattr(esmc.ttnn, wdtype)))["config"]
    mn8 = bf16_acc and on
    assert (cfg["M_block_size"], cfg["N_block_size"]) == ((8, 8) if mn8 else (16, 4))
    assert (cfg["K_block_size"], cfg["subblock_w"]) == (4, 4 if bf16_acc else 2)


def test_esmc_ffn_ln_l1_default_on():
    assert esmc._ESMC_FFN_LN_L1 is True


@pytest.mark.parametrize("width", [480, 1280])
def test_saprot_hifi2_ffn_projections_only(width, monkeypatch):
    """SaProt --fast lowers only fc1 and fc2 to HiFi2, and only from width _HIFI2_MIN_WIDTH up (qkv/out at HiFi2 failed
    the fast bar)."""
    from tt_bio import saprot as sp
    monkeypatch.setattr(sp, "_SAPROT_HIFI2", None)
    lin = sp._SaprotLin()
    lin.compute_kernel_config = _Ck(math_fidelity=ttnn.MathFidelity.HiFi3)
    lin.norm_weight = _T(1, width, ttnn.bfloat16)
    want = {"qkv": (width, 3 * width), "out": (width, width), "fc1": (width, 4 * width), "fc2": (4 * width, width)}
    for role, (k, n) in want.items():
        assert sp._role(width, k, n) == role
        lowered = role in ("fc1", "fc2") and width >= sp._HIFI2_MIN_WIDTH
        fid = lin._ck(_T(k, n, ttnn.bfloat8_b)).math_fidelity
        assert fid == (ttnn.MathFidelity.HiFi2 if lowered else ttnn.MathFidelity.HiFi3), role
        assert lin._ck(_T(k, n, ttnn.bfloat16)).math_fidelity == ttnn.MathFidelity.HiFi3
