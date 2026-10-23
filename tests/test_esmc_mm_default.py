"""ESMC's minimal_matmul projections default on for both archs unless lm_linear is active; the switch overrides."""
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
    assert esmc._esmc_mm() is (True if switch is None else switch)
    with tenstorrent.levers("lm_linear"):
        assert esmc._esmc_mm() is (False if switch is None else switch)



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
@pytest.mark.parametrize("large,rows,wdtype", [(True, 544, "bfloat16"), (True, 544, "bfloat8_b"),
                                               (True, 1088, "bfloat16"), (False, 544, "bfloat16")])
def test_mm_6b_bf16_default(wormhole, switch, large, rows, wdtype, monkeypatch):
    """The 6B's page rows under bf16 weights take minimal_matmul for its qkv split and gate on Blackhole only."""
    monkeypatch.setattr(tenstorrent, "is_wormhole", lambda: wormhole)
    monkeypatch.setattr(esmc, "_ESMC_MM_6B_BF16", switch)
    monkeypatch.setattr(esmc, "_ESMC_MM", None)
    lin = esmc._MMLin()
    lin.lm_large = large
    x, w = _T(rows, 2560, esmc.ttnn.bfloat16), _T(2560, 7680, getattr(esmc.ttnn, wdtype))
    regime = large and rows == 544 and wdtype == "bfloat16"
    assert lin._mm_6b_bf16(x, w) is (regime and (not wormhole if switch is None else switch))


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
    """qk_rope_heads defaults on for both archs unless lm_qk_chain is active; the switch overrides both."""
    monkeypatch.setattr(tenstorrent, "is_wormhole", lambda: wormhole)
    monkeypatch.setattr(esmc, "_ESMC_QKRH", switch)
    assert esmc._qkrh() is (True if switch is None else switch)
    with tenstorrent.levers("lm_qk_chain"):
        assert esmc._qkrh() is (False if switch is None else switch)


@pytest.mark.parametrize("wormhole", [True, False])
def test_qkrh_dtypes_fit_l1(wormhole, monkeypatch):
    """Block-fp8 q, k, v from qk_rope_heads, bf16 q on Wormhole bf16-accumulation forwards, and all bf16 where that mix
    would not fit the kernel's L1 (the 6B width), so qk_rope_heads never declines a dtype choice made here."""
    from tt_bio import qk_rope_heads
    monkeypatch.setattr(tenstorrent, "is_wormhole", lambda: wormhole)
    monkeypatch.setattr(esmc, "_ESMC_QKRH_BF8", None)
    monkeypatch.setattr(esmc, "_ESMC_QKRH_Q_BF16", None)
    monkeypatch.setattr(esmc, "_ESMC_MM_BF16_ACC", None)
    b8 = type("W", (), {"dtype": ttnn.bfloat8_b})()
    for d in (960, 1152, 2560):
        for exact in (False, True):
            got = esmc._qkrh_dtypes(b8, exact, d)
            q_bf16 = wormhole and not exact
            if q_bf16 and d == 2560:
                assert got == (None, None)
            else:
                assert got == (ttnn.bfloat8_b, ttnn.bfloat16 if q_bf16 else None)
            assert got == (None, None) or qk_rope_heads.fits(d, *got)


@pytest.mark.parametrize("wormhole", [True, False])
@pytest.mark.parametrize("switch", [None, True, False])
def test_saprot_qkrh_default(wormhole, switch, monkeypatch):
    """SaProt's packed RoPE + head split program defaults on with block-fp8 weights (--fast) on both arches, and in
    normal mode where its RoPE sums run on the SFPU (Wormhole); the switch overrides."""
    from tt_bio import qk_rope_heads, saprot
    monkeypatch.setattr(tenstorrent, "is_wormhole", lambda: wormhole)
    monkeypatch.setattr(saprot, "_SAPROT_QKRH", switch)
    monkeypatch.setattr(qk_rope_heads, "SFPU", None)
    for dtype in (ttnn.bfloat8_b, ttnn.bfloat16):
        w = type("W", (), {"dtype": dtype})()
        want = (dtype == ttnn.bfloat8_b or wormhole) if switch is None else switch
        assert saprot._saprot_qkrh(w) is want


@pytest.mark.parametrize("wormhole", [True, False])
@pytest.mark.parametrize("switch", [None, True, False])
def test_saprot_fc1_gelu_f32_default(wormhole, switch, monkeypatch):
    """SaProt's fc1 asks for approx gelu (calculate_gelu_f32 under the gelu_f32 overlay) on Wormhole and on Blackhole --fast
    at width >= 1280 (block-fp8 weight); the switch overrides."""
    from tt_bio import saprot
    monkeypatch.setattr(tenstorrent, "is_wormhole", lambda: wormhole)
    monkeypatch.setattr(saprot, "_SAPROT_GELU_F32", switch)
    on = wormhole if switch is None else switch
    assert saprot._fc1_act() == ("gelu_approx" if on else "gelu")
    for dtype in (ttnn.bfloat8_b, ttnn.bfloat16):
        for d in (480, 1280):
            w = types.SimpleNamespace(dtype=dtype, shape=(d, 4 * d))
            on = (wormhole or (dtype == ttnn.bfloat8_b and d >= 1280)) if switch is None else switch
            assert saprot._fc1_act(w) == ("gelu_approx" if on else "gelu")


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
    """The LM FFN's fc1 and gate run as one gated minimal_matmul on both archs unless lm_gate_ng is active."""
    monkeypatch.setattr(tenstorrent, "is_wormhole", lambda: wormhole)
    monkeypatch.setattr(esmc, "_ESMC_MM_GATE", switch)
    assert esmc._mm_gate_on() is (True if switch is None else switch)
    with tenstorrent.levers("lm_gate_ng"):
        assert esmc._mm_gate_on() is (False if switch is None else switch)


# ESMC 300M / 600M (3808 rows, page batch) and 6B (544 rows) FFN gates: rows, n2 = 2 x hidden, K tiles.
_GATE_SHAPES = {"300m": (3808, 5120, 30), "600m": (3808, 6144, 36), "6b": (544, 13824, 80)}


@pytest.mark.parametrize("bf16_acc", [False, True])
@pytest.mark.parametrize("model,want", [("300m", ((8, 2, 10, 2, 2), (8, 2, 10, 1, 5))),
                                        ("600m", ((8, 2, 12, 1, 4), (8, 4, 12, 1, 6))),
                                        ("6b", ((2, 4, 6, 1, 3), (2, 4, 6, 1, 6)))])
def test_mm_gate_config_wormhole_unchanged(model, want, bf16_acc):
    """Wormhole's 8 grid columns divide every ESMC gate: the factory's layout and the block configs graded there."""
    from tt_bio import mm_gate
    rows, n2, kt = _GATE_SHAPES[model]
    x = types.SimpleNamespace(volume=lambda: rows * 64, shape=(rows, 64))
    assert mm_gate.config(x, n2, kt, (8, 9), bf16_acc) == want[bf16_acc]
    assert mm_gate.layout(rows, n2, want[bf16_acc][2], (8, 9))[0] is (rows > n2)


@pytest.mark.parametrize("grid", [(11, 10), (13, 10)])
@pytest.mark.parametrize("model", list(_GATE_SHAPES))
def test_mm_gate_layout_blackhole(model, grid):
    """Blackhole's 11 / 13 columns divide no ESMC gate; the padded layout still gives every core whole N blocks,
    covers N, and is the cheaper of the two orientations."""
    from tt_bio import mm_gate
    rows, n2, kt = _GATE_SHAPES[model]
    x = types.SimpleNamespace(volume=lambda: rows * 64, shape=(rows, 64))
    cfg = mm_gate.config(x, n2, kt, grid, False)
    assert cfg is not None and mm_gate.plan(x, n2, cfg, grid) is None
    transpose, n_core, m_core = mm_gate.layout(rows, n2, cfg[2], grid)
    n_cores, m_cores = (grid[1], grid[0]) if transpose else grid
    assert n_core % cfg[2] == 0 and (n2 // 32) % cfg[2] == 0
    assert n_core * n_cores >= n2 // 32 and m_core * m_cores >= rows // 32


@pytest.mark.parametrize("wdtype,bf16_acc", [("bfloat16", False), ("bfloat8_b", True)])
@pytest.mark.parametrize("on", [True, False])
def test_mm_config_blocks_8x8_under_bf16_acc(wdtype, bf16_acc, on, monkeypatch):
    """--fast's bf16-accumulated block-fp8 projections take 8 x 8 blocks; every other minimal_matmul keeps 16 x 4."""
    monkeypatch.setattr(esmc, "_ESMC_MM_CFG", True)
    monkeypatch.setattr(esmc, "_ESMC_MM_CFG_BF16_ACC_8X8", on)
    monkeypatch.setattr(esmc, "_ESMC_MM_KB6", False)  # K 6 is Wormhole's default since 1c4b96533; not what this pins
    monkeypatch.setattr(esmc.ttnn, "MinimalMatmulConfig", lambda **kw: kw, raising=False)
    x = types.SimpleNamespace(device=lambda: types.SimpleNamespace(
        compute_with_storage_grid_size=lambda: types.SimpleNamespace(x=8, y=9)))
    cfg = esmc._mm_config(x, _T(1152, 3456, getattr(esmc.ttnn, wdtype)))["config"]
    mn8 = bf16_acc and on
    assert (cfg["M_block_size"], cfg["N_block_size"]) == ((8, 8) if mn8 else (16, 4))
    assert (cfg["K_block_size"], cfg["subblock_w"]) == (4, 4 if bf16_acc else 2)


def test_esmc_ffn_ln_l1_default_on():
    assert esmc._ESMC_FFN_LN_L1 is True


@pytest.mark.parametrize("wormhole", [True, False])
@pytest.mark.parametrize("width", [480, 1280])
def test_saprot_hifi2_ffn_projections_only(width, wormhole, monkeypatch):
    """SaProt --fast lowers only fc1 and fc2 to HiFi2, only from width _HIFI2_MIN_WIDTH up (qkv/out at HiFi2 failed
    the fast bar), on both arches."""
    from tt_bio import saprot as sp
    monkeypatch.setattr(sp, "_SAPROT_HIFI2", None)
    monkeypatch.setattr(sp, "_SAPROT_IN_B8", False)
    monkeypatch.setattr("tt_bio.tenstorrent.is_wormhole", lambda: wormhole)
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


@pytest.mark.parametrize("wormhole", [True, False])
@pytest.mark.parametrize("switch, want", [(None, None), (False, ()), (True, ("qkv", "out", "fc1")), (("qkv",), ("qkv",))])
def test_saprot_in_b8_default(wormhole, switch, want, monkeypatch):
    """--fast's block-fp8 activation serves out alone by default on Wormhole (qkv failed the fast bar, fc1 bought
    nothing); the roles it serves run at HiFi2, and a bf16 weight or a narrow model never takes it."""
    from tt_bio import saprot as sp
    monkeypatch.setattr(sp, "_SAPROT_IN_B8", switch)
    monkeypatch.setattr("tt_bio.tenstorrent.is_wormhole", lambda: wormhole)
    want = (("out",) if wormhole else ()) if want is None else want
    assert sp._in_b8_roles() == want
    assert sp._hifi2_roles() == ("fc1", "fc2") + tuple(r for r in ("qkv", "out") if r in want)
    w = sp._HIFI2_MIN_WIDTH
    assert sp._in_b8_on(_T(w, w, ttnn.bfloat8_b), w) is ("out" in want)
    assert sp._in_b8_on(_T(w, w, ttnn.bfloat16), w) is False
    assert sp._in_b8_on(_T(480, 480, ttnn.bfloat8_b), 480) is False


@pytest.mark.parametrize("wdtype, want_l1", [("bfloat8_b", True), ("bfloat16", False)])
def test_saprot_ln_l1_fast_only(wdtype, want_l1, monkeypatch):
    """SaProt's block layer_norm writes to L1 by default only when the next projection's weight is block-fp8 (--fast)."""
    from tt_bio import saprot as sp
    monkeypatch.setattr(sp, "_SAPROT_LN_L1", None)
    monkeypatch.setattr(sp, "_LN_L1_REFUSED", set())
    seen = []
    monkeypatch.setattr(sp.ttnn, "layer_norm", lambda x, **kw: seen.append(kw.get("memory_config")) or x)
    x = types.SimpleNamespace(padded_shape=(7, 544, 1280))
    sp._block_ln(x, None, None, None, _T(1280, 3840, getattr(ttnn, wdtype)))
    assert (seen == [ttnn.L1_MEMORY_CONFIG]) == want_l1 and len(seen) == 1


@pytest.mark.parametrize("wdtype, want_l1", [("bfloat8_b", True), ("bfloat16", False)])
def test_saprot_x_l1_fast_only(wdtype, want_l1, monkeypatch):
    """SaProt's unfused residual add writes the stream to L1 by default only with block-fp8 weights (--fast)."""
    from tt_bio import saprot as sp
    monkeypatch.setattr(sp, "_SAPROT_X_L1", None)
    monkeypatch.setattr(sp, "_SAPROT_RESID", False)
    monkeypatch.setattr(sp, "_SAPROT_MM_RESID", False)   # the unfused add; mm_resid is on by default on Wormhole
    monkeypatch.setattr(sp, "_X_L1_REFUSED", set())
    seen = []
    monkeypatch.setattr(sp.ttnn, "add", lambda a, b, **kw: seen.append(kw.get("memory_config")) or a)
    monkeypatch.setattr(sp.ttnn, "deallocate", lambda t: None)
    lin = sp._SaprotLin()
    lin._lin = lambda x, w, bias=None: x
    resid = types.SimpleNamespace(padded_shape=(7, 544, 1280))
    lin._lin_add(resid, _T(1280, 1280, getattr(ttnn, wdtype)), None, resid)
    assert (seen == [ttnn.L1_MEMORY_CONFIG]) == want_l1 and len(seen) == 1


@pytest.mark.parametrize("switch", [None, True, False])
def test_saprot_sdpa_fp32_dest_with_qkrh(switch, monkeypatch):
    """SaProt's page attention runs at HiFi4 fp32 dest behind the packed RoPE program or normal's HiFi3 projections by
    default; the switch overrides."""
    from tt_bio import saprot as sp
    monkeypatch.setattr(sp, "_SAPROT_SDPA", switch)
    for qkrh in (True, False):
        for h3 in (True, False):
            want = (qkrh or h3) if switch is None else switch
            assert sp._sdpa_ckc(qkrh, h3) == (sp._SDPA_FP32 if want else None)


@pytest.mark.parametrize("k, n, wdtype, want", [(480, 1920, "bfloat8_b", True), (480, 1920, "bfloat16", True),
                                                (1280, 3840, "bfloat8_b", False), (1280, 5120, "bfloat16", False)])
def test_saprot_out_l1_small_only(k, n, wdtype, want, monkeypatch):
    """A projection writes to L1 by default only when its output fits the L1 budget, in either mode."""
    from tt_bio import saprot as sp
    monkeypatch.setattr(sp, "_SAPROT_OUT_L1", True)
    assert sp._out_l1(_T(7 * 544, k, ttnn.bfloat16), _T(k, n, getattr(ttnn, wdtype))) is want


@pytest.mark.parametrize("hifi2, want", [(False, (4, 7, 20, 1, 4)), (True, (8, 7, 10, 1, 2))])
def test_saprot_wh_fc1_out_blocks(hifi2, want, monkeypatch):
    """Wormhole fc1 at the page shape on Galaxy's grid runs the 2D program on out blocks: half the rows per core in
    normal mode, half the rows and columns at --fast's HiFi2 (placeable next to the L1 LN output); a grid where the
    blocks do not divide, and the switch, drop them."""
    from tt_bio import saprot as sp
    monkeypatch.setattr(sp, "_SAPROT_PROG", None)
    monkeypatch.setattr(sp, "_SAPROT_OUT_SPLIT", True)
    monkeypatch.setattr(sp, "_SAPROT_FC1_MM", False)
    monkeypatch.setattr("tt_bio.tenstorrent.is_wormhole", lambda: True)
    x, w = _T(3808, 1280, ttnn.bfloat16), _T(1280, 5120, ttnn.bfloat8_b)
    assert sp._saprot_prog(x, w, "gelu_approx", hifi2) is None   # 8 x 8: per_core_M 15 does not halve
    x.device = lambda: types.SimpleNamespace(compute_with_storage_grid_size=lambda: types.SimpleNamespace(x=8, y=9))
    kind, pc = sp._saprot_prog(x, w, "gelu_approx", hifi2)
    assert kind == "2d" and (pc.per_core_M, pc.per_core_N) == (14, 20)
    assert (pc.in0_block_w, pc.out_block_h, pc.out_block_w, pc.out_subblock_h, pc.out_subblock_w) == want
    monkeypatch.setattr(sp, "_SAPROT_OUT_SPLIT", False)
    assert sp._saprot_prog(x, w, "gelu_approx", hifi2) is None


@pytest.mark.parametrize("hifi2, want", [(False, (8, 8, 8, 1, 4)), (True, (4, 8, 8, 1, 4))])
def test_saprot_wh_fc1_minimal_matmul(hifi2, want, monkeypatch):
    """Wormhole fc1 (GELU fused) runs on minimal_matmul by default, at the block measured for its fidelity."""
    from tt_bio import saprot as sp
    monkeypatch.setattr(sp, "_SAPROT_PROG", None)
    monkeypatch.setattr("tt_bio.tenstorrent.is_wormhole", lambda: True)
    x, w = _T(3808, 1280, ttnn.bfloat16), _T(1280, 5120, ttnn.bfloat8_b)
    x.device = lambda: types.SimpleNamespace(compute_with_storage_grid_size=lambda: types.SimpleNamespace(x=8, y=9))
    kind, c = sp._saprot_prog(x, w, "gelu_approx", hifi2)
    assert kind == "mm"
    assert (c.M_block_size, c.K_block_size, c.N_block_size, c.subblock_h, c.subblock_w) == want


class _NoSliceShape(tuple):
    """ttnn.Shape: indexes by int, takes no slice."""
    def __getitem__(self, i):
        if isinstance(i, slice):
            raise TypeError("ttnn.Shape takes no slice")
        return tuple.__getitem__(self, i)


@pytest.mark.parametrize("fid,dtype,want", [("HiFi3", ttnn.bfloat16, (8, 8, 8, 2, 2)), ("HiFi4", ttnn.bfloat16, None),
                                            ("HiFi3", ttnn.bfloat8_b, None)])
def test_saprot_wh_fc2_mm_resid_hifi3(fid, dtype, want, monkeypatch):
    """Wormhole normal fc2 (bf16 weight at HiFi3) takes the residual add in mm_resid at its measured block; other fc2s keep
    the 2D pick, which mm_resid does not serve."""
    from tt_bio import saprot as sp
    monkeypatch.setattr(sp, "_SAPROT_PROG", None)
    monkeypatch.setattr("tt_bio.tenstorrent.is_wormhole", lambda: True)
    x, w = _T(3808, 5120, ttnn.bfloat16), _T(5120, 1280, dtype)
    w.shape = _NoSliceShape(w.shape)
    x.device = lambda: types.SimpleNamespace(compute_with_storage_grid_size=lambda: types.SimpleNamespace(x=8, y=9))
    lin = sp._SaprotLin.__new__(sp._SaprotLin)
    lin._ck = lambda _w: types.SimpleNamespace(math_fidelity=getattr(ttnn.MathFidelity, fid))
    seen = []
    monkeypatch.setattr("tt_bio.mm_resid.matmul_add", lambda *a, **k: seen.append((a[4], a[5])) or "out")
    out = lin._mm_resid(x, w, None, None, False)
    assert (out, seen) == (("out", [(want, (8, 9))]) if want else (None, []))
