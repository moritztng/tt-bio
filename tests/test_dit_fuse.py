"""The `dit_fuse` lever (OpenFold3's bf16 token-DiT block in 23 ttnn calls), host side, no device.

Checked here: the lever is named, ungraded and in no mode; it engages only on the bf16 `_sdpa32` DiT with no
other DiT lever owning the s path, and with it off a block keeps `fuse = None` and never enters `_call_fused`;
the fused call issues the ttnn calls the design says (and the masked transition tail when the token mask is not
all ones); and the fused algebra -- folded adaLN gammas, sigmoid in the epilogue, the gated residuals as addcmul,
the mask multiply dropped where it is 1 -- is the block's own math in float64. The device A/B and the numerics
against float64 on a card are perf/spd_gap_of3/dit_ops.py.
"""
import inspect

import pytest
import torch

ttnn = pytest.importorskip("ttnn")
from tt_bio import tenstorrent as T  # noqa: E402
from tt_bio import openfold3_diffusion_transformer as M  # noqa: E402

FAST = "normal+dit_s16+diffusion_bf16+trimul_t1+trimul_t2"


def _sd(seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64) * 0.05  # noqa: E731
    d = {}
    for p in ("attention_pair_bias.layer_norm_a.", "conditioned_transition.layer_norm."):
        d[p + "layer_norm_s.weight"] = 1 + r(384)
        d[p + "linear_g.weight"], d[p + "linear_g.bias"] = r(768, 384), r(768)
        d[p + "linear_s.weight"] = r(768, 384)
    a, c = "attention_pair_bias.", "conditioned_transition."
    d[a + "linear_ada_out.weight"], d[a + "linear_ada_out.bias"] = r(768, 384), r(768)
    d[a + "layer_norm_z.weight"], d[a + "linear_z.weight"] = 1 + r(128), r(16, 128)
    for k in "qkvog":
        d[a + f"mha.linear_{k}.weight"] = r(768, 768)
    d[a + "mha.linear_q.bias"] = r(768)
    d[c + "swiglu.linear_a.weight"], d[c + "swiglu.linear_b.weight"] = r(1536, 768), r(1536, 768)
    d[c + "linear_out.weight"] = r(768, 1536)
    d[c + "linear_g.weight"], d[c + "linear_g.bias"] = r(768, 384), r(768)
    return d


@pytest.fixture
def hostblock(monkeypatch):
    """`_DiTBlock` built off device: uploads become placeholders."""
    monkeypatch.setattr(T, "get_device", lambda *a, **k: None)
    monkeypatch.setattr(ttnn, "from_torch", lambda t, **k: ("tt", tuple(t.shape), k.get("dtype")))
    monkeypatch.setattr(M._DiTBlock, "_round", lambda self, t, dtype: t)

    def build(spec, dtype=None):
        with T.lever_scope(spec):
            if dtype is None:
                return M._DiTBlock({k: v.float() for k, v in _sd().items()}, None)
            with T.device_dtype_override(dtype):
                return M._DiTBlock({k: v.float() for k, v in _sd().items()}, None)
    return build


def test_lever_named_and_ungraded():
    assert "dit_fuse" in T.LEVERS
    assert "dit_fuse" in T.UNGRADED_LEVERS
    assert "dit_fuse" not in T.NORMAL_LEVERS | T.FAST_LEVERS
    from tt_bio import openfold3_fold as of3
    assert "dit_fuse" in of3.FAST_LEVERS and "dit_fuse" not in of3.NORMAL_LEVERS
    assert T.parse_levers(FAST + "+dit_fuse") - T.parse_levers(FAST) == {"dit_fuse"}


def test_sdpa32_fused_writes_fp32_unless_asked():
    p = inspect.signature(T._sdpa32_fused).parameters["out_dtype"]
    assert p.default == ttnn.float32


def test_engages_only_on_the_bf16_sdpa32_dit(hostblock):
    assert hostblock(FAST).fuse is None                                   # off: the incumbent block
    assert hostblock(FAST + "+dit_fuse").fuse is not None                 # OpenFold3 --fast + the lever
    assert hostblock(FAST + "-dit_sdpa32+dit_fuse").fuse is None          # needs _sdpa32
    assert hostblock("dit_sdpa32+dit_s16+dit_fuse", ttnn.float32).fuse is None   # fp32 DiT: dit_s16 owns the s path
    assert hostblock("dit_sdpa32+dit_fuse", ttnn.float32).fuse is None        # fp32 DiT without it: still bf16-only
    b8 = hostblock(FAST + "+dit_b8+dit_fuse")                             # dit_b8 joins: bfp8 weights through _lin
    assert b8.fuse is not None and b8.lp is None


class _Rec:
    """Records the ttnn calls one block makes; every op returns a fresh placeholder."""

    def __init__(self, monkeypatch, blk):
        self.calls = []
        n = iter(range(10 ** 6))

        def op(name):
            def f(*a, **k):
                self.calls.append((name, k.get("activation")))
                return ("t", next(n))
            return f
        for name in ("layer_norm", "addcmul", "multiply", "unsqueeze", "reshape", "typecast", "add"):
            monkeypatch.setattr(ttnn, name, op(name))
        monkeypatch.setattr(ttnn, "deallocate", lambda t: None)
        monkeypatch.setattr(M, "mask_add", op("mask_add"))
        monkeypatch.setattr(blk, "_lin", lambda x, w, bias=None, activation=None:
                            op("linear")(x, w, activation=activation))

        class _O(tuple):
            shape = (1, 1, 512, 1024)

        def sdpa(qkv, bias, scale, heads, out_dtype=ttnn.float32):
            self.calls.append(("sdpa32", out_dtype))
            return _O()
        monkeypatch.setattr(T, "_sdpa32_fused", sdpa)


def test_fused_call_issues_the_designed_ops(hostblock, monkeypatch):
    blk = hostblock(FAST + "+dit_fuse")
    rec = _Rec(monkeypatch, blk)
    cache = {(id(blk), "pair_bias32"): ("bias",)}
    blk("a", "s", "z", "mb", "tmc", cache=cache, s16=("s_hat",), tok_ones=True)
    names = [c[0] for c in rec.calls]
    assert ("sdpa32", ttnn.bfloat16) in rec.calls and "typecast" not in names
    assert names.count("linear") == 12 and names.count("layer_norm") == 2
    assert names.count("addcmul") == 4 and names.count("multiply") == 2 and "add" not in names
    assert [c[1] for c in rec.calls if c[0] == "linear"].count("sigmoid") == 4    # 2 adaLN scales + 2 s gates
    assert len(names) == 23
    rec.calls.clear()
    blk("a", "s", "z", "mb", "tmc", cache=cache, s16=("s_hat",), tok_ones=False)
    names = [c[0] for c in rec.calls]
    assert names.count("addcmul") == 3 and names.count("multiply") == 3 and names.count("mask_add") == 1


def test_off_never_enters_the_fused_call(hostblock, monkeypatch):
    blk = hostblock(FAST)
    monkeypatch.setattr(blk, "_call_fused", lambda *a, **k: pytest.fail("fused call with the lever off"))
    sentinel = RuntimeError("incumbent path")

    def boom(*a, **k):
        raise sentinel
    monkeypatch.setattr(blk, "adaln_a", boom)
    with pytest.raises(RuntimeError, match="incumbent path"):
        blk("a", "s", "z", "mb", "tmc", cache={}, s16=None)


def test_fused_algebra_is_the_blocks_math_in_float64():
    """The rewrite in float64: s_hat @ (gamma * W) for LN(s; gamma) @ W, sigmoid on the accumulator, the
    gated residuals and the adaLN modulation as one multiply-add, and the mask multiply dropped at 1."""
    d = _sd(1)
    g = torch.Generator().manual_seed(2)
    N = 64
    s = torch.randn(N, 384, generator=g, dtype=torch.float64)
    a = torch.randn(N, 768, generator=g, dtype=torch.float64)
    ln = lambda x, w=None: torch.nn.functional.layer_norm(x, x.shape[-1:], w, None, 1e-5)  # noqa: E731
    p = "attention_pair_bias.layer_norm_a."

    # incumbent: AdaLN(LN_s with gamma), multiply then add
    sn = ln(s, d[p + "layer_norm_s.weight"])
    ref = torch.sigmoid(sn @ d[p + "linear_g.weight"].T + d[p + "linear_g.bias"]) * ln(a) \
        + sn @ d[p + "linear_s.weight"].T
    # fused: shared s_hat, gamma folded (AdaLN.folded_weights' formula), one addcmul
    gam = d[p + "layer_norm_s.weight"].reshape(-1, 1)
    w_sc, w_sb = d[p + "linear_g.weight"].T * gam, d[p + "linear_s.weight"].T * gam
    s_hat = ln(s)
    sc = torch.sigmoid(s_hat @ w_sc + d[p + "linear_g.bias"])
    fused = torch.addcmul(s_hat @ w_sb, ln(a), sc)
    torch.testing.assert_close(fused, ref, rtol=1e-10, atol=1e-10)

    # gated residual + masked transition tail with an all-ones mask
    o, tr = torch.randn(N, 768, generator=g, dtype=torch.float64), torch.randn(N, 768, generator=g, dtype=torch.float64)
    gate = torch.sigmoid(s @ d["conditioned_transition.linear_g.weight"].T + d["conditioned_transition.linear_g.bias"])
    mask = torch.ones(N, 1, dtype=torch.float64)
    torch.testing.assert_close(torch.addcmul(a, tr, gate), a + (tr * gate) * mask, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(torch.addcmul(a, o, gate), a + o * gate, rtol=1e-12, atol=1e-12)
