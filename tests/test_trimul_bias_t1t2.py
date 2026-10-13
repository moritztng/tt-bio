"""T1 / T2 projection biases (AF2's triangle multiplication) host side, no device.

Checked here: the bias tiles T1 and T2 take have the layout their kernels seed DST from (T1's [a | b]
tiles reuse `gin_ragged_bias`'s rows-are-channels layout, g_out and T2's p take every-row-equal tiles),
seeding the accumulator with them gives `linear(..., bias=)` against torch, the program-cache keys
without a bias are exactly the keys used before, every bias line in the kernels sits behind the
T1_BIAS / T2_BIAS define, and `_t1_in` no longer declines a biased module for "g_in" / "g_out_bias".

Run: TT_VISIBLE_DEVICES= python3 -m pytest tests/test_trimul_bias_t1t2.py
"""
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ttnn = pytest.importorskip("ttnn")
from tt_bio import tenstorrent as T
from tt_bio import trimul_tail as TTL

KDIR = Path(TTL.__file__).resolve().parent / "kernels"
TILE = 32


def _tile(t2d, i):
    """Page i of a [R, 32] tiled tensor: rows 32 i .. 32 i + 31."""
    return t2d[i * TILE:(i + 1) * TILE]


# ---------------------------------------------------------------- bias tile layouts

def test_gin_ragged_bias_host_is_what_gin_ragged_bias_built():
    torch.manual_seed(0)
    bp, bg = torch.randn(128), torch.randn(128)
    h = TTL.gin_ragged_bias_host(bp, bg, 195)
    assert h.shape == (256 + 32, 32)
    rows = torch.cat([bp, bg])
    assert torch.equal(h[:256], rows[:, None].expand(-1, 32))
    last = 195 - 32 * 6
    assert torch.equal(h[256], (torch.arange(32) < last).float())


def test_row_bias_tile_n_is_bias_slice_in_every_row():
    b = torch.randn(128)
    h = TTL.row_bias_host(b)
    assert h.shape == (128, 32)
    for n in range(4):
        assert torch.equal(_tile(h, n), b[32 * n:32 * n + 32][None, :].expand(32, -1))


def test_t1_bias_layout_and_seeded_projection_matches_linear():
    """Emulate the kernel: each output tile's accumulator starts as its bias page, then + the K sum."""
    torch.manual_seed(1)
    K, C2, N, S = 128, 256, 128, 224
    bp, bg, bgo = torch.randn(C2), torch.randn(C2), torch.randn(N)
    h = TTL.t1_bias_host(bp, bg, bgo, S)
    ct2, ntg = C2 // TILE, N // TILE            # wpT [C2, K]: CT2 channel tiles per pass
    assert h.shape[0] == TILE * TTL.t1_bias_ntiles(ct2, ntg)
    assert torch.equal(_tile(h, 2 * ct2), torch.ones(32, 32))   # S whole tiles: the mask is unused
    xn = torch.randn(32, K, dtype=torch.float64)                # 32 positions of one row tile
    wp, wg, wo = (torch.randn(C2, K, dtype=torch.float64), torch.randn(C2, K, dtype=torch.float64),
                  torch.randn(N, K, dtype=torch.float64))
    for pass_, (w, b) in enumerate(((wp, bp), (wg, bg))):
        ref = torch.nn.functional.linear(xn, w, b.double()).t()          # [C2, 32]: rows = channels
        for q in range(ct2):
            seed = _tile(h, pass_ * ct2 + q).double()
            acc = seed + w[32 * q:32 * q + 32] @ xn.t()
            torch.testing.assert_close(acc, ref[32 * q:32 * q + 32])
    ref = torch.nn.functional.linear(xn, wo, bgo.double())               # [32, N]: cols = channels
    for n in range(ntg):
        seed = _tile(h, 2 * ct2 + 1 + n).double()
        torch.testing.assert_close(seed + xn @ wo[32 * n:32 * n + 32].t(), ref[:, 32 * n:32 * n + 32])


def test_t2_bias_seeded_projection_matches_linear():
    torch.manual_seed(2)
    xn, w, b = torch.randn(64, 128, dtype=torch.float64), torch.randn(128, 128, dtype=torch.float64), torch.randn(128)
    h = TTL.row_bias_host(b).double()
    ref = torch.nn.functional.linear(xn, w, b.double())
    for m in range(2):
        for n in range(4):
            acc = _tile(h, n) + xn[32 * m:32 * m + 32] @ w[32 * n:32 * n + 32].t()
            torch.testing.assert_close(acc, ref[32 * m:32 * m + 32, 32 * n:32 * n + 32])


# ---------------------------------------------------------------- no bias: nothing changes

class _Fake:
    def __init__(self, shape, padded=None):
        self.shape, self.padded_shape = tuple(shape), tuple(padded or shape)
        self.dtype, self.layout = ttnn.bfloat16, ttnn.TILE_LAYOUT

    def memory_config(self):
        return ttnn.DRAM_MEMORY_CONFIG


def test_cache_keys_without_bias_are_the_old_keys():
    x, wpT, wgo = _Fake([1, 256, 256, 128]), _Fake([256, 128]), _Fake([128, 128])
    ckc, grid = ("HiFi4", False, True, False), (8, 8)
    old_t1 = ("t1", str(x.padded_shape), str(wpT.padded_shape), str(wgo.padded_shape), tuple(grid),
              tuple(str(c) for c in ckc), False, 1e-5, TTL.SIGPOLY, TTL.RNE, TTL.GIN_MOVE_NB,
              TTL.T1_RB, TTL.T1_SBW, TTL.T1_ABL)
    assert TTL.t1_key(x, wpT, wgo, grid, ckc, None, 1e-5) == old_t1
    assert TTL.t1_key(x, wpT, wgo, grid, ckc, None, 1e-5, bias=object()) != old_t1
    lw, lb, wp = _Fake([128], [32, 128]), _Fake([128], [32, 128]), _Fake([128, 128])
    spec = lambda t: (str(t.padded_shape), str(t.dtype), str(t.memory_config()))  # noqa: E731
    old_t2 = ("t2", spec(x), spec(wp), spec(lw), spec(lb), tuple(grid), tuple(str(c) for c in ckc),
              1e-5, TTL.SIGPOLY, TTL.RNE)
    assert TTL.t2_key(x, wp, lw, lb, grid, ckc, 1e-5) == old_t2
    assert TTL.t2_key(x, wp, lw, lb, grid, ckc, 1e-5, pbias=_Fake([128, 32])) != old_t2


def _outside_ifdef(path, define):
    """Lines of `path` outside every `#ifdef define ... #endif` block."""
    out, depth = [], 0
    for line in path.read_text().splitlines():
        s = line.strip()
        if depth:
            if s.startswith("#if"):
                depth += 1
            elif s.startswith("#endif"):
                depth -= 1
            continue
        if s == f"#ifdef {define}":
            depth = 1
            continue
        out.append(line)
    return out


@pytest.mark.parametrize("rel,define", [("trimul_in_t1/reader.cpp", "T1_BIAS"),
                                        ("trimul_in_t1/compute.cpp", "T1_BIAS"),
                                        ("trimul_tail_res/reader_ln.cpp", "T2_BIAS"),
                                        ("trimul_tail_res/compute_ln.cpp", "T2_BIAS")])
def test_bias_code_only_behind_its_define(rel, define):
    code = [ln.split("//")[0] for ln in _outside_ifdef(KDIR / rel, define)]
    assert not [ln for ln in code if re.search(r"bias|c_20", ln, re.I)], rel


def test_bias_cb_free_in_both_plans():
    bf, f32 = TTL.t1_cb_plan(4, 8, 4, True)
    assert TTL.T1_CB_BIAS not in {i for i, _ in bf + f32}
    assert TTL.T2_CB_BIAS not in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17,
                                  TTL.T2CM_CB_GATHER, TTL.T2CM_CB_SCRATCH}


def _module(biased):
    m = T.TriangleMultiplication.__new__(T.TriangleMultiplication)
    data = {"g_out.weight": torch.zeros(128, 128)}
    if biased:
        data["g_out.bias"] = torch.zeros(128)
    m.weights = type("W", (), {"data": data, "__getitem__": lambda self, k: data[k]})()
    m._g_in_b = torch.zeros(256) if biased else None
    m._p_in_b = torch.zeros(256) if biased else None
    m._g_in_t = torch.zeros(128, 256)
    m.g_in_weight = None
    m.g_out_bias = object() if biased else None
    m.p_out_bias = object() if biased else None
    m.gated_move = True
    return m


def test_unbiased_module_passes_no_bias():
    m = _module(False)
    assert m._t1_bias(256) is None and m._t2_pbias() is None


@pytest.mark.parametrize("biased", [False, True])
def test_t1_in_no_longer_declines_for_biases(monkeypatch, biased):
    m = _module(biased)
    monkeypatch.setattr(T, "_FAST_MODE", False)
    monkeypatch.setattr(T, "_TRIMUL_RAW_CHANNEL_MOVES", False)
    monkeypatch.setattr(T, "_TRIMUL_INPROJ_ROWBLOCK", False)
    # The first decline after the bias checks: proves none before it fired.
    monkeypatch.setattr(T, "_triangle_mul_memory_config",
                        lambda H: SimpleNamespace(buffer_type=ttnn.BufferType.L1))
    TTL.T1_WHY.clear()
    with T.levers("trimul_t1"):
        assert m._t1_in(None, None, 256, (1, 256, 256, 128)) is None
    assert TTL.T1_WHY == {"not_dram": 1}
    m.g_in_weight = object()                      # the training leaves still decline
    TTL.T1_WHY.clear()
    with T.levers("trimul_t1"):
        assert m._t1_in(None, None, 256, (1, 256, 256, 128)) is None
    assert TTL.T1_WHY == {"g_in": 1}


def test_t1_still_declines_ragged_af2_size():
    """AF2-IG folds at 195 tokens: T1 reads whole tiles only, so it declines there ("act_shape")
    even with the biases and trimul_t1_dram. Pinned so a fix to it is a visible change."""
    x = _Fake([1, 195, 195, 128], [1, 224, 224, 128])
    with T.levers("trimul_t1"):
        assert TTL.t1_eligible(x, None, None, _Fake([256, 128]), _Fake([256, 128]), _Fake([128, 128])) \
            == "act_shape"
