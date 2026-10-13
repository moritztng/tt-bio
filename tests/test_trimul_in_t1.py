"""T1 (`trimul_tail.fused_in_ln`) host-side plumbing, no device.

What can be checked without a card: the CB budget per core at c_z = 128 against Blackhole's L1, that
every circular buffer the three kernels name is in the descriptor's plan, that T1's weights are the
incumbent in-projection's columns, and the kernels' page / slot / gather arithmetic (mirrored in
`trimul_tail.t1_*_page`, tied to the C++ by the exact expressions below) by emulating the whole data
flow tile by tile in float64 against the model's math. The kernels themselves have never run; the
device check is perf/spd_trimul/in_t1.py.

Run: TT_VISIBLE_DEVICES= python3 -m pytest tests/test_trimul_in_t1.py
"""
import re
from pathlib import Path

import pytest
import torch
import ttnn

from tt_bio import tenstorrent as T
from tt_bio import trimul_tail as TTL

KDIR = Path(TTL.__file__).resolve().parent / "kernels" / "trimul_in_t1"
L1_PER_CORE = 1_572_864          # Blackhole worker L1; a CB total above it TT_THROWs at program creation
L1_ALLOCATOR_BANK = 1_461_760    # what `_l1_bank_bytes` reads on Blackhole (its docstring)
KT, CT2, NTG = 4, 8, 4           # c_z 128, hidden 128, g_out 128: openfold3's trimul


# ---------------------------------------------------------------- CB budget

@pytest.mark.parametrize("mask", [False, True])
def test_cb_budget_fits_l1(mask):
    total = TTL.t1_cb_bytes(KT, CT2, NTG, mask)
    assert total == (911_360 if not mask else 978_944)
    assert total < L1_PER_CORE
    assert total <= 0.85 * L1_ALLOCATOR_BANK      # t1_eligible's own gate
    # N only sets units per core: at 512 tokens, 256 units of 32 row tiles over the 11 x 10 grid.
    split = TTL.t1_split((512 // 32) ** 2, 110)
    assert sum(n for _, n in split) == 256 and max(n for _, n in split) == 3


def test_cb_plan_covers_every_cb_the_kernels_use():
    used = set()
    for f in ("reader.cpp", "compute.cpp", "writer.cpp"):
        used |= {int(i) for i in re.findall(r"CBIndex::c_(\d+)", (KDIR / f).read_text())}
    bf, f32 = TTL.t1_cb_plan(KT, CT2, NTG, True)
    planned = [i for i, _ in bf] + [i for i, _ in f32]
    assert len(planned) == len(set(planned)), "a CB index is planned twice"
    assert used == set(planned)
    bf0, _ = TTL.t1_cb_plan(KT, CT2, NTG, False)
    assert {3, 6}.isdisjoint(i for i, _ in bf0)


def test_kernel_page_expressions_match_the_python_mirror():
    reader = (KDIR / "reader.cpp").read_text()
    writer = (KDIR / "writer.cpp").read_text()
    assert "uint32_t page = ((b * S + xt * 32 + h0) * St + yt) * Kt;" in reader
    assert "page += St * Kt" in reader
    assert "uint32_t page = ((b * S + xt * 32 + h * R) * St + yt) * NTG;" in writer
    assert "page += St * NTG" in writer
    assert "uint32_t page = (b * CT + (is_a ? q : q - CT)) * 32 * SS + cube;" in writer
    assert "page += SS" in writer


# ---------------------------------------------------------------- weights

def _ckpt(K, h, cz, g):
    return {"g_in.weight": torch.randn(2 * h, K, generator=g) / K ** 0.5,
            "p_in.weight": torch.randn(2 * h, K, generator=g) / K ** 0.5,
            "g_out.weight": torch.randn(cz, K, generator=g) / K ** 0.5}


def test_t1_weights_are_the_incumbents_columns():
    g = torch.Generator().manual_seed(1)
    w = _ckpt(128, 128, 128, g)
    g_in_t, p_in_t = w["g_in.weight"].t(), w["p_in.weight"].t()
    wpT, wgT, wgo = TTL.t1_host_weights(g_in_t, p_in_t, w["g_out.weight"].t())
    assert tuple(wpT.shape) == tuple(wgT.shape) == (2 * 128, 128)
    assert tuple(wgo.shape) == (128, 128)
    # `_gp_in_gated(hidden, 1)` on the torch side: the fused chunk, regrouped [p_a | p_b], [g_a | g_b].
    for split in (False, True):
        prev = T.set_trimul_gp_bank_split(split)
        try:
            (fused,) = T.TriangleMultiplication._gp_fused_order(None, (g_in_t, p_in_t), 128, 1)
            col = lambda r: fused[:, T.gp_off(r, 128):T.gp_off(r, 128) + 128]  # noqa: E731
            assert torch.equal(torch.cat([col("p_a"), col("p_b")], -1).t(), wpT)
            assert torch.equal(torch.cat([col("g_a"), col("g_b")], -1).t(), wgT)
        finally:
            T.set_trimul_gp_bank_split(prev)


# ---------------------------------------------------------------- tile emulation

FACE, TB_EL = 16, 1024


def _tile_to_faces(t):
    """A 32 x 32 tile in the device's face order: faces (0,0) (0,1) (1,0) (1,1), 16 x 16 row-major."""
    return torch.cat([t[r:r + 16, c:c + 16].reshape(-1) for r in (0, 16) for c in (0, 16)])


def _faces_to_tile(f):
    t = torch.empty(32, 32, dtype=f.dtype)
    for i, (r, c) in enumerate(((0, 0), (0, 16), (16, 0), (16, 16))):
        t[r:r + 16, c:c + 16] = f[i * 256:(i + 1) * 256].reshape(16, 16)
    return t


def _pages(x):
    """Interleaved tile pages of a tiled tensor, row-major over (rows / 32, cols / 32)."""
    rows = x.reshape(-1, x.shape[-1])
    R, Cc = rows.shape
    return [rows[r:r + 32, c:c + 32] for r in range(0, R, 32) for c in range(0, Cc, 32)]


def _mask_cut(mtile_faces, il):
    """reader.cpp's mask cut, in 32-bit words of a bf16 tile: row il of the mask tile to row 0."""
    words = mtile_faces.reshape(-1, 2)            # two bf16 per uint32
    dst = torch.zeros(512, 2, dtype=mtile_faces.dtype)
    s0 = (((il >> 4) * 2) * 256 + (il & 15) * 16) // 2
    for k in range(8):
        dst[k] = words[s0 + k]
        dst[128 + k] = words[s0 + 128 + k]
    return _faces_to_tile(dst.reshape(-1))


def _gather(src_tiles_faces, c):
    """writer.cpp's gather (element offsets = byte offsets / 2): row c of the item's R = 32 or 16
    tiles -> the first R * 32 elements of a staged tile (all four faces, or faces 0-1)."""
    TB, FB, RB = 1024, 256, 16
    src = torch.cat(src_tiles_faces)
    R = len(src_tiles_faces)
    st = torch.empty(R * 32, dtype=src.dtype)
    sc = (c >> 4) * 2 * FB + (c & 15) * RB
    s0, s1 = sc, sc + FB
    for f in range(R // 16):
        d0, d1 = 2 * f * FB, (2 * f + 1) * FB
        for _ in range(16):
            st[d0:d0 + RB] = src[s0:s0 + RB]
            st[d1:d1 + RB] = src[s1:s1 + RB]
            s0, s1, d0, d1 = s0 + TB, s1 + TB, d0 + RB, d1 + RB
    return st


@pytest.mark.parametrize("R", [32, 16])
@pytest.mark.parametrize("S,masked", [(64, True), (96, False)])
def test_emulated_data_flow_matches_the_model(S, masked, R):
    d = torch.float64
    K, h, cz = 128, 128, 128
    St, Kt, ct2, ct, ntg = S // 32, K // 32, 2 * h // 32, h // 32, cz // 32
    g = torch.Generator().manual_seed(S)
    w = {k: v.to(d) for k, v in _ckpt(K, h, cz, g).items()}
    z = (torch.randn(1, S, S, K, generator=g) * 2 + torch.randn(1, S, S, 1, generator=g)).to(d)
    gamma = (1 + 0.1 * torch.randn(K, generator=g)).to(d)
    beta = (0.1 * torch.randn(K, generator=g)).to(d)
    m = (torch.rand(1, S, S, generator=g) > 0.2).to(d) if masked else None

    # The model (openfold3 / the incumbent route): mask on a only, before the (0,3,1,2) move.
    zn = torch.nn.functional.layer_norm(z, (K,), gamma, beta, 1e-5)
    gi, pi = zn @ w["g_in.weight"].t(), zn @ w["p_in.weight"].t()
    a_ref = pi[..., :h] * torch.sigmoid(gi[..., :h])
    b_ref = pi[..., h:] * torch.sigmoid(gi[..., h:])
    if masked:
        a_ref = a_ref * m[..., None]
    a_ref, b_ref = a_ref.permute(0, 3, 1, 2), b_ref.permute(0, 3, 1, 2)
    go_ref = zn @ w["g_out.weight"].t()

    wpT, wgT, wgo = TTL.t1_host_weights(w["g_in.weight"].t(), w["p_in.weight"].t(),
                                        w["g_out.weight"].t())
    W = _pages(torch.cat([wpT, wgT]))             # c_0: Wp^T then Wg^T, (2 * ct2) x Kt tiles
    WO = _pages(wgo)                              # c_18: Kt x ntg tiles
    gam = _pages(torch.nn.functional.pad(gamma[None], (0, 0, 0, 31)))   # row 0 carries the vector
    bet = _pages(torch.nn.functional.pad(beta[None], (0, 0, 0, 31)))
    Z = _pages(z)
    M = _pages(m) if masked else None
    a_pages, b_pages, g_pages = {}, {}, {}

    NH = 32 // R
    for c0, n in TTL.t1_split(St * St * NH, 7):   # an odd core count, so splits are uneven
        for v in range(c0, c0 + n):
            # item v = half h of unit u (h = 0 when R = 32): global row tiles h0 + il
            u, h0 = v // NH, (v % NH) * R
            b, xt, yt = u // (St * St), (u // St) % St, u % St
            # reader + compute 1: layer_norm per row tile into slot dt * R + il
            xn = [None] * (R * Kt)
            for il in range(R):
                row = torch.cat([Z[TTL.t1_x_page(b, xt, yt, h0 + il, dt, St, Kt)] for dt in range(Kt)], 1)
                mu = row.mean(1, keepdim=True)
                xc = row - mu
                rstd = 1 / torch.sqrt((xc * xc).mean(1, keepdim=True) + 1e-5)
                for dt in range(Kt):
                    t = xc[:, dt * 32:(dt + 1) * 32] * rstd
                    xn[dt * R + il] = t * gam[dt][0:1] + bet[dt][0:1]      # bcast row 0
            # compute 2 + writer: g_out in z's layout
            for il in range(R):
                for nn in range(ntg):
                    acc = sum(xn[k * R + il] @ WO[k * ntg + nn] for k in range(Kt))
                    g_pages[TTL.t1_g_page(b, xt, yt, h0 + il, nn, St, ntg)] = acc
            # compute 3: per channel tile q, R transposed products, gate, mask, then the gather
            for q in range(ct2):
                P = [sum(W[q * Kt + k] @ xn[k * R + il].t() for k in range(Kt)) for il in range(R)]
                G = [torch.sigmoid(sum(W[(ct2 + q) * Kt + k] @ xn[k * R + il].t() for k in range(Kt)))
                     for il in range(R)]
                out = []
                for il in range(R):
                    o = P[il] * G[il]
                    if masked and q < ct:
                        mt = _mask_cut(_tile_to_faces(M[(b * St + xt) * St + yt]), h0 + il)
                        o = o * mt[0:1]                                       # mul_bcast_rows
                    out.append(_tile_to_faces(o))
                for c in range(32):
                    page = TTL.t1_ab_page(b, q, c, xt, yt, ct, St)
                    pages = a_pages if q < ct else b_pages
                    # the page write: R * 32 elements at element offset h * R * 32 of the tile
                    faces = pages.setdefault(page, torch.full((TB_EL,), float("nan"), dtype=d))
                    off = (v % NH) * R * 32
                    assert faces[off:off + R * 32].isnan().all()          # each byte written once
                    faces[off:off + R * 32] = _gather(out, c)
    a_pages = {p: _faces_to_tile(f) for p, f in a_pages.items()}
    b_pages = {p: _faces_to_tile(f) for p, f in b_pages.items()}

    def unpages(pages, shape):
        cols, R = shape[-1], 1
        for s in shape[:-1]:
            R *= s
        full = torch.empty(R, cols, dtype=d)
        nc = cols // 32
        assert sorted(pages) == list(range(R // 32 * nc))
        for p, t in pages.items():
            full[(p // nc) * 32:(p // nc + 1) * 32, (p % nc) * 32:(p % nc + 1) * 32] = t
        return full.reshape(shape)

    assert torch.allclose(unpages(a_pages, (1, h, S, S)), a_ref, rtol=0, atol=1e-10)
    assert torch.allclose(unpages(b_pages, (1, h, S, S)), b_ref, rtol=0, atol=1e-10)
    assert torch.allclose(unpages(g_pages, (1, S, S, cz)), go_ref, rtol=0, atol=1e-10)


# ---------------------------------------------------------------- eligibility, no device

class _Fake:
    def __init__(self, shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mem=None, padded=None):
        self.shape, self.padded_shape = tuple(shape), tuple(padded or shape)
        self.dtype, self.layout = dtype, layout
        self._mem = mem or ttnn.DRAM_MEMORY_CONFIG

    def memory_config(self):
        return self._mem


def _args(S=512, K=128, h=128, N=128):
    return dict(x=_Fake([1, S, S, K]), ln_w=_Fake([K], padded=[32, K]), ln_b=_Fake([K], padded=[32, K]),
                wpT=_Fake([2 * h, K]), wgT=_Fake([2 * h, K]), wgo=_Fake([K, N]))


def test_eligibility(monkeypatch):
    monkeypatch.setattr(T, "_l1_bank_bytes", lambda: L1_ALLOCATOR_BANK)
    monkeypatch.setattr(TTL, "T1", False)
    assert TTL.t1_eligible(**_args()) == "off"
    with T.levers("trimul_t1"):           # the lever alone serves the call (it was inert on 2227e169f)
        assert TTL.t1_eligible(**_args()) is None
    monkeypatch.setattr(TTL, "T1", True)
    assert TTL.t1_eligible(**_args()) is None
    assert TTL.t1_eligible(**_args(), mask=_Fake([1, 512, 512])) is None
    assert TTL.t1_eligible(**_args(), mask=_Fake([1, 512, 500], padded=[1, 512, 512])) == "mask"
    assert TTL.t1_eligible(**_args(K=256)) == "key=8x8x4"            # protenix-v2's c_z
    assert TTL.t1_eligible(**_args(h=64)) == "key=4x4x4"
    a = _args()
    a["ln_w"] = _Fake([128], dtype=ttnn.float32)
    assert TTL.t1_eligible(**a) == "ln_params"
    a = _args()
    a["x"] = _Fake([1, 500, 500, 128], padded=[1, 512, 512, 128])
    assert TTL.t1_eligible(**a) == "act_shape"
    a = _args()
    a["x"] = _Fake([1, 512, 512, 128], mem=ttnn.L1_MEMORY_CONFIG)
    assert TTL.t1_eligible(**a) == "memcfg"
    monkeypatch.setattr(T, "_l1_bank_bytes", lambda: 1_000_000)      # a smaller part: refuse, not throw
    assert TTL.t1_eligible(**_args(), mask=_Fake([1, 512, 512])) == "l1"


def test_call_site_is_inert_when_off(monkeypatch):
    monkeypatch.setattr(TTL, "T1", False)
    tm = object.__new__(T.TriangleMultiplication)     # no weights, no device: must not be touched
    before = list(TTL.T1_STATS)
    assert tm._t1_in(None, None, 512, [1, 512, 512, 128]) is None
    assert TTL.T1_STATS == before
