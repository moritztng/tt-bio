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
    assert total == (911_360 if not mask else 987_136)
    assert total < L1_PER_CORE
    assert total <= 0.85 * L1_ALLOCATOR_BANK      # t1_eligible's own gate
    # N only sets units per core: at 512 tokens, 256 units of 32 row tiles over the 11 x 10 grid.
    split = TTL.t1_split((512 // 32) ** 2, 110)
    assert sum(n for _, n in split) == 256 and max(n for _, n in split) == 3


@pytest.mark.parametrize("mask", [False, True])
def test_cb_budget_rfd4_gathers_8_per_barrier(mask):
    """RFD4's key (c_z 384, hidden 256) stages 8 gathered tiles per read barrier (T1_NB), the others gin_moved's 2."""
    assert TTL._t1_nb_pref((12, 16, 12)) == 8 and TTL._t1_nb_pref((8, 16, 8)) == TTL.GIN_MOVE_NB
    b8, b2 = (TTL.t1_cb_bytes(12, 16, 12, mask, nb=nb, w8=True) for nb in (8, 2))
    assert TTL.t1_cb_bytes(12, 16, 12, mask, w8=True) == b8 == b2 + 12 * 2048
    assert b8 <= 0.85 * L1_ALLOCATOR_BANK


@pytest.mark.parametrize("mask", [False, True])
def test_rfd4_splits_the_gather_at_nb_8(mask, monkeypatch):
    """RFD4's key gathers on both data-movement cores at NB 8 with or without the mask (c_2's two channel tiles become
    one per core, the reader stages its own 2 * NB); the other keys keep the writer alone."""
    import tt_bio.tenstorrent as T_
    monkeypatch.setattr(T_, "_l1_bank_bytes", lambda: L1_ALLOCATOR_BANK)
    one = TTL.t1_cb_bytes(12, 16, 12, mask, w8=True)
    assert TTL.t1_cb_bytes(12, 16, 12, mask, w8=True, split=True) == one + 16 * 2048
    assert TTL.t1_dm((12, 16, 12), mask, True) == (8, True)
    assert TTL.t1_dm((12, 16, 12), mask, True, b8=True) == (8, True)   # bfp8 a / b: the split and the holds fit
    assert TTL.t1_cb_bytes(12, 16, 12, mask, w8=True, split=True, b8=True, il=8) < one + 16 * 2048
    assert TTL.t1_dm((12, 16, 12), mask, False)[1] is False          # bf16 weights: no room
    for key in [(4, 8, 4), (8, 16, 8), (12, 24, 12)]:
        assert TTL.t1_dm(key, mask, True)[1] is False


@pytest.mark.parametrize("mask", [False, True])
def test_cb_budget_fits_l1_c256(mask):
    """c_z 256 in one launch of 32-row units would need ~1.9 MB a core; its plan fits the gate."""
    assert TTL.T1_PLANS[(8, 16, 8)] == (2, 16)
    assert TTL.t1_cb_bytes(8, 16, 8, mask) <= 0.85 * L1_ALLOCATOR_BANK


@pytest.mark.parametrize("mask", [False, True])
def test_cb_budget_fits_l1_c384(mask):
    """c_z 384 (OpenDDE): three channel groups of 8-row units, about the c_z 256 plan's footprint."""
    assert TTL.T1_PLANS[(12, 24, 12)] == (3, 8)
    assert TTL.t1_cb_bytes(12, 24, 12, mask) <= 0.85 * L1_ALLOCATOR_BANK


@pytest.mark.parametrize("mask", [False, True])
def test_cb_budget_fits_l1_c384_w8(mask):
    """bfp8 resident weights (trimul_t1w8): c_z 384 fits two channel groups, c_z 256 one launch."""
    assert TTL.t1_plan((12, 24, 12), w8=True) == (2, 8)
    assert TTL.t1_plan((12, 24, 12)) == (3, 8) and TTL.t1_plan((8, 16, 8), w8=True) == (1, 16)
    w8 = TTL.t1_cb_bytes(12, 24, 12, mask, w8=True)
    assert w8 <= 0.85 * L1_ALLOCATOR_BANK
    assert w8 < TTL.t1_cb_bytes(12, 24, 12, mask)   # 1.5x the weight tiles a core, at 1088 B a tile not 2048
    assert TTL.t1_cb_bytes(8, 16, 8, mask, w8=True) <= 0.85 * L1_ALLOCATOR_BANK


def test_cb_plan_covers_every_cb_the_kernels_use():
    used = set()
    for f in ("reader.cpp", "compute.cpp", "writer.cpp"):
        used |= {int(i) for i in re.findall(r"CBIndex::c_(\d+)", (KDIR / f).read_text())}
    bf, f32 = TTL.t1_cb_plan(KT, CT2, NTG, True)
    planned = [i for i, _ in bf] + [i for i, _ in f32]
    assert len(planned) == len(set(planned)), "a CB index is planned twice"
    bfg, _ = TTL.t1_cb_plan(12, 24, 12, True)   # c_z 384: channel groups add the T1_LN1 xn CB
    bfs, _ = TTL.t1_cb_plan(12, 16, 12, True, w8=True, split=True)   # T1_SPLIT adds the reader's CBs
    assert len(bfs) == len({i for i, _ in bfs})
    bfh, _ = TTL.t1_cb_plan(12, 16, 12, True, w8=True, split=True, b8=True)   # bfp8 at 8-row items adds the holds
    assert len(bfh) == len({i for i, _ in bfh}) and {i for i, _ in bfh} - {i for i, _ in bfs} == {26, 27}
    assert used == set(planned) | {i for i, _ in bfg} | {i for i, _ in bfh}
    bf0, _ = TTL.t1_cb_plan(KT, CT2, NTG, False)
    assert {3, 6, 15}.isdisjoint(i for i, _ in bf0)


def test_kernel_page_expressions_match_the_python_mirror():
    reader = (KDIR / "reader.cpp").read_text()
    writer = (KDIR / "writer.cpp").read_text()
    gather = (KDIR / "gather.hpp").read_text()
    assert "return ((b * S + xt * 32 + h * IL) * St + yt) * Kt;" in reader
    assert "page += St * Kt" in reader
    assert "noc_async_read_page(k * NTGT + n, wo, w)" in reader
    assert "uint32_t page = ((b * S + xt * 32 + h * IL) * St + yt) * NTGT + N0;" in writer
    assert "page += St * NTGT" in writer
    assert "const uint32_t off0 = IL == 32 ? 0 : (r0 >> 4) * 2 * FB + (r0 & 15) * RB;" in gather
    assert "get_noc_addr(page, off0)" in gather
    assert "page += SS" in gather
    assert "const uint32_t page = (b * CT + (q < CT ? q : q - CT)) * 32 * SS + cube;" in writer
    assert "const uint32_t pg = (b * CT + (q < CT ? q : q - CT)) * 32 * St * St + cube;" in reader
    # T1_SPLIT: compute's even channel tiles (from the launch's first) go to the reader, the odd ones to the writer.
    assert "for (uint32_t q = Q0 + T1SPLIT; q < Q0 + QN; q += 1 + T1SPLIT)" in writer
    assert "for (uint32_t q = Q0; q < Q0 + QN; q += 2)" in reader
    assert "T1SPLIT && !(q & 1) ? out_rd_cb : out_cb" in (KDIR / "compute.cpp").read_text()


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


def _gather(src_tiles_faces, c, il=32, r0=0):
    """writer.cpp's gather (element offsets = byte offsets / 2): row c of the `il` tiles -> rows
    r0 .. r0 + il - 1 of the staged output tile, in faces; then its writes, as (offset, length)."""
    TB, FB, RB = 1024, 256, 16
    src = torch.cat(src_tiles_faces)
    st = torch.full((TB_EL,), float("nan"), dtype=src.dtype)
    s0 = (c >> 4) * 2 * FB + (c & 15) * RB
    for i in range(il):
        r = r0 + i
        d = (r >> 4) * 2 * FB + (r & 15) * RB
        st[d:d + RB] = src[s0 + i * TB:s0 + i * TB + RB]
        st[d + FB:d + FB + RB] = src[s0 + i * TB + FB:s0 + i * TB + FB + RB]
    off0 = 0 if il == 32 else (r0 >> 4) * 2 * FB + (r0 & 15) * RB
    wlen = TB if il == 32 else (2 * FB if il == 16 else il * RB)
    writes = [(off0, wlen)] + ([(off0 + FB, wlen)] if il == 8 else [])
    return st, writes


@pytest.mark.parametrize("S,masked,K,rows", [(64, True, 128, 32), (96, False, 128, 32), (64, True, 128, 16),
                                             (96, False, 128, 16), (64, True, 256, 16), (96, False, 256, 16),
                                             (64, True, 384, 8), (96, False, 384, 8)])
def test_emulated_data_flow_matches_the_model(S, masked, K, rows):
    """The kernels' loops, page for page, against float64 torch: one launch of 32-row units at
    c_z 128 (or of 16-row halves, t1_rows), two channel groups of 16-row units writing half tiles
    at c_z 256, three of 8-row units at 384."""
    d = torch.float64
    h = cz = K
    St, Kt, ct2, ct, ntg = S // 32, K // 32, 2 * h // 32, h // 32, cz // 32
    groups, IL = TTL.T1_PLANS[(Kt, ct2, ntg)][0], rows
    H, qn, ng = 32 // IL, ct2 // groups, ntg // groups
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
    WP, WG = _pages(wpT), _pages(wgT)             # ct2 x Kt tiles each
    WOP = _pages(wgo)                             # Kt x ntg tiles
    gam = _pages(torch.nn.functional.pad(gamma[None], (0, 0, 0, 31)))   # row 0 carries the vector
    bet = _pages(torch.nn.functional.pad(beta[None], (0, 0, 0, 31)))
    Z = _pages(z)
    M = _pages(m) if masked else None
    a_pages, b_pages, g_pages = {}, {}, {}

    ab_faces = {}                                 # (is_a, page) -> the output tile in faces
    for grp in range(groups):
        Q0, N0 = grp * qn, grp * ng
        # reader: c_0 = pages Q0 * Kt .. of Wp^T then of Wg^T; c_18 slot k * ng + n = page k * ntg + N0 + n
        W = WP[Q0 * Kt:(Q0 + qn) * Kt] + WG[Q0 * Kt:(Q0 + qn) * Kt]
        WO = [WOP[k * ntg + N0 + n] for k in range(Kt) for n in range(ng)]
        for c0, n in TTL.t1_split(St * St * H, 7):    # an odd core count, so splits are uneven
            for u in range(c0, c0 + n):
                b, xt, hh, yt = u // (St * St * H), (u // (St * H)) % St, (u // St) % H, u % St
                # reader + compute 1: layer_norm per row tile h * IL + il into slot dt * IL + il
                xn = [None] * (IL * Kt)
                for il in range(IL):
                    row = torch.cat([Z[TTL.t1_x_page(b, xt, yt, hh * IL + il, dt, St, Kt)]
                                     for dt in range(Kt)], 1)
                    mu = row.mean(1, keepdim=True)
                    xc = row - mu
                    rstd = 1 / torch.sqrt((xc * xc).mean(1, keepdim=True) + 1e-5)
                    for dt in range(Kt):
                        t = xc[:, dt * 32:(dt + 1) * 32] * rstd
                        xn[dt * IL + il] = t * gam[dt][0:1] + bet[dt][0:1]      # bcast row 0
                # compute 2 + writer: g_out in z's layout
                for il in range(IL):
                    for nn in range(ng):
                        acc = sum(xn[k * IL + il] @ WO[k * ng + nn] for k in range(Kt))
                        g_pages[TTL.t1_g_page(b, xt, yt, hh * IL + il, N0 + nn, St, ntg)] = acc
                # compute 3: per channel tile q, IL transposed products, gate, mask, then the gather
                for qq in range(qn):
                    q = Q0 + qq
                    P = [sum(W[qq * Kt + k] @ xn[k * IL + il].t() for k in range(Kt)) for il in range(IL)]
                    G = [torch.sigmoid(sum(W[(qn + qq) * Kt + k] @ xn[k * IL + il].t() for k in range(Kt)))
                         for il in range(IL)]
                    out = []
                    for il in range(IL):
                        o = P[il] * G[il]
                        if masked and q < ct:
                            mt = _mask_cut(_tile_to_faces(M[(b * St + xt) * St + yt]), hh * IL + il)
                            o = o * mt[0:1]                                       # mul_bcast_rows
                        out.append(_tile_to_faces(o))
                    for c in range(32):
                        key = (q < ct, TTL.t1_ab_page(b, q, c, xt, yt, ct, St))
                        buf = ab_faces.setdefault(key, torch.full((TB_EL,), float("nan"), dtype=d))
                        st, writes = _gather(out, c, IL, hh * IL)
                        for o, n in writes:
                            buf[o:o + n] = st[o:o + n]
    for (is_a, page), f in ab_faces.items():
        (a_pages if is_a else b_pages)[page] = _faces_to_tile(f)

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
    assert TTL.t1_eligible(**_args(K=256)) == "key=8x8x4"            # c_z 256 with hidden 128: no plan
    assert TTL.t1_eligible(**_args(K=256, h=256, N=256)) == "c256_off"  # protenix / esmfold2: own lever until graded
    with T.levers("trimul_t1_c256"):     # a wide plan's lever turns T1 on for that plan alone
        assert TTL.t1_eligible(**_args(K=256, h=256, N=256)) is None
        monkeypatch.setattr(TTL, "T1", False)
        assert TTL.t1_eligible(**_args(K=256, h=256, N=256)) is None
        assert TTL.t1_eligible(**_args()) == "off"                        # c_z 128 stays off
        monkeypatch.setattr(TTL, "T1", True)
    assert TTL.t1_eligible(**_args(K=384, h=384, N=384)) == "c384_off"  # opendde: own lever until graded
    with T.levers("trimul_t1_c384"):
        assert TTL.t1_eligible(**_args(K=384, h=384, N=384)) is None
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


# ---------------------------------------------------------------- work split

@pytest.mark.parametrize("units,ncores,want", [
    (256, 110, 2),   # 512 tokens on a p150a: 78 % -> 93 %
    (256, 64, 1),    # 512 tokens on a full 8x8 grid: already balanced
    (256, 56, 1),    # Wormhole 8x7: 91 % either way
    (1024, 110, 1),  # 1024 tokens: 93 % -> 98 %, under the 10 % bar
    (144, 110, 2),   # 384 tokens: 65 % -> 87 %
])
def test_balance_split(units, ncores, want):
    assert TTL.balance_split(units, ncores) == want


@pytest.mark.parametrize("plan_il,units,ncores,want", [
    (32, 256, 110, 16),   # c_z 128 at 512 tokens on a p150a: halves
    (32, 256, 64, 32),
    (16, 256, 110, 16),   # c_z 256 / 384 plans keep their own rows
    (8, 256, 110, 8),
])
def test_t1_rows(plan_il, units, ncores, want, monkeypatch):
    monkeypatch.setattr(TTL, "T1_ROWS", 0)
    assert TTL.t1_rows(plan_il, units, ncores) == want
    monkeypatch.setattr(TTL, "T1_ROWS", 16)
    assert TTL.t1_rows(plan_il, units, ncores) == min(plan_il, 16)


@pytest.mark.parametrize("St,cores", [(16, 110), (16, 130), (8, 110), (12, 7)])
def test_t1_pair_items_cover_each_face_pair_on_one_core(St, cores):
    """bfp8 at 8-row items (kernels/trimul_in_t1/unit.hpp): every (b, xt, h, yt) once, items 2k and 2k + 1 the two
    halves of one face pair, and every core runs whole pairs, so the second item can write both items' exponents."""
    H = 4
    nu = 2 * St * St * H
    units = [TTL.t1_unit(u, St, H, pair=True) for u in range(nu)]
    assert sorted(units) == sorted(TTL.t1_unit(u, St, H) for u in range(nu))
    for k in range(0, nu, 2):
        (b0, x0, h0, y0), (b1, x1, h1, y1) = units[k], units[k + 1]
        assert (b0, x0, y0) == (b1, x1, y1) and h0 % 2 == 0 and h1 == h0 + 1
    split = TTL.t1_split(nu, cores, pair=True)
    assert sum(n for _, n in split) == nu and all(u0 % 2 == 0 and n % 2 == 0 for u0, n in split)
    assert all(split[i][0] + split[i][1] == split[i + 1][0] for i in range(len(split) - 1))


def test_t1_pairs_only_where_balance_is_free():
    """bfp8 at 8-row items needs whole face pairs per core; T1 pairs only where that keeps the busiest core's count."""
    items = lambda n: (n // 32) ** 2 * 4  # noqa: E731
    assert TTL.t1_pair_free(items(512), 110) and TTL.t1_pair_free(items(384), 110)
    assert not TTL.t1_pair_free(items(256), 110)                    # 3 items -> 4 on the busiest p300c core
    assert TTL.t1_pair_free(items(256), 130) and TTL.t1_pair_free(items(512), 130)
    assert not TTL.t1_pair_free(items(384), 130)                    # 5 -> 6 on a p150a
