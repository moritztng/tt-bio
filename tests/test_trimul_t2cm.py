"""T2CM (`trimul_tail.fused_tail_ln_cm`, the `trimul_t2cm` lever) host side, no device.

T2 reading the trimul einsum's channel-major [1, K, S, S] output itself, so `reblock_permute_back` never
runs. Checked here: the lever is named and in no mode, eligibility declines cleanly (a reason string,
nothing thrown or touched), the CB plan fits L1, every CB the kernels name is planned, the core split covers
each cube once, and the reader's gather byte arithmetic, emulated on the real face layout of a tile, hands
compute exactly the tile whose transpose is the permuted x tile T2 would read. The device A/B is
perf/spd_trimul/t2cm.py.

Run: TT_VISIBLE_DEVICES= python3 -m pytest tests/test_trimul_t2cm.py
"""
import re
from pathlib import Path

import pytest
import torch

ttnn = pytest.importorskip("ttnn")
from tt_bio import tenstorrent as T
from tt_bio import trimul_tail as TTL

KDIR = Path(TTL.__file__).resolve().parent / "kernels" / "trimul_tail_res"
L1_BH_BANK = 1_461_760           # `_l1_bank_bytes` on Blackhole
L1_WH_BANK = 1_499_136 * 0.98    # the Wormhole figure tests/test_transition_t3.py budgets against


def test_lever_named_and_fast_only():
    assert "trimul_t2cm" in T.LEVERS
    assert "trimul_t2cm" not in T.UNGRADED_LEVERS | T.NORMAL_LEVERS
    from tt_bio import openfold3_fold as of3
    assert "trimul_t2cm" in of3.FAST_LEVERS and "trimul_t2cm" not in of3.NORMAL_LEVERS


# ---------------------------------------------------------------- eligibility

class _Fake:
    def __init__(self, shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mem=None, padded=None):
        self.shape, self.padded_shape = tuple(shape), tuple(padded or shape)
        self.dtype, self.layout = dtype, layout
        self._mem = mem or ttnn.DRAM_MEMORY_CONFIG

    def memory_config(self):
        return self._mem


def _args(S=512, K=128, N=128):
    return dict(xcm=_Fake([1, K, S, S]), ln_w=_Fake([K], padded=[32, K]), ln_b=_Fake([K], padded=[32, K]),
                wp=_Fake([K, N]), g=_Fake([1, S, S, N]), resid=_Fake([1, S, S, N]))


def test_eligibility_declines_cleanly(monkeypatch):
    monkeypatch.setattr(T, "_l1_bank_bytes", lambda: L1_BH_BANK)
    monkeypatch.setattr(TTL, "T2", False)
    monkeypatch.setattr(TTL, "T2CM", False)
    assert TTL.t2cm_eligible(**_args()) == "off"
    with T.levers("trimul_t2cm"):                 # without T2 there is nothing to feed
        assert TTL.t2cm_eligible(**_args()) == "off"
    with T.levers("trimul_t2"):                   # T2 alone keeps its permuted input
        assert TTL.t2cm_eligible(**_args()) == "off"
    with T.levers("trimul_t2,trimul_t2cm"):
        assert TTL.t2cm_eligible(**_args()) is None
        assert TTL.t2cm_eligible(**_args(S=256)) is None
        assert TTL.t2cm_eligible(**_args(K=256, N=256)) == "l1"          # protenix-v2's c_z: T2 serves (8, 8), the gather does not fit
        a = _args()
        a["xcm"] = _Fake([1, 128, 500, 500], padded=[1, 128, 512, 512])
        assert TTL.t2cm_eligible(**a) == "cm_shape"                      # ragged: the permute keeps it
        a = _args()
        a["xcm"] = _Fake([1, 512, 512, 128])                             # the permuted layout is not T2CM's
        assert TTL.t2cm_eligible(**a) in ("cm_shape", "act_k")
        a = _args()
        a["xcm"] = _Fake([1, 128, 512, 512], mem=ttnn.L1_MEMORY_CONFIG)
        assert TTL.t2cm_eligible(**a) == "memcfg"
        a = _args()
        a["resid"] = _Fake([1, 256, 256, 128])
        assert TTL.t2cm_eligible(**a) == "out_shape"
        a = _args()
        a["ln_w"] = _Fake([128], dtype=ttnn.float32)
        assert TTL.t2cm_eligible(**a) == "ln_params"
        a = _args()
        a["g"] = _Fake([1, 512, 512, 128], dtype=ttnn.bfloat8_b)
        assert TTL.t2cm_eligible(**a) == "dtype"
        monkeypatch.setattr(T, "_l1_bank_bytes", lambda: 600_000)        # a smaller part: refuse, not throw
        assert TTL.t2cm_eligible(**_args()) == "l1"


def test_off_declines_without_counting_or_touching(monkeypatch):
    monkeypatch.setattr(TTL, "T2", False)
    monkeypatch.setattr(TTL, "T2CM", False)
    before, rej = list(TTL.T2CM_STATS), dict(TTL.REJECTS)
    assert TTL.fused_tail_ln_cm(None, None, None, None, None, None, (0, 0, 0, 0), (1, 1)) is None
    assert TTL.T2CM_STATS == before and TTL.REJECTS == rej


# ---------------------------------------------------------------- CB plan

def test_cb_plan_fits_l1():
    kt, nt = 4, 4
    Mb = T._MM_BLOCK[(kt, nt)][0]
    assert TTL.TILE % Mb == 0
    tiles = TTL.t2cm_tiles(kt, nt, Mb)
    assert tiles == TTL._t2_tiles(kt, nt, Mb) + 2 * Mb * kt + 32 * kt
    total = tiles * 2048
    assert total == 755_712                       # 369 bf16 tiles: T2's 209 + 32 gather + 128 scratch
    assert total <= 0.85 * L1_BH_BANK
    assert total <= 0.85 * L1_WH_BANK


def test_every_cb_the_kernels_name_is_planned():
    used = set()
    for f in ("reader_ln_cm.cpp", "writer_ln_cm.cpp", "compute_ln.cpp"):
        used |= {int(i) for i in re.findall(r"CBIndex::c_(\d+)", (KDIR / f).read_text())}
    # _build_t2's plan: T2's sixteen, T2m's moved-tile gather (c_15) and the two T2CM adds
    planned = {0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, TTL.T2CM_CB_GATHER, TTL.T2CM_CB_SCRATCH}
    assert used <= planned, used - planned
    assert {TTL.T2CM_CB_GATHER, TTL.T2CM_CB_SCRATCH} <= used
    src = Path(TTL.__file__).read_text()
    assert "_cb(T2CM_CB_GATHER, core_grid, 2 * Mb * kt), _cb(T2CM_CB_SCRATCH, core_grid, TILE * kt)" in src


@pytest.mark.parametrize("units,cores", [(256, 64), (256, 130), (256, 110), (64, 130), (5, 64)])
def test_split_covers_each_cube_once(units, cores):
    parts = TTL.t2cm_split(units, cores)
    assert len(parts) == cores
    assert [u for u0, n in parts for u in range(u0, u0 + n)] == list(range(units))
    assert max(n for _, n in parts) - min(n for _, n in parts) <= 1


# ---------------------------------------------------------------- the gather, emulated on the face layout

def _to_faces(t):
    """A 32x32 tile as tt-metal stores it: faces f0 (rows 0-15, cols 0-15), f1 (0-15, 16-31), f2, f3,
    each row-major."""
    return torch.cat([t[r:r + 16, c:c + 16].reshape(-1) for r in (0, 16) for c in (0, 16)])


def _from_faces(v):
    t = torch.empty(32, 32, dtype=v.dtype)
    for f, (r, c) in enumerate([(0, 0), (0, 16), (16, 0), (16, 16)]):
        t[r:r + 16, c:c + 16] = v[f * 256:(f + 1) * 256].reshape(16, 16)
    return t


def test_reader_gather_then_transpose_is_the_permuted_tile():
    """reader_ln_cm.cpp's step 2 run on element offsets (bf16: 2 bytes an element, so every byte offset
    in the kernel is halved here), then compute's transpose_wh, against x = xcm.permute(0, 2, 3, 1)."""
    src = (KDIR / "reader_ln_cm.cpp").read_text()
    # the expressions this emulation mirrors
    assert "uint32_t page = it * St + jt;" in src and "page += StSt;" in src
    assert "const uint32_t src_il = scratch + (il / FACE) * 2 * FACE_BYTES + (il % FACE) * FACE_ROW_BYTES;" in src
    assert "uint32_t s0 = src_il + ct * TILE * tb;" in src
    assert "const uint32_t dst_tile = gbase + (mi * Kt + ct) * tb;" in src
    assert "const uint32_t m = (it * TILE + il0 + mi) * St + jt;" in src
    wsrc = (KDIR / "writer_ln_cm.cpp").read_text()
    assert "const uint32_t row = ((it * TILE + il0 + mi) * St + jt) * Nt;" in wsrc

    g = torch.Generator().manual_seed(3)
    K, S = 64, 96
    Kt, St, Mb = K // 32, S // 32, 4
    TB, FR, FB = 1024, 16, 256                    # tile, face row, face: in elements
    xcm = torch.randn(1, K, S, S, generator=g)
    x = xcm.permute(0, 2, 3, 1)                   # [1, S, S, K], what T2 reads
    # xcm's tile pages, face layout: page = (c * St + i_t) * St + j_t
    pages = {}
    for c in range(K):
        for a in range(St):
            for b in range(St):
                pages[(c * St + a) * St + b] = _to_faces(xcm[0, c, a * 32:(a + 1) * 32, b * 32:(b + 1) * 32])
    seen = set()
    for u in range(St * St):
        it, jt = u // St, u % St
        scratch = torch.cat([pages[it * St + jt + c * St * St] for c in range(K)])   # step 1
        for il0 in range(0, 32, Mb):
            gath = torch.zeros(Mb * Kt * TB)
            for mi in range(Mb):
                il = il0 + mi
                src_il = (il // 16) * 2 * FB + (il % 16) * FR
                for ct in range(Kt):
                    dst = (mi * Kt + ct) * TB
                    s0, s1, d0, d1 = src_il + ct * 32 * TB, src_il + ct * 32 * TB + FB, dst, dst + FB
                    for cl in range(32):
                        if cl == 16:
                            d0, d1 = dst + 2 * FB, dst + 3 * FB
                        gath[d0:d0 + FR] = scratch[s0:s0 + FR]
                        gath[d1:d1 + FR] = scratch[s1:s1 + FR]
                        s0, s1, d0, d1 = s0 + TB, s1 + TB, d0 + FR, d1 + FR
            for mi in range(Mb):
                m = (it * 32 + il0 + mi) * St + jt
                i, jb = m // St, m % St
                for ct in range(Kt):
                    tile = _from_faces(gath[(mi * Kt + ct) * TB:(mi * Kt + ct + 1) * TB]).t()   # transpose_wh
                    want = x[0, i, jb * 32:(jb + 1) * 32, ct * 32:(ct + 1) * 32]
                    assert torch.equal(tile, want), (u, il0, mi, ct)
                    seen.add((m, ct))
    assert seen == {(m, ct) for m in range(S * S // 32) for ct in range(Kt)}
