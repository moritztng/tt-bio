"""Drive the head-major call sites with the flags ON, without a device.

Every guard in this row has been read; none of its SERVING path has ever executed. The card window
it is waiting for is the scarce resource, and a `TypeError` on the first call would burn it. So run
the paths for real -- real `ttnn` constants, the real `_common_ok`, the real `_mm_block_for`, the
real key guard -- and patch only the three calls that need hardware: `allocate_tensor_on_device`,
`reshape`, and `mm_generic.generic_minimal_matmul`, whose arguments are recorded instead.

What that pins is exactly what the lever's correctness rests on and what no shape check can see:
which define each destination is written with, and how many destinations there are. It is also the
check that would have caught the HEAD_MAJOR_MT-on-a-single-output bug found by eye in pass 3 -- q is
one output, so its writer reads HEAD_MAJOR_OUT_MT and never looks at HEAD_MAJOR_MT.

Not covered, by construction: the descriptor builder and the kernels. `build()` is exercised by the
shipped triangle attention on every fold, and its one untested addition, the bias, is pinned against
the installed wheel's own kernel sources by `perf/c12_diffusion_head/argcheck.py`.
"""
import pathlib
import sys

import pytest

WT = str(pathlib.Path(__file__).resolve().parent.parent)
if WT not in sys.path:
    sys.path.insert(0, WT)

TILE = 32


class Tensor:
    """A stand-in carrying only what the guards and the destination arithmetic read."""

    def __init__(self, dims, dtype=None, mc=None, layout=None):
        import ttnn
        self.shape = tuple(int(d) for d in dims)
        self.padded_shape = self.shape
        self.dtype = dtype if dtype is not None else ttnn.bfloat16
        self.layout = layout if layout is not None else ttnn.TILE_LAYOUT
        self._mc = mc if mc is not None else ttnn.DRAM_MEMORY_CONFIG

    def memory_config(self):
        return self._mc

    def device(self):
        return "DEV"

    def __repr__(self):
        return f"T{self.shape}"


class Ckc:
    """`mm_generic.ckc_args` reads these four off a DeviceComputeKernelConfig by getattr."""
    import ttnn as _t
    math_fidelity = _t.MathFidelity.HiFi4
    math_approx_mode = False
    fp32_dest_acc_en = True
    dst_full_sync_en = False


@pytest.fixture
def rig(monkeypatch):
    """Flags on, the three device calls recorded."""
    import ttnn
    from tt_bio import mm_generic as G
    from tt_bio import triatt_qkv as TQ

    calls = []

    def fake_mm(dev, in0, in1, outs, cfg, ckc, defines=(), kernel_dir=None, m_k=None,
                noc_mode=None, n_widths=None, bias=None):
        calls.append({"in0": in0, "in1": in1,
                      "outs": [o.shape for o in (outs if isinstance(outs, list) else [outs])],
                      "n_outs": len(outs) if isinstance(outs, list) else 1,
                      "cfg": cfg, "defines": dict(defines), "m_k": m_k,
                      "n_widths": n_widths, "bias": bias})
        return outs

    monkeypatch.setattr(G, "generic_minimal_matmul", fake_mm)
    monkeypatch.setattr(TQ.G, "generic_minimal_matmul", fake_mm, raising=False)
    monkeypatch.setattr(ttnn, "allocate_tensor_on_device",
                        lambda shape, dtype, layout, dev, mc: Tensor(list(shape)))
    monkeypatch.setattr(ttnn, "reshape", lambda t, dims, *a, **k: Tensor(dims))
    monkeypatch.setattr(TQ, "_APB_ENABLED", True)
    monkeypatch.setattr(TQ, "_ATOM_ENABLED", True)
    TQ.APB_REJECTS.clear()
    TQ.ATOM_REJECTS.clear()
    return TQ, calls


def _cfg():
    """Any non-None value: the descriptor is built from the block entry, not from this."""
    return object()


@pytest.mark.parametrize("c_s,n_heads,phd,mt,dt", [
    (768, 16, 64, 16, 2),        # the diffusion token transformer, 4800 programs per fold
    (384, 16, 32, 16, 1),        # the trunk attention, 264 programs per fold
])
def test_fused_qkv_writes_three_head_major_destinations(rig, c_s, n_heads, phd, mt, dt):
    TQ, calls = rig
    s = Tensor((1, 512, c_s))
    w = Tensor((c_s, 3 * n_heads * phd))
    bias = Tensor((3 * n_heads * phd,))
    served = TQ.APB_STATS[0]

    out = TQ.qkv_heads(s, w, Ckc(), n_heads, phd, __import__("ttnn").bfloat16, _cfg(),
                       bias=bias, allow_m_le_n=True, site="apb")

    assert out is not None, f"declined: {TQ.APB_REJECTS}"
    assert len(out) == 3 and all(t.shape == (1, n_heads, 512, phd) for t in out), [t.shape for t in out]
    assert len(calls) == 1, calls
    c = calls[0]
    assert c["in0"] is s and c["in1"] is w and c["bias"] is bias
    assert c["n_outs"] == 3, "q, k and v are three N chunks of one pass"
    assert c["m_k"] is None and c["n_widths"] is None
    want = {"HEAD_MAJOR_MT": mt} | ({"HEAD_MAJOR_DT": dt} if dt != 1 else {})
    assert c["defines"] == want, c["defines"]
    assert TQ.APB_STATS[0] == served + 1


def test_a_multi_tile_head_passes_DT_and_a_single_tile_one_does_not(rig):
    """HEAD_MAJOR_DT must appear exactly when the head is more than one tile.

    Passing it as 1 would be harmless; omitting it when it is 2 would send every tile to the wrong
    address, and the header's default of 1 is what makes the omission silent.
    """
    TQ, calls = rig
    for phd, expect_dt in ((64, True), (32, False)):
        calls.clear()
        s, w = Tensor((1, 512, 768)), Tensor((768, 3 * 16 * phd))
        # (24, 96) is the only one of these two widths with a block entry; give the other one the
        # same c_s so the test is about the define, not about the table.
        if TQ.qkv_heads(s, w, Ckc(), 16, phd, __import__("ttnn").bfloat16, _cfg(),
                        allow_m_le_n=True, site="apb") is None:
            assert {r for r, _ in TQ.APB_REJECTS} == {"no_block_entry"}, TQ.APB_REJECTS
            continue
        assert ("HEAD_MAJOR_DT" in calls[0]["defines"]) is expect_dt, calls[0]["defines"]


def test_atom_writes_q_with_the_single_output_define_and_kv_with_the_split_one(rig):
    TQ, calls = rig
    import ttnn
    B, K, W, ADIM, D_S, H, HD = 1, 140, 32, 128, 128, 4, 32
    s = Tensor((B, K, W, D_S))
    s_kv = Tensor((B, K, ADIM, D_S))
    w_q, b_q = Tensor((D_S, H * HD)), Tensor((H * HD,))
    w_kv = Tensor((D_S, 2 * H * HD))
    served = TQ.ATOM_STATS[0]

    out = TQ.atom_qkv_heads(s, w_q, b_q, s_kv, w_kv, Ckc(), H, HD, ttnn.bfloat16,
                            _cfg(), _cfg())

    assert out is not None, f"declined: {TQ.ATOM_REJECTS}"
    q, k, v = out
    # what `_attention` is handed: q keeps the window's 32 rows, k and v the gathered 128
    assert q.shape == (B, K * H, W, HD), q.shape
    assert k.shape == v.shape == (B, K * H, ADIM, HD), (k.shape, v.shape)

    assert len(calls) == 2, calls
    qc, kvc = calls
    # q: ONE output, so the kernel takes write_block_sync and reads MM_OUT_TILE_ID
    assert qc["in0"] is s and qc["in1"] is w_q and qc["bias"] is b_q
    assert qc["n_outs"] == 1 and qc["outs"] == [(B * K, H, W, HD)], qc["outs"]
    assert qc["defines"] == {"HEAD_MAJOR_OUT_MT": W // TILE}, qc["defines"]
    assert "HEAD_MAJOR_MT" not in qc["defines"], \
        "the single-output writer never reads HEAD_MAJOR_MT; passing it is inert"
    # kv: TWO outputs, so the split writer and MM_SPLIT_TILE_ID
    assert kvc["in0"] is s_kv and kvc["in1"] is w_kv and kvc["bias"] is None
    assert kvc["n_outs"] == 2 and kvc["outs"] == [(B * K, H, ADIM, HD)] * 2, kvc["outs"]
    assert kvc["defines"] == {"HEAD_MAJOR_MT": ADIM // TILE}, kvc["defines"]
    assert TQ.ATOM_STATS[0] == served + 1


def test_atom_q_define_is_one_at_the_window_and_tracks_a_wider_one(rig):
    """`HEAD_MAJOR_OUT_MT` is the window's row tiles, not a constant 1.

    At ATOM_WINDOW = 32 it is 1 and the head-major id equals the plain id, which is why a wrong
    define would still have produced the right answer there. A wider window is what separates
    "correct" from "correct by accident", so drive one.
    """
    TQ, calls = rig
    import ttnn
    B, K, W, ADIM, D_S, H, HD = 1, 35, 128, 128, 128, 4, 32
    out = TQ.atom_qkv_heads(Tensor((B, K, W, D_S)), Tensor((D_S, H * HD)), None,
                            Tensor((B, K, ADIM, D_S)), Tensor((D_S, 2 * H * HD)),
                            Ckc(), H, HD, ttnn.bfloat16, _cfg(), _cfg())
    assert out is not None, f"declined: {TQ.ATOM_REJECTS}"
    assert calls[0]["defines"] == {"HEAD_MAJOR_OUT_MT": W // TILE}, calls[0]["defines"]
    assert calls[0]["bias"] is None, "no q bias was passed, so none may be plumbed"


def test_atom_refuses_a_multi_tile_head_rather_than_passing_a_define_that_is_ignored(rig):
    TQ, _calls = rig
    import ttnn
    B, K, W, ADIM, D_S, H, HD = 1, 140, 32, 128, 256, 4, 64
    out = TQ.atom_qkv_heads(Tensor((B, K, W, D_S)), Tensor((D_S, H * HD)), None,
                            Tensor((B, K, ADIM, D_S)), Tensor((D_S, 2 * H * HD)),
                            Ckc(), H, HD, ttnn.bfloat16, _cfg(), _cfg())
    assert out is None
    assert {r for r, _ in TQ.ATOM_REJECTS} == {"multi_tile_head_on_single_output"}, TQ.ATOM_REJECTS


def test_kq_norm_declines_at_both_sites_and_says_so(rig):
    TQ, calls = rig
    import ttnn
    assert TQ.qkv_heads(Tensor((1, 512, 768)), Tensor((768, 3072)), Ckc(), 16, 64,
                        ttnn.bfloat16, _cfg(), allow_m_le_n=True, site="apb",
                        refuse="kq_norm") is None
    assert TQ.atom_qkv_heads(Tensor((1, 140, 32, 128)), Tensor((128, 128)), None,
                             Tensor((1, 140, 128, 128)), Tensor((128, 256)), Ckc(), 4, 32,
                             ttnn.bfloat16, _cfg(), _cfg(), refuse="kq_norm") is None
    assert {r for r, _ in TQ.APB_REJECTS} == {"kq_norm"}, TQ.APB_REJECTS
    assert {r for r, _ in TQ.ATOM_REJECTS} == {"kq_norm"}, TQ.ATOM_REJECTS
    assert not calls, "a refused call must not reach the device"


def test_flags_off_serve_nothing_and_record_nothing(monkeypatch):
    """The shipped default. An off lever must not even count a decline."""
    from tt_bio import triatt_qkv as TQ
    import ttnn
    monkeypatch.setattr(TQ, "_APB_ENABLED", False)
    monkeypatch.setattr(TQ, "_ATOM_ENABLED", False)
    TQ.APB_REJECTS.clear()
    TQ.ATOM_REJECTS.clear()
    before = (list(TQ.APB_STATS), list(TQ.ATOM_STATS))
    assert TQ.qkv_heads(Tensor((1, 512, 768)), Tensor((768, 3072)), Ckc(), 16, 64,
                        ttnn.bfloat16, _cfg(), allow_m_le_n=True, site="apb") is None
    assert TQ.atom_qkv_heads(Tensor((1, 140, 32, 128)), Tensor((128, 128)), None,
                             Tensor((1, 140, 128, 128)), Tensor((128, 256)), Ckc(), 4, 32,
                             ttnn.bfloat16, _cfg(), _cfg()) is None
    assert (list(TQ.APB_STATS), list(TQ.ATOM_STATS)) == before
    assert not TQ.APB_REJECTS and not TQ.ATOM_REJECTS
