"""`AtomTransformer._windows_tiled` / `_merge_heads` against the ROW_MAJOR window path they replace.

Both are data movement only, so every output must be torch.equal to the old path, in fp32 (the
normal-mode diffusion) and bf16 (OpenDDE, the LPX diffusion), at one sample and at five, at a
size whose atom count is not a multiple of 32 and at the 730-token complex's 5,919 atoms.
Also times both paths per call.

Run: TT_VISIBLE_DEVICES=<card> python3 -m pytest -s tests/test_atom_windows_tiled_hw.py
"""
import time

import pytest
import torch
import ttnn

from tt_bio import protenix as PX
from tt_bio.main import ensure_p300_mesh_descriptor

pytestmark = pytest.mark.device


def _at(dev):
    at = PX.AtomTransformer.__new__(PX.AtomTransformer)
    at.device, at._kv_widx = dev, {}
    return at


def _old(at, Q, K, V, M, N, NP):
    if M == 1:
        q, k, v = at._windows_q(Q, N, NP), at._windows_kv(K, N, NP), at._windows_kv(V, N, NP)
    else:
        q, k, v = at._windows_q_m(Q, M, N, NP), at._windows_kv_m(K, M, N, NP), at._windows_kv_m(V, M, N, NP)
    return q, ttnn.permute(k, (0, 1, 3, 2)), v


def _old_merge(o, M, N, NP, C):
    o = ttnn.permute(o, (0, 2, 1, 3))
    if M == 0:
        o = ttnn.slice(ttnn.to_layout(ttnn.reshape(o, (NP, C)), ttnn.ROW_MAJOR_LAYOUT), [0, 0], [N, C])
    else:
        o = ttnn.slice(ttnn.to_layout(ttnn.reshape(o, (M, NP, C)), ttnn.ROW_MAJOR_LAYOUT), [0, 0, 0], [M, N, C])
    return ttnn.to_layout(o, ttnn.TILE_LAYOUT)


def _timed(fn, reps=5):
    fn()
    ttnn.synchronize_device(_timed.dev)
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    ttnn.synchronize_device(_timed.dev)
    return (time.perf_counter() - t) / reps * 1e3


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("M,N", [(1, 300), (5, 300), (5, 5919)])
def test_windows_tiled_equal_and_faster(dtype, M, N):
    ensure_p300_mesh_descriptor()
    from tt_bio.tenstorrent import get_device
    dev = get_device()
    _timed.dev = dev
    at = _at(dev)
    H, dh, nq = at.N_HEADS, at.HEAD_DIM, at.N_QUERIES
    C, NP = H * dh, -(-N // nq) * nq
    g = torch.Generator().manual_seed(N + M)
    up = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dtype)
    Q, K, V = (up(torch.randn(M, N, C, generator=g)) for _ in range(3))
    host = lambda t: ttnn.to_torch(t).float()
    old = _old(at, Q, K, V, M, N, NP)
    new = at._windows_tiled(Q, K, V, M, N, NP)
    for name, a, b in zip(("q", "kT", "v"), old, new):
        assert tuple(a.shape) == tuple(b.shape), (name, a.shape, b.shape)
        assert torch.equal(host(a), host(b)), name
    o = up(torch.randn(M * NP // nq, H, nq, dh, generator=g))
    Mo = 0 if M == 1 else M
    a, b = _old_merge(o, Mo, N, NP, C), at._merge_heads(o, Mo, N, NP)
    assert tuple(a.shape) == tuple(b.shape)
    assert torch.equal(host(a), host(b))
    t_old = _timed(lambda: (_old(at, Q, K, V, M, N, NP), _old_merge(o, Mo, N, NP, C)))
    t_new = _timed(lambda: (at._windows_tiled(Q, K, V, M, N, NP), at._merge_heads(o, Mo, N, NP)))
    print(f"\n[windows] {dtype} M={M} N={N}: old {t_old:.3f} ms, tiled {t_new:.3f} ms, x{t_old / t_new:.2f}")
