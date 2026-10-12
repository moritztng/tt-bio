"""The ending-node tail's address algebra (kernels/triatt_tail/reader_tr.cpp, writer_tr.cpp), device-free.

Emulates both kernels' page arithmetic and face-row shuffle over tiles stored in face order, runs the
unit math (z + (o * sigmoid(g)) @ Wo per output row tile) in float64, and checks the result against
z[b, a] + u[a, b] computed directly; also that every z tile is written exactly once.
"""
import numpy as np
import pytest

T = 32
ROW = 16  # elements in a face row


def to_faces(t):
    """32x32 tile -> 1024 elements in face order (four 16x16 faces, row-major in each)."""
    return np.concatenate([t[:16, :16].ravel(), t[:16, 16:].ravel(), t[16:, :16].ravel(), t[16:, 16:].ravel()])


def from_faces(f):
    t = np.empty((T, T))
    t[:16, :16], t[:16, 16:] = f[:256].reshape(16, 16), f[256:512].reshape(16, 16)
    t[16:, :16], t[16:, 16:] = f[512:768].reshape(16, 16), f[768:].reshape(16, 16)
    return t


def face_row(r):
    """Offset (in elements) of tile row r's first face row; its second half is FACE later."""
    return ((r >> 4) * 2) * 256 + (r & 15) * ROW


def head_major_pages(x):
    """[A, H, B, 32] -> pages in TILE order: page (a*H + h)*Bt + bt is tile rows b of (a, h)."""
    A, H, B, _ = x.shape
    return [to_faces(x[a, h, bt * T:(bt + 1) * T]) for a in range(A) for h in range(H) for bt in range(B // T)]


def pair_pages(z):
    """[S1, S2, N] -> pages ((i * S2t + jt) * Nt + n)."""
    S1, S2, N = z.shape
    return [to_faces(z[i, jt * T:(jt + 1) * T, n * T:(n + 1) * T])
            for i in range(S1) for jt in range(S2 // T) for n in range(N // T)]


@pytest.mark.parametrize("S,H,N", [(64, 4, 128), (96, 2, 64), (128, 4, 128)])
def test_reader_tr_writer_tr_put_the_update_into_z(S, H, N):
    rng = np.random.default_rng(S + H)
    o, g = rng.standard_normal((2, S, H, S, T))
    w = rng.standard_normal((H * T, N)) / 8
    z = rng.standard_normal((S, S, N))
    Kt, Nt, St, At_n = H, N // T, S // T, S // T
    op, gp, zp = head_major_pages(o), head_major_pages(g), pair_pages(z)
    written = np.zeros(len(zp), int)
    for blk in range(St * At_n):                       # every core's blocks, in any order
        bt, At = blk // At_n, blk % At_n
        raw_o = [None] * (32 * Kt)
        raw_g = [None] * (32 * Kt)
        for r in range(32):
            for h in range(Kt):
                page = ((At * 32 + r) * Kt + h) * St + bt
                raw_o[r * Kt + h], raw_g[r * Kt + h] = op[page], gp[page]
        for q in range(32):
            ot, gt = [np.zeros(1024) for _ in range(Kt)], [np.zeros(1024) for _ in range(Kt)]
            for h in range(Kt):
                for r in range(32):
                    s, d = face_row(q), face_row(r)
                    for half in (0, 256):
                        ot[h][d + half:d + half + ROW] = raw_o[r * Kt + h][s + half:s + half + ROW]
                        gt[h][d + half:d + half + ROW] = raw_g[r * Kt + h][s + half:s + half + ROW]
            u = (bt * 32 + q) * At_n + At
            x = np.concatenate([from_faces(ot[h]) * (1 / (1 + np.exp(-from_faces(gt[h])))) for h in range(Kt)], 1)
            out = x @ w                                  # rows: the 32 a of At
            for n in range(Nt):
                zp[u * Nt + n] = to_faces(from_faces(zp[u * Nt + n]) + out[:, n * T:(n + 1) * T])
                written[u * Nt + n] += 1
    assert (written == 1).all()
    got = np.empty_like(z)
    for i in range(S):
        for jt in range(St):
            for n in range(Nt):
                got[i, jt * T:(jt + 1) * T, n * T:(n + 1) * T] = from_faces(zp[(i * St + jt) * Nt + n])
    upd = np.einsum("ahbd,hdn->abn", o * (1 / (1 + np.exp(-g))), w.reshape(H, T, N))
    np.testing.assert_allclose(got, z + upd.transpose(1, 0, 2), rtol=1e-12, atol=1e-12)
