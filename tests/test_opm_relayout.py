"""opm_relayout's page and face arithmetic, no device: emulate the reader's copies tile by tile
(faces of 16 x 16, 32-element rows as two 16-element face rows) and check the result is
y[i, j, 32 c + d] = z[32 i + c, 32 j + d] exactly. The kernel strings are tied to the emulation below.
Device check: perf/spd/opm_relayout_probe.py.

Run: TT_VISIBLE_DEVICES= python3 -m pytest tests/test_opm_relayout.py
"""
from pathlib import Path

import pytest
import torch

KDIR = Path(__file__).resolve().parents[1] / "tt_bio" / "kernels" / "opm_relayout"
ROW, FACE, TILE = 16, 256, 1024     # elements


def pages(m):
    out = []
    for r in range(m.shape[0] // 32):
        for c in range(m.shape[1] // 32):
            t = m[32 * r:32 * r + 32, 32 * c:32 * c + 32]
            out.append(torch.cat([t[:16, :16].flatten(), t[:16, 16:].flatten(),
                                  t[16:, :16].flatten(), t[16:, 16:].flatten()]))
    return out


@pytest.mark.parametrize("I,J", [(32, 32), (64, 96), (96, 64)])
def test_relayout_emulation(I, J):
    blk = torch.arange(I * J * 32 * 32, dtype=torch.float64).reshape(I, J, 32, 32)
    src = pages(blk.permute(0, 2, 1, 3).reshape(I * 32, J * 32))     # z = [(i c), (j d)]
    want = pages(blk.reshape(I * J, 1024))                          # y = [(i j), (c d)]
    for g in range(I * J // 32):
        scratch = torch.cat(src[32 * g:32 * g + 32])
        for c in range(32):
            o = torch.full((TILE,), float("nan"), dtype=torch.float64)
            s0 = (c // 16) * 2 * FACE + (c % 16) * ROW
            s1 = s0 + FACE
            d0, d1 = 0, FACE
            for jl in range(32):
                if jl == 16:
                    d0, d1 = 2 * FACE, 3 * FACE
                o[d0:d0 + ROW] = scratch[s0:s0 + ROW]
                o[d1:d1 + ROW] = scratch[s1:s1 + ROW]
                s0, s1, d0, d1 = s0 + TILE, s1 + TILE, d0 + ROW, d1 + ROW
            assert torch.equal(o, want[32 * g + c]), (g, c)


def test_kernel_matches_emulation():
    r = (KDIR / "reader.cpp").read_text()
    assert "uint32_t s0 = base + (c / FACE_H) * 2 * FACE_BYTES + (c % FACE_H) * ROW_BYTES;" in r
    assert "uint32_t page = g * TILES;" in r and "noc_async_read_page(page + jl, s, dst);" in r
    assert "d0 = t + 2 * FACE_BYTES;" in r and "d1 = t + 3 * FACE_BYTES;" in r
    w = (KDIR / "writer.cpp").read_text()
    assert "s.get_noc_addr(g * TILES + c)" in w
