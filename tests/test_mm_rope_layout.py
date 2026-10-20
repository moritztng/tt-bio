"""mm_rope's per-core column order: every core gets whole q / k heads then its share of v, and the writer's map back
(rope_col in kernels/mm_gate/matmul_dataflow_common.hpp, mirrored here) inverts `permute`. CPU only."""
import pytest
import torch

pytest.importorskip("ttnn")
from tt_bio import mm_rope  # noqa: E402


def _rope_col(n, t, p, v0):
    r, j = divmod(n, t)
    return r * p + j if j < p else v0 + r * (t - p) + j - p


def test_layout_saprot_wh():
    assert mm_rope.layout(120, 8, 8) == (15, 10, 80)
    assert mm_rope.layout(120, 9, 8) is None    # 13.3 tiles a core
    assert mm_rope.layout(120, 8, 7) is None    # odd N block: a head could straddle two blocks
    assert mm_rope.layout(120, 11, 8) is None


def test_permute_roundtrip_and_whole_heads():
    d, ncols, nb = 1280, 8, 8
    w = torch.arange(3 * d, dtype=torch.float32)[None].repeat(2, 1)
    wp = mm_rope.permute(w, ncols, nb)
    t, p, v0 = mm_rope.layout(3 * d // 32, ncols, nb)
    back = torch.empty_like(wp)
    for n in range(3 * d // 32):
        c = _rope_col(n, t, p, v0)
        back[:, c * 32:(c + 1) * 32] = wp[:, n * 32:(n + 1) * 32]
    assert torch.equal(back, w)
    for n in range(0, 3 * d // 32):   # q / k tiles sit in head pairs at even core-local offsets
        r, j = divmod(n, t)
        if j < p and j % 2 == 0:
            c = _rope_col(n, t, p, v0)
            assert c % 2 == 0 and _rope_col(n + 1, t, p, v0) == c + 1 and c < v0
