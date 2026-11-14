"""The ending triangle attention's output transpose has to see an L1-resident input beside it.

RF3 on a Blackhole p300c (11x10) died at 778 tokens in its template pairformer (c=64), every seed,
both modes: the output projection left the untransposed update in L1 (`l1_out`), the ending
transpose put its result in L1 as well, underneath it, because its gate prices the whole grid, and
the caller's `add_` threw "L1 buffer allocated at 122880 and static circular buffer region ends at
123392" (9W8A, 649 + 129 aa). The diag on the fold showed the residual in DRAM and the result as
the only L1 buffer left, 12 KB above the CB base. At 768 tokens the two tensors still leave the add
its room, so it folds. The residual is priced the same way when it is the resident one.
"""
from types import SimpleNamespace

import pytest

pytest.importorskip("ttnn", reason="tenstorrent.py imports ttnn at module scope")

import ttnn  # noqa: E402

from tt_bio import tenstorrent as tt  # noqa: E402

BH_L1_PER_CORE = 1532448         # p300c, the throw's CB base 40416 under 1572864
ADD_CB_END = 123392              # the add_'s static CB region in the throw


class _T:
    def __init__(self, s, c, l1):
        self.shape = self.padded_shape = (1, s, s, c)
        self.dtype = ttnn.bfloat16
        self._l1 = l1

    def is_allocated(self):
        return True

    def memory_config(self):
        return ttnn.L1_MEMORY_CONFIG if self._l1 else ttnn.DRAM_MEMORY_CONFIG


@pytest.fixture
def p300c(monkeypatch):
    monkeypatch.setattr(ttnn, "get_max_worker_l1_unreserved_size", lambda: BH_L1_PER_CORE)
    monkeypatch.setattr(tt, "COMPUTE_GRID_MAIN", (11, 10))


def route(s, c, resident, add_to_input=True, which="input"):
    att = SimpleNamespace(transpose_l1_reserve=0)
    x, x_in = _T(s, c, resident and which == "input"), _T(s, c, resident and which == "residual")
    mc = tt.TriangleAttention._out_transpose_memory_config(att, x, x_in, add_to_input)
    return mc.buffer_type


def test_the_crash_shape_goes_to_dram(p300c):
    # Premise: the whole-grid gate alone admits the update at 778.
    assert tt._transpose_memory_config(_T(778, 64, False)).buffer_type == ttnn.BufferType.L1
    for which in ("input", "residual"):
        assert route(778, 64, resident=True, which=which) == ttnn.BufferType.DRAM


def test_768_keeps_its_l1_route_and_fits(p300c):
    assert route(768, 64, resident=True) == ttnn.BufferType.L1
    per_core = tt._chunk_l1_per_core(_T(768, 64, True))
    assert 1572864 - 2 * per_core >= ADD_CB_END


def test_a_dram_input_keeps_the_old_gate(p300c):
    for s in (512, 768, 778):
        old = tt._transpose_memory_config(_T(s, 64, False)).buffer_type
        assert route(s, 64, resident=False) == old
        assert route(s, 64, resident=True, add_to_input=False, which="residual") == old


def test_both_resident_counts_both(p300c):
    att = SimpleNamespace(transpose_l1_reserve=0)
    mc = tt.TriangleAttention._out_transpose_memory_config(
        att, _T(512, 64, True), _T(512, 64, True), True)
    assert mc.buffer_type == ttnn.BufferType.L1          # 3 x 0.30 MB fits
    mc = tt.TriangleAttention._out_transpose_memory_config(
        att, _T(640, 64, True), _T(640, 64, True), True)
    assert mc.buffer_type == ttnn.BufferType.DRAM        # 3 x 0.48 MB does not


def test_monotone_in_tokens(p300c):
    for which in ("input", "residual"):
        seen_dram = False
        for s in range(32, 1025, 2):
            if route(s, 64, resident=True, which=which) == ttnn.BufferType.DRAM:
                seen_dram = True
            else:
                assert not seen_dram, (which, s)
