"""trimul_t1_small: T1 / T2 serve RFD4's 128-token trimul (card-free: routing only)."""
import pytest

ttnn = pytest.importorskip("ttnn")

from tt_bio import tenstorrent as tt  # noqa: E402
from tt_bio import trimul_tail  # noqa: E402


def test_floor_moves_only_under_the_lever():
    with tt.levers("trimul_t1"):
        assert trimul_tail.min_n() == trimul_tail.MIN_N == 160
    with tt.levers("trimul_t1,trimul_t1_small"):
        assert trimul_tail.min_n() == trimul_tail.SMALL_MIN_N == 128


@pytest.mark.parametrize("n", [128, 192])
def test_small_lengths_take_the_dram_route(n):
    with tt.levers("trimul_t1"):
        assert tt._triangle_mul_memory_config(n).buffer_type == ttnn.BufferType.L1
    with tt.levers("trimul_t1,trimul_t1_small"):
        assert tt._triangle_mul_memory_config(n).buffer_type == ttnn.BufferType.DRAM
    with tt.levers("trimul_t1,trimul_t1_small"):
        assert tt._triangle_mul_memory_config(96).buffer_type == ttnn.BufferType.L1


def test_not_in_the_shared_fast_set():
    assert "trimul_t1_small" in tt.LEVERS
    assert "trimul_t1_small" not in tt.FAST_LEVERS
