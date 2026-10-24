"""`_core_grid_1d_plan`: the `core_grid=` leg of `_pair_proj_linear` re-blocks a plan ttnn under-prices.

The measured case is RFD4's trimul gate projection at 192 tokens, 32 samples wide, on a p150a:
(32,192,192,384) bfp8 @ (384,384) bf16 with fp32 accumulation. ttnn picks drain block (71, 3) and
the program build refuses it: 1586560 B of CB region against 1572864 B, i.e. 1475456 B of circular
buffers above the 111104 B L1 base on a 1461760 B bank (perf/rfd4/l1decline/evidence).
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ttnn = pytest.importorskip("ttnn")

import tt_bio.tenstorrent as T  # noqa: E402

BANK = 1461760  # Blackhole, what the L1 allocator reports per bank


def _t(shape, dtype):
    mc = SimpleNamespace(is_sharded=lambda: False)
    return SimpleNamespace(padded_shape=tuple(shape), dtype=dtype, memory_config=lambda: mc)


def _ckc(fp32=True, l1acc=True):
    return SimpleNamespace(fp32_dest_acc_en=fp32, packer_l1_acc=l1acc)


@pytest.fixture(autouse=True)
def p150a(monkeypatch):
    monkeypatch.setattr(T, "COMPUTE_GRID_MAIN", (13, 10))
    monkeypatch.setattr(T, "_matmul_cb_budget", lambda: BANK)


def _true_bytes(cfg, t0=1088, t1=2048, out=2048, interm=4096):
    h, w, bw = cfg.out_block_h, cfg.out_block_w, cfg.in0_block_w
    return 2 * h * bw * t0 + 2 * w * bw * t1 + h * w * (out + interm)


def test_ttnn_choice_reproduces_the_measured_refusal():
    """ttnn's estimate (out priced at the bfp8 input tile) picks (71, 3), which really costs 1475456 B."""
    def ttnn_price(h, w):
        return 2 * h * 1088 + 2 * w * 2048 + h * w * (1088 + 4096)
    assert T._ttnn_1d_drain(284, 12, ttnn_price, BANK) == (71, 3)
    assert T._ttnn_1d_drain(142, 12, ttnn_price, BANK) == (71, 3)
    assert 2 * 71 * 1088 + 2 * 3 * 2048 + 213 * (2048 + 4096) + 111104 == 1586560


@pytest.mark.parametrize("width", [16, 32])
def test_bfp8_input_gets_a_plan_that_fits(width):
    x = _t((width, 192, 192, 384), ttnn.bfloat8_b)
    w = _t((384, 384), ttnn.bfloat16)
    cfg = T._core_grid_1d_plan(x, w, _ckc(), ttnn.bfloat16)
    assert cfg is not None
    assert cfg.in0_block_w == 1                      # ttnn's K order, so ttnn's arithmetic
    assert cfg.per_core_M == -(-width * 192 * 6 // 130) and cfg.per_core_N == 12
    assert cfg.per_core_M % cfg.out_block_h == 0 and 12 % cfg.out_block_w == 0
    assert _true_bytes(cfg) < BANK


def test_a_plan_ttnn_prices_right_is_left_alone():
    """bf16 in, bf16 out: ttnn's estimate is exact, so the call keeps `core_grid=`."""
    x = _t((32, 192, 192, 384), ttnn.bfloat16)
    w = _t((384, 384), ttnn.bfloat16)
    assert T._core_grid_1d_plan(x, w, _ckc(), ttnn.bfloat16) is None


def test_without_fp32_accumulation_the_measured_call_fits():
    """Measured: the same call with fp32_dest_acc_en=False builds; the gate must agree."""
    x = _t((32, 192, 192, 384), ttnn.bfloat8_b)
    w = _t((384, 384), ttnn.bfloat16)
    assert T._core_grid_1d_plan(x, w, _ckc(fp32=False), ttnn.bfloat16) is None


def test_not_a_tall_1d_plan_is_left_alone():
    x = _t((1, 32, 64, 384), ttnn.bfloat8_b)       # 2048 rows over 384 columns: ratio 5, ttnn goes 2D
    w = _t((384, 384), ttnn.bfloat16)
    assert T._core_grid_1d_plan(x, w, _ckc(), ttnn.bfloat16) is None
