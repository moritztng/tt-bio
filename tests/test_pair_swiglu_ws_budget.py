"""ws_l1acc / ws_resid L1 accounting: the hidden split `weights()` picks for bf16 weights must still fit once the
fp32 accumulator CB joins, or the lever would decline silently on exactly the shapes it exists for
(Protenix c_z 256 and OpenDDE c_z 384 in normal mode). Host only: fake tensors, no device."""
from types import SimpleNamespace

import pytest

ttnn = pytest.importorskip("ttnn")
from tt_bio import pair_swiglu_ws as W  # noqa: E402


def _t(shape, dtype=ttnn.bfloat16, buffer_type=None):
    mc = SimpleNamespace(memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
                         buffer_type=ttnn.BufferType.DRAM if buffer_type is None else buffer_type)
    return SimpleNamespace(shape=shape, padded_shape=shape, dtype=dtype, layout=ttnn.TILE_LAYOUT,
                           memory_config=lambda: mc)


def _parts(c, hidden, dtype):
    """The chunk count `weights()` chooses, from its own rule."""
    kt, nh, ct = c // 32, hidden // 32, c // 32
    return next((p for p in range(1, nh // W._SW + 1) if nh % (p * W._SW) == 0
                 and W._l1_bytes(kt, nh // p, ct, dtype) <= W._L1_BUDGET), 1)


@pytest.mark.parametrize("c,hidden", [(128, 512), (256, 1024), (384, 1536)])
def test_l1acc_fits_the_split_weights_picks(c, hidden):
    p = _parts(c, hidden, ttnn.bfloat16)
    h = hidden // p
    x, w12, w3 = _t([1, 64, 512, c]), _t([c, 2 * h]), _t([h, c])
    assert W._l1_bytes(c // 32, h // 32, c // 32, ttnn.bfloat16) <= W._L1_BUDGET
    assert W.eligible(x, w12, w3)
    assert W.eligible(x, w12, w3, l1acc=True)


def test_l1acc_charges_its_accumulator():
    # A shape that fits with less than one accumulator's worth of L1 to spare must decline under l1acc.
    acc = W._SW * W._tile_bytes(ttnn.float32)
    budget = W._L1_BUDGET
    try:
        W._L1_BUDGET = W._l1_bytes(4, 16, 4, ttnn.bfloat16) + acc - 1
        x, w12, w3 = _t([1, 1, 512, 128]), _t([128, 1024]), _t([512, 128])
        assert W.eligible(x, w12, w3)
        assert not W.eligible(x, w12, w3, l1acc=True)
    finally:
        W._L1_BUDGET = budget


@pytest.mark.parametrize("c,hidden", [(128, 512), (256, 1024), (384, 1536)])
def test_resid_fits_the_split_weights_picks(c, hidden):
    # ws_resid adds two row tiles of the residual; it must not push the chunk weights() chose out of L1.
    for dtype in (ttnn.bfloat16, ttnn.bfloat8_b):
        h = hidden // _parts(c, hidden, dtype)
        x, w12, w3 = _t([1, 64, 512, c]), _t([c, 2 * h], dtype), _t([h, c], dtype)
        r = _t([1, 64, 512, c])
        assert W.eligible(x, w12, w3, resid=r, l1acc=dtype == ttnn.bfloat16)


def test_resid_must_match_the_output():
    x, w12, w3 = _t([1, 64, 512, 128]), _t([128, 1024]), _t([512, 128])
    assert W.eligible(x, w12, w3, resid=_t([1, 64, 512, 128]))
    assert not W.eligible(x, w12, w3, resid=_t([1, 64, 512, 256]))                        # wrong width
    assert not W.eligible(x, w12, w3, resid=_t([1, 64, 512, 128], ttnn.bfloat8_b))        # not bf16
    assert not W.eligible(x, w12, w3, resid=_t([1, 64, 512, 128], buffer_type=ttnn.BufferType.L1))


@pytest.mark.parametrize("wormhole", [True, False])
def test_ln_resid_budget_is_blackhole_only(monkeypatch, wormhole):
    # ws_ln_resid: ESMFold2's c_z 256 bfp8 transition with the LN prologue and x as its residual takes 1230848 B.
    # Blackhole admits it under T1's rule (0.85 of a p150a bank, 1461760 B); Wormhole keeps 1100 KiB and declines.
    from tt_bio import tenstorrent as T
    monkeypatch.setattr(T, "is_wormhole", lambda: wormhole)
    monkeypatch.setattr(T, "_l1_bank_bytes", lambda: 1461760)
    b8 = ttnn.bfloat8_b
    x, w12, w3 = _t([1, 512, 512, 256]), _t([256, 2048], b8), _t([1024, 256], b8)
    g = _t([1, 256])
    assert W.eligible(x, w12, w3, ln=(g, g), resid=x) is not wormhole
    assert W.eligible(x, w12, w3, ln=(g, g), resid=x, l1acc=True) is not wormhole
    # Plans without the prologue keep the shared budget on both arches.
    assert W.eligible(x, w12, w3, resid=x)
    assert W._ln_budget() == (W._L1_BUDGET if wormhole else int(0.85 * 1461760))
