"""`autograd.bmm_program_config` prices its K block against the core's L1 before it plans.

BindCraft 2 crashed at the 768 token axis on Blackhole because the plan took in0_block_w = 8 of
Kt = 24 for the dA product of its attention VJP and the circular buffers came to 1753088 B on a
1461760 B core. The pricing below is fitted to the device's own refusals
(`perf/bcw_bmm/out/cb_probe2_bh.json`), and the hardware half checks the narrowed plan against
float64 and that the axes it does not touch keep their plan.

Run: TT_VISIBLE_DEVICES=<card> python3 -m pytest tests/test_bmm_program_config.py
"""
import pytest
import torch
import ttnn

from tt_bio import autograd as ag

B16, F32 = 2048, 4096
#: CB base on a p300c: refusals quote a 1572864 B ceiling and the bank above the base is 1461760.
CB_BASE = 1572864 - 1461760

# (Mt, Nt, w, transpose_a, a, b, out tile bytes, fp32 partials, refused CB end)
REFUSALS = [
    (24, 1, 8, True, B16, B16, B16, True, 1864192),    # the crash, as the model issues it
    (24, 1, 12, True, B16, B16, B16, True, 2667008),
    (24, 2, 12, True, B16, B16, B16, True, 2863616),
    (24, 1, 12, True, B16, B16, F32, False, 2617856),  # fp32 out carries its own partials
    (24, 1, 12, True, F32, F32, F32, False, 5026304),
    (24, 1, 24, False, B16, B16, B16, True, 2716160),
    (24, 2, 24, False, B16, B16, B16, True, 2961920),
    (24, 1, 24, False, F32, B16, F32, False, 5026304),
    (24, 1, 24, False, B16, F32, B16, True, 2814464),
    (24, 1, 8, True, B16, B16, B16, False, 1765888),   # no compute config: no partials
    (32, 1, 8, True, B16, B16, B16, False, 2306560),
    (16, 1, 16, True, B16, B16, B16, False, 2306560),
]


@pytest.mark.parametrize("Mt,Nt,w,ta,a,b,o,p,end", REFUSALS)
def test_cb_bytes_reproduce_the_devices_refusals(Mt, Nt, w, ta, a, b, o, p, end):
    assert CB_BASE + ag.bmm_cb_bytes(Mt, Nt, w, ta, a, b, o, p) == end


def test_the_768_plan_does_not_fit_and_six_does():
    bank = 1461760
    assert ag.bmm_cb_bytes(24, 1, 8, True, B16, B16, B16, True) > bank
    assert ag.bmm_cb_bytes(24, 1, 6, True, B16, B16, B16, True) <= bank


def _dev_tensor(t, dev, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev)


@pytest.mark.device
@pytest.mark.parametrize("n,w", [(768, 6), (1024, 4)])
def test_narrowed_plan_runs_and_matches_float64(n, w):
    """The dA product at an axis whose first-choice block refuses: it runs, as close to float64
    as ttnn's own plan."""
    from tt_bio import tenstorrent as T
    dev = T.get_device()
    ag.BMM_NARROWED.clear()
    torch.manual_seed(0)
    x, g = torch.randn(2, 8, n, n), torch.randn(2, 8, n, 32)
    a, b = _dev_tensor(x, dev), _dev_tensor(g, dev)
    cfg = ag.precise_config()
    pc = ag.bmm_program_config(a, b, True, False, compute_kernel_config=cfg)
    assert pc is not None and pc.in0_block_w == w and pc.per_core_M == n // 32
    assert ag.BMM_NARROWED == {(n // 32, 1, n // 32, True): (8, w)}
    ref = ttnn.to_torch(a).double().transpose(-1, -2) @ ttnn.to_torch(b).double()
    got = ttnn.to_torch(ag.bmm(a, b, True, False, compute_kernel_config=cfg)).double()
    own = ttnn.to_torch(ag._matmul(a, b, transpose_a=True, compute_kernel_config=cfg)).double()
    rel = lambda y: float((y - ref).norm() / ref.norm())
    assert torch.isfinite(got).all()
    assert rel(got) < 1e-2 and rel(got) <= 1.5 * rel(own) + 1e-6, (rel(got), rel(own))


@pytest.mark.device
@pytest.mark.parametrize("n", [288, 544, 800, 864])
def test_axes_that_ran_keep_their_plan(n):
    from tt_bio import tenstorrent as T
    dev = T.get_device()
    ag.BMM_NARROWED.clear()
    a = _dev_tensor(torch.randn(2, 8, n, n), dev)
    b = _dev_tensor(torch.randn(2, 8, n, 32), dev)
    pc = ag.bmm_program_config(a, b, True, False, compute_kernel_config=ag.precise_config())
    Kt = n // 32
    assert pc.in0_block_w == max(d for d in range(1, 9) if Kt % d == 0)
    assert ag.BMM_NARROWED == {}
