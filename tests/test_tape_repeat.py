"""The tape's repeat backward, against float64 torch.

The outer product mean's small-depth path (`tenstorrent.py`, `_small_depth`) repeats the b
projection over the I batch, and the float64 stack grade differentiates through it. A repeat
of a length-1 axis is a broadcast, so its backward is a sum over that axis; the second case
repeats two axes at once, where a backward that summed only the first would come back the
wrong shape.
"""
import pytest
import torch
import ttnn

pytestmark = [pytest.mark.device]

tt = pytest.importorskip("tt_bio.taped_ttnn").taped_ttnn()
BAR = 1.0e-2  # bf16 cotangent summed over at most 64 copies in fp32 tiles


@pytest.mark.parametrize("shape,reps", [((1, 64, 32), (64, 1, 1)), ((1, 1, 32, 32), (3, 2, 1, 1))])
def test_repeat_backward_sums_over_the_repeated_axes(shape, reps):
    from tt_bio import autograd as ag
    from tt_bio.tenstorrent import get_device

    dev = get_device()
    torch.manual_seed(0)
    out_shape = [n * r for n, r in zip(shape, reps)]
    cot = torch.randn(out_shape)
    ref = cot.double()
    for ax, r in enumerate(reps):
        if r > 1:
            ref = ref.sum(dim=ax, keepdim=True)

    x = ag.Tensor(ttnn.from_torch(torch.randn(shape), layout=ttnn.TILE_LAYOUT, device=dev,
                                  dtype=ttnn.bfloat16), requires_grad=True)
    g = ttnn.from_torch(cot, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    with ag.tape():
        out = tt.repeat(x, reps)
        ag.backward([out], [g])

    got = torch.Tensor(ttnn.to_torch(x.grad)).double().reshape(shape)
    rel = float((got - ref).norm() / ref.norm())
    assert rel < BAR, f"repeat backward rel={rel:.3e} over {BAR:.1e} ({shape} x {reps})"
