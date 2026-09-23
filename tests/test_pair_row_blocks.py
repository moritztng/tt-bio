"""`pair_row_blocks` is the single pass, bit for bit, for a per-position pair op.

OpenFold3's diffusion runs three such ops on the whole pair (the conditioning branch, the noisy
position embedder's projection and every DiT block's pair bias), and at 1536 tokens each one's
fp32 layer norm is 1207959552 B that a Wormhole chip holding the trunk outputs refused. After a
refusal they run through this helper, so what it returns has to be exactly what the single
pass would have: same op, same rows, a join that moves bytes and nothing else.
"""
import pytest
import torch
import ttnn

from tt_bio import tenstorrent as T

pytestmark = pytest.mark.device


@pytest.fixture(scope="module")
def dev():
    return T.get_device()


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("rows", [32, 64])
def test_layer_norm_linear_in_row_blocks_is_the_single_pass(dev, dtype, rows):
    torch.manual_seed(0)
    n, c, out = 96, 128, 16                     # 96 = 3 blocks of 32, or one of 64 plus a tail
    ft = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dtype)
    z = ft(torch.randn(1, n, n, c))
    ln_w = ft(torch.randn(c) * 0.5 + 1.0)
    w = ft(torch.randn(c, out) / c ** 0.5)
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)

    def fn(x):
        xn = ttnn.layer_norm(x, weight=ln_w, epsilon=1e-5, compute_kernel_config=ckc)
        y = ttnn.linear(xn, w, compute_kernel_config=ckc)
        ttnn.deallocate(xn)
        return y

    single = ttnn.to_torch(fn(z))
    blocked = T.pair_row_blocks(fn, (z,), rows)
    assert tuple(blocked.shape) == (1, n, n, out)
    assert torch.equal(ttnn.to_torch(blocked), single)


def test_every_tensor_is_sliced_on_the_same_rows(dev):
    """The conditioning branch passes three pair-shaped inputs; each block must see the same
    rows of all of them."""
    a = torch.randn(1, 64, 64, 32)
    b = torch.randn(1, 64, 64, 32)
    ft = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)
    ta, tb = ft(a), ft(b)
    out = T.pair_row_blocks(lambda x, y: ttnn.add(x, y), (ta, tb), 32)
    assert torch.equal(ttnn.to_torch(out), ttnn.to_torch(ttnn.add(ta, tb)))


@pytest.mark.parametrize("rows", [32, 64])
def test_a_residual_in_row_blocks_is_the_single_pass_and_frees_its_input(dev, rows):
    """The protenix template embedder's `z + linear_u(relu(u))`: after a refusal the old `z` is
    consumed at the join, so the joined result can take the room it leaves."""
    torch.manual_seed(1)
    n, c_t, c_z = 96, 64, 128
    ft = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    zh, uh = torch.randn(1, n, n, c_z), torch.randn(1, n, n, c_t)
    w = ft(torch.randn(c_t, c_z) / c_t ** 0.5)
    res = lambda zr, ur: ttnn.add(zr, ttnn.linear(ttnn.relu(ur), w))
    single = ttnn.to_torch(res(ft(zh), ft(uh)))
    z = ft(zh)
    out = T.pair_row_blocks(res, (z, ft(uh)), rows, consume=z)
    assert not z.is_allocated()
    assert torch.equal(ttnn.to_torch(out), single)
