"""The row-blocked diffusion pair conditioning is the whole-tensor chain.

OpenDDE's structural pair at 1536 residues (2987 tokens) was refused a 2300133376 B fp32 relative
position projection, and its fp32 cast would have been 13.9 GB, both built whole before the row
blocks began. After a refusal the whole chain (cast, compression, concat + LN + projection, both
transitions) now runs per row block and each block goes to the host. Every step is per (i, j)
position, so the blocked result must equal the single pass.
"""
import types

import pytest
import torch
import ttnn

from tt_bio import tenstorrent as T
from tt_bio.protenix import Protenix

pytestmark = pytest.mark.device

C = "diffusion_module.diffusion_conditioning."


def _cond_weights(model):
    from tt_bio import weights

    ckpt = weights.resolve(model)
    if ckpt is None:
        pytest.skip(f"{model} checkpoint not present")
    sd = torch.load(ckpt, map_location="cpu", weights_only=True, mmap=True)
    sd = sd.get("model", sd)
    return {k[len("module."):]: v.float() for k, v in sd.items()
            if k.startswith("module." + C)}


@pytest.mark.parametrize("model", ["protenix-v2", "opendde"])
@pytest.mark.parametrize("fp32", [True, False])
def test_row_blocked_pair_conditioning_is_the_single_pass(model, fp32):
    w = _cond_weights(model)
    dev = T.get_device()
    dtype = ttnn.float32 if fp32 else ttnn.bfloat16
    ck = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    up = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dtype)
    stub = types.SimpleNamespace(
        _w=w, compute_kernel_config=ck, dev=dev, _paircond_rows_refused={},
        diffusion=types.SimpleNamespace(_up=up, dtype=dtype, _diffusion_fp32=fp32))
    # The trunk pair's width: OpenDDE compresses it first, protenix-v2 concatenates it as is.
    c_z = (w[C + "layernorm_z_trunk.weight"].shape[0] if C + "layernorm_z_trunk.weight" in w
           else w[C + "layernorm_z.weight"].shape[0] - w[C + "relpe.linear_no_bias.weight"].shape[0])
    n = 160                                      # rb=64 below: two blocks and a 32-row tail
    torch.manual_seed(3)
    z = torch.randn(1, n, n, c_z)
    relp = (torch.rand(n, n, 139) < 0.05).float()
    ft = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    z_tt = ft(z)
    single = Protenix._diffusion_pair_cond(stub, z_tt, relp)
    assert stub._paircond_rows_refused == {}, "the single pass was refused at 160 tokens"
    assert not z_tt.is_allocated(), "the input pair outlived its only reader"

    class _Refused(dict):
        def __contains__(self, key):
            return True

        def get(self, key, default=None):
            return 64

    stub._paircond_rows_refused = _Refused()
    blocked = Protenix._diffusion_pair_cond(stub, ft(z), relp)
    assert blocked.shape == (n, n, single.shape[-1]) and single.numel() == blocked.numel()
    assert torch.equal(blocked, single.reshape(blocked.shape)), \
        (blocked - single.reshape(blocked.shape)).abs().max()
