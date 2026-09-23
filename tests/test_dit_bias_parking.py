"""A parked DiT pair bias gives the token DiT the same bytes as a resident one.

The 24 per-block pair biases are precomputed once per fold and read once each per diffusion
step. At 2987 structural tokens (OpenDDE at 1536 residues) they are 13.9 GB of fp32 on a 12 GiB
Wormhole chip, so `place_by_reserve` keeps what fits and parks the rest in device layout, and the
DiT uploads a parked one for its read. Real protenix-v2 diffusion weights, 96 tokens.
"""
import pytest
import torch
import ttnn

from tt_bio import tenstorrent as T

pytestmark = pytest.mark.device


def test_place_by_reserve_parks_past_the_reserve_and_brings_it_back():
    dev = T.get_device()
    x = torch.randn(1, 16, 96, 96)
    t = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)
    assert T.place_by_reserve(t, 0) is t
    parked = T.place_by_reserve(t, 1 << 50)
    assert parked.storage_type() != ttnn.StorageType.DEVICE and not t.is_allocated()
    back = T.place_by_reserve(parked, 0)
    assert back.storage_type() == ttnn.StorageType.DEVICE
    assert torch.equal(ttnn.to_torch(back), x)


@pytest.mark.parametrize("fp32", [True, False])
def test_token_dit_is_the_same_with_parked_biases(monkeypatch, fp32):
    from tt_bio import weights
    from tt_bio.protenix import DiffusionModule

    ckpt = weights.resolve("protenix-v2")
    if ckpt is None:
        pytest.skip("protenix-v2 checkpoint not present")
    sd = torch.load(ckpt, map_location="cpu", weights_only=True, mmap=True)
    sd = sd.get("model", sd)
    pfx = "module.diffusion_module."
    sd = {k[len(pfx):]: v for k, v in sd.items() if k.startswith(pfx)}
    dev = T.get_device()
    ck = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    dm = DiffusionModule(sd, dev, ck, diffusion_fp32=fp32)
    torch.manual_seed(4)
    NT = 96
    pair_z = torch.randn(NT, NT, sd["diffusion_conditioning.layernorm_z.weight"].shape[0] // 2)
    a = torch.randn(1, NT, 768)
    s = torch.randn(1, NT, sd["diffusion_conditioning.linear_no_bias_s.weight"].shape[0])
    up = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dm._dit_dtype)

    def dit():
        biases = dm._dit_block_biases(dm._dit_z_device(pair_z))
        return biases, ttnn.to_torch(dm._token_dit_device(up(a), up(s), biases, NT))

    biases, resident = dit()
    assert all(b.storage_type() == ttnn.StorageType.DEVICE for b in biases)
    real = T.place_by_reserve
    monkeypatch.setattr(T, "place_by_reserve", lambda t, reserve: real(t, 1 << 50))
    biases, parked = dit()
    assert all(b.storage_type() != ttnn.StorageType.DEVICE for b in biases)
    assert torch.equal(parked, resident), (parked - resident).abs().max()


@pytest.mark.parametrize("fp32", [True, False])
def test_row_blocked_compute_bias_is_the_whole_one(monkeypatch, fp32):
    """Each of the 24 precomputes norms the whole DiT pair beside the biases already built; after
    a refusal it runs on row blocks, in the pair's own dtype. 96 tokens = a 64-row block and a
    32-row tail."""
    from tt_bio import weights
    from tt_bio.protenix import DiffusionModule

    ckpt = weights.resolve("protenix-v2")
    if ckpt is None:
        pytest.skip("protenix-v2 checkpoint not present")
    sd = torch.load(ckpt, map_location="cpu", weights_only=True, mmap=True)
    sd = sd.get("model", sd)
    pfx = "module.diffusion_module."
    sd = {k[len(pfx):]: v for k, v in sd.items() if k.startswith(pfx)}
    dev = T.get_device()
    ck = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    dm = DiffusionModule(sd, dev, ck, diffusion_fp32=fp32)
    torch.manual_seed(5)
    NT = 96
    z = dm._dit_z_device(torch.randn(NT, NT, sd["diffusion_conditioning.layernorm_z.weight"].shape[0] // 2))
    apb = dm._dit[0][1]
    whole = apb.compute_bias(z)
    assert T._APB_BIAS_REFUSED == {}, "the whole norm was refused at 96 tokens"

    class _Refused(dict):
        def __contains__(self, key):
            return True

        def get(self, key, default=None):
            return 64

    monkeypatch.setattr(T, "_APB_BIAS_REFUSED", _Refused())
    blocked = apb.compute_bias(z)
    assert blocked.dtype == whole.dtype and tuple(blocked.shape) == tuple(whole.shape)
    assert torch.equal(ttnn.to_torch(blocked), ttnn.to_torch(whole))
