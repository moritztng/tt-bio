"""Protenix `_pair_cond_device_terms` against float64: the DiT's LN(pair_z) and the atom
encoder's W_z(LN_z(pair_z)) made on the device from the device pair, next to the host fp32 path
they replace. Both are measured against the same float64 reference; the device path must stay
within fp32 noise of it. Random weights: the test is about the device ops, not the checkpoint."""
import types

import pytest
import torch
import torch.nn.functional as F

ttnn = pytest.importorskip("ttnn")
pytestmark = pytest.mark.device


def _err(x, ref):
    d = (x.double() - ref).abs()
    return float(d.max()), float(d.max() / ref.abs().max())


@pytest.mark.parametrize("nt", [128, 736])
def test_pair_cond_device_terms_match_float64(nt):
    from tt_bio.protenix import DiffusionModule, Protenix
    from tt_bio.tenstorrent import get_device

    dev = get_device()
    g = torch.Generator().manual_seed(0)
    c = 256
    # A conditioned pair has per-row and per-channel structure on top of noise; give the norm both.
    pz = (torch.randn(nt, nt, c, generator=g) * 2 + torch.randn(nt, 1, 1, generator=g)
          + torch.randn(1, 1, c, generator=g) * 0.5)
    E = "atom_attention_encoder."
    w = {E + "layernorm_z.weight": 1 + 0.1 * torch.randn(c, generator=g),
         E + "linear_no_bias_z.weight": torch.randn(16, c, generator=g) / c ** 0.5}
    ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                 fp32_dest_acc_en=True, packer_l1_acc=True)
    D = types.SimpleNamespace(_w=w, _dit_dtype=ttnn.float32, _dit_ckc=ckc)
    D._ln_dit = DiffusionModule._ln_dit.__get__(D)
    D._w_tt_dit = DiffusionModule._w_tt_dit.__get__(D)
    host = types.SimpleNamespace(diffusion=D, _to_host=Protenix._to_host)

    pz_dev = ttnn.from_torch(pz.unsqueeze(0), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)
    dit_z, ztok = Protenix._pair_cond_device_terms(host, pz_dev)
    dit_z = torch.Tensor(ttnn.to_torch(dit_z)).float().reshape(nt, nt, c)

    p64 = pz.double()
    dit_ref = F.layer_norm(p64, (c,))
    ztok_ref = F.linear(F.layer_norm(p64, (c,)) * w[E + "layernorm_z.weight"].double(),
                        w[E + "linear_no_bias_z.weight"].double())
    dit_host = F.layer_norm(pz, (c,))
    ztok_host = F.linear(F.layer_norm(pz, (c,)) * w[E + "layernorm_z.weight"],
                         w[E + "linear_no_bias_z.weight"])

    errs = {"dit_z device": _err(dit_z, dit_ref), "dit_z host": _err(dit_host, dit_ref),
            "ztok device": _err(ztok, ztok_ref), "ztok host": _err(ztok_host, ztok_ref)}
    print(f"nt={nt} " + "  ".join(f"{k}: abs {a:.3e} rel {r:.3e}" for k, (a, r) in errs.items()))
    assert errs["dit_z device"][1] < 1e-4
    assert errs["ztok device"][1] < 1e-4
