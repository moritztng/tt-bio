"""BoltzGen's diffusion-conditioning bias stacks run through Boltz-2's _bias_stack.

Each of the three stacks (atom encoder, atom decoder, token transformer) is LayerNorm + a
bias-free Linear per layer over one shared input. With TT_BIO_HOST_LEVERS=0 the helper is the
reference concatenation itself; with the fused stack (the default) it must agree to float
rounding, which is what Boltz-2 measured for its own 24-layer stack.
"""
import pytest
import torch

from tt_bio.boltz2 import _bias_stack
from tt_bio.boltzgen.model.modules.diffusion_conditioning import DiffusionConditioning


@pytest.fixture(scope="module")
def cond():
    torch.manual_seed(0)
    dc = DiffusionConditioning(token_s=32, token_z=16, atom_s=16, atom_z=8,
                               token_transformer_depth=6, atom_encoder_depth=2,
                               atom_decoder_depth=2)
    for stack in (dc.atom_enc_proj_z, dc.atom_dec_proj_z, dc.token_trans_proj_z):
        for norm, _ in stack:
            torch.nn.init.normal_(norm.weight, 1.0, 0.1)
            torch.nn.init.normal_(norm.bias, 0.0, 0.1)
    return dc


@pytest.mark.parametrize("name,dim", [("atom_enc_proj_z", 8), ("atom_dec_proj_z", 8),
                                      ("token_trans_proj_z", 16)])
def test_stack_matches_the_per_layer_reference(cond, name, dim, monkeypatch):
    stack = getattr(cond, name)
    x = torch.randn(1, 40, 40, dim)
    with torch.no_grad():
        ref = torch.cat([layer(x) for layer in stack], dim=-1)
        monkeypatch.setenv("TT_BIO_HOST_LEVERS", "0")
        assert torch.equal(_bias_stack(stack, x), ref)
        monkeypatch.setenv("TT_BIO_HOST_LEVERS", "1")
        monkeypatch.setenv("TT_BIO_FUSE_BIAS_STACKS", "1")
        fused = _bias_stack(stack, x)
    assert fused.shape == ref.shape
    torch.testing.assert_close(fused, ref, rtol=1e-5, atol=1e-5)
