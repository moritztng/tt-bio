"""On-device parity for the OpenFold3 TemplateEmbedderAllAtom (P8).

Golden: ~/of3_ref_out.pkl["intermediates"]["template_embedder_real"], captured by
scripts/of3_template_embedder_golden.py. The golden carries the cycle-0 trunk z (the
embedder input, a constant shift of z_init), the per-template feature tensors with the
mask products precomputed on host (multichain / pseudo-beta / backbone-frame pair
masks), and the reference sub-outputs t_embed (TemplatePairEmbedderAllAtom),
t_stack (TemplatePairStack), and z_template (full TemplateEmbedderAllAtom).

Three sub-leg gates isolate the device precision of each leg:
  A. TemplatePairFeatureEmbedder (8 linears + linear_z(layer_norm_z(z)) + add) -> t_embed.
     Pure linears + adds + a layer_norm, so this is the byte-correct linear leg.
  B. TemplatePairStack (2 AF2 PairBlocks + final layer_norm) -> t_stack. Reuses the same
     TriangleMultiplication/TriangleAttention/Transition primitives as the MSA pair_stack
     (tests/test_openfold3_msa.py). The template pair_stack runs at
     c_hidden_tri_att=16 / no_heads=4 (head_dim=16, sub-tile), which needs the
     TriangleAttention sub-tile head_dim path (pad qkv head_dim to 32, slice + manual
     concat the SDPA output); see tt_bio/tenstorrent.py TriangleAttention. The OF3
     template regime (c_t=64, smaller pair magnitudes than the MSA c=128 stack) does NOT
     hit the MSA pair_stack's device-precision gap, so this leg clears >0.98 on device.
  C. TemplateEmbedderAllAtom full (A + B + mean/relu/linear_t) -> z_template. Inherits
     leg B's behavior; clears >0.98 on device.

The pair_mask is all-ones for the single-chain ubiquitin golden, so masking is a no-op
and PairformerLayer is called with mask=None (mirrors tests/test_openfold3_msa.py).
"""
import os, pickle, pytest, torch, ttnn

_CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
_GOLD = os.path.expanduser("~/of3_ref_out.pkl")
pytestmark = pytest.mark.skipif(not (os.path.exists(_CKPT) and os.path.exists(_GOLD)),
                                reason="of3 ckpt or golden pkl missing")


def _pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum()
                 / ((a - a.mean()).norm() * (b - b.mean()).norm()))


def _cfg(dev):
    return ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)


def _feat_to_device(feat, dev):
    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    return {k: ft(v) for k, v in feat.items()}


def test_of3_template_feature_embedder_on_device():
    """Sub-leg A: device TemplatePairFeatureEmbedder (feat + z -> t_embed) vs golden.
    Eight bias-free linears + a layer_norm + linear_z + add -- the byte-correct linear
    leg. Gated tight (PCC > 0.98)."""
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_template import TemplatePairFeatureEmbedder
    from tt_bio.openfold3_weights import _sub

    sd = torch.load(_CKPT, map_location="cpu", weights_only=False)
    fe_sd = _sub(sd, "template_embedder.template_pair_embedder")
    g = pickle.load(open(_GOLD, "rb"))["intermediates"]["template_embedder_real"]
    feat, z_ref, t_embed_ref = g["feat"], g["z"], g["t_embed"]

    dev = get_device()
    fe = TemplatePairFeatureEmbedder(fe_sd, _cfg(dev))
    feat_d = _feat_to_device(feat, dev)
    z_d = ttnn.from_torch(z_ref.unsqueeze(0).float(), layout=ttnn.TILE_LAYOUT,
                          device=dev, dtype=ttnn.bfloat16)
    t_embed_d, _ = fe(feat_d, z_d)
    t_embed = torch.Tensor(ttnn.to_torch(t_embed_d)).float().reshape(t_embed_ref.shape)
    pcc = _pcc(t_embed, t_embed_ref.float())
    print(f"\nOF3 TemplatePairFeatureEmbedder: t_embed_pcc={pcc:.5f}")
    assert pcc > 0.98, f"t_embed_pcc={pcc:.5f} below 0.98"


def test_of3_template_pair_stack_on_device():
    """Sub-leg B: device TemplatePairStack (t_embed -> t_stack) vs golden. Reuses the
    MSA pair_stack primitives at c_hidden_tri_att=16 / no_heads=4 (head_dim=16,
    sub-tile), which needs the TriangleAttention sub-tile head_dim path. The OF3
    template regime (c_t=64) does not hit the MSA pair_stack's c=128 device-precision
    gap, so this leg clears >0.98 on device. Gated tight (PCC > 0.98)."""
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_template import TemplatePairStack
    from tt_bio.openfold3_weights import _sub

    sd = torch.load(_CKPT, map_location="cpu", weights_only=False)
    te_sd = _sub(sd, "template_embedder")
    g = pickle.load(open(_GOLD, "rb"))["intermediates"]["template_embedder_real"]
    t_embed_ref, t_stack_ref = g["t_embed"], g["t_stack"]

    dev = get_device()
    ps = TemplatePairStack(te_sd, _cfg(dev))
    t_embed_d = ttnn.from_torch(t_embed_ref.float(), layout=ttnn.TILE_LAYOUT,
                                device=dev, dtype=ttnn.bfloat16)
    t_stack_d = ps(t_embed_d)
    t_stack = torch.Tensor(ttnn.to_torch(t_stack_d)).float().reshape(t_stack_ref.shape)
    pcc = _pcc(t_stack, t_stack_ref.float())
    print(f"\nOF3 TemplatePairStack (2 blocks): t_stack_pcc={pcc:.5f}")
    assert pcc > 0.98, f"t_stack_pcc={pcc:.5f} below 0.98"


def test_of3_template_embedder_on_device():
    """Sub-leg C: device TemplateEmbedderAllAtom full (feat + z -> z_template) vs
    golden. Composes legs A + B + mean/relu/linear_t; clears >0.98 on device.
    Gated tight (PCC > 0.98)."""
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_template import TemplateEmbedder
    from tt_bio.openfold3_weights import _sub

    sd = torch.load(_CKPT, map_location="cpu", weights_only=False)
    te_sd = _sub(sd, "template_embedder")
    g = pickle.load(open(_GOLD, "rb"))["intermediates"]["template_embedder_real"]
    feat, z_ref, z_template_ref = g["feat"], g["z"], g["z_template"]

    dev = get_device()
    te = TemplateEmbedder(te_sd, _cfg(dev))
    feat_d = _feat_to_device(feat, dev)
    z_d = ttnn.from_torch(z_ref.unsqueeze(0).float(), layout=ttnn.TILE_LAYOUT,
                          device=dev, dtype=ttnn.bfloat16)
    z_template_d = te(feat_d, z_d)
    z_template = torch.Tensor(ttnn.to_torch(z_template_d)).float().reshape(z_template_ref.shape)
    pcc = _pcc(z_template, z_template_ref.float())
    print(f"\nOF3 TemplateEmbedderAllAtom: z_template_pcc={pcc:.5f}")
    assert pcc > 0.98, f"z_template_pcc={pcc:.5f} below 0.98"
