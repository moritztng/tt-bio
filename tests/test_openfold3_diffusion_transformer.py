"""On-device parity for the OpenFold3 token-level DiffusionTransformer (Algorithm 23,
non-cross-attention path): the DiT block used inside DiffusionModule.

Golden: ~/of3_ref_out.pkl["intermediates"]["diffusion_transformer_real"], captured by
scripts/of3_diffusion_transformer_golden.py. The golden carries the reference DiT
inputs (a, s=si, z=zij, mask=token_mask) and outputs (block 0 + the full 24-block
stack), so the device DiT -- AdaLN + per-block pair bias + fused padded-qkv MHA with
a query gate and an ``linear_ada_out`` output gate, then a SwiGLU zero-gated
ConditionedTransitionBlock -- is gated against the exact reference artifacts,
isolating the device block precision from the atom-encoder/conditioning host math
(same discipline as the other OF3 golden legs).

Two gates: block 0 (unit correctness, the block topology in isolation) and the full
24-block stack (no accumulation blowup). Both must clear PCC > 0.98.
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


def _run(n_blocks, gold, sd, dev):
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_diffusion_transformer import OF3DiffusionTransformer
    from tt_bio.openfold3_weights import _sub

    dt_sd = _sub(_sub(sd, "diffusion_module"), "diffusion_transformer")
    dit = OF3DiffusionTransformer(dt_sd, _cfg(dev), n_blocks=n_blocks)

    a_in, s, z, tok = gold["a_in"], gold["s"], gold["z"], gold["token_mask"]
    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    N = tok.shape[0]
    tok_mask = tok.reshape(1, N)
    tok_mask_col = tok.reshape(1, N, 1)
    a_d = dit(ft(a_in.unsqueeze(0)), ft(s.unsqueeze(0)), ft(z.unsqueeze(0)),
              ft(tok_mask), ft(tok_mask_col))
    return torch.Tensor(ttnn.to_torch(a_d)).float().reshape(a_in.shape)


def test_of3_diffusion_transformer_block0_on_device():
    """Device DiT block 0 vs the reference on the real DiT input. PCC > 0.98."""
    from tt_bio.tenstorrent import get_device
    sd = torch.load(_CKPT, map_location="cpu", weights_only=False)
    g = pickle.load(open(_GOLD, "rb"))["intermediates"]["diffusion_transformer_real"]
    dev = get_device()
    out = _run(1, g, sd, dev)
    pcc = _pcc(out, g["a_block0"].float())
    print(f"\nOF3 DiffusionTransformer block0: a_pcc={pcc:.5f}")
    assert pcc > 0.98, f"block0 a_pcc={pcc:.5f} below 0.98"


def test_of3_diffusion_transformer_stack_on_device():
    """Device 24-block DiT stack vs the reference on the real DiT input. PCC > 0.98."""
    from tt_bio.tenstorrent import get_device
    sd = torch.load(_CKPT, map_location="cpu", weights_only=False)
    g = pickle.load(open(_GOLD, "rb"))["intermediates"]["diffusion_transformer_real"]
    dev = get_device()
    out = _run(24, g, sd, dev)
    pcc = _pcc(out, g["a_stack"].float())
    print(f"\nOF3 DiffusionTransformer 24-block stack: a_pcc={pcc:.5f}")
    assert pcc > 0.98, f"stack a_pcc={pcc:.5f} below 0.98"
