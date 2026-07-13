"""On-device parity for the OpenFold3 DiffusionConditioning leg (Algorithm 21).

Golden: ~/of3_ref_out.pkl["intermediates"]["diffusion_conditioning_real"], captured by
scripts/of3_diffusion_conditioning_golden.py. The golden carries the reference relpos
(139-dim relpos_complex) and the post-Fourier noise embedding (256-dim), so the device
conditioning -- pair/single linears + weight-only top LNs + 2x SwiGLU transition on each
of s/z + the noise-embedding broadcast add -- is gated against the exact reference
artifacts, isolating the device linear/LN/SwiGLU precision from the relpos/Fourier host
math (same discipline as the other OF3 golden legs).

This is the first device sub-leg of the OF3 DiffusionModule (P8): the conditioned
(si, zij) it produces are what the OF3 DiffusionTransformer (token-level DiT, to be
ported next) consumes. The atom enc/dec inside DiffusionModule reuse the already-gated
P7 AtomTransformer.
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


def test_of3_diffusion_conditioning_on_device():
    """Device DiffusionConditioning vs the reference on real ubiquitin trunk tensors.
    si (conditioned single) and zij (conditioned pair) gated tight (PCC > 0.98)."""
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_diffusion import OF3DiffusionConditioning
    from tt_bio.openfold3_weights import _sub

    sd = torch.load(_CKPT, map_location="cpu", weights_only=False)
    dc_sd = _sub(_sub(sd, "diffusion_module"), "diffusion_conditioning")
    g = pickle.load(open(_GOLD, "rb"))["intermediates"]["diffusion_conditioning_real"]
    si_input, si_trunk, zij_trunk = g["si_input"], g["si_trunk"], g["zij_trunk"]
    relpos, n_emb, tok = g["relpos"], g["n_emb"], g["token_mask"]
    si_ref, zij_ref = g["si_ref"], g["zij_ref"]

    dev = get_device()
    dc = OF3DiffusionConditioning(dc_sd, _cfg(dev))

    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    si_d, zij_d = dc(
        ft(zij_trunk.unsqueeze(0)), ft(relpos.unsqueeze(0)),
        ft(si_trunk.unsqueeze(0)), ft(si_input.unsqueeze(0)),
        ft(n_emb.reshape(1, 1, 256)),
        ft((tok[:, None] * tok[None, :]).reshape(76, 76, 1).unsqueeze(0)),  # pair mask
        ft(tok.reshape(76, 1).unsqueeze(0)),                                   # token mask
    )
    si = torch.Tensor(ttnn.to_torch(si_d)).float().reshape(si_ref.shape)
    zij = torch.Tensor(ttnn.to_torch(zij_d)).float().reshape(zij_ref.shape)
    si_pcc = _pcc(si, si_ref.float())
    zij_pcc = _pcc(zij, zij_ref.float())
    print(f"\nOF3 DiffusionConditioning: si_pcc={si_pcc:.5f} zij_pcc={zij_pcc:.5f}")
    assert si_pcc > 0.98, f"si_pcc={si_pcc:.5f} below 0.98"
    assert zij_pcc > 0.98, f"zij_pcc={zij_pcc:.5f} below 0.98"
