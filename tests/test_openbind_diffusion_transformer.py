"""On-device parity for the OpenBind token-level DiffusionTransformer (the hoisted pair
LayerNorm, the one architectural difference between OF3-preview2 and OpenBind).

OpenBind applies ``layer_norm_z`` once before the 24-block stack instead of once inside
each block's ``AttentionPairBias``. tt-bio caches that single norm across diffusion steps,
since ``z`` is fixed across them, and drops the per-block norm when the checkpoint says to
(``openfold3_diffusion_transformer.py``, selected by ``is_openbind``). This gates the
device against the upstream v0.5.0 reference run on the same real inputs.

Golden: ``~/of3_ob_ref.pkl["intermediates"]["diffusion_transformer_ob0"]``, from
``scripts/ob0_diffusion_transformer_golden.py``.

HOW MUCH THIS LEG CAN PROVE, measured rather than assumed. The reference capture also
computes the same stack with the shared norm SKIPPED -- the failure this leg is nominally
here to catch -- and it lands at update PCC 0.995 for block 0 and 0.972 for the 24-block
stack. So the DiT pair bias is a weak lever at this magnitude of ``z``, and a >0.99 device
match is only weakly discriminating: it confirms the topology, the weight mapping and the
step cache, but it would NOT catch a dropped norm on its own. The gate is written to say
exactly that -- it requires the device to beat the norm-skipped control by a margin rather
than merely clear an absolute threshold. The decisive A1 evidence is elsewhere and is not
pretended to be here: the checkpoint loads strict against this topology and nothing else
(48 per-block keys traded for 1, verified on both files), and end-to-end structure.
"""
import os

import pytest
import torch
import ttnn

import of3_golden

_CKPT = os.path.expanduser("~/.boltz/of3-ob-2025-06-30-174k.pt")
_GOLD = os.path.expanduser("~/of3_ob_ref.pkl")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(_CKPT) and os.path.exists(_GOLD)),
    reason="OpenBind ckpt or ~/of3_ob_ref.pkl missing "
           "(scripts/ob0_diffusion_transformer_golden.py writes it)")


def _pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


def _cfg(dev):
    return ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)


def _run(n_blocks, g, dev):
    from tt_bio.openfold3_diffusion_transformer import OF3DiffusionTransformer
    from tt_bio.openfold3_weights import _sub

    sd = torch.load(_CKPT, map_location="cpu", weights_only=False)
    for key in ("state_dict", "model", "ema"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
    dt_sd = _sub(sd, "diffusion_module.diffusion_transformer")
    assert "layer_norm_z.weight" in dt_sd, "not the OpenBind checkpoint"
    dit = OF3DiffusionTransformer(dt_sd, _cfg(dev), n_blocks=n_blocks)

    a_in, s, z, tok = g["a_in"], g["s"], g["z"], g["token_mask"]
    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev,
                                   dtype=ttnn.bfloat16)
    N = tok.shape[0]
    out = dit(ft(a_in.unsqueeze(0)), ft(s.unsqueeze(0)), ft(z.unsqueeze(0)),
              ft(tok.reshape(1, N)), ft(tok.reshape(1, N, 1)))
    return torch.Tensor(ttnn.to_torch(out)).float().reshape(a_in.shape)


@pytest.mark.parametrize("leg,n_blocks", [("a_block0", 1), ("a_stack", 24)])
def test_openbind_diffusion_transformer_on_device(leg, n_blocks):
    from tt_bio.tenstorrent import get_device

    g = of3_golden.intermediates(_GOLD)["diffusion_transformer_ob0"]
    got = _run(n_blocks, g, get_device())
    ref, base = g[leg].float(), g["a_in"].float()
    ctrl = g[leg + "_unnormed"].float()

    pcc_out = _pcc(got, ref)
    pcc_upd = _pcc(got - base, ref - base)
    # What the same comparison scores for the norm-skipped reference, i.e. the floor this
    # leg can discriminate against.
    ctrl_upd = _pcc(ctrl - base, ref - base)
    print(f"\nOB0 DiffusionTransformer [{leg}, {n_blocks} block(s)]: "
          f"out_pcc={pcc_out:.5f} update_pcc={pcc_upd:.5f} "
          f"(norm-skipped control update_pcc={ctrl_upd:.5f})")

    assert pcc_out > 0.98, f"{leg} out_pcc={pcc_out:.5f} below 0.98"
    # Beat the control, not just an absolute bar. Without this the gate would pass on a
    # device that dropped the shared norm entirely.
    assert pcc_upd > ctrl_upd, (
        f"{leg} device update_pcc={pcc_upd:.5f} does not beat the norm-skipped control "
        f"{ctrl_upd:.5f}, so this leg cannot tell the two apart")
