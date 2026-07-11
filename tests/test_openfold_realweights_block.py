"""On-device parity with REAL AF2 weights: block-0 of finetuning_ptm_1.pt.

Loads the real released OpenFold pTM checkpoint's Evoformer block 0 into the vendored
reference sub-modules, composes the reference EvoformerBlock forward, and compares to
the device tt_bio.openfold.EvoformerBlock built from the same real weights via
openfold_weights.evoformer_block_subs. Real (well-conditioned) weights — the true test
of the weight-loader + block numerics. Also reports the impact of the currently-dropped
o/g gate/output biases (tri-att + MSA), the tracked real-weight follow-up.
"""
import os
import torch
import ttnn

from tt_bio._vendor.openfold.model.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming)
from tt_bio._vendor.openfold.model.triangular_attention import (
    TriangleAttentionStartingNode, TriangleAttentionEndingNode)
from tt_bio._vendor.openfold.model.outer_product_mean import OuterProductMean
from tt_bio._vendor.openfold.model.msa import (
    MSARowAttentionWithPairBias as RefRow, MSAColumnAttention as RefCol)
from tt_bio._vendor.openfold.model.pair_transition import PairTransition

from tt_bio.openfold import EvoformerBlock
from tt_bio.openfold_weights import evoformer_block_subs
from tt_bio.tenstorrent import get_device

CKPT = "/home/ttuser/openfold_ckpt/finetuning_ptm_1.pt"
C_M, C_Z, C_HID_MSA, C_HID_MUL, C_HID_PAIR = 256, 128, 32, 128, 32
H_MSA, H_PAIR, N_SEQ, N_RES = 8, 4, 8, 64


def _pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum()
                 / ((a - a.mean()).norm() * (b - b.mean()).norm()))


def _sc(sd, p):
    return {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}


def _load(mod, sd, zero_og=False):
    mod.load_state_dict(sd, strict=False)
    if zero_og:  # zero the o/g gate/output biases the device blocks currently drop
        for name in ("mha.linear_o", "mha.linear_g"):
            m = mod
            for part in name.split("."):
                m = getattr(m, part, None)
                if m is None:
                    break
            if m is not None and getattr(m, "bias", None) is not None:
                m.bias.data.zero_()
    return mod.eval()


def _ref_forward(b, m, z):
    with torch.no_grad():
        m = m + b["row"](m, z); m = m + b["col"](m); m = m + b["mtr"](m)
        z = z + b["opm"](m)
        z = z + b["tmo"](z); z = z + b["tmi"](z); z = z + b["tas"](z); z = z + b["tae"](z)
        z = z + b["ptr"](z)
    return m, z


def _build_ref(blk0, zero_og):
    return {
        "row": _load(RefRow(C_M, C_Z, C_HID_MSA, H_MSA), _sc(blk0, "msa_att_row."), zero_og),
        "col": _load(RefCol(C_M, C_HID_MSA, H_MSA), _sc(blk0, "msa_att_col."), zero_og),
        "mtr": _load(PairTransition(C_M, 4), _sc(blk0, "core.msa_transition.")),
        "opm": _load(OuterProductMean(C_M, C_Z, C_HID_MSA), _sc(blk0, "core.outer_product_mean.")),
        "tmo": _load(TriangleMultiplicationOutgoing(C_Z, C_HID_MUL), _sc(blk0, "core.tri_mul_out.")),
        "tmi": _load(TriangleMultiplicationIncoming(C_Z, C_HID_MUL), _sc(blk0, "core.tri_mul_in.")),
        "tas": _load(TriangleAttentionStartingNode(C_Z, C_HID_PAIR, H_PAIR), _sc(blk0, "core.tri_att_start."), zero_og),
        "tae": _load(TriangleAttentionEndingNode(C_Z, C_HID_PAIR, H_PAIR), _sc(blk0, "core.tri_att_end."), zero_og),
        "ptr": _load(PairTransition(C_Z, 4), _sc(blk0, "core.pair_transition.")),
    }


def main(seed=0):
    assert os.path.exists(CKPT), CKPT
    torch.manual_seed(seed)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    blk0 = {k[len("evoformer.blocks.0."):]: v for k, v in sd.items() if k.startswith("evoformer.blocks.0.")}

    m0 = torch.randn(1, N_SEQ, N_RES, C_M) * 0.5
    z0 = torch.randn(1, N_RES, N_RES, C_Z) * 0.5

    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    blk = EvoformerBlock(evoformer_block_subs(blk0), C_HID_PAIR, H_PAIR, C_HID_MSA, H_MSA, cfg)
    ft = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    mo, zo = blk(ft(m0), ft(z0))
    mo = torch.Tensor(ttnn.to_torch(mo)).float().reshape(m0.shape)
    zo = torch.Tensor(ttnn.to_torch(zo)).float().reshape(z0.shape)

    for tag, zero in [("full real weights", False), ("o/g biases zeroed", True)]:
        mr, zr = _ref_forward(_build_ref(blk0, zero), m0.clone(), z0.clone())
        print(f"[real block0 | {tag}] m PCC={_pcc(mo, mr):.5f}  z PCC={_pcc(zo, zr):.5f}")


if __name__ == "__main__":
    main()
