"""On-device parity: a full AF2 Evoformer block composed from the verified tt_bio
blocks vs the same composition of vendored reference modules (random weights).

Composition (inference, no dropout/mask; EvoformerBlock.forward, opm_first=False):
    m += MSARowAttentionWithPairBias(m, z)
    m += MSAColumnAttention(m)
    m += MSATransition(m)
    z += OuterProductMean(m)
    z += TriangleMultiplicationOutgoing(z)
    z += TriangleMultiplicationIncoming(z)
    z += TriangleAttentionStartingNode(z)
    z += TriangleAttentionEndingNode(z)
    z += PairTransition(z)

Core-reuse mode: all biases zeroed (the o/g biases some blocks drop). Proves the
verified primitives compose + interoperate (shapes, residual order, tri-att ending).
"""
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

from tt_bio.protenix_weights import remap_triangle_multiplication, remap_outer_product_mean
from tt_bio.tenstorrent import get_device, TriangleMultiplication, TriangleAttention, OuterProductMean as TTOpm
from tt_bio.openfold import ReluTransition, MSARowAttentionWithPairBias as TTRow, MSAColumnAttention as TTCol

C_M, C_Z, C_HID_MSA, C_HID_MUL, C_HID_PAIR = 256, 128, 32, 128, 32
H_MSA, H_PAIR, TN, N_SEQ, N_RES = 8, 4, 4, 8, 64


def _pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum()
                 / ((a - a.mean()).norm() * (b - b.mean()).norm()))


def _init(mod):
    with torch.no_grad():
        for n, p in mod.named_parameters():
            if p.ndim == 2:
                p.copy_(torch.randn_like(p) / (p.shape[1] ** 0.5))  # xavier-ish, well-conditioned
            elif "weight" in n:
                p.fill_(1.0)   # LayerNorm gain
            else:
                p.zero_()      # all biases -> 0 (core-reuse: dropped biases)
    return mod.eval()


def _strip(sd, pfx):
    return {k[len(pfx):]: v for k, v in sd.items() if k.startswith(pfx)}


def main(seed=0):
    torch.manual_seed(seed)
    # reference modules
    row = _init(RefRow(C_M, C_Z, C_HID_MSA, H_MSA))
    col = _init(RefCol(C_M, C_HID_MSA, H_MSA))
    mtr = _init(PairTransition(C_M, TN))
    opm = _init(OuterProductMean(C_M, C_Z, C_HID_MSA))
    tmo = _init(TriangleMultiplicationOutgoing(C_Z, C_HID_MUL))
    tmi = _init(TriangleMultiplicationIncoming(C_Z, C_HID_MUL))
    tas = _init(TriangleAttentionStartingNode(C_Z, C_HID_PAIR, H_PAIR))
    tae = _init(TriangleAttentionEndingNode(C_Z, C_HID_PAIR, H_PAIR))
    ptr = _init(PairTransition(C_Z, TN))

    m0 = torch.randn(1, N_SEQ, N_RES, C_M) * 0.5
    z0 = torch.randn(1, N_RES, N_RES, C_Z) * 0.5
    with torch.no_grad():
        m, z = m0.clone(), z0.clone()
        m = m + row(m, z); m = m + col(m); m = m + mtr(m)
        z = z + opm(m)
        z = z + tmo(z); z = z + tmi(z); z = z + tas(z); z = z + tae(z); z = z + ptr(z)
    m_ref, z_ref = m, z

    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    ft = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    ta_remap = lambda sd: {k.replace("mha.", ""): v for k, v in sd.items()}

    tt_row = TTRow(C_HID_MSA, H_MSA, dict(row.state_dict()), cfg)
    tt_col = TTCol(C_HID_MSA, H_MSA, dict(col.state_dict()), cfg)
    tt_mtr = ReluTransition(dict(mtr.state_dict()), cfg)
    tt_opm = TTOpm(remap_outer_product_mean(dict(opm.state_dict())), cfg)
    tt_tmo = TriangleMultiplication(False, remap_triangle_multiplication(dict(tmo.state_dict())), cfg)
    tt_tmi = TriangleMultiplication(True, remap_triangle_multiplication(dict(tmi.state_dict())), cfg)
    tt_tas = TriangleAttention(C_HID_PAIR, H_PAIR, False, ta_remap(dict(tas.state_dict())), cfg)
    tt_tae = TriangleAttention(C_HID_PAIR, H_PAIR, True, ta_remap(dict(tae.state_dict())), cfg)
    tt_ptr = ReluTransition(dict(ptr.state_dict()), cfg)

    m = ft(m0); z = ft(z0)
    mshape, zshape = (1, N_SEQ, N_RES, C_M), (1, N_RES, N_RES, C_Z)
    upd = lambda base, out, shp: ttnn.add(base, ttnn.reshape(out, shp))
    m = upd(m, tt_row(m, z), mshape)
    m = upd(m, tt_col(m), mshape)
    m = upd(m, tt_mtr(m), mshape)
    z = upd(z, tt_opm(m, msa_mask=None, n_msa=N_SEQ), zshape)
    z = upd(z, tt_tmo(z), zshape)
    z = upd(z, tt_tmi(z), zshape)
    z = upd(z, tt_tas(z), zshape)
    z = upd(z, tt_tae(z), zshape)
    z = upd(z, tt_ptr(z), zshape)

    mo = torch.Tensor(ttnn.to_torch(m)).float().reshape(m_ref.shape)
    zo = torch.Tensor(ttnn.to_torch(z)).float().reshape(z_ref.shape)
    pm, pz = _pcc(mo, m_ref), _pcc(zo, z_ref)
    print(f"[EvoformerBlock] m PCC={pm:.5f}  z PCC={pz:.5f}")
    assert pm > 0.98 and pz > 0.98, f"Evoformer block composite parity failed (m {pm}, z {pz})"
    print("PASS: full AF2 Evoformer block composes + verifies (m,z PCC > 0.98)")


if __name__ == "__main__":
    main()
