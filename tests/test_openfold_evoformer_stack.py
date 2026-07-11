"""On-device parity: a 2-block tt_bio.openfold.EvoformerStack (device trunk module)
vs the same composition of vendored reference blocks + the single-rep projection
s = Linear(c_m->c_s)(m[...,0,:,:]). Verifies block chaining and the s-projection.
Core-reuse (biases zeroed)."""
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
from tt_bio.tenstorrent import get_device
from tt_bio.openfold import EvoformerStack

C_M, C_Z, C_S = 256, 128, 384
C_HID_MSA, C_HID_MUL, C_HID_PAIR = 32, 128, 32
H_MSA, H_PAIR, TN, N_SEQ, N_RES = 8, 4, 4, 8, 64


def _pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum()
                 / ((a - a.mean()).norm() * (b - b.mean()).norm()))


def _init(mod):
    with torch.no_grad():
        for n, p in mod.named_parameters():
            if p.ndim == 2:
                p.copy_(torch.randn_like(p) / (p.shape[1] ** 0.5))
            elif "weight" in n:
                p.fill_(1.0)
            else:
                p.zero_()
    return mod.eval()


def _mk_block():
    return {
        "row": _init(RefRow(C_M, C_Z, C_HID_MSA, H_MSA)),
        "col": _init(RefCol(C_M, C_HID_MSA, H_MSA)),
        "msa_transition": _init(PairTransition(C_M, TN)),
        "opm": _init(OuterProductMean(C_M, C_Z, C_HID_MSA)),
        "tri_mul_out": _init(TriangleMultiplicationOutgoing(C_Z, C_HID_MUL)),
        "tri_mul_in": _init(TriangleMultiplicationIncoming(C_Z, C_HID_MUL)),
        "tri_att_start": _init(TriangleAttentionStartingNode(C_Z, C_HID_PAIR, H_PAIR)),
        "tri_att_end": _init(TriangleAttentionEndingNode(C_Z, C_HID_PAIR, H_PAIR)),
        "pair_transition": _init(PairTransition(C_Z, TN)),
    }


def _ref_forward(b, m, z):
    with torch.no_grad():
        m = m + b["row"](m, z); m = m + b["col"](m); m = m + b["msa_transition"](m)
        z = z + b["opm"](m)
        z = z + b["tri_mul_out"](z); z = z + b["tri_mul_in"](z)
        z = z + b["tri_att_start"](z); z = z + b["tri_att_end"](z)
        z = z + b["pair_transition"](z)
    return m, z


def _sub(b):
    ta = lambda sd: {k.replace("mha.", ""): v for k, v in sd.items()}
    return {
        "row": dict(b["row"].state_dict()),
        "col": dict(b["col"].state_dict()),
        "msa_transition": dict(b["msa_transition"].state_dict()),
        "opm": remap_outer_product_mean(dict(b["opm"].state_dict())),
        "tri_mul_out": remap_triangle_multiplication(dict(b["tri_mul_out"].state_dict())),
        "tri_mul_in": remap_triangle_multiplication(dict(b["tri_mul_in"].state_dict())),
        "tri_att_start": ta(dict(b["tri_att_start"].state_dict())),
        "tri_att_end": ta(dict(b["tri_att_end"].state_dict())),
        "pair_transition": dict(b["pair_transition"].state_dict()),
    }


def main(seed=0):
    torch.manual_seed(seed)
    b1, b2 = _mk_block(), _mk_block()
    s_lin = _init(torch.nn.Linear(C_M, C_S))

    m0 = torch.randn(1, N_SEQ, N_RES, C_M) * 0.5
    z0 = torch.randn(1, N_RES, N_RES, C_Z) * 0.5
    m, z = _ref_forward(b1, m0.clone(), z0.clone())
    m, z = _ref_forward(b2, m, z)
    with torch.no_grad():
        s_ref = s_lin(m[..., 0, :, :])
    m_ref, z_ref = m, z

    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    stack = EvoformerStack([_sub(b1), _sub(b2)], dict(s_lin.state_dict()),
                           C_HID_PAIR, H_PAIR, C_HID_MSA, H_MSA, cfg)
    ft = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    mo, zo, so = stack(ft(m0), ft(z0))
    mo = torch.Tensor(ttnn.to_torch(mo)).float().reshape(m_ref.shape)
    zo = torch.Tensor(ttnn.to_torch(zo)).float().reshape(z_ref.shape)
    so = torch.Tensor(ttnn.to_torch(so)).float().reshape(s_ref.shape)
    pm, pz, ps = _pcc(mo, m_ref), _pcc(zo, z_ref), _pcc(so, s_ref)
    print(f"[EvoformerStack x2] m PCC={pm:.5f}  z PCC={pz:.5f}  s PCC={ps:.5f}")
    assert pm > 0.98 and pz > 0.98 and ps > 0.98, f"stack parity failed ({pm},{pz},{ps})"
    print("PASS: device EvoformerStack (2 blocks + s-projection) verified (PCC > 0.98)")


if __name__ == "__main__":
    main()
