"""On-device parity: tt_bio.openfold.EvoformerStack built via openfold_weights from a
REAL reference EvoformerStack checkpoint tree (blocks.{i}.pair_stack.*, msa_att_col.
_msa_att.*, linear.*) vs that reference stack's own forward. Random weights, no
download/MSA needed. Validates the real module-tree scoping/remap path (the integration
risk my earlier per-module composites sidestepped)."""
import torch
import ttnn

from tt_bio._vendor.openfold.model.evoformer import EvoformerStack as RefStack
from tt_bio.openfold import EvoformerStack as TTStack
from tt_bio.openfold_weights import evoformer_stack_subs
from tt_bio.tenstorrent import get_device

C_M, C_Z, C_S = 256, 128, 384
C_HID_MSA, C_HID_OPM, C_HID_MUL, C_HID_PAIR = 32, 32, 128, 32
H_MSA, H_PAIR, TN, NB, N_SEQ, N_RES = 8, 4, 4, 2, 8, 64


def _pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum()
                 / ((a - a.mean()).norm() * (b - b.mean()).norm()))


def main(seed=0):
    torch.manual_seed(seed)
    ref = RefStack(
        c_m=C_M, c_z=C_Z, c_hidden_msa_att=C_HID_MSA, c_hidden_opm=C_HID_OPM,
        c_hidden_mul=C_HID_MUL, c_hidden_pair_att=C_HID_PAIR, c_s=C_S,
        no_heads_msa=H_MSA, no_heads_pair=H_PAIR, no_blocks=NB, transition_n=TN,
        msa_dropout=0.15, pair_dropout=0.25, no_column_attention=False, opm_first=False,
        fuse_projection_weights=False, blocks_per_ckpt=None, inf=1e9, eps=1e-10,
    ).eval()
    with torch.no_grad():
        for n, p in ref.named_parameters():
            if p.ndim == 2:
                p.copy_(torch.randn_like(p) / (p.shape[1] ** 0.5))
            elif "weight" in n:
                p.fill_(1.0)
            else:
                p.zero_()

    m0 = torch.randn(1, N_SEQ, N_RES, C_M) * 0.5
    z0 = torch.randn(1, N_RES, N_RES, C_Z) * 0.5
    msa_mask = torch.ones(1, N_SEQ, N_RES)
    pair_mask = torch.ones(1, N_RES, N_RES)
    with torch.no_grad():
        m_ref, z_ref, s_ref = ref(m0.clone(), z0.clone(), msa_mask, pair_mask, chunk_size=None)

    subs, s_lin = evoformer_stack_subs(dict(ref.state_dict()), NB)
    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    stack = TTStack(subs, s_lin, C_HID_PAIR, H_PAIR, C_HID_MSA, H_MSA, cfg)
    ft = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    mo, zo, so = stack(ft(m0), ft(z0))
    mo = torch.Tensor(ttnn.to_torch(mo)).float().reshape(m_ref.shape)
    zo = torch.Tensor(ttnn.to_torch(zo)).float().reshape(z_ref.shape)
    so = torch.Tensor(ttnn.to_torch(so)).float().reshape(s_ref.shape)
    pm, pz, ps = _pcc(mo, m_ref), _pcc(zo, z_ref), _pcc(so, s_ref)
    print(f"[EvoformerStack real-tree] m PCC={pm:.5f}  z PCC={pz:.5f}  s PCC={ps:.5f}")
    assert pm > 0.98 and pz > 0.98 and ps > 0.98, f"real-tree stack parity failed ({pm},{pz},{ps})"
    print("PASS: device EvoformerStack built from a real reference checkpoint tree (PCC > 0.98)")


if __name__ == "__main__":
    main()
