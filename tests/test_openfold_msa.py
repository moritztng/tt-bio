"""On-device parity: OpenFold MSA row (with pair bias) + column attention (vendored
reference, random weights) vs tt_bio.openfold.MSA{Row,Column} blocks. q/k/v bias-free
(match); o/g biases zeroed for the core-reuse check (gated o/g bias = tracked
follow-up). Full mask -> mask_bias 0.
"""
import torch
import ttnn

from tt_bio._vendor.openfold.model.msa import (
    MSARowAttentionWithPairBias as RefRow,
    MSAColumnAttention as RefCol,
)
from tt_bio.openfold import MSARowAttentionWithPairBias as TTRow, MSAColumnAttention as TTCol
from tt_bio.tenstorrent import get_device


def _pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum()
                 / ((a - a.mean()).norm() * (b - b.mean()).norm()))


def _cfg(dev):
    return ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)


def _rand(ref, *drop):
    with torch.no_grad():
        for p in ref.parameters():
            p.copy_(torch.randn_like(p) * 0.1)
        for name in drop:
            mod = ref
            for part in name.split("."):
                mod = getattr(mod, part)
            mod.bias.data.zero_()


def main(C_M=256, C_Z=128, C_HID=32, HEADS=8, N_SEQ=8, N_RES=64, seed=0):
    torch.manual_seed(seed)
    dev = get_device()
    cfg = _cfg(dev)
    ft = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    # --- row (with pair bias) ---
    row = RefRow(C_M, C_Z, C_HID, HEADS).eval()
    _rand(row, "mha.linear_o", "mha.linear_g")
    m = torch.randn(1, N_SEQ, N_RES, C_M); z = torch.randn(1, N_RES, N_RES, C_Z)
    with torch.no_grad():
        out_r = row(m.clone(), z.clone())
    ttr = TTRow(C_HID, HEADS, dict(row.state_dict()), cfg)
    or_ = torch.Tensor(ttnn.to_torch(ttr(ft(m), ft(z)))).float().reshape(out_r.shape)
    pcc_r = _pcc(or_, out_r)
    print(f"[MSARowAttentionWithPairBias] PCC={pcc_r:.5f}")

    # --- column (no bias) ---
    col = RefCol(C_M, C_HID, HEADS).eval()
    _rand(col, "_msa_att.mha.linear_o", "_msa_att.mha.linear_g")
    m2 = torch.randn(1, N_SEQ, N_RES, C_M)
    with torch.no_grad():
        out_c = col(m2.clone())
    ttc = TTCol(C_HID, HEADS, dict(col.state_dict()), cfg)
    oc_ = torch.Tensor(ttnn.to_torch(ttc(ft(m2)))).float().reshape(out_c.shape)
    pcc_c = _pcc(oc_, out_c)
    print(f"[MSAColumnAttention]          PCC={pcc_c:.5f}")

    assert pcc_r > 0.98 and pcc_c > 0.98, f"MSA attention parity failed (row {pcc_r}, col {pcc_c})"
    print("PASS: AF2 MSA row+column attention verified (PCC > 0.98)")


if __name__ == "__main__":
    main()
