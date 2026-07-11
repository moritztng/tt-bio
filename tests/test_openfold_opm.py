"""On-device parity: OpenFold OuterProductMean (vendored reference, random weights)
vs the shared tt_bio.tenstorrent.OuterProductMean block via remap_outer_product_mean.

Full (all-ones) MSA mask, so OpenFold's per-pair norm einsum(mask,mask)=N_seq matches
the block's scalar 1/n_msa. linear_1/linear_2 biases are dropped by the block (zeroed
here for the core-reuse check); linear_out bias is kept and mapped.
"""
import torch
import ttnn

from tt_bio._vendor.openfold.model.outer_product_mean import OuterProductMean
from tt_bio.protenix_weights import remap_outer_product_mean
from tt_bio.tenstorrent import get_device, OuterProductMean as TTOuterProductMean


def _pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum()
                 / ((a - a.mean()).norm() * (b - b.mean()).norm()))


def main(C_M=256, C_Z=128, C_HID=32, N_SEQ=8, N_RES=64, seed=0):
    torch.manual_seed(seed)
    ref = OuterProductMean(C_M, C_Z, C_HID).eval()
    with torch.no_grad():
        for p in ref.parameters():
            p.copy_(torch.randn_like(p))
        ref.linear_1.bias.data.zero_()  # dropped by the fused block
        ref.linear_2.bias.data.zero_()
    m = torch.randn(1, N_SEQ, N_RES, C_M)
    with torch.no_grad():
        out = ref(m.clone())  # full mask -> norm = N_SEQ

    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    opm = TTOuterProductMean(remap_outer_product_mean(dict(ref.state_dict())), cfg)
    mt = ttnn.from_torch(m, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    o = torch.Tensor(ttnn.to_torch(opm(mt, msa_mask=None, n_msa=N_SEQ))).float().reshape(out.shape)
    pcc = _pcc(o, out)
    print(f"[OuterProductMean] PCC={pcc:.5f}")
    assert pcc > 0.98, f"OPM PCC {pcc} <= 0.98 -- shared-block reuse broken"
    print("PASS: OuterProductMean reuse verified (PCC > 0.98)")


if __name__ == "__main__":
    main()
