"""On-device parity: OpenFold PairTransition (vendored reference, random weights) vs
the net-new tt_bio.openfold.ReluTransition. Weight keys match directly (no remap)."""
import torch
import ttnn

from tt_bio._vendor.openfold.model.pair_transition import PairTransition
from tt_bio.openfold import ReluTransition
from tt_bio.tenstorrent import get_device


def _pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum()
                 / ((a - a.mean()).norm() * (b - b.mean()).norm()))


def main(C_Z=128, N=4, L=64, seed=0):
    torch.manual_seed(seed)
    ref = PairTransition(C_Z, N).eval()
    with torch.no_grad():
        for p in ref.parameters():
            p.copy_(torch.randn_like(p))
    z = torch.randn(1, L, L, C_Z)
    with torch.no_grad():
        out = ref(z.clone(), mask=None)  # [1, L, L, C_Z]

    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    tr = ReluTransition(dict(ref.state_dict()), cfg)
    zt = ttnn.from_torch(z, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    o = torch.Tensor(ttnn.to_torch(tr(zt))).float().reshape(out.shape)
    pcc = _pcc(o, out)
    print(f"[ReluTransition] PCC={pcc:.5f}")
    assert pcc > 0.98, f"ReluTransition PCC {pcc} <= 0.98"
    print("PASS: AF2 ReLU-MLP transition verified (PCC > 0.98)")


if __name__ == "__main__":
    main()
