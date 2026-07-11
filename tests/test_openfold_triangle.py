"""On-device parity: OpenFold TriangleMultiplication{Outgoing,Incoming} (vendored
reference, random weights) vs the shared tt_bio.tenstorrent.TriangleMultiplication
block, wired through remap_triangle_multiplication (protenix_weights).

Proves (1) the AF2/OpenFold triangle-multiplicative-update op reuses the shared ttnn
primitive, now including AF2's biased linears + gating bias, and (2) the bias-free
code path (AF3-family: Protenix-v2 / Boltz-2 checkpoints) is unchanged — bias support
is gated on state_dict key presence.
"""
import copy
import torch
import ttnn

from tt_bio._vendor.openfold.model.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
)
from tt_bio.protenix_weights import remap_triangle_multiplication
from tt_bio.tenstorrent import get_device, TriangleMultiplication

_DROPPED_BIAS = ["linear_a_p", "linear_a_g", "linear_b_p", "linear_b_g", "linear_g", "linear_z"]
_STRIP = {"g_in.bias", "p_in.bias", "g_out.bias", "p_out.bias"}


def _pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum()
                 / ((a - a.mean()).norm() * (b - b.mean()).norm()))


def _run(ref_cls, ending, C_Z=128, C_HID=128, L=64, seed=0):
    torch.manual_seed(seed)
    ref = ref_cls(C_Z, C_HID).eval()
    # AF init zeros the "final" linear and sets gating bias=1 -> default-init emits
    # all-zeros (degenerate PCC). Randomize every parameter so the op is genuinely
    # exercised, identically on both sides.
    with torch.no_grad():
        for p in ref.parameters():
            p.copy_(torch.randn_like(p) * 0.1)
    z = torch.randn(1, L, L, C_Z)
    with torch.no_grad():
        out_full = ref(z.clone())
    ref_nb = copy.deepcopy(ref)
    with torch.no_grad():
        for n in _DROPPED_BIAS:
            getattr(ref_nb, n).bias.data.zero_()
        out_nb = ref_nb(z.clone())

    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    zt = lambda: ttnn.from_torch(z, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    remapped = remap_triangle_multiplication(dict(ref.state_dict()))  # includes biases
    tm_full = TriangleMultiplication(ending, remapped, cfg)
    o_full = torch.Tensor(ttnn.to_torch(tm_full(zt()))).float().reshape(out_full.shape)

    remapped_nb = {k: v for k, v in remapped.items() if k not in _STRIP}  # gated-off path
    tm_nb = TriangleMultiplication(ending, remapped_nb, cfg)
    o_nb = torch.Tensor(ttnn.to_torch(tm_nb(zt()))).float().reshape(out_nb.shape)

    return _pcc(o_nb, out_nb), _pcc(o_full, out_full)


def main():
    for name, cls, ending in [
        ("Outgoing", TriangleMultiplicationOutgoing, False),
        ("Incoming", TriangleMultiplicationIncoming, True),
    ]:
        biasfree, full = _run(cls, ending)
        print(f"[{name}] ending={ending}  PCC bias-free-path={biasfree:.5f}  full(biased)={full:.5f}")
        assert biasfree > 0.98, f"{name} bias-free path PCC {biasfree} <= 0.98 -- AF3 path regressed"
        assert full > 0.98, f"{name} full biased PCC {full} <= 0.98 -- AF2 bias support broken"
    print("PASS: TriangleMultiplication reuse verified (biased AF2 + bias-free AF3, both PCC > 0.98)")


if __name__ == "__main__":
    main()
