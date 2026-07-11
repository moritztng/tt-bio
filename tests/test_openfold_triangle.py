"""On-device parity: OpenFold TriangleMultiplication{Outgoing,Incoming} (vendored
reference, random weights) vs the shared tt_bio.tenstorrent.TriangleMultiplication
block, wired through the EXISTING remap_triangle_multiplication (protenix_weights).

Proves the AF2/OpenFold triangle-multiplicative-update op reuses the shared ttnn
primitive with zero new device code. Reports two PCCs:
  * core  — OpenFold projection/gate/output biases zeroed (the math the fused
            tt-bio block implements; protenix-v2 checkpoints are bias-free here);
  * full  — biases kept (classic AF2 uses biased linears + gating bias=1.0), which
            quantifies what real-weight parity will need the fused block to add.
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


def _pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum()
                 / ((a - a.mean()).norm() * (b - b.mean()).norm()))


def _run(ref_cls, ending, C_Z=128, C_HID=128, L=64, seed=0):
    torch.manual_seed(seed)
    ref = ref_cls(C_Z, C_HID).eval()
    # AF init zeros the "final" linear (linear_z) and sets gating bias=1 -> a
    # default-init module emits all-zeros (degenerate PCC). Randomize every
    # parameter so the op is genuinely exercised, identically on both sides.
    with torch.no_grad():
        for p in ref.parameters():
            p.copy_(torch.randn_like(p) * 0.1)
    z = torch.randn(1, L, L, C_Z)
    with torch.no_grad():
        out_full = ref(z.clone())
    ref_nb = copy.deepcopy(ref)
    for n in _DROPPED_BIAS:
        getattr(ref_nb, n).bias.data.zero_()
    with torch.no_grad():
        out_nb = ref_nb(z.clone())

    remapped = remap_triangle_multiplication(dict(ref.state_dict()))
    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    tm = TriangleMultiplication(ending, remapped, cfg)
    zt = ttnn.from_torch(z, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    ot = tm(zt)
    ot = torch.Tensor(ttnn.to_torch(ot)).float().reshape(out_full.shape)
    return _pcc(ot, out_nb), _pcc(ot, out_full)


def main():
    # OpenFold outgoing == AF2 "starting" edges -> tt-bio ending=False; incoming -> True.
    for name, cls, ending in [
        ("Outgoing", TriangleMultiplicationOutgoing, False),
        ("Incoming", TriangleMultiplicationIncoming, True),
    ]:
        core, full = _run(cls, ending)
        print(f"[{name}] ending={ending}  PCC core(no-bias)={core:.5f}  full(biased)={full:.5f}")
        assert core > 0.98, f"{name} core PCC {core} <= 0.98 -- shared-block reuse broken"
    print("PASS: TriangleMultiplication reuse verified (core PCC > 0.98)")


if __name__ == "__main__":
    main()
