"""On-device parity: OpenFold TriangleAttention{Starting,Ending}Node (vendored
reference, random weights) vs the shared tt_bio.tenstorrent.TriangleAttention block.

Remap is a pure rename (strip the `mha.` prefix onto the block's flat q/k/v/o/g keys).
q/k/v are bias-free (matches the block); linear_o/linear_g carry biases in AF2 that
the block currently drops — verified here as the core-reuse PCC with those biases
zeroed. (Gated o/g bias is the same mechanical follow-up applied to
TriangleMultiplication; tracked in docs/openfold-port.md.)
"""
import copy
import torch
import ttnn

from tt_bio._vendor.openfold.model.triangular_attention import (
    TriangleAttentionStartingNode,
    TriangleAttentionEndingNode,
)
from tt_bio.tenstorrent import get_device, TriangleAttention

_DROPPED_BIAS = ["mha.linear_o", "mha.linear_g"]


def _pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum()
                 / ((a - a.mean()).norm() * (b - b.mean()).norm()))


def _remap(sd):
    # strip the mha. prefix; linear_q/k/v/o/g + layer_norm + linear land flat.
    return {k.replace("mha.", ""): v for k, v in sd.items()}


def _run(ref_cls, ending, C_IN=128, C_HID=32, HEADS=4, L=64, seed=0):
    torch.manual_seed(seed)
    ref = ref_cls(C_IN, C_HID, HEADS).eval()
    with torch.no_grad():
        for p in ref.parameters():
            p.copy_(torch.randn_like(p) * 0.1)
        # zero the biases the shared block drops (q/k/v already bias-free)
        for n in _DROPPED_BIAS:
            mod = ref
            for part in n.split("."):
                mod = getattr(mod, part)
            if mod.bias is not None:
                mod.bias.data.zero_()
    x = torch.randn(1, L, L, C_IN)
    with torch.no_grad():
        out = ref(x.clone())

    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    ta = TriangleAttention(C_HID, HEADS, ending, _remap(dict(ref.state_dict())), cfg)
    xt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    o = torch.Tensor(ttnn.to_torch(ta(xt))).float().reshape(out.shape)
    return _pcc(o, out)


def main():
    for name, cls, ending in [
        ("StartingNode", TriangleAttentionStartingNode, False),
        ("EndingNode", TriangleAttentionEndingNode, True),
    ]:
        pcc = _run(cls, ending)
        print(f"[{name}] ending={ending}  PCC core(no o/g bias)={pcc:.5f}")
        assert pcc > 0.98, f"{name} core PCC {pcc} <= 0.98 -- shared-block reuse broken"
    print("PASS: TriangleAttention reuse verified (core PCC > 0.98)")


if __name__ == "__main__":
    main()
