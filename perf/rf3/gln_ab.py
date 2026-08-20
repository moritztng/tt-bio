#!/usr/bin/env python3
"""A/B the confidence head's global layer norm against the torch reference it ports.

Two questions, not one. Is the new flatten bit-exact with the old one (it is not, above
128 tokens), and which of the two is closer to what it is a port OF -- torch's
`F.layer_norm(x, normalized_shape=x.shape)` in fp32. A reduction over 134M elements is
where a flat accumulation loses, so "differs" and "worse" are not the same finding.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import ttnn

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

EPS = 1e-5


def old_gln(x):
    shape = tuple(x.shape)
    n = 1
    for d in shape:
        n *= d
    flat = ttnn.reshape(x, (1, 1, 1, n))
    m = ttnn.mean(flat, dim=-1, keepdim=True)
    xc = ttnn.subtract(flat, m)
    v = ttnn.mean(ttnn.multiply(xc, xc), dim=-1, keepdim=True)
    out = ttnn.multiply(xc, ttnn.rsqrt(ttnn.add(v, EPS)))
    return ttnn.reshape(out, shape)


def rel_rms(a, b):
    return float((a - b).pow(2).mean().sqrt() / b.std())


def main():
    from tt_bio.rf3.confidence_head import global_layer_norm
    from tt_bio.tenstorrent import get_device
    device = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    torch.manual_seed(0)
    print(f"{'case':26s} {'new vs torch':>14s} {'old vs torch':>14s}  winner")
    for I in (128, 256, 512, 768):
        for name, shape in ((f"s_inputs I={I}", (1, I, 449)),
                            (f"s_trunk  I={I}", (1, I, 384)),
                            (f"z_trunk  I={I}", (1, I, I, 128))):
            t = torch.randn(*shape)
            # the port feeds bf16, so the reference arm reads the same bf16 values in fp32
            tb = t.bfloat16().float()
            ref = F.layer_norm(tb, normalized_shape=tuple(tb.shape), eps=EPS)
            a = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device,
                                dtype=ttnn.bfloat16)
            new = torch.Tensor(ttnn.to_torch(global_layer_norm(a, cfg))).float()
            old = torch.Tensor(ttnn.to_torch(old_gln(a))).float()
            rn, ro = rel_rms(new, ref), rel_rms(old, ref)
            print(f"{name:26s} {rn:14.3e} {ro:14.3e}  "
                  f"{'NEW' if rn < ro else ('OLD' if ro < rn else 'tie')}")
            ttnn.deallocate(a)


if __name__ == "__main__":
    main()
