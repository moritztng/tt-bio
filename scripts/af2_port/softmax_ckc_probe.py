"""`ttnn.softmax` loses up to 2.9e-2 on a confident row unless it is given a compute kernel config.

Found while planning AF2's MSA column attention, which softmaxes over an MSA depth of 2 and so
sits in the worst regime by construction. The error is keyed on the exp-sum, not the width: it is
worst when the sum is at or just above a power of two, which is where a peaky softmax always lands
(sum -> 1). Passing the trunk's own HiFi4 / fp32_dest_acc config drops it 10-60x.

`_fp32_softmax_tail` (tenstorrent.py) passes no config, so this is live on every model that takes
the fp32-softmax path. Fixing it moves shipped numbers, so it is release-gated: run this, then A/B
the real sites through `device_gate.py`, and leave the default alone.

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:<slug> PYTHONPATH=. \
        env/bin/python3 scripts/af2_port/softmax_ckc_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ttnn  # noqa: E402

from tt_bio.tenstorrent import get_device  # noqa: E402


def kernel_config() -> ttnn.DeviceComputeKernelConfig:
    device = get_device()
    cls = (ttnn.types.WormholeComputeKernelConfig
           if device.arch() == ttnn.Arch.WORMHOLE_B0
           else ttnn.types.BlackholeComputeKernelConfig)
    return cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
               fp32_dest_acc_en=True, packer_l1_acc=True)


def softmax(x: torch.Tensor, **kw) -> torch.Tensor:
    tt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=get_device(), dtype=ttnn.float32)
    return torch.Tensor(ttnn.to_torch(ttnn.softmax(tt, dim=-1, **kw))).float()


def main() -> int:
    ckc = kernel_config()

    print("all-equal logits, so the answer is exactly 1/w and the exp-sum is exactly w")
    print(f"  {'w':>5} {'rel err':>11} {'rel err + ckc':>15}")
    for w in (2, 3, 4, 8, 16, 32, 64, 208):
        x = torch.zeros(1, 8, 32, w, dtype=torch.float32)
        p0 = float(softmax(x)[0, 0, 0, 0]) * w - 1.0
        p1 = float(softmax(x, compute_kernel_config=ckc)[0, 0, 0, 0]) * w - 1.0
        print(f"  {w:5d} {p0:+11.3e} {p1:+15.3e}")

    print()
    print("fixed width 208, peakiness swept: one dominant logit `gap` above 207 zeros")
    print(f"  {'gap':>6} {'exp-sum':>9} {'max abs err':>12} {'+ ckc':>12}")
    for gap in (50.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.05, 0.0):
        x = torch.zeros(1, 8, 32, 208, dtype=torch.float32)
        x[..., 0] = gap
        want = torch.softmax(x.double(), dim=-1)
        total = float(torch.exp(x[0, 0, 0].double() - gap).sum())
        e0 = float((softmax(x).double() - want).abs().max())
        e1 = float((softmax(x, compute_kernel_config=ckc).double() - want).abs().max())
        print(f"  {gap:6.2f} {total:9.4f} {e0:12.3e} {e1:12.3e}")

    print()
    print("the padding is already masked: an explicit -1e9 fill changes nothing, a 0 fill collapses")
    torch.manual_seed(0)
    x = (torch.randn(1, 8, 32, 2) * 3.0).float()
    want = torch.softmax(x.double(), dim=-1)
    for fill, name in ((None, "implicit (logical width 2)"), (-1e9, "explicit -1e9"),
                       (0.0, "explicit 0")):
        if fill is None:
            got = softmax(x)
        else:
            wide = torch.full((1, 8, 32, 32), fill, dtype=torch.float32)
            wide[..., :2] = x
            got = softmax(wide)[..., :2]
        print(f"  {name:28s} max abs err={float((got.double() - want).abs().max()):.3e}  "
              f"rowsum={float(got[0, 0, 0].sum()):.6f}")

    print()
    print("softmax_in_place takes the same argument (this is the call _fp32_softmax_tail makes)")
    x = torch.zeros(1, 8, 32, 32, dtype=torch.float32)
    tt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=get_device(), dtype=ttnn.float32)
    for kw, name in (({}, "default"), ({"compute_kernel_config": ckc}, "with ckc")):
        got = torch.Tensor(ttnn.to_torch(ttnn.softmax_in_place(ttnn.clone(tt), **kw))).float()
        print(f"  {name:12s} rel err={float(got[0, 0, 0, 0]) * 32 - 1.0:+.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
