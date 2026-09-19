#!/usr/bin/env python3
"""What precision does the card actually give the IPA, and which form of the point term survives?

Written because the obvious build of Alg. 22's point term does not fit: upstream materialises a
`[B, N, N, H, P, 3]` difference, which is 75M elements at micro-batch 8 and 256 tokens, and the
textbook fix is the identity `|q-k|^2 = |q|^2 + |k|^2 - 2 q.k`, one matmul instead of a broadcast
subtract. That trade is only safe if the matmul carries enough mantissa to survive the
cancellation: the points are global coordinates in Angstrom, so the terms reach ~2e4 while the
distances that matter are ~1.

Three questions, one device context:

1. What relative error does a matmul carry on fp32 inputs, across kernel config and dtype?
2. Is that error input rounding or accumulation? A K-ladder answers it: K=1 has no accumulation.
3. Does the explicit eltwise difference carry the point term, and at what logit error?

Run: TT_VISIBLE_DEVICES=<card> TT_BIO_LEASE_CARDS=<card> PYTHONPATH=$PWD python3 \
        scripts/abb3_port/precision_probe.py
"""
from __future__ import annotations

import torch
import ttnn

from tt_bio.tenstorrent import get_device

F32, BF16 = ttnn.float32, ttnn.bfloat16
FIDELITY = {"HiFi4": ttnn.MathFidelity.HiFi4, "HiFi2": ttnn.MathFidelity.HiFi2,
            "LoFi": ttnn.MathFidelity.LoFi}


def _cfg(fid, acc: bool):
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=acc, packer_l1_acc=acc)


def _tt(dev, x: torch.Tensor, dtype=F32):
    return ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=dtype)


def matmul_grid(dev) -> None:
    torch.manual_seed(0)
    a = torch.randn(256, 32, dtype=torch.float64)
    b = torch.randn(32, 256, dtype=torch.float64)
    ref = a @ b
    scale = ref.abs().max().item()
    print(f"1. matmul [256,32]x[32,256], |ref|max {scale:.3f}, fp32 would be "
          f"{1.2e-7 * scale:.1e} absolute")
    for fname, fid in FIDELITY.items():
        for acc in (True, False):
            for dt, dname in ((F32, "f32 "), (BF16, "bf16")):
                out = ttnn.to_torch(ttnn.matmul(_tt(dev, a, dt), _tt(dev, b, dt), dtype=dt,
                                                compute_kernel_config=_cfg(fid, acc))).double()
                err = (out - ref).abs().max().item()
                print(f"   {fname:<5} fp32_dest_acc={int(acc)} {dname}  max abs {err:.3e}"
                      f"  rel {err / scale:.2e}")


def k_ladder(dev) -> None:
    print("\n2. K ladder at HiFi4 on fp32 inputs. Flat in K means input rounding, not accumulation")
    torch.manual_seed(0)
    for k in (1, 2, 8, 32, 128, 512):
        a = torch.randn(256, k, dtype=torch.float64)
        b = torch.randn(k, 256, dtype=torch.float64)
        ref = a @ b
        out = ttnn.to_torch(ttnn.matmul(_tt(dev, a), _tt(dev, b), dtype=F32,
                                        compute_kernel_config=_cfg(FIDELITY["HiFi4"], True))
                            ).double()
        rel = ((out - ref).abs() / ref.abs().clamp(min=1e-12)).median().item()
        print(f"   K={k:<4} median relative {rel:.2e}   "
              f"(bf16 eps 3.9e-3, tf32 eps 1.0e-3, fp32 eps 1.2e-7)")


def point_term(dev) -> None:
    """Both forms of the point term against float64, at the magnitudes the model runs at."""
    print("\n3. the point term, both forms, against float64 at realistic magnitudes")
    torch.manual_seed(0)
    n = 256
    trans = torch.randn(n, 3, dtype=torch.float64) * 12.0       # translations, Angstrom
    q = trans[:, None, :] + torch.randn(n, 4, 3, dtype=torch.float64) * 2.0
    k = trans[:, None, :] + torch.randn(n, 4, 3, dtype=torch.float64) * 2.0
    ref = (q[:, None] - k[None, :]).square().sum(-1).sum(-1)
    near = ref < 100.0
    # softplus(0) * sqrt(1/(3 * P * 9/2)): the head weight at the released initialisation, which
    # is what turns a squared-distance error into a logit error.
    w = 0.693147 * (1.0 / (3 * (4 * 9.0 / 2))) ** 0.5

    cfg = _cfg(FIDELITY["HiFi4"], True)
    for centred in (False, True):
        off = trans.mean(0) if centred else torch.zeros(3, dtype=torch.float64)
        qf = (q - off).reshape(1, 1, n, 12)
        kf = (k - off).reshape(1, 1, n, 12)
        qsq = ttnn.to_torch(ttnn.sum(ttnn.multiply(_tt(dev, qf), _tt(dev, qf)), dim=-1,
                                     keepdim=True)).double()
        ksq = ttnn.to_torch(ttnn.sum(ttnn.multiply(_tt(dev, kf), _tt(dev, kf)), dim=-1,
                                     keepdim=True)).double()
        cross = ttnn.to_torch(ttnn.matmul(_tt(dev, qf), _tt(dev, kf), transpose_b=True,
                                          dtype=F32, compute_kernel_config=cfg)).double()
        got = (qsq + ksq.transpose(-1, -2) - 2.0 * cross)[0, 0]
        err = (got - ref).abs()
        print(f"   identity{', centred' if centred else '        '}: max abs {err.max():.3e},"
              f" {err[near].max():.3e} on the {int(near.sum())} pairs under 100 A^2"
              f" -> logit error {0.5 * w * err[near].max():.2e} near")

    # The explicit form: 12 two-sided broadcast subtracts, squared and accumulated. Every op is
    # eltwise, and eltwise on this card is fp32 (measured 3.0e-07 on the subtract itself).
    zero = torch.zeros(1, n, n, dtype=torch.float64)
    acc = None
    for p in range(4):
        for d in range(3):
            diff = ttnn.subtract(_tt(dev, zero + q[:, p, d].reshape(1, n, 1)),
                                 _tt(dev, k[:, p, d].reshape(1, 1, n)))
            sq = ttnn.multiply(diff, diff)
            acc = sq if acc is None else ttnn.add(acc, sq)
    err = (ttnn.to_torch(acc)[0].double() - ref).abs()
    print(f"   explicit           : max abs {err.max():.3e},"
          f" {err[near].max():.3e} on the same pairs"
          f" -> logit error {0.5 * w * err[near].max():.2e} near")
    print(f"   d2 range {ref.min():.2f}..{ref.max():.0f} A^2, head weight at init {w:.4f}")


def main() -> int:
    dev = get_device()
    print(f"arch {dev.arch()}")
    try:
        matmul_grid(dev)
        k_ladder(dev)
        point_term(dev)
    finally:
        ttnn.close_device(dev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
