"""Which ttnn primitive is not bit-exact across batch size?

Every op below gets identical data replicated into every batch lane. A
batch-invariant op must return, for lane i at B=D, exactly what it returns at
B=1 -- the mathematical inputs are identical, so any difference is the op's
tiling/blocking changing with the leading (M) dimension.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import ttnn  # noqa: E402
from tt_bio.tenstorrent import CORE_GRID_MAIN, get_device  # noqa: E402

L, C, NH, HD = 419, 128, 4, 32
BATCHES = [2, 4, 8]


def ckc():
    dev = get_device()
    cls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
           else ttnn.types.BlackholeComputeKernelConfig)
    return cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
               fp32_dest_acc_en=True, packer_l1_acc=True)


def tt(x, dtype=ttnn.bfloat16):
    return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=get_device(), dtype=dtype)


def report(name, run, make_inputs):
    """run(*dev_inputs) -> device tensor; make_inputs(D) -> list of host tensors."""
    base = make_inputs(1)
    out1 = ttnn.to_torch(run(*[tt(x) for x in base])).float()
    line = [f"{name:38s}"]
    for D in BATCHES:
        rep = [x.expand(D, *([-1] * (x.ndim - 1))).contiguous() if x.shape[0] == 1 else x
               for x in base]
        outD = ttnn.to_torch(run(*[tt(x) for x in rep])).float()
        ma = float((outD[0] - out1[0]).abs().max())
        line.append(f"D={D}:{ma:.3e}")
    print("  ".join(line))


def main():
    K = ckc()
    torch.manual_seed(0)
    w = torch.randn(C, C) * 0.05
    w_dev = tt(w)
    ln_w = torch.randn(C).abs()
    ln_dev = tt(ln_w)

    print("op                                        maxabs(lane0@B=D - out@B=1)")

    report("linear[B,L,C]@[C,C] core_grid=MAIN",
           lambda x: ttnn.linear(x, w_dev, compute_kernel_config=K,
                                 dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN),
           lambda D: [torch.randn(1, L, C)])

    report("linear[B,L,C]@[C,C] no core_grid",
           lambda x: ttnn.linear(x, w_dev, compute_kernel_config=K, dtype=ttnn.bfloat16),
           lambda D: [torch.randn(1, L, C)])

    report("rms_norm[B,L,C]",
           lambda x: ttnn.rms_norm(x, weight=ln_dev, epsilon=1e-6, compute_kernel_config=K),
           lambda D: [torch.randn(1, L, C)])

    report("matmul QK [B,H,L,HD]@[B,H,HD,L]",
           lambda q, k: ttnn.matmul(q, ttnn.permute(k, (0, 1, 3, 2)), compute_kernel_config=K),
           lambda D: [torch.randn(1, NH, L, HD), torch.randn(1, NH, L, HD)])

    report("matmul AV [B,H,L,L]@[B,H,L,HD]",
           lambda a, v: ttnn.matmul(a, v, compute_kernel_config=K, dtype=ttnn.bfloat16),
           lambda D: [torch.randn(1, NH, L, L).abs(), torch.randn(1, NH, L, HD)])

    report("softmax[B,H,L,L] dim=-1",
           lambda s: ttnn.softmax(s, dim=-1),
           lambda D: [torch.randn(1, NH, L, L)])

    report("sigmoid/multiply[B,L,C]",
           lambda a, b: ttnn.multiply(a, ttnn.sigmoid(b)),
           lambda D: [torch.randn(1, L, C), torch.randn(1, L, C)])

    # GatedCrossAttention folds batch into the leading matmul dim as B*T.
    report("linear[B*T,Q,C]@[C,C] core_grid=MAIN",
           lambda x: ttnn.linear(x, w_dev, compute_kernel_config=K,
                                 dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN),
           lambda D: [torch.randn(1, 40, 24, C)])


if __name__ == "__main__":
    main()
