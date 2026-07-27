"""Is a `core_grid=` linear's batch-exactness a property of the CALL SITE or of the SHAPE?

p14 measured `core_grid=CORE_GRID_MAIN` on RFD3's four hot pair-tensor linears at one token
count (I=250) and found two bit-exact and two not:

    [D,250,250,128] @ [128,512]   2.08x   BREAKS (2.5e-1)
    [D,250,250,128] @ [128,256]   3.17x   EXACT
    [D,250,250,512] @ [512,128]   4.17x   EXACT
    [D,250,250,128] @ [128,32]    4.83x   BREAKS

Hinting the two exact ones would be a real 1.58-2.30x at D=8 -- but only if exactness is a
property of the call site. I is fixture-dependent, so if exactness flips with I the hint is
unsafe at any call site and the only defensible rule is the structural one (K in a single tile,
where there is exactly one K-blocking and no freedom to lose).

This sweeps I and reports, per (K,N), which token counts are exact. Any flip settles it.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import ttnn  # noqa: E402
from tt_bio.tenstorrent import CORE_GRID_MAIN, get_device  # noqa: E402

# real RFD3 token counts: contigs used by the parity/bench scripts land around these
TOKENS = [40, 96, 128, 180, 250, 256, 300]
KN = [(128, 512), (128, 256), (512, 128), (128, 32)]
D = 4


def ckc():
    dev = get_device()
    cls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
           else ttnn.types.BlackholeComputeKernelConfig)
    return cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
               fp32_dest_acc_en=True, packer_l1_acc=True)


def tt(x):
    return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=get_device(),
                           dtype=ttnn.bfloat16)


def main():
    kw = dict(compute_kernel_config=ckc(), dtype=ttnn.bfloat16)
    print(f"D={D}; lane 0 of a D-batched forward must equal the D=1 forward exactly\n")
    print(f"{'K x N':>12s} | " + " ".join(f"I={i:<4d}" for i in TOKENS))
    print("-" * (15 + 7 * len(TOKENS)))
    for K, N in KN:
        cells = []
        for I in TOKENS:
            torch.manual_seed(0)
            a1 = torch.randn(1, I, I, K)
            b = tt(torch.randn(K, N))
            r1 = ttnn.to_torch(ttnn.matmul(tt(a1), b, core_grid=CORE_GRID_MAIN, **kw))
            aD = tt(a1.repeat(D, 1, 1, 1))
            rD = ttnn.to_torch(ttnn.matmul(aD, b, core_grid=CORE_GRID_MAIN, **kw))
            d = (rD.float().flatten()[:I * I * N] - r1.float().flatten()).abs().max().item()
            cells.append("EXACT" if d == 0.0 else "BREAK")
            ttnn.deallocate(aD)
            ttnn.deallocate(b)
        print(f"{K:5d} x {N:<4d} | " + " ".join(f"{c:<6s}" for c in cells))


if __name__ == "__main__":
    main()
