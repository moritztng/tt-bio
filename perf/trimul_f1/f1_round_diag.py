#!/usr/bin/env python3
"""Which rounding point in F1's gate epilogue misses ttnn, and by how much.

F1 runs fp32 DST because its GEMM must, so every value that production rounds to bf16 through a
16-bit DST store has to be rounded explicitly here. Three legs, all against the same reference:

    round=0   no explicit round, the packer does it (E6 measured this wrong: ties away from zero)
    round=1   SFPSTOCHRND round-to-nearest-even
    round=2   the same rounding written out in integer arithmetic

and one leg with the gate dropped, scored against `ttnn.multiply(p, g)` with no activation, which
separates a sigmoid miss from a multiply miss the way E6's `skip_sigmoid` did.
"""
import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tt_bio.mm_generic as MG                                             # noqa: E402
import tt_bio.trimul_tail as F1                                           # noqa: E402
from tt_bio.tenstorrent import COMPUTE_GRID_MAIN, get_device              # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
CZ = 256

dev = get_device()
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
gx, gy = COMPUTE_GRID_MAIN
cfg = ttnn.MinimalMatmulConfig(
    M_block_size=F1.BLOCK[0], K_block_size=F1.BLOCK[1], N_block_size=F1.BLOCK[2],
    subblock_h=F1.BLOCK[3], subblock_w=F1.BLOCK[4],
    compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy))

torch.manual_seed(7)
to = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
xa, xb = to(torch.randn(1, N, N, CZ)), to(torch.randn(1, N, N, CZ))
wa, wb = to(torch.randn(CZ, CZ) * 0.05), to(torch.randn(CZ, CZ) * 0.05)

mm = lambda x, w: ttnn.experimental.minimal_matmul(
    input_tensor=x, weight_tensor=w, compute_kernel_config=ckc, dtype=ttnn.bfloat16, config=cfg)


def reference(sigmoid: bool):
    p, g = mm(xa, wa), mm(xb, wb)
    acts = [ttnn.UnaryOpType.SIGMOID] if sigmoid else []
    return ttnn.to_torch(ttnn.multiply_(p, g, input_tensor_b_activations=acts))


def leg(name, rnd, skip_sig):
    F1.ROUND, F1.SKIP_SIGMOID = rnd, skip_sig
    ref = reference(sigmoid=not skip_sig)
    got = ttnn.to_torch(F1.fused_tail(xa, xb, wa, wb, MG.ckc_args(ckc), (gx, gy)))
    d = (got.float() - ref.float()).abs()
    n = int((d > 0).sum())
    print(f"{name:26s} torch.equal={bool(torch.equal(got, ref))!s:5s} "
          f"mismatched={n:9d}/{d.numel()} ({100.0 * n / d.numel():6.3f}%) "
          f"max_abs_diff={float(d.max()):.3e}")
    return n


print(f"N={N}, qb1 card 3, ttnn 0.67.4, HiFi4 + fp32_dest_acc + packer_l1_acc")
leg("round=0 packer", 0, 0)
leg("round=1 SFPSTOCHRND even", 1, 0)
leg("round=2 integer RNE", 2, 0)
leg("round=1 no sigmoid", 1, 1)
leg("round=0 no sigmoid", 0, 1)
