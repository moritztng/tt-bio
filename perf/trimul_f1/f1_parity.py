#!/usr/bin/env python3
"""F1's isolation gate (state/protenix-beat-b200.md 6.4.1): `torch.equal` against the exact three
ops it replaces, at the 512 aa shape and at least eight others.

Reference is production's own sequence, not a torch model:

    p = minimal_matmul(xa, Wp)              # _pair_proj_minimal_matmul
    g = minimal_matmul(xb, Wg)
    out = multiply_(p, g, SIGMOID on b)

Anything short of `torch.equal` is dead on the spot and is not released to a timed leg.
"""
import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tt_bio.mm_generic as MG                                             # noqa: E402
import tt_bio.tenstorrent as T                                            # noqa: E402
import tt_bio.trimul_tail as F1                                           # noqa: E402
from tt_bio.tenstorrent import get_device                                 # noqa: E402

CZ = 256
SHAPES = [int(a) for a in sys.argv[1:]] or [512, 32, 64, 96, 128, 160, 224, 288, 384, 448, 576]

dev = get_device()
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
# Read the grid AFTER get_device: `get_device` widens COMPUTE_GRID_MAIN to the device's actual
# compute grid, so a name bound at import time is the stale 11-column declaration.
gx, gy = T.COMPUTE_GRID_MAIN
print(f"grid {gx}x{gy}  ttnn {__import__('importlib.metadata', fromlist=['x']).version('ttnn')}")
cfg = ttnn.MinimalMatmulConfig(
    M_block_size=F1.BLOCK[0], K_block_size=F1.BLOCK[1], N_block_size=F1.BLOCK[2],
    subblock_h=F1.BLOCK[3], subblock_w=F1.BLOCK[4],
    compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy))

fails = 0
for i, N in enumerate(SHAPES):
    torch.manual_seed(100 + i)
    to = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    xa = to(torch.randn(1, N, N, CZ))
    xb = to(torch.randn(1, N, N, CZ))
    wa = to(torch.randn(CZ, CZ) * 0.05)
    wb = to(torch.randn(CZ, CZ) * 0.05)

    mm = lambda x, w: ttnn.experimental.minimal_matmul(
        input_tensor=x, weight_tensor=w, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
        config=cfg)
    p, g = mm(xa, wa), mm(xb, wb)
    ref = ttnn.to_torch(
        ttnn.multiply_(p, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]))

    try:
        out = F1.fused_tail(xa, xb, wa, wb, MG.ckc_args(ckc), (gx, gy))
    except Exception as e:                                                # noqa: BLE001
        print(f"N={N:4d}  RAISED {type(e).__name__}: {str(e)[:400]}")
        fails += 1
        continue
    if out is None:
        print(f"N={N:4d}  DECLINED (out of eligible scope)")
        continue

    got = ttnn.to_torch(out)
    eq = bool(torch.equal(got, ref))
    d = (got.float() - ref.float()).abs()
    nmis = int((d > 0).sum())
    print(f"N={N:4d}  torch.equal={eq}  mismatched={nmis}/{d.numel()} "
          f"({100.0 * nmis / d.numel():.3f}%)  max_abs_diff={float(d.max()):.3e}")
    fails += 0 if eq else 1

print(f"served={F1.STATS[0]} declined={F1.STATS[1]}")
print("VERDICT:", "BIT-EXACT AT EVERY SHAPE" if fails == 0 else f"FAILED AT {fails} SHAPES")
sys.exit(0 if fails == 0 else 1)
