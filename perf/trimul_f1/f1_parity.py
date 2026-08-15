#!/usr/bin/env python3
"""F1's isolation gate: `torch.equal` against the exact three ops it replaces, at the fold's shape
and at least eight others.

The reference is `_trimul_out_proj` itself, the function the caller falls through to when
`fused_tail` returns None, and not a `minimal_matmul` configured to match F1. Those are the same
thing at c_z = 256 and they are NOT the same thing at c_z = 384: `_pair_proj_minimal_matmul` hard
-guards `k_tiles == 8`, so a wider pair track leaves the `minimal_matmul` path entirely and lands on
`ttnn.linear`, which folds the contraction differently. Scoring F1 against a config-matched
`minimal_matmul` would have called that bit-exact.

Anything short of `torch.equal` is dead on the spot and is not released to a timed leg.
"""
import argparse
import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tt_bio.mm_generic as MG                                             # noqa: E402
import tt_bio.tenstorrent as T                                            # noqa: E402
import tt_bio.trimul_tail as F1                                           # noqa: E402
from tt_bio.tenstorrent import get_device                                 # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cz", type=int, default=256, help="pair channel width; 256 protenix-v2, 384 opendde")
ap.add_argument("sizes", type=int, nargs="*",
                default=[512, 32, 64, 96, 128, 160, 224, 288, 384, 448, 576])
a = ap.parse_args()
CZ = a.cz

dev = get_device()
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
# Read the grid AFTER get_device: `get_device` widens COMPUTE_GRID_MAIN to the device's actual
# compute grid, so a name bound at import time is the stale 11-column declaration.
gx, gy = T.COMPUTE_GRID_MAIN
kt = -(-CZ // 32)
print(f"grid {gx}x{gy}  c_z={CZ}  key=({kt},{kt})  F1 block={F1._block_for(kt, kt)}  "
      f"ttnn {__import__('importlib.metadata', fromlist=['x']).version('ttnn')}")

fails = 0
for i, N in enumerate(a.sizes):
    torch.manual_seed(100 + i)
    to = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    xa = to(torch.randn(1, N, N, CZ))
    xb = to(torch.randn(1, N, N, CZ))
    wa = to(torch.randn(CZ, CZ) * 0.05)
    wb = to(torch.randn(CZ, CZ) * 0.05)

    if i == 0:
        probe = T._pair_proj_minimal_matmul(xa, wa, ckc, ttnn.bfloat16)
        print(f"production tail path: {'minimal_matmul' if probe is not None else 'ttnn.linear'}"
              f"  (_pair_proj_minimal_matmul {'served' if probe is not None else 'returned None'})")
        if probe is not None:
            ttnn.deallocate(probe)

    p = T._trimul_out_proj(xa, wa, ckc)
    g = T._trimul_out_proj(xb, wb, ckc)
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

print(f"served={F1.STATS[0]} declined={F1.STATS[1]} rejects={F1.REJECTS}")
print("VERDICT:", "BIT-EXACT AT EVERY SHAPE" if fails == 0 else f"FAILED AT {fails} SHAPES")
sys.exit(0 if fails == 0 else 1)
