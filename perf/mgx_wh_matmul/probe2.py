#!/usr/bin/env python3
"""Two follow-ups to knobs.py / element.py.

(a) throttle_level 0..5 at the band's in0_block_w: the LLK runs the HiFi phases through a different
    MOP above level 3 (`llk_math_matmul.h` `_llk_math_matmul_`, THROTTLE_LEVEL > 3 && high_fidelity).
(b) Which operand bits the miss needs, on the reduced single-row / single-column case of element.py:
    the FPU splits srcA as 5 + 3 significand bits and srcB as 7 + 1 across the HiFi phases, so
    clearing an operand's low bits removes the phases that read them. Each operand is cleared both
    ways (low 3 and low 1 bits), because which of in0 / in1 lands in srcA is what this measures.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch  # noqa: E402
import ttnn  # noqa: E402

import tt_bio.tenstorrent as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--kt", type=int, default=48)
ap.add_argument("--channels", type=int, default=8)
ap.add_argument("--out", default=None)
a = ap.parse_args()
dev = T.get_device()
kcls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
        else ttnn.types.BlackholeComputeKernelConfig)
gx, gy = T.COMPUTE_GRID_MAIN


def ckc(throttle=0, fid="HiFi4"):
    c = kcls(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=False, fp32_dest_acc_en=True,
             packer_l1_acc=True)
    c.throttle_level = list(ttnn.ThrottleLevel.__members__.values())[throttle]
    return c


def mm(A, B, w, cfg, grid=(1, 1), M=1, N=1):
    ta = ttnn.from_torch(A, layout=ttnn.TILE_LAYOUT, device=dev)
    tb = ttnn.from_torch(B, layout=ttnn.TILE_LAYOUT, device=dev)
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid, in0_block_w=w, out_subblock_h=1, out_subblock_w=1,
        out_block_h=M, out_block_w=N, per_core_M=M, per_core_N=N, transpose_mcast=False,
        fused_activation=None, fuse_batch=False)
    out = ttnn.matmul(ta, tb, compute_kernel_config=cfg, program_config=pc, dtype=ttnn.float32, transpose_b=True)
    o = ttnn.to_torch(out).double()
    for t in (ta, tb, out):
        ttnn.deallocate(t)
    return o


def clear_low(x, bits):
    """Zero the lowest `bits` significand bits of a bf16 tensor."""
    i = x.view(torch.int16)
    return (i & ~((1 << bits) - 1)).view(torch.bfloat16)


rows = []
kt, s = a.kt, a.kt * 32
g = torch.Generator().manual_seed(2000 + kt)
A = torch.randn(1, a.channels, s, s, generator=g).bfloat16()
B = torch.randn(1, a.channels, s, s, generator=g).bfloat16()
ref = torch.matmul(A.double(), B.double().transpose(-1, -2))
thr = 0.05 * ref.abs().max()
w = T._trimul_in0_block_w(kt)
M, N = -(-kt // gy), -(-kt // gx)
bad = None
for lvl in range(6):
    try:
        e = mm(A, B, w, ckc(lvl), (gx, gy), M, N) - ref
    except Exception as ex:
        r = {"arm": "throttle", "level": lvl, "error": str(ex).splitlines()[0][:160]}
    else:
        big = torch.nonzero(e.abs() > thr)
        r = {"arm": "throttle", "level": lvl, "w": w, "gross": int(big.shape[0])}
        if lvl == 0:
            bad = big[:, 1:].tolist()
    rows.append(r)
    print(json.dumps(r), flush=True)

# (b) on each wrong element, reduced to its own A row and B column inside its tile
for c, i, j in (bad or [])[:6]:
    ti, tj, li, lj = i // 32, j // 32, i % 32, j % 32
    Ar = torch.zeros(1, 1, 32, s, dtype=torch.bfloat16)
    Br = torch.zeros(1, 1, 32, s, dtype=torch.bfloat16)
    Ar[0, 0, li], Br[0, 0, lj] = A[0, c, i], B[0, c, j]
    r = {"arm": "bits", "c": c, "i": i, "j": j}
    for name, (xa, xb) in {"asis": (Ar, Br), "in0_lo3": (clear_low(Ar, 3), Br), "in0_lo1": (clear_low(Ar, 1), Br),
                           "in1_lo3": (Ar, clear_low(Br, 3)), "in1_lo1": (Ar, clear_low(Br, 1)),
                           "swap": (Br, Ar)}.items():
        o = mm(xa, xb, w, ckc(0))
        r[name] = round(float((o - torch.matmul(xa.double(), xb.double().transpose(-1, -2)))[0, 0, li, lj]), 3)
    rows.append(r)
    print(json.dumps(r), flush=True)
if a.out:
    Path(a.out).write_text(json.dumps({"arch": str(dev.arch()), "kt": kt, "rows": rows}, indent=1) + "\n")
