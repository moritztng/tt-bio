#!/usr/bin/env python3
"""Two controls for tri_matmul's verdict.

C1 occupancy control. Same K (512), same dtype, same kernel config, same tile-level dataflow, but an
output big enough that the ceil-division program config engages the whole 13x10 grid instead of 64 of
it. If the rate scales with the core count, occupancy is the binding term and not K.

C2 residency control. The production shape with operands and result in L1 instead of DRAM. If DRAM
were binding, this moves; §7.3 of perfwar-trimul-kernel measured 22% at 298 aa.
"""
import json, statistics as st, sys, time
from pathlib import Path

sys.path.insert(0, "/home/ttuser/.coworker/wt/trimul-bottleneck-rootcause")
import torch, ttnn
from tt_bio import tenstorrent as T
from tt_bio.tenstorrent import get_device


def med(dev, fn, n=5, warm=2):
    out = []
    for _ in range(n):
        for _ in range(warm):
            r = fn(); del r
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        k = fn()
        ttnn.synchronize_device(dev)
        out.append(time.perf_counter() - t0)
        del k
    return st.median(out)


dev = get_device()
gx, gy = T.COMPUTE_GRID_MAIN
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                            math_approx_mode=False, fp32_dest_acc_en=True,
                                            packer_l1_acc=True)
DR, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
R = {"grid": [gx, gy], "runs": []}
torch.manual_seed(0)

# C1: batch 1, M = N = 512 * f, K = 512.  f=1 is the production geometry (Mt=Nt=16).
for f in (1, 2, 4):
    M = 512 * f
    K = 512
    at = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    bt = torch.randn(1, 1, K, M, dtype=torch.bfloat16)
    A = ttnn.from_torch(at, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=DR)
    B = ttnn.from_torch(bt, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=DR)
    Mt, Nt, Kt = M // 32, M // 32, K // 32
    pcM, pcN = -(-Mt // gy), -(-Nt // gx)
    ibw = max(d for d in range(min(10, Kt), 0, -1) if Kt % d == 0)
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy), in0_block_w=ibw, out_subblock_h=1, out_subblock_w=1,
        out_block_h=pcM, out_block_w=pcN, per_core_M=pcM, per_core_N=pcN, transpose_mcast=False,
        fused_activation=None, fuse_batch=False)
    t = med(dev, lambda: ttnn.matmul(A, B, compute_kernel_config=ckc, memory_config=DR,
                                     dtype=ttnn.bfloat16, program_config=pc))
    fl = 2.0 * M * K * M
    R["runs"].append({"ctl": "C1_occupancy", "M": M, "N": M, "K": K, "Mt": Mt, "Nt": Nt, "Kt": Kt,
                      "in0_block_w": ibw, "per_core_M": pcM, "per_core_N": pcN,
                      "cores": (-(-Mt // pcM)) * (-(-Nt // pcN)), "grid_cores": gx * gy,
                      "s": t, "tflops": fl / t / 1e12,
                      "read_MB": (M * K + K * M) * 2 / 1e6, "write_MB": M * M * 2 / 1e6})
    ttnn.deallocate(A); ttnn.deallocate(B)

# C2: production shape, operands+out in L1 vs DRAM
C, S = 32, 512
St = S // 32
pc = T._triangle_mul_program_config(St)
for tag, mc in (("DRAM", DR), ("L1", L1)):
    A = ttnn.from_torch(torch.randn(1, C, S, S, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                        device=dev, dtype=ttnn.bfloat16, memory_config=mc)
    B = ttnn.from_torch(torch.randn(1, C, S, S, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                        device=dev, dtype=ttnn.bfloat16, memory_config=mc)
    t = med(dev, lambda mc=mc: ttnn.matmul(A, B, compute_kernel_config=ckc, memory_config=mc,
                                           dtype=ttnn.bfloat16, program_config=pc))
    fl = 2.0 * C * S * S * S
    R["runs"].append({"ctl": "C2_residency", "where": tag, "s": t, "tflops": fl / t / 1e12,
                      "cores": 64, "grid_cores": gx * gy})
    ttnn.deallocate(A); ttnn.deallocate(B)

Path("perf/trimul_root/occ_qb1c0.json").write_text(json.dumps(R, indent=1))
for r in R["runs"]:
    print(r)
