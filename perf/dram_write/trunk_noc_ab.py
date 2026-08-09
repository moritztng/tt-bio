"""Does the NOC misassignment cost real time on a real 298 aa trunk op, on the production pin?

Shape: the trimul output projection at N=320, [1, 320*320, 256] @ [256, 256], 52.4 MB bf16 result to
DRAM. ttnn.linear(core_grid) lands on the 1D-mcast factory whose writer is on NOC 0 (probed); the same
matmul through ttnn.experimental.minimal_matmul has its writer on NOC 1 (probed). Same arithmetic, same
bytes, different write NOC -- so the ratio between them should be the ratio between this card's two
per-NOC write roofs (171.1 GB/s NOC 0, 283.3 GB/s NOC 1) rather than anything about the matmuls.

W6 measured this pair on qb2 at ttnn 0.68.0 and shipped the swap; per warroom discipline that is a
ratio, not a number for this pin. This re-measures on qb1 card 2 at 0.67.4 and adds the bit-exactness
check, because the swap changes accumulation order and the tt-metal-side fix would not.
"""
import json, time, statistics as st
import torch, ttnn

DEV = ttnn.open_device(device_id=0)
g = DEV.compute_with_storage_grid_size()
torch.manual_seed(0)
N, CZ = 320, 256
ROWS = N * N
OUT_BYTES = ROWS * CZ * 2

a = ttnn.from_torch(torch.randn(1, ROWS, CZ), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                    device=DEV, memory_config=ttnn.DRAM_MEMORY_CONFIG)
w = ttnn.from_torch(torch.randn(CZ, CZ), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                    device=DEV, memory_config=ttnn.DRAM_MEMORY_CONFIG)
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=True)


def bench(name, fn, reps=6):
    y = fn()
    ttnn.synchronize_device(DEV)
    out = ttnn.to_torch(y).clone()
    ttnn.deallocate(y)
    for _ in range(3):
        ttnn.deallocate(fn())
    ts = []
    for _ in range(5):
        ttnn.synchronize_device(DEV)
        t0 = time.perf_counter()
        for _ in range(reps):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(DEV)
        ts.append((time.perf_counter() - t0) / reps)
    ms = st.median(ts) * 1e3
    print("TRUNK " + json.dumps({"arm": name, "ms": round(ms, 4),
                                 "write_gbps": round(OUT_BYTES / (ms * 1e-3) / 1e9, 1)}), flush=True)
    return out


lin = bench("ttnn.linear_core_grid_writerNOC0",
            lambda: ttnn.linear(a, w, core_grid=ttnn.CoreGrid(y=g.y, x=g.x),
                                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                compute_kernel_config=ckc, dtype=ttnn.bfloat16))
mm = bench("minimal_matmul_writerNOC1",
           lambda: ttnn.experimental.minimal_matmul(input_tensor=a, weight_tensor=w,
                                                    compute_kernel_config=ckc, dtype=ttnn.bfloat16))
print("TRUNK " + json.dumps({
    "bit_exact_linear_vs_minimal": torch.equal(lin, mm),
    "max_abs_diff": round((lin.float() - mm.float()).abs().max().item(), 6),
    "out_MB": round(OUT_BYTES / 1e6, 2)}), flush=True)
ttnn.close_device(DEV)
