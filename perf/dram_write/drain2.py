"""Round 2: complete the {1 RISC, 2 RISCs} x {NOC 0, NOC 1} square, and sweep barrier depth.

Round 1 showed the issuing RISC is irrelevant (R0/NOC0 152.4 vs R1/NOC0 151.6 GB/s; R0/NOC1 252.8 vs
R1/NOC1 255.1) and the NOC is worth 1.62x. This asks whether a second RISC adds anything once both
are on the same NOC -- if the resource is the NOC, two RISCs on one NOC should match one RISC on that
NOC, and only a NOC-0 + NOC-1 pair should beat it.
"""
import json, time, statistics as st
import torch, ttnn

pd = ttnn._ttnn.program_descriptor
NOC0 = ttnn._ttnn.types.NOC.RISCV_0_default
NOC1 = ttnn._ttnn.types.NOC.RISCV_1_default
R0 = ttnn._ttnn.types.DataMovementProcessor.RISCV_0
R1 = ttnn._ttnn.types.DataMovementProcessor.RISCV_1
KSRC = "/home/ttuser/.coworker/wt/perfwar-tworisc-drain/perf/dram_write/kernels/drain_l1_to_dram.cpp"

DEV = ttnn.open_device(device_id=0)
g = DEV.compute_with_storage_grid_size()
GX, GY = g.x, g.y
NCORES = GX * GY
TILES_PER_CORE = 96
TILE_BYTES = 32 * 32 * 2
ROWS = NCORES * TILES_PER_CORE * 32
BYTES = NCORES * TILES_PER_CORE * TILE_BYTES

grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GX - 1, GY - 1))])
shard = ttnn.ShardSpec(grid, [TILES_PER_CORE * 32, 32], ttnn.ShardOrientation.ROW_MAJOR)
l1_sharded = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard)

torch.manual_seed(0)
src = ttnn.from_torch(torch.randn(ROWS, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=DEV,
                      memory_config=l1_sharded)
src_ref = ttnn.to_torch(src)
dst = ttnn.from_torch(torch.zeros(ROWS, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=DEV,
                      memory_config=ttnn.DRAM_MEMORY_CONFIG)
OUT_ADDR, SRC_ADDR = dst.buffer_address(), src.buffer_address()


def run(name, arms, bar_every, reps=8):
    ks = []
    for proc, noc, src_off, stride, ntiles in arms:
        rt = [(ttnn.CoreCoord(x, y),
               pd.VectorUInt32([OUT_ADDR, SRC_ADDR, (y * GX + x) * TILES_PER_CORE, ntiles, src_off]))
              for y in range(GY) for x in range(GX)]
        ks.append(pd.KernelDescriptor(
            kernel_source=KSRC, core_ranges=grid,
            compile_time_args=pd.VectorUInt32([TILE_BYTES, stride, bar_every]),
            runtime_args=rt,
            config=pd.DataMovementConfigDescriptor(processor=proc, noc=noc)))
    prog = pd.ProgramDescriptor(kernels=ks, semaphores=[], cbs=[])
    ttnn.generic_op([src, dst], prog)
    ttnn.synchronize_device(DEV)
    exact = torch.equal(ttnn.to_torch(dst), src_ref)
    for _ in range(3):
        ttnn.generic_op([src, dst], prog)
    ts = []
    for _ in range(5):
        ttnn.synchronize_device(DEV)
        t0 = time.perf_counter()
        for _ in range(reps):
            ttnn.generic_op([src, dst], prog)
        ttnn.synchronize_device(DEV)
        ts.append((time.perf_counter() - t0) / reps)
    us = st.median(ts) * 1e6
    print("DRAIN " + json.dumps({"arm": name, "bar": bar_every, "us": round(us, 2),
                                 "gbps": round(BYTES / (us * 1e-6) / 1e9, 1), "bit_exact": exact}),
          flush=True)


A, H = TILES_PER_CORE, TILES_PER_CORE // 2



run("1RISC_NOC1_bar16", [(R0, NOC1, 0, 1, A)], 16)
run("1RISC_NOC0_bar16", [(R0, NOC0, 0, 1, A)], 16)
ttnn.close_device(DEV)
