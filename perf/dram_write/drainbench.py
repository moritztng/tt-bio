"""Measure the DRAM write roof per issuing RISC and per NOC, with nothing but the write in the way.

The question this settles: the DRAM-write root-cause claimed the matmul's 25.166 MB writeback is slow
(163.9 GB/s, 59% of the 277.6 GB/s write roof) because one RISC issues it while the other idles, and
predicted 1.60x from splitting the drain across BRISC and NCRISC. ttnn.clone reaches 274.8 GB/s with a
single-RISC writer, so that mechanism is suspect; what the clone does differently is run its writer on
NOC 1 (WriterDataMovementConfig takes preferred_noc_for_dram_write, which is NOC_1 for every arch)
while the matmul's writer is built with NOC_INDEX=0.

Same bytes, same tiles per core, same kernel, four (processor, noc) pairs plus two two-RISC splits.
Source is L1 height-sharded so reads are core-local; the only NOC traffic is the write. Output is
checked bit-exact against the input on every arm, so a fast arm that lost writes cannot pass.

  python3 drainbench.py            -> one JSON line per arm
"""
import json, time, statistics as st
import torch, ttnn

pd = ttnn._ttnn.program_descriptor
NOC0 = ttnn._ttnn.types.NOC.RISCV_0_default   # NOC::NOC_0 and RISCV_0_default share the value 0
NOC1 = ttnn._ttnn.types.NOC.RISCV_1_default   # NOC::NOC_1 and RISCV_1_default share the value 1
R0 = ttnn._ttnn.types.DataMovementProcessor.RISCV_0
R1 = ttnn._ttnn.types.DataMovementProcessor.RISCV_1
KSRC = "/home/ttuser/.coworker/wt/perfwar-tworisc-drain/perf/dram_write/kernels/drain_l1_to_dram.cpp"

DEV = ttnn.open_device(device_id=0)
g = DEV.compute_with_storage_grid_size()
GX, GY = g.x, g.y
NCORES = GX * GY
TILES_PER_CORE = 96                       # the matmul writer's per-core output, 196 KB
TILE_BYTES = 32 * 32 * 2                  # bf16
ROWS = NCORES * TILES_PER_CORE * 32
BYTES = NCORES * TILES_PER_CORE * TILE_BYTES

grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GX - 1, GY - 1))])
shard = ttnn.ShardSpec(grid, [TILES_PER_CORE * 32, 32], ttnn.ShardOrientation.ROW_MAJOR)
l1_sharded = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard)

torch.manual_seed(0)
src = ttnn.from_torch(torch.randn(ROWS, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=DEV,
                      memory_config=l1_sharded)
src_ref = ttnn.to_torch(src)


def kernels(arms, bar_every, out_addr, src_addr):
    """arms: list of (processor, noc, src_off, stride, num_tiles) -> one KernelDescriptor each."""
    out = []
    for proc, noc, src_off, stride, ntiles in arms:
        rt = []
        for y in range(GY):
            for x in range(GX):
                core = ttnn.CoreCoord(x, y)
                start = (y * GX + x) * TILES_PER_CORE
                rt.append((core, pd.VectorUInt32([out_addr, src_addr, start, ntiles, src_off])))
        out.append(pd.KernelDescriptor(
            kernel_source=KSRC, core_ranges=grid,
            compile_time_args=pd.VectorUInt32([TILE_BYTES, stride, bar_every]),
            runtime_args=rt,
            config=pd.DataMovementConfigDescriptor(processor=proc, noc=noc)))
    return out


def run(name, arms, bar_every, reps=8):
    dst = ttnn.from_torch(torch.zeros(ROWS, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                          device=DEV, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ks = kernels(arms, bar_every, dst.buffer_address(), src.buffer_address())
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
    ttnn.deallocate(dst)
    print("DRAIN " + json.dumps({"arm": name, "us": round(us, 2),
                                 "gbps": round(BYTES / (us * 1e-6) / 1e9, 1), "bit_exact": exact,
                                 "MB": round(BYTES / 1e6, 3), "cores": NCORES}), flush=True)


ALL = TILES_PER_CORE
HALF = TILES_PER_CORE // 2
run("R0_NOC0_bar1", [(R0, NOC0, 0, 1, ALL)], 1)
run("R0_NOC1_bar1", [(R0, NOC1, 0, 1, ALL)], 1)
run("R1_NOC0_bar1", [(R1, NOC0, 0, 1, ALL)], 1)
run("R1_NOC1_bar1", [(R1, NOC1, 0, 1, ALL)], 1)
run("R0_NOC0_bar4", [(R0, NOC0, 0, 1, ALL)], 4)
run("R0_NOC1_bar4", [(R0, NOC1, 0, 1, ALL)], 4)
run("split_R0NOC0_R1NOC1", [(R0, NOC0, 0, 2, HALF), (R1, NOC1, 1, 2, HALF)], 1)
run("split_R0NOC1_R1NOC1", [(R0, NOC1, 0, 2, HALF), (R1, NOC1, 1, 2, HALF)], 1)
ttnn.close_device(DEV)
