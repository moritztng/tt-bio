"""tt-metal PR #56057 (DIIM-448) screen, run on the installed ttnn without a tt-metal checkout.

The body of test_mm_mcast_write_ack_failure_scan, verbatim in shape, grid, program config and
compute config (upstream copy beside this file, PR head 4c6dc24d). Only the device open differs:
tt-metal's conftest mesh_device fixture becomes open_mesh_device(1x1, l1_small_size=1152) after
tt-bio's p300 mesh descriptor. A healthy chip finishes in seconds; a bad one parks the in1 sender in
noc_async_write_barrier and tt-metal's dispatch watchdog raises after WATCHDOG seconds.
    TT_VISIBLE_DEVICES=N mcast_screen.py OUT_DIR [ITERS]"""
import json, math, os, subprocess, sys, time
from pathlib import Path

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
iters = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("MATMUL_ITERS", 20))
os.environ.setdefault("TT_METAL_OPERATION_TIMEOUT_SECONDS", "30")
chip = int(os.environ["TT_VISIBLE_DEVICES"])
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn


def identity():
    node = Path(f"/sys/class/tenstorrent/tenstorrent!{chip}")
    t = {k: (node / k).read_text().strip() for k in ("tt_aiclk", "tt_asic_id", "tt_serial", "tt_card_type")
         if (node / k).exists()}
    return dict(chip=chip, bdf=(node / "device").resolve().name, dev=f"/dev/tenstorrent/{chip}", **t)


def log(**kw):
    kw["t"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = json.dumps(kw)
    print(line, flush=True)
    with open(out / "screen.jsonl", "a") as f:
        f.write(line + "\n")


log(event="start", iters=iters, watchdog=os.environ["TT_METAL_OPERATION_TIMEOUT_SECONDS"], **identity())
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=1152)
grid = mesh.compute_with_storage_grid_size()
gx, gy = (int(v) for v in os.environ.get("MATMUL_GRID", f"{grid.x - 1}x{grid.y}").split("x"))
heads = int(os.environ.get("MATMUL_HEADS", 64))
seq = int(os.environ.get("MATMUL_SEQ", 5120))
m_tiles = seq // 32
per_core_m = math.ceil(m_tiles / (gx * gy))
while m_tiles % per_core_m:
    per_core_m += 1
log(event="shape", grid=f"{grid.x}x{grid.y}", used=f"{gx}x{gy}", heads=heads, seq=seq, per_core_m=per_core_m)

dram = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
to_mesh = dict(layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=dram, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
in0 = ttnn.from_torch(torch.randn([1, heads, seq, 128]), dtype=ttnn.bfloat16, **to_mesh)
in1 = ttnn.from_torch(torch.randn([1, heads, 128, 512]), dtype=ttnn.bfloat8_b, **to_mesh)
pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
    compute_with_storage_grid_size=(gx, gy), in0_block_w=4, out_subblock_h=1, out_subblock_w=8,
    per_core_M=per_core_m, per_core_N=16, fuse_batch=False, fused_activation=None, mcast_in0=False)
ck = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2,
                                            math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True)
t0 = time.time()
i = 0
try:
    for i in range(iters):
        o = ttnn.linear(in0, in1, memory_config=dram, dtype=ttnn.bfloat16, program_config=pc, compute_kernel_config=ck)
        ttnn.synchronize_device(mesh)
        if i == 0:  # the op really ran: PCC of the first output against torch on the same (dequantized) inputs
            a, b = (ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float() for t in (in0, in1))
            got = ttnn.to_torch(o, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()
            ref = a @ b
            pcc = float(torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1])
            log(event="check", pcc=round(pcc, 6), shape=list(got.shape), max_abs_ref=round(float(ref.abs().max()), 2))
            if pcc < 0.99:
                log(event="WRONG", pcc=pcc); os._exit(2)
        ttnn.deallocate(o)
        if i % 100 == 0 or i == iters - 1:
            log(event="iter", i=i, s=round(time.time() - t0, 2), aiclk=identity().get("tt_aiclk"))
except Exception as e:
    log(event="HUNG", i=i, s=round(time.time() - t0, 2), error=str(e)[:400], **identity())
    os._exit(1)  # never close a mesh on a hung chip; the board reset clears it
log(event="PASS", iters=iters, s=round(time.time() - t0, 2), **identity())
ttnn.close_mesh_device(mesh)
